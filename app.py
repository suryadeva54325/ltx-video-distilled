import gradio as gr
import torch
import numpy as np
import random
import os
import yaml
from pathlib import Path
import imageio
import tempfile
from PIL import Image
from huggingface_hub import hf_hub_download
import shutil

# --- Import necessary classes from the provided files ---
from inference import (
    create_ltx_video_pipeline,
    create_latent_upsampler,
    load_image_to_tensor_with_resize_and_crop,
    seed_everething,
    get_device,
    calculate_padding,
    load_media_file
)
from ltx_video.pipelines.pipeline_ltx_video import ConditioningItem, LTXMultiScalePipeline, LTXVideoPipeline
from ltx_video.utils.skip_layer_strategy import SkipLayerStrategy

# --- Global constants from user's request and YAML ---
YAML_CONFIG_STRING = """
pipeline_type: multi-scale
checkpoint_path: "ltxv-13b-0.9.7-distilled.safetensors" # This will be replaced by the rc3 version
downscale_factor: 0.6666666
spatial_upscaler_model_path: "ltxv-spatial-upscaler-0.9.7.safetensors"
stg_mode: "attention_values" # options: "attention_values", "attention_skip", "residual", "transformer_block"
decode_timestep: 0.05
decode_noise_scale: 0.025
text_encoder_model_name_or_path: "PixArt-alpha/PixArt-XL-2-1024-MS"
precision: "bfloat16"
sampler: "from_checkpoint" # options: "uniform", "linear-quadratic", "from_checkpoint"
prompt_enhancement_words_threshold: 120
prompt_enhancer_image_caption_model_name_or_path: "MiaoshouAI/Florence-2-large-PromptGen-v2.0"
prompt_enhancer_llm_model_name_or_path: "unsloth/Llama-3.2-3B-Instruct"
stochastic_sampling: false

first_pass:
  timesteps: [1.0000, 0.9937, 0.9875, 0.9812, 0.9750, 0.9094, 0.7250]
  guidance_scale: 1
  stg_scale: 0
  rescaling_scale: 1
  skip_block_list: [42]

second_pass:
  timesteps: [0.9094, 0.7250, 0.4219]
  guidance_scale: 1
  stg_scale: 0
  rescaling_scale: 1
  skip_block_list: [42]
"""
PIPELINE_CONFIG_YAML = yaml.safe_load(YAML_CONFIG_STRING)

# Model specific paths (to be downloaded)
DISTILLED_MODEL_REPO = "LTX-Colab/LTX-Video-Preview"
DISTILLED_MODEL_FILENAME = "ltxv-13b-0.9.7-distilled-rc3.safetensors"

UPSCALER_REPO = "Lightricks/LTX-Video"

MAX_IMAGE_SIZE = PIPELINE_CONFIG_YAML.get("max_resolution", 1280)
MAX_NUM_FRAMES = 257

# --- Global variables for loaded models ---
pipeline_instance = None
latent_upsampler_instance = None
models_dir = "downloaded_models_gradio_cpu_init"
Path(models_dir).mkdir(parents=True, exist_ok=True)

print("Downloading models (if not present)...")
distilled_model_actual_path = hf_hub_download(
    repo_id=DISTILLED_MODEL_REPO,
    filename=DISTILLED_MODEL_FILENAME,
    local_dir=models_dir,
    local_dir_use_symlinks=False
)
PIPELINE_CONFIG_YAML["checkpoint_path"] = distilled_model_actual_path
print(f"Distilled model path: {distilled_model_actual_path}")

SPATIAL_UPSCALER_FILENAME = PIPELINE_CONFIG_YAML["spatial_upscaler_model_path"]
spatial_upscaler_actual_path = hf_hub_download(
    repo_id=UPSCALER_REPO,
    filename=SPATIAL_UPSCALER_FILENAME,
    local_dir=models_dir,
    local_dir_use_symlinks=False
)
PIPELINE_CONFIG_YAML["spatial_upscaler_model_path"] = spatial_upscaler_actual_path
print(f"Spatial upscaler model path: {spatial_upscaler_actual_path}")

print("Creating LTX Video pipeline on CPU...")
pipeline_instance = create_ltx_video_pipeline(
    ckpt_path=PIPELINE_CONFIG_YAML["checkpoint_path"],
    precision=PIPELINE_CONFIG_YAML["precision"],
    text_encoder_model_name_or_path=PIPELINE_CONFIG_YAML["text_encoder_model_name_or_path"],
    sampler=PIPELINE_CONFIG_YAML["sampler"],
    device="cpu",
    enhance_prompt=False,
    prompt_enhancer_image_caption_model_name_or_path=PIPELINE_CONFIG_YAML["prompt_enhancer_image_caption_model_name_or_path"],
    prompt_enhancer_llm_model_name_or_path=PIPELINE_CONFIG_YAML["prompt_enhancer_llm_model_name_or_path"],
)
print("LTX Video pipeline created on CPU.")

if PIPELINE_CONFIG_YAML.get("spatial_upscaler_model_path"):
    print("Creating latent upsampler on CPU...")
    latent_upsampler_instance = create_latent_upsampler(
        PIPELINE_CONFIG_YAML["spatial_upscaler_model_path"],
        device="cpu"
    )
    print("Latent upsampler created on CPU.")


def generate(prompt, negative_prompt, input_image_filepath, input_video_filepath,
             height_ui, width_ui, mode,
             ui_steps, num_frames_ui,
             ui_frames_to_use,
             seed_ui, randomize_seed, ui_guidance_scale, improve_texture_flag,
             progress=gr.Progress(track_ τότε=True)):

    target_inference_device = get_device()
    print(f"Target inference device: {target_inference_device}")

    if randomize_seed:
        seed_ui = random.randint(0, 2**32 - 1)
    seed_everething(int(seed_ui))
    
    actual_height = int(height_ui)
    actual_width = int(width_ui)
    actual_num_frames = int(num_frames_ui)

    height_padded = ((actual_height - 1) // 32 + 1) * 32
    width_padded = ((actual_width - 1) // 32 + 1) * 32
    num_frames_padded = ((actual_num_frames - 2) // 8 + 1) * 8 + 1
    
    padding_values = calculate_padding(actual_height, actual_width, height_padded, width_padded)

    call_kwargs = {
        "prompt": prompt,
        "negative_prompt": negative_prompt,
        "height": height_padded,
        "width": width_padded,
        "num_frames": num_frames_padded,
        "frame_rate": 30, 
        "generator": torch.Generator(device=target_inference_device).manual_seed(int(seed_ui)),
        "output_type": "pt", # Crucial: pipeline will output [0,1] range tensors
        "conditioning_items": None,
        "media_items": None,
        "decode_timestep": PIPELINE_CONFIG_YAML["decode_timestep"],
        "decode_noise_scale": PIPELINE_CONFIG_YAML["decode_noise_scale"],
        "stochastic_sampling": PIPELINE_CONFIG_YAML["stochastic_sampling"],
        "image_cond_noise_scale": 0.15,
        "is_video": True,
        "vae_per_channel_normalize": True,
        "mixed_precision": (PIPELINE_CONFIG_YAML["precision"] == "mixed_precision"),
        "offload_to_cpu": False,
        "enhance_prompt": False,
    }

    stg_mode_str = PIPELINE_CONFIG_YAML.get("stg_mode", "attention_values")
    if stg_mode_str.lower() in ["stg_av", "attention_values"]:
        call_kwargs["skip_layer_strategy"] = SkipLayerStrategy.AttentionValues
    elif stg_mode_str.lower() in ["stg_as", "attention_skip"]:
        call_kwargs["skip_layer_strategy"] = SkipLayerStrategy.AttentionSkip
    elif stg_mode_str.lower() in ["stg_r", "residual"]:
        call_kwargs["skip_layer_strategy"] = SkipLayerStrategy.Residual
    elif stg_mode_str.lower() in ["stg_t", "transformer_block"]:
        call_kwargs["skip_layer_strategy"] = SkipLayerStrategy.TransformerBlock
    else:
        raise ValueError(f"Invalid stg_mode: {stg_mode_str}")

    if mode == "image-to-video" and input_image_filepath:
        try:
            media_tensor = load_image_to_tensor_with_resize_and_crop(
                input_image_filepath, actual_height, actual_width
            )
            media_tensor = torch.nn.functional.pad(media_tensor, padding_values)
            call_kwargs["conditioning_items"] = [ConditioningItem(media_tensor.to(target_inference_device), 0, 1.0)]
        except Exception as e:
            print(f"Error loading image {input_image_filepath}: {e}")
            raise gr.Error(f"Could not load image: {e}")
    elif mode == "video-to-video" and input_video_filepath:
        try:
            call_kwargs["media_items"] = load_media_file(
                media_path=input_video_filepath,
                height=actual_height, 
                width=actual_width,
                max_frames=int(ui_frames_to_use),
                padding=padding_values
            ).to(target_inference_device)
        except Exception as e:
            print(f"Error loading video {input_video_filepath}: {e}")
            raise gr.Error(f"Could not load video: {e}")

    print(f"Moving models to {target_inference_device} for inference...")
    pipeline_instance.to(target_inference_device)
    active_latent_upsampler = None
    if improve_texture_flag and latent_upsampler_instance:
        latent_upsampler_instance.to(target_inference_device)
        active_latent_upsampler = latent_upsampler_instance
    print("Models moved.")

    result_images_tensor = None
    try:
        if improve_texture_flag:
            if not active_latent_upsampler:
                raise gr.Error("Spatial upscaler model not loaded or improve_texture not selected, cannot use multi-scale.")
            
            multi_scale_pipeline_obj = LTXMultiScalePipeline(pipeline_instance, active_latent_upsampler)
            
            first_pass_args = PIPELINE_CONFIG_YAML.get("first_pass", {}).copy()
            first_pass_args["guidance_scale"] = float(ui_guidance_scale)
            if "timesteps" not in first_pass_args:
                first_pass_args["num_inference_steps"] = int(ui_steps)

            second_pass_args = PIPELINE_CONFIG_YAML.get("second_pass", {}).copy()
            second_pass_args["guidance_scale"] = float(ui_guidance_scale)
            
            multi_scale_call_kwargs = call_kwargs.copy()
            multi_scale_call_kwargs.update({
                "downscale_factor": PIPELINE_CONFIG_YAML["downscale_factor"],
                "first_pass": first_pass_args,
                "second_pass": second_pass_args,
            })
            
            print(f"Calling multi-scale pipeline (eff. HxW: {actual_height}x{actual_width}) on {target_inference_device}")
            result_images_tensor = multi_scale_pipeline_obj(**multi_scale_call_kwargs).images
        else:
            single_pass_call_kwargs = call_kwargs.copy()
            single_pass_call_kwargs["guidance_scale"] = float(ui_guidance_scale)
            single_pass_call_kwargs["num_inference_steps"] = int(ui_steps)
            single_pass_call_kwargs.pop("first_pass", None)
            single_pass_call_kwargs.pop("second_pass", None)
            single_pass_call_kwargs.pop("downscale_factor", None)
            
            print(f"Calling base pipeline (padded HxW: {height_padded}x{width_padded}) on {target_inference_device}")
            result_images_tensor = pipeline_instance(**single_pass_call_kwargs).images
    
    finally:
        print(f"Moving models back to CPU...")
        pipeline_instance.to("cpu")
        if active_latent_upsampler:
            active_latent_upsampler.to("cpu")
        
        if target_inference_device == "cuda":
            torch.cuda.empty_cache()
        print("Models moved back to CPU and cache cleared (if CUDA).")

    if result_images_tensor is None:
        raise gr.Error("Generation failed.")

    pad_left, pad_right, pad_top, pad_bottom = padding_values
    slice_h_end = -pad_bottom if pad_bottom > 0 else None
    slice_w_end = -pad_right if pad_right > 0 else None
    result_images_tensor = result_images_tensor[
        :, :, :actual_num_frames, pad_top:slice_h_end, pad_left:slice_w_end
    ]

    # The pipeline with output_type="pt" should return tensors in the [0, 1] range.
    video_np = result_images_tensor[0].permute(1, 2, 3, 0).cpu().float().numpy()
    
    # Clip to ensure values are indeed in [0, 1] before scaling to uint8
    video_np = np.clip(video_np, 0, 1) 
    video_np = (video_np * 255).astype(np.uint8)

    temp_dir = tempfile.mkdtemp()
    timestamp = random.randint(10000,99999)
    output_video_path = os.path.join(temp_dir, f"output_{timestamp}.mp4")
    
    try:
        with imageio.get_writer(output_video_path, fps=call_kwargs["frame_rate"], macro_block_size=1) as video_writer:
            for frame_idx in range(video_np.shape[0]):
                progress(frame_idx / video_np.shape[0], desc="Saving video")
                video_writer.append_data(video_np[frame_idx])
    except Exception as e:
        print(f"Error saving video with macro_block_size=1: {e}")
        try:
            with imageio.get_writer(output_video_path, fps=call_kwargs["frame_rate"], format='FFMPEG', codec='libx264', quality=8) as video_writer:
                 for frame_idx in range(video_np.shape[0]):
                    progress(frame_idx / video_np.shape[0], desc="Saving video (fallback ffmpeg)")
                    video_writer.append_data(video_np[frame_idx])
        except Exception as e2:
            print(f"Fallback video saving error: {e2}")
            raise gr.Error(f"Failed to save video: {e2}")

    if isinstance(input_image_filepath, tempfile._TemporaryFileWrapper):
        if os.path.exists(input_image_filepath.name): # Check if it's already closed by Gradio
             try:
                input_image_filepath.close()
                os.remove(input_image_filepath.name)
             except: pass # May already be closed/removed
    elif input_image_filepath and os.path.exists(input_image_filepath) and input_image_filepath.startswith(tempfile.gettempdir()):
        try: os.remove(input_image_filepath) # If Gradio passed a path to a temp file
        except: pass

    if isinstance(input_video_filepath, tempfile._TemporaryFileWrapper):
        if os.path.exists(input_video_filepath.name):
            try:
                input_video_filepath.close()
                os.remove(input_video_filepath.name)
            except: pass
    elif input_video_filepath and os.path.exists(input_video_filepath) and input_video_filepath.startswith(tempfile.gettempdir()):
        try: os.remove(input_video_filepath)
        except: pass
            
    return output_video_path

# --- Gradio UI Definition ---
css="""
#col-container {
    margin: 0 auto;
    max-width: 900px;
}
"""

with gr.Blocks(css=css, theme=gr.themes.Glass()) as demo:
    gr.Markdown("# LTX Video 0.9.7 Distilled (using LTX-Video lib)")
    gr.Markdown("Generates a short video based on text prompt, image, or existing video. Models are moved to GPU during generation and back to CPU afterwards to save VRAM.")
    with gr.Row():
        with gr.Column():
            with gr.Group():
                with gr.Tab("text-to-video") as text_tab:
                    image_n_hidden = gr.Textbox(label="image_n", visible=False, value=None)
                    video_n_hidden = gr.Textbox(label="video_n", visible=False, value=None)
                    t2v_prompt = gr.Textbox(label="Prompt", value="A majestic dragon flying over a medieval castle", lines=3)
                    t2v_button = gr.Button("Generate Text-to-Video", variant="primary")
                with gr.Tab("image-to-video") as image_tab:
                    video_i_hidden = gr.Textbox(label="video_i", visible=False, value=None)
                    image_i2v = gr.Image(label="Input Image", type="filepath", sources=["upload", "webcam"])
                    i2v_prompt = gr.Textbox(label="Prompt", value="The creature from the image starts to move", lines=3)
                    i2v_button = gr.Button("Generate Image-to-Video", variant="primary")
                with gr.Tab("video-to-video") as video_tab:
                    image_v_hidden = gr.Textbox(label="image_v", visible=False, value=None)
                    video_v2v = gr.Video(label="Input Video", sources=["upload", "webcam"])
                    frames_to_use = gr.Slider(label="Frames to use from input video", minimum=9, maximum=MAX_NUM_FRAMES, value=9, step=8, info="Number of initial frames to use for conditioning/transformation. Must be N*8+1.")
                    v2v_prompt = gr.Textbox(label="Prompt", value="Change the style to cinematic anime", lines=3)
                    v2v_button = gr.Button("Generate Video-to-Video", variant="primary")

            improve_texture = gr.Checkbox(label="Improve Texture (multi-scale)", value=True, info="Uses a two-pass generation for better quality, but is slower. Recommended for final output.")

        with gr.Column():
            output_video = gr.Video(label="Generated Video", interactive=False)
            gr.Markdown("Note: Generation can take a few minutes depending on settings and hardware.")

    with gr.Accordion("Advanced settings", open=False):
        negative_prompt_input = gr.Textbox(label="Negative Prompt", value="worst quality, inconsistent motion, blurry, jittery, distorted", lines=2)
        with gr.Row():
            seed_input = gr.Number(label="Seed", value=42, precision=0, minimum=0, maximum=2**32-1)
            randomize_seed_input = gr.Checkbox(label="Randomize Seed", value=False)
        with gr.Row():
            guidance_scale_input = gr.Slider(label="Guidance Scale (CFG)", minimum=1.0, maximum=10.0, value=PIPELINE_CONFIG_YAML.get("first_pass", {}).get("guidance_scale", 1.0), step=0.1, info="Controls how much the prompt influences the output. Higher values = stronger influence.")
            default_steps = len(PIPELINE_CONFIG_YAML.get("first_pass", {}).get("timesteps", [1]*7))
            steps_input = gr.Slider(label="Inference Steps (for first pass if multi-scale)", minimum=1, maximum=30, value=default_steps, step=1, info="Number of denoising steps. More steps can improve quality but increase time. If YAML defines 'timesteps' for a pass, this UI value is ignored for that pass.")
        with gr.Row():
            num_frames_input = gr.Slider(label="Number of Frames to Generate", minimum=9, maximum=MAX_NUM_FRAMES, value=25, step=8, info="Total frames in the output video. Should be N*8+1 (e.g., 9, 17, 25...).")
        with gr.Row():
            height_input = gr.Slider(label="Height", value=512, step=32, minimum=256, maximum=MAX_IMAGE_SIZE, info="Must be divisible by 32.")
            width_input = gr.Slider(label="Width", value=704, step=32, minimum=256, maximum=MAX_IMAGE_SIZE, info="Must be divisible by 32.")
    
    t2v_inputs = [t2v_prompt, negative_prompt_input, image_n_hidden, video_n_hidden,
                  height_input, width_input, gr.State("text-to-video"),
                  steps_input, num_frames_input, gr.State(0),
                  seed_input, randomize_seed_input, guidance_scale_input, improve_texture]
    
    i2v_inputs = [i2v_prompt, negative_prompt_input, image_i2v, video_i_hidden,
                  height_input, width_input, gr.State("image-to-video"),
                  steps_input, num_frames_input, gr.State(0),
                  seed_input, randomize_seed_input, guidance_scale_input, improve_texture]

    v2v_inputs = [v2v_prompt, negative_prompt_input, image_v_hidden, video_v2v,
                  height_input, width_input, gr.State("video-to-video"),
                  steps_input, num_frames_input, frames_to_use,
                  seed_input, randomize_seed_input, guidance_scale_input, improve_texture]

    t2v_button.click(fn=generate, inputs=t2v_inputs, outputs=[output_video], api_name="text_to_video")
    i2v_button.click(fn=generate, inputs=i2v_inputs, outputs=[output_video], api_name="image_to_video")
    v2v_button.click(fn=generate, inputs=v2v_inputs, outputs=[output_video], api_name="video_to_video")

if __name__ == "__main__":
    if os.path.exists(models_dir) and os.path.isdir(models_dir):
        print(f"Model directory: {Path(models_dir).resolve()}")
    
    demo.queue().launch(debug=True, share=False)