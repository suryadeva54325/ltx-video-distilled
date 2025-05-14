import gradio as gr
import spaces
import torch
import numpy as np
import os
import yaml
import random
from PIL import Image
import imageio # For export_to_video and reading video frames
from pathlib import Path
from huggingface_hub import hf_hub_download

# --- LTX-Video Imports (from your provided codebase) ---
from ltx_video.pipelines.pipeline_ltx_video import (
    ConditioningItem,
    LTXVideoPipeline,
    LTXMultiScalePipeline,
)
from ltx_video.models.autoencoders.vae_encode import vae_decode, vae_encode, un_normalize_latents, normalize_latents
from inference import (
    create_ltx_video_pipeline,
    create_latent_upsampler,
    load_image_to_tensor_with_resize_and_crop, # Re-using for image conditioning
    load_media_file, # Re-using for video conditioning
    get_device,
    seed_everething,
    calculate_padding,
)
from ltx_video.utils.skip_layer_strategy import SkipLayerStrategy
from ltx_video.models.autoencoders.latent_upsampler import LatentUpsampler
# --- End LTX-Video Imports ---

# --- Diffusers/Original utils (keeping export_to_video for convenience if it works) ---
from diffusers.utils import export_to_video # Keep if it works with PIL list
# ---

# --- Global Configuration & Model Loading ---
DEVICE = get_device()
MODEL_DIR = "downloaded_models" # Directory to store downloaded models
Path(MODEL_DIR).mkdir(parents=True, exist_ok=True)

# Load YAML configuration
YAML_CONFIG_PATH = "configs/ltxv-13b-0.9.7-distilled.yaml" # Place this file in the same directory
with open(YAML_CONFIG_PATH, "r") as f:
    PIPELINE_CONFIG_YAML = yaml.safe_load(f)

# Download and prepare model paths from YAML
LTXV_MODEL_FILENAME = PIPELINE_CONFIG_YAML["checkpoint_path"]
SPATIAL_UPSCALER_FILENAME = PIPELINE_CONFIG_YAML["spatial_upscaler_model_path"]
TEXT_ENCODER_PATH = PIPELINE_CONFIG_YAML["text_encoder_model_name_or_path"] # This is usually a repo name

try:
    # Main LTX-Video model
    if not os.path.isfile(os.path.join(MODEL_DIR, LTXV_MODEL_FILENAME)):
        print(f"Downloading {LTXV_MODEL_FILENAME}...")
        ltxv_checkpoint_path = hf_hub_download(
            repo_id="LTX-Colab/LTX-Video-Preview", # Assuming the distilled model is also here or adjust repo_id
            filename=LTXV_MODEL_FILENAME,
            local_dir=MODEL_DIR,
            repo_type="model",
        )
    else:
        ltxv_checkpoint_path = os.path.join(MODEL_DIR, LTXV_MODEL_FILENAME)

    # Spatial Upsampler model
    if not os.path.isfile(os.path.join(MODEL_DIR, SPATIAL_UPSCALER_FILENAME)):
        print(f"Downloading {SPATIAL_UPSCALER_FILENAME}...")
        spatial_upsampler_path = hf_hub_download(
            repo_id="Lightricks/LTX-Video",
            filename=SPATIAL_UPSCALER_FILENAME,
            local_dir=MODEL_DIR,
            repo_type="model",
        )
    else:
        spatial_upsampler_path = os.path.join(MODEL_DIR, SPATIAL_UPSCALER_FILENAME)
except Exception as e:
    print(f"Error downloading models: {e}")
    print("Please ensure model files are correctly specified and accessible.")
    # Depending on severity, you might want to exit or disable GPU features
    # For now, we'll let it proceed and potentially fail later if paths are invalid.
    ltxv_checkpoint_path = LTXV_MODEL_FILENAME # Fallback to filename if download fails
    spatial_upsampler_path = SPATIAL_UPSCALER_FILENAME


print(f"Using LTX-Video checkpoint: {ltxv_checkpoint_path}")
print(f"Using Spatial Upsampler: {spatial_upsampler_path}")
print(f"Using Text Encoder: {TEXT_ENCODER_PATH}")

# Create LTX-Video pipeline
pipe = create_ltx_video_pipeline(
    ckpt_path=ltxv_checkpoint_path,
    precision=PIPELINE_CONFIG_YAML["precision"],
    text_encoder_model_name_or_path=TEXT_ENCODER_PATH,
    sampler=PIPELINE_CONFIG_YAML["sampler"], # "from_checkpoint" or specific sampler
    device=DEVICE,
    enhance_prompt=False, # Assuming Gradio controls this, or set based on YAML later
).to(torch.bfloat16)

# Create Latent Upsampler
latent_upsampler = create_latent_upsampler(
    latent_upsampler_model_path=spatial_upsampler_path,
    device=DEVICE
)
latent_upsampler = latent_upsampler.to(torch.bfloat16)


# Multi-scale pipeline (wrapper)
multi_scale_pipe = LTXMultiScalePipeline(
    video_pipeline=pipe,
    latent_upsampler=latent_upsampler
).to(torch.bfloat16)
# --- End Global Configuration & Model Loading ---


MAX_SEED = np.iinfo(np.int32).max
MAX_IMAGE_SIZE = 2048 # Not strictly used here, but good to keep in mind


def round_to_nearest_resolution_acceptable_by_vae(height, width, vae_scale_factor):
    # print("before rounding",height, width)
    height = height - (height % vae_scale_factor)
    width = width - (width % vae_scale_factor)
    # print("after rounding",height, width)
    return height, width

@spaces.GPU
def generate(prompt,
             negative_prompt,
             image_path, # Gradio gives filepath for Image component
             video_path, # Gradio gives filepath for Video component
             height,
             width,
             mode,
             steps,      # This will map to num_inference_steps for the first pass
             num_frames,
             frames_to_use,
             seed,
             randomize_seed,
             guidance_scale,
             improve_texture=False, progress=gr.Progress(track_tqdm=True)):

    if randomize_seed:
        seed = random.randint(0, MAX_SEED)
    seed_everething(seed)
    
    generator = torch.Generator(device=DEVICE).manual_seed(seed)

    # --- Prepare conditioning items ---
    conditioning_items_list = []
    input_media_for_vid2vid = None # For the specific vid2vid mode in LTX pipeline

    # Pad target dimensions
    # VAE scale factor is typically 8 for spatial, but LTX might have its own specific factor.
    # CausalVideoAutoencoder has spatial_downscale_factor and temporal_downscale_factor
    vae_spatial_scale_factor = pipe.vae.spatial_downscale_factor
    vae_temporal_scale_factor = pipe.vae.temporal_downscale_factor

    # Ensure target height/width are multiples of VAE spatial scale factor
    height_padded_target = ((height - 1) // vae_spatial_scale_factor + 1) * vae_spatial_scale_factor
    width_padded_target = ((width - 1) // vae_spatial_scale_factor + 1) * vae_spatial_scale_factor
    
    # Ensure num_frames is multiple of VAE temporal scale factor + 1 (for causal VAE)
    # (num_frames - 1) should be multiple of temporal_scale_factor for non-causal parts
    # For CausalVAE, it's often (N * temporal_factor) + 1 frames.
    # The inference script uses: num_frames_padded = ((num_frames - 2) // 8 + 1) * 8 + 1
    # Assuming 8 is the temporal scale factor here for simplicity, adjust if different
    num_frames_padded_target = ((num_frames - 2) // vae_temporal_scale_factor + 1) * vae_temporal_scale_factor + 1


    padding_target = calculate_padding(height, width, height_padded_target, width_padded_target)


    if mode == "video-to-video" and video_path:
        # LTX pipeline's vid2vid uses `media_items` argument for the full video to transform
        # and `conditioning_items` for specific keyframes if needed.
        # Here, the Gradio's "video-to-video" seems to imply transforming the input video.
        input_media_for_vid2vid = load_media_file(
            media_path=video_path,
            height=height, # Original height before padding for loading
            width=width,   # Original width
            max_frames=min(num_frames_padded_target, frames_to_use if frames_to_use > 0 else num_frames_padded_target),
            padding=padding_target, # Padding to make it compatible with VAE of target size
        )
        # If we also want to strongly condition on the first frame(s) of this video:
        conditioning_media = load_media_file(
            media_path=video_path,
            height=height, width=width,
            max_frames=min(frames_to_use if frames_to_use > 0 else 1, num_frames_padded_target), # Use specified frames or just the first
            padding=padding_target,
            just_crop=True # Crop to aspect ratio, then resize
        )
        conditioning_items_list.append(ConditioningItem(media_item=conditioning_media, media_frame_number=0, conditioning_strength=1.0))

    elif mode == "image-to-video" and image_path:
        conditioning_media = load_image_to_tensor_with_resize_and_crop(
            image_input=image_path,
            target_height=height, # Original height
            target_width=width    # Original width
        )
        # Apply padding to the loaded tensor
        conditioning_media = torch.nn.functional.pad(conditioning_media, padding_target)
        conditioning_items_list.append(ConditioningItem(media_item=conditioning_media, media_frame_number=0, conditioning_strength=1.0))
    
    # else mode is "text-to-video", no explicit conditioning items unless defined elsewhere

    # --- Get pipeline parameters from YAML ---
    first_pass_config = PIPELINE_CONFIG_YAML.get("first_pass", {})
    second_pass_config = PIPELINE_CONFIG_YAML.get("second_pass", {})
    downscale_factor = PIPELINE_CONFIG_YAML.get("downscale_factor", 2/3)

    # Override steps from Gradio if provided, for the first pass
    if steps:
        # The YAML timesteps are specific, so overriding num_inference_steps might not be what we want
        # If YAML has `timesteps`, `num_inference_steps` is ignored by LTXVideoPipeline.
        # If YAML does not have `timesteps`, then `num_inference_steps` from Gradio will be used for the first pass.
        first_pass_config["num_inference_steps"] = steps
        # For distilled model, the second pass steps are usually very few, defined by its timesteps.
        # We won't override second_pass_config["num_inference_steps"] from the Gradio `steps`
        # as it's meant for the primary generation.

    # Determine initial generation dimensions (downscaled)
    # These are the dimensions for the *first pass* of the multi-scale pipeline
    initial_gen_height = int(height_padded_target * downscale_factor)
    initial_gen_width = int(width_padded_target * downscale_factor)

    initial_gen_height, initial_gen_width = round_to_nearest_resolution_acceptable_by_vae(
        initial_gen_height, initial_gen_width, vae_spatial_scale_factor
    )
    
    shared_pipeline_args = {
        "prompt": prompt,
        "negative_prompt": negative_prompt,
        "num_frames": num_frames_padded_target, # Always generate padded num_frames
        "frame_rate": 30, # Example, or get from UI if available
        "guidance_scale": guidance_scale,
        "generator": generator,
        "conditioning_items": conditioning_items_list if conditioning_items_list else None,
        "skip_layer_strategy": SkipLayerStrategy.AttentionValues, # Default or from YAML
        "offload_to_cpu": False, # Managed by global DEVICE
        "is_video": True,
        "vae_per_channel_normalize": True, # Common default
        "mixed_precision": (PIPELINE_CONFIG_YAML["precision"] == "bfloat16"),
        "enhance_prompt": False, # Controlled by Gradio app logic if needed for full LTX script
        "image_cond_noise_scale": 0.025, # from YAML decode_noise_scale, or make it a param
        "media_items": input_media_for_vid2vid if mode == "video-to-video" else None,
        # "decode_timestep" and "decode_noise_scale" are part of first_pass/second_pass or direct call
    }

    # --- Generation ---
    if improve_texture:
        print("Using LTXMultiScalePipeline for generation...")
        # Ensure first_pass_config and second_pass_config have necessary overrides
        # The 'steps' from Gradio applies to the first pass's num_inference_steps if timesteps not set
        if "timesteps" not in first_pass_config:
             first_pass_config["num_inference_steps"] = steps
        
        first_pass_config.setdefault("decode_timestep", PIPELINE_CONFIG_YAML.get("decode_timestep", 0.05))
        first_pass_config.setdefault("decode_noise_scale", PIPELINE_CONFIG_YAML.get("decode_noise_scale", 0.025))
        second_pass_config.setdefault("decode_timestep", PIPELINE_CONFIG_YAML.get("decode_timestep", 0.05))
        second_pass_config.setdefault("decode_noise_scale", PIPELINE_CONFIG_YAML.get("decode_noise_scale", 0.025))

        # The multi_scale_pipe's __call__ expects width and height for the *initial* (downscaled) generation
        result_frames_tensor = multi_scale_pipe(
            **shared_pipeline_args,
            width=initial_gen_width,
            height=initial_gen_height,
            downscale_factor=downscale_factor, # This might be used internally by multi_scale_pipe
            first_pass=first_pass_config,
            second_pass=second_pass_config,
            output_type="pt" # Get tensor for further processing
        ).images
        
        # LTXMultiScalePipeline should return images at 2x the initial_gen_width/height
        # So, result_frames_tensor is at initial_gen_width*2, initial_gen_height*2

    else:
        print("Using LTXVideoPipeline (first pass) + Manual Upsample + Decode...")
        # 1. First pass generation at downscaled resolution
        if "timesteps" not in first_pass_config:
             first_pass_config["num_inference_steps"] = steps

        first_pass_args = {
            **shared_pipeline_args,
            **first_pass_config,
            "width": initial_gen_width,
            "height": initial_gen_height,
            "output_type": "latent"
        }
        latents = pipe(**first_pass_args).images # .images here is actually latents
        print("First pass done!")
        # 2. Upsample latents manually
        # Need to handle normalization around latent upsampler if it expects unnormalized latents
        latents_unnorm = un_normalize_latents(latents, pipe.vae, vae_per_channel_normalize=True)
        upsampled_latents_unnorm = latent_upsampler(latents_unnorm)
        upsampled_latents = normalize_latents(upsampled_latents_unnorm, pipe.vae, vae_per_channel_normalize=True)
        
        # 3. Decode upsampled latents
        # The upsampler typically doubles the spatial dimensions
        upscaled_height_for_decode = initial_gen_height * 2
        upscaled_width_for_decode = initial_gen_width * 2
        
        # Prepare target_shape for VAE decoder
        # batch_size, channels, num_frames, height, width
        # Latents are (B, C, F_latent, H_latent, W_latent)
        # Target shape for vae.decode is pixel space
        # num_video_frames_final = upsampled_latents.shape[2] * pipe.vae.temporal_downscale_factor
        # if causal, it might be (F_latent - 1) * factor + 1
        num_video_frames_final = (upsampled_latents.shape[2] -1) * pipe.vae.temporal_downscale_factor + 1


        decode_kwargs = {
            "target_shape": (
                upsampled_latents.shape[0], # batch
                3, # out channels
                num_video_frames_final,
                upscaled_height_for_decode,
                upscaled_width_for_decode
            )
        }
        if pipe.vae.decoder.timestep_conditioning:
            decode_kwargs["timestep"] = torch.tensor([PIPELINE_CONFIG_YAML.get("decode_timestep", 0.05)] * upsampled_latents.shape[0]).to(DEVICE)
            # Add noise for decode if specified, similar to LTXVideoPipeline's call
            noise = torch.randn_like(upsampled_latents)
            decode_noise_val = PIPELINE_CONFIG_YAML.get("decode_noise_scale", 0.025)
            upsampled_latents = upsampled_latents * (1 - decode_noise_val) + noise * decode_noise_val

        print("before vae decoding")
        result_frames_tensor = pipe.vae.decode(upsampled_latents, **decode_kwargs).sample
        print("after vae decoding?")
        # result_frames_tensor shape: (B, C, F_video, H_video, W_video)

    # --- Post-processing: Cropping and Converting to PIL ---
    # Crop to original num_frames (before padding)
    result_frames_tensor = result_frames_tensor[:, :, :num_frames, :, :]

    # Unpad to target height and width
    _, _, _, current_h, current_w = result_frames_tensor.shape
    
    # Calculate crop needed if current dimensions are larger than padded_target
    # This happens if multi_scale_pipe output is larger than height_padded_target
    crop_y_start = (current_h - height_padded_target) // 2
    crop_x_start = (current_w - width_padded_target) // 2
    
    result_frames_tensor = result_frames_tensor[
        :, :, :, 
        crop_y_start : crop_y_start + height_padded_target, 
        crop_x_start : crop_x_start + width_padded_target
    ]
    
    # Now remove the padding added for VAE compatibility
    pad_left, pad_right, pad_top, pad_bottom = padding_target
    unpad_bottom = -pad_bottom if pad_bottom > 0 else result_frames_tensor.shape[3]
    unpad_right = -pad_right if pad_right > 0 else result_frames_tensor.shape[4]

    result_frames_tensor = result_frames_tensor[
        :, :, :,
        pad_top : unpad_bottom,
        pad_left : unpad_right
    ]


    # Convert tensor to list of PIL Images
    video_pil_list = []
    # result_frames_tensor shape: (B, C, F, H, W)
    # We expect B=1 from typical generation
    video_single_batch = result_frames_tensor[0] # Shape: (C, F, H, W)
    video_single_batch = (video_single_batch / 2 + 0.5).clamp(0, 1) # Normalize to [0,1]
    video_single_batch = video_single_batch.permute(1, 2, 3, 0).cpu().numpy() # F, H, W, C
    
    for frame_idx in range(video_single_batch.shape[0]):
        frame_np = (video_single_batch[frame_idx] * 255).astype(np.uint8)
        video_pil_list.append(Image.fromarray(frame_np))

    # Save video
    output_video_path = "output.mp4" # Gradio handles temp files
    export_to_video(video_pil_list, output_video_path, fps=24) # Assuming fps from original script
    return output_video_path


css="""
#col-container {
    margin: 0 auto;
    max-width: 900px;
}
"""

with gr.Blocks(css=css, theme=gr.themes.Ocean()) as demo:
    gr.Markdown("# LTX Video 0.9.7 Distilled (using LTX-Video lib)")
    with gr.Row():
        with gr.Column():
            with gr.Group():
                with gr.Tab("text-to-video") as text_tab:
                    image_n = gr.Image(label="", visible=False, value=None) # Ensure None for path
                    video_n = gr.Video(label="", visible=False, value=None) # Ensure None for path
                    t2v_prompt = gr.Textbox(label="prompt", value="A majestic dragon flying over a medieval castle")
                    t2v_button = gr.Button("Generate Text-to-Video")
                with gr.Tab("image-to-video") as image_tab:
                    video_i = gr.Video(label="", visible=False, value=None)
                    image_i2v = gr.Image(label="input image", type="filepath")
                    i2v_prompt = gr.Textbox(label="prompt", value="The creature from the image starts to move")
                    i2v_button = gr.Button("Generate Image-to-Video")
                with gr.Tab("video-to-video") as video_tab:
                    image_v = gr.Image(label="", visible=False, value=None)
                    video_v2v = gr.Video(label="input video")
                    frames_to_use = gr.Number(label="num frames to use",info="first # of frames to use from the input video for conditioning/transformation", value=9)
                    v2v_prompt = gr.Textbox(label="prompt", value="Change the style to cinematic anime")
                    v2v_button = gr.Button("Generate Video-to-Video")

                improve_texture = gr.Checkbox(label="improve texture (multi-scale)", value=True, info="Uses a two-pass generation for better quality, but is slower.")

        with gr.Column():
            output = gr.Video(interactive=False)

    with gr.Accordion("Advanced settings", open=False):
        negative_prompt_input = gr.Textbox(label="negative prompt", value="worst quality, inconsistent motion, blurry, jittery, distorted")
        with gr.Row():
            seed_input = gr.Number(label="seed", value=42, precision=0)
            randomize_seed_input = gr.Checkbox(label="randomize seed", value=False)
        with gr.Row():
            guidance_scale_input = gr.Slider(label="guidance scale", minimum=0, maximum=10, value=1.0, step=0.1, info="For distilled models, CFG is often 1.0 (disabled) or very low.") # Distilled model might not need high CFG
            steps_input = gr.Slider(label="Steps (for first pass if multi-scale)", minimum=1, maximum=30, value=PIPELINE_CONFIG_YAML.get("first_pass", {}).get("timesteps", [1]*8).__len__(), step=1, info="Number of inference steps. If YAML defines timesteps, this is ignored for that pass.") # Default to length of first_pass timesteps
            num_frames_input = gr.Slider(label="# frames", minimum=9, maximum=121, value=25, step=8, info="Should be N*8+1, e.g., 9, 17, 25...") # Adjusted for LTX structure
        with gr.Row():
            height_input = gr.Slider(label="height", value=512, step=8, minimum=256, maximum=MAX_IMAGE_SIZE) # Step by VAE factor
            width_input = gr.Slider(label="width", value=704, step=8, minimum=256, maximum=MAX_IMAGE_SIZE) # Step by VAE factor

    t2v_button.click(fn=generate,
                      inputs=[t2v_prompt,
                              negative_prompt_input,
                              image_n, # Pass None for image
                              video_n, # Pass None for video
                              height_input,
                              width_input,
                              gr.State("text-to-video"),
                              steps_input,
                              num_frames_input,
                              gr.State(0), # frames_to_use not relevant for t2v
                              seed_input,
                              randomize_seed_input, guidance_scale_input, improve_texture],
                      outputs=[output])

    i2v_button.click(fn=generate,
                      inputs=[i2v_prompt,
                              negative_prompt_input,
                              image_i2v,
                              video_i, # Pass None for video
                              height_input,
                              width_input,
                              gr.State("image-to-video"),
                              steps_input,
                              num_frames_input,
                              gr.State(0), # frames_to_use not relevant for i2v initial frame
                              seed_input,
                              randomize_seed_input, guidance_scale_input, improve_texture],
                      outputs=[output])

    v2v_button.click(fn=generate,
                      inputs=[v2v_prompt,
                              negative_prompt_input,
                              image_v, # Pass None for image
                              video_v2v,
                              height_input,
                              width_input,
                              gr.State("video-to-video"),
                              steps_input,
                              num_frames_input,
                              frames_to_use,
                              seed_input,
                              randomize_seed_input, guidance_scale_input, improve_texture],
                      outputs=[output])

demo.launch()