import gradio as gr
import spaces 
import torch
# from pipeline_ltx_condition import LTXVideoCondition, LTXConditionPipeline
# from diffusers import LTXLatentUpsamplePipeline
from diffusers import LTXConditionPipeline, LTXLatentUpsamplePipeline
from diffusers.pipelines.ltx.pipeline_ltx_condition import LTXVideoCondition
from diffusers.utils import export_to_video, load_video
import numpy as np


pipe = LTXConditionPipeline.from_pretrained("linoyts/LTX-Video-0.9.7-distilled-diffusers", torch_dtype=torch.bfloat16)
pipe_upsample = LTXLatentUpsamplePipeline.from_pretrained("a-r-r-o-w/LTX-Video-0.9.7-Latent-Spatial-Upsampler-diffusers", vae=pipe.vae, torch_dtype=torch.bfloat16)
pipe.to("cuda")
pipe_upsample.to("cuda")
pipe.vae.enable_tiling()

MAX_SEED = np.iinfo(np.int32).max
MAX_IMAGE_SIZE = 2048


def round_to_nearest_resolution_acceptable_by_vae(height, width):
    height = height - (height % pipe.vae_temporal_compression_ratio)
    width = width - (width % pipe.vae_temporal_compression_ratio)
    return height, width

def change_mode_to_text():
  return gr.update(value="text-to-video")

def change_mode_to_image():
  return gr.update(value="image-to-video")

def change_mode_to_video():
  return gr.update(value="video-to-video")
    
@spaces.GPU
def generate(prompt,
             negative_prompt,
             image,
             video,
             mode,
             steps,
             num_frames,
             frames_to_use,
             seed,
             randomize_seed,
             improve_texture=False, progress=gr.Progress(track_tqdm=True)):
    
    if randomize_seed:
        seed = random.randint(0, MAX_SEED)
        
    # Part 1. Generate video at smaller resolution
    # Text-only conditioning is also supported without the need to pass `conditions`
    expected_height, expected_width = 768, 1152 #todo make configurable
    downscale_factor = 2 / 3
    downscaled_height, downscaled_width = int(expected_height * downscale_factor), int(expected_width * downscale_factor)
    downscaled_height, downscaled_width = round_to_nearest_resolution_acceptable_by_vae(downscaled_height, downscaled_width)

    if mode == "text-to-video" and video is not None:
        video = load_video(video)[:frames_to_use]
        condition = True
    elif mode == "image-to-video" and image is not None:
        video = [image]
        condition = True
    else:
       condition=False

    if condition:
        condition1 = LTXVideoCondition(video=video, frame_index=0)
    else:
        condition1 = None
    
    latents = pipe(
        conditions=condition1,
        prompt=prompt,
        negative_prompt=negative_prompt,
        width=downscaled_width,
        height=downscaled_height,
        num_frames=num_frames,
        num_inference_steps=steps,
        decode_timestep = 0.05,
        decode_noise_scale = 0.025,
        guidance_scale=1.0,
        generator=torch.Generator(device="cuda").manual_seed(seed),
        output_type="latent",
    ).frames
   

   
    
    # latents = pipe(
    #         conditions=condition1,
    #         prompt=prompt,
    #         negative_prompt=negative_prompt,
    #         # width=downscaled_width,
    #         # height=downscaled_height,
    #         num_frames=num_frames,
    #         num_inference_steps=steps,
    #         decode_timestep = 0.05,
    #         decode_noise_scale = 0.025,
    #         generator=torch.Generator().manual_seed(seed),
    #         #output_type="latent",
    #     ).frames
        
    # Part 2. Upscale generated video using latent upsampler with fewer inference steps
    # The available latent upsampler upscales the height/width by 2x
    if improve_texture:
        upscaled_height, upscaled_width = downscaled_height * 2, downscaled_width * 2
        upscaled_latents = pipe_upsample(
            latents=latents,
            output_type="latent"
        ).frames
        
        # Part 3. Denoise the upscaled video with few steps to improve texture (optional, but recommended)  
        video = pipe(
            conditions=condition1,
            prompt=prompt,
            negative_prompt=negative_prompt,
            width=upscaled_width,
            height=upscaled_height,
            num_frames=num_frames,
            denoise_strength=0.4,  # Effectively, 4 inference steps out of 10
            num_inference_steps=10,
            latents=upscaled_latents,
            decode_timestep=0.05,
            image_cond_noise_scale=0.025,
            generator=torch.Generator().manual_seed(seed),
            output_type="pil",
        ).frames[0]
    else:
        upscaled_height, upscaled_width = downscaled_height * 2, downscaled_width * 2
        video = pipe_upsample(
            latents=latents,
            # output_type="latent"
        ).frames[0]
    
    # Part 4. Downscale the video to the expected resolution
    video = [frame.resize((expected_width, expected_height)) for frame in video]
    export_to_video(video, "output.mp4", fps=24)
    return "output.mp4"



css="""
#col-container {
    margin: 0 auto;
    max-width: 900px;
}
"""

js_func = """
function refresh() {
    const url = new URL(window.location);

    if (url.searchParams.get('__theme') !== 'dark') {
        url.searchParams.set('__theme', 'dark');
        window.location.href = url.href;
    }
}
"""

with gr.Blocks(css=css, theme=gr.themes.Ocean()) as demo:

  gr.Markdown("# LTX Video 0.9.7 Distilled")
  mode = gr.State(value="text-to-video")
  with gr.Row():
    with gr.Column():
      with gr.Group():
        with gr.Tab("text-to-video") as text_tab:
          image = gr.Image(label="", visible=False)
          #prompt = gr.Textbox(label="prompt")
        with gr.Tab("image-to-video") as image_tab:
          image = gr.Image(label="")
        with gr.Tab("video-to-video") as video_tab:
          video = gr.Video(label="")
          frames_to_use = gr.Number(label="num frames to use",info="first # of frames to use from the input video", value=1)
        prompt = gr.Textbox(label="prompt")
        improve_texture = gr.Checkbox(label="improve texture", value=False, info="note it slows generation")
      run_button = gr.Button()
    with gr.Column():
      output = gr.Video(interactive=False)
      

  with gr.Accordion("Advanced settings", open=False):
     negative_prompt = gr.Textbox(label="negative prompt", value="worst quality, inconsistent motion, blurry, jittery, distorted", visible=False)  
     with gr.Row():
      seed = gr.Number(label="seed", value=0, precision=0)
      randomize_seed = gr.Checkbox(label="randomize seed")
     with gr.Row():
      steps = gr.Slider(label="Steps", minimum=1, maximum=30, value=8, step=1)
      num_frames = gr.Slider(label="# frames", minimum=1, maximum=161, value=96, step=1)
     with gr.Row():
       height = gr.Slider(label="height", value=512, step=1)
       width = gr.Slider(label="width", value=704, step=1)
    

  text_tab.select(fn=change_mode_to_text, inputs=[], outputs=[mode])
  image_tab.select(fn=change_mode_to_image, inputs=[], outputs=[mode])
  video_tab.select(fn=change_mode_to_video, inputs=[], outputs=[mode])
    
  run_button.click(fn=generate, 
                   inputs=[prompt,
             negative_prompt,
             image,
             video,
             mode,
             steps,
             num_frames,
             frames_to_use,
             seed,
             randomize_seed, improve_texture], 
                   outputs=[output])


  

demo.launch()
