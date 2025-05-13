import gradio as gr
import spaces 
import torch
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
    
@spaces.GPU
def generate(prompt,
             negative_prompt,
             image, 
             steps,
             num_frames,
             seed,
             randomize_seed,
             t2v, progress=gr.Progress(track_tqdm=True)):
    
    expected_height, expected_width = 768, 1152
    downscale_factor = 2 / 3

    if randomize_seed:
        seed = random.randint(0, MAX_SEED)

    if image is not None or t2v:
        condition1 = LTXVideoCondition(video=image, frame_index=0)
    else:
        condition1 = None

    # Part 1. Generate video at smaller resolution
    # Text-only conditioning is also supported without the need to pass `conditions`
    downscaled_height, downscaled_width = int(expected_height * downscale_factor), int(expected_width * downscale_factor)
    downscaled_height, downscaled_width = round_to_nearest_resolution_acceptable_by_vae(downscaled_height, downscaled_width)
    
    latents = pipe(
            conditions=condition1,
            prompt=prompt,
            negative_prompt=negative_prompt,
            # width=downscaled_width,
            # height=downscaled_height,
            num_frames=num_frames,
            num_inference_steps=steps,
            decode_timestep = 0.05,
            decode_noise_scale = 0.025,
            generator=torch.Generator().manual_seed(seed),
            #output_type="latent",
        ).frames
        
    # Part 2. Upscale generated video using latent upsampler with fewer inference steps
    # The available latent upsampler upscales the height/width by 2x
    # upscaled_height, upscaled_width = downscaled_height * 2, downscaled_width * 2
    # upscaled_latents = pipe_upsample(
    #     latents=latents,
    #     output_type="latent"
    # ).frames
    
    # # Part 3. Denoise the upscaled video with few steps to improve texture (optional, but recommended)
    # video = pipe(
    #     conditions=condition1,
    #     prompt=prompt,
    #     negative_prompt=negative_prompt,
    #     width=upscaled_width,
    #     height=upscaled_height,
    #     num_frames=num_frames,
    #     denoise_strength=0.4,  # Effectively, 4 inference steps out of 10
    #     num_inference_steps=10,
    #     latents=upscaled_latents,
    #     decode_timestep=0.05,
    #     image_cond_noise_scale=0.025,
    #     generator=torch.Generator().manual_seed(seed),
    #     output_type="pil",
    # ).frames[0]
    
    # Part 4. Downscale the video to the expected resolution
    video = [frame.resize((expected_width, expected_height)) for frame in latents[0]]
    export_to_video(latents, "output.mp4", fps=24)
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

  with gr.Row():
    with gr.Column():
      with gr.Group():
        image = gr.Image(label="")
        prompt = gr.Textbox(label="prompt")
        t2v = gr.Checkbox(label="run text-to-video", value=False)
      run_button = gr.Button()
    with gr.Column():
      output = gr.Video(interactive=False)
      

  with gr.Accordion("Advanced settings", open=False):
     negative_prompt = gr.Textbox(label="negative prompt", value="", visible=False)  
     with gr.Row():
      seed = gr.Number(label="seed", value=0, precision=0)
      randomize_seed = gr.Checkbox(label="randomize seed")
     with gr.Row():
      steps = gr.Slider(label="Steps", minimum=1, maximum=30, value=8, step=1)
      num_frames = gr.Slider(label="# frames", minimum=1, maximum=200, value=161, step=1)

  
  run_button.click(fn=generate, 
                   inputs=[prompt,
             negative_prompt,
             image, 
             steps,
             num_frames,
             seed,
             randomize_seed, t2v], 
                   outputs=[output])
  

demo.launch()
