import gradio as gr
import spaces 
import torch
from diffusers import LTXConditionPipeline, LTXLatentUpsamplePipeline
from diffusers.pipelines.ltx.pipeline_ltx_condition import LTXVideoCondition
from diffusers.utils import export_to_video, load_video

pipe = LTXConditionPipeline.from_pretrained("a-r-r-o-w/LTX-Video-0.9.7-diffusers", torch_dtype=torch.bfloat16)
pipe_upsample = LTXLatentUpsamplePipeline.from_pretrained("a-r-r-o-w/LTX-Video-0.9.7-Latent-Spatial-Upsampler-diffusers", vae=pipe.vae, torch_dtype=torch.bfloat16)
pipe.to("cuda")
pipe_upsample.to("cuda")
pipe.vae.enable_tiling()


def round_to_nearest_resolution_acceptable_by_vae(height, width):
    height = height - (height % pipe.vae_temporal_compression_ratio)
    width = width - (width % pipe.vae_temporal_compression_ratio)
    return height, width
    
@spaces.GPU
def generate(prompt,
             negative_prompt,
             steps,
             seed):
    return


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

with gr.Blocks(css=css, theme=gr.themes.Ocean(), js=js_func) as demo:

  gr.Markdown("# LTX Video 0.9.7 Distilled")

  with gr.Row():
    with gr.Column():
      with gr.Group():
        image = gr.Image(label="")
        prompt = gr.Textbox(label="prompt")
      run_button = gr.Button()
    with gr.Column():
      output = gr.Video(interactive=False)
      

  with gr.Accordion("Advanced settings", open=False):
     n_prompt = gr.Textbox(label="negative prompt", value="", visible=False)  
     with gr.Row():
      seed = gr.Number(label="seed", value=0, precision=0)
      randomize_seed = gr.Checkbox(label="randomize seed")
     with gr.Row():
      steps = gr.Slider(label="Steps", minimum=1, maximum=30, value=8, step=1)
      num_frames = gr.Slider(label="# frames", minimum=1, maximum=30, value=8, step=1)
    
    
  

demo.launch()
