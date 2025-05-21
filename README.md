---
title: LTX Video Fast
emoji: 🎥
colorFrom: yellow
colorTo: pink
sdk: gradio
sdk_version: 5.29.1
app_file: app.py
pinned: false
short_description: ultra-fast video model, LTX 0.9.7 13B distilled
---

# LTX Video Fast

This project provides an ultra-fast video generation model, LTX 0.9.7 13B distilled.

## Installation

1.  **Clone the repository:**

    ```bash
    git clone <repository_url>
    cd <repository_directory>
    ```

2.  **Install the dependencies:**

    ```bash
    pip install -r requirements.txt
    ```

    The `requirements.txt` file includes the following dependencies:

    ```
    accelerate
transformers
sentencepiece
pillow
numpy
torchvision
huggingface_hub
spaces
opencv-python
imageio
imageio-ffmpeg
einops
timm
av
git+https://github.com/huggingface/diffusers.git@main
    ```

## Configuration

The project uses YAML configuration files to define model parameters and pipeline settings. Example configuration files are located in the `configs` directory.

*   `configs/ltxv-13b-0.9.7-dev.yaml`
*   `configs/ltxv-13b-0.9.7-distilled.yaml`
*   `configs/ltxv-2b-0.9.1.yaml`
*   `configs/ltxv-2b-0.9.5.yaml`
*   `configs/ltxv-2b-0.9.6-dev.yaml`
*   `configs/ltxv-2b-0.9.6-distilled.yaml`
*   `configs/ltxv-2b-0.9.yaml`

To use a specific configuration, you can specify its path when running the generation scripts.

## Usage

The project supports different generation modes: text-to-video, image-to-video, and video-to-video.

### Text-to-Video

To generate a video from a text prompt, use the `inference.py` script with the `--mode text2video` argument.  You must also specify the path to the desired config file using `--config` and a text prompt using `--prompt`. For example:

### Image-to-Video

To generate a video from an image, use the `inference.py` script with the `--mode image2video` argument.  You must also specify the path to the desired config file using `--config` and the path to the input image using `--image_path`. For example:

### Video-to-Video

To generate a video from another video, use the `inference.py` script with the `--mode video2video` argument.  You must also specify the path to the desired config file using `--config` and the path to the input video using `--video_path`. For example:

Check out the configuration reference at https://huggingface.co/docs/hub/spaces-config-reference