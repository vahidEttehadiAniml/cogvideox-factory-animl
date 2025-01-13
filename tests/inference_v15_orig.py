import os, glob

import numpy as np
from PIL import Image
import torch
from diffusers import (
    CogVideoXDPMScheduler,
    CogVideoXImageToVideoPipeline,
    CogVideoXFramesToVideoPipeline,
    CogVideoXPipeline,
    CogVideoXVideoToVideoPipeline,
)
from diffusers.utils import export_to_video, load_image, load_video

from diffusers import AutoencoderKLCogVideoX, CogVideoXVideoToVideoPipeline, CogVideoXTransformer3DModel, CogVideoXDPMScheduler
from transformers import T5EncoderModel


def get_bounding_box(image):
    """
    Calculates the bounding box of the non-transparent region of an image.
    Returns (left, upper, right, lower) or None if the image is fully transparent.
    """
    if image.mode not in ("RGBA", "LA"):
        return 0, 0, image.width, image.height
    alpha_channel = image.getchannel("A")
    bbox = alpha_channel.getbbox()
    if bbox is None:
        return None  # Fully transparent image
    return bbox


def crop_to_multiple_of_8(image, target_width, target_height):
    """Crops a PIL image to the target dimensions (multiples of 8) using center crop."""
    img_width, img_height = image.size

    left = (img_width - target_width) // 2
    top = (img_height - target_height) // 2
    right = (img_width + target_width) // 2
    bottom = (img_height + target_height) // 2

    # Adjust crop region if it falls outside of the image
    left = max(0, left)
    top = max(0, top)
    right = min(img_width, right)
    bottom = min(img_height, bottom)

    return image.crop((left, top, right, bottom)), (left, top)


def paste_cropped_back(original_image, cropped_image, offset):
    """
    Pastes a cropped image back onto the original image at the correct position.

    Args:
        original_image: The original PIL Image object.
        cropped_image: The cropped PIL Image object.
        offset: The tuple (left, top) that indicates the position of the crop relative to the original image.

    Returns:
        A new PIL Image object with the cropped image pasted back.
    """

    left, top = offset

    # Create a copy of the original image to prevent inplace modifications.
    new_image = original_image.copy()

    new_image.paste(cropped_image, (left, top))
    return new_image


def process_images(image_paths, img_size=1024):
    """
    Processes a series of images, cropping them based on their mask bounding boxes.

    Args:
      image_paths: A list of paths to image files.

    Returns:
      A list of dictionaries. Each dictionary contains:
      - original: the original image
      - cropped: the cropped image
      - offset: the crop offset
    """
    max_bbox_width = 0
    max_bbox_height = 0

    images = []

    # Get Maximum bounding box
    for path in image_paths:
        try:
            img = Image.open(path).resize((img_size, img_size))
            images.append(img)
        except FileNotFoundError:
            print(f"Error: Could not find image: {path}")
            continue
        except Exception as e:
            print(f"Error: Could not load image: {path}. Details: {e}")
            continue

        bbox = get_bounding_box(img)
        if bbox:
            left, upper, right, lower = bbox
            bbox_width = right - left
            bbox_height = lower - upper
            max_bbox_width = max(max_bbox_width, bbox_width)
            max_bbox_height = max(max_bbox_height, bbox_height)

    # If any image is not transparent, assume the entire image is needed.
    if max_bbox_width == 0 or max_bbox_height == 0:
        max_bbox_width = max(img.width for img in images) if images else 0
        max_bbox_height = max(img.height for img in images) if images else 0

    ### create a margin to be sure all of the obejct is inside
    max_bbox_width *= 1.25
    max_bbox_height *= 1.25

    # Calculate target size.
    target_width = (round(max_bbox_width / 16) + 1) * 16
    target_height = (round(max_bbox_height / 16) + 1) * 16

    result = []
    for image in images:
        cropped_img, offset = crop_to_multiple_of_8(image, target_width, target_height)
        result.append({"original": image, "cropped": cropped_img, "offset": offset})

    return result



num_frames = 81
use_noise_condition = False
infer_img = 'real'

prompt = "A nice sneaker on the some rocks with a background of an active volcano splashing llavas while camera trajectory rotate around sneaker, ultrarealistic detail and breath-taking movie-like camera shot."
# prompt = "A nice sneaker on the betonic with a background of a modern city while camera trajectory rotate around sneaker, ultrarealistic detail and breath-taking movie-like camera shot."
negative_prompt = 'blurry, low-quality'

vid_path = f'/home/vahid/Downloads/Professional_Mode_beautiful_rotation_around_the_sn.mp4'
# vid_path = f'/home/vahid/Downloads/fd03b8c9-fd6f-48b0-8eec-55ff5db6e7d1_video.mp4'


video_inp = load_video(vid_path)


width, height = video_inp[0].size

# Calculate target size.
target_width = (round(width / 16) + 1) * 16
target_height = (round(height / 16) + 1) * 16

# img_inp = video_inp[0].resize((target_width, target_height))

width, height = (1360, 768)
# width, height = (768, 1360)
img_inp = video_inp[0].resize((width, height))

video_inp = []
for n in range(num_frames):
    video_inp.append(img_inp)


model_path = "/media/vahid/DATA/projects/CogVideo/models/CogVideoX1.5-5B-I2V"

# pipe = CogVideoXFramesToVideoPipeline.from_pretrained(model_path, torch_dtype=torch.bfloat16).to("cuda")
pipe = CogVideoXImageToVideoPipeline.from_pretrained(model_path, torch_dtype=torch.bfloat16).to("cuda")
pipe.vae.enable_slicing()
pipe.vae.enable_tiling()

# video = pipe(img_inp, frames=video_inp[:num_frames], prompt=prompt, negative_prompt=negative_prompt, num_frames=num_frames,
#              height=height, width=width, use_dynamic_cfg=True, num_inference_steps=50, use_noise_condition=use_noise_condition).frames[0]
video = pipe(image=img_inp, prompt=prompt, negative_prompt=negative_prompt, num_frames=num_frames,
             height=height, width=width, use_dynamic_cfg=True, num_inference_steps=50).frames[0]

vid_out = []
for v in video:
    vid_out.append(v)
export_to_video(vid_out, f"outputs_orig/volcano_81.mp4", fps=16)
