import os, glob

import numpy as np
from PIL import Image
import torch
from diffusers import CogVideoXFramesToVideoPipeline
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



num_frames = 33
use_noise_condition = False
infer_img = 'real'

traj = '0/bottom'
cap_id = 'dev/superuser_z7v9nureXRtGqa2aJl9K/Bouteille_de_vin_sur_table_quick_180_XdiO7Z4LmiEreHwF7xTr'
prompt = "A nice Air-jordan sneaker on the beach under heavy sun while camera trajectory to bottom, High quality, ultrarealistic detail and breath-taking movie-like camera shot."
negative_prompt = 'blurry, low-quality'

imgs_path = f'/media/vahid/DATA/data/animl_data/trainings/{cap_id}/aug_views/novel_views_0/{traj}/gs_main'
img_path = f'{imgs_path}/001.png'

img_size = 2048

# video_inp = []
# for n, im_path in enumerate(sorted(glob.glob(f"{imgs_path}/*.png"))):
#     image = load_image(im_path).resize((img_size,img_size))
#     video_inp.append(image)
#
# for n in range(len(video_inp), num_frames):
#     video_inp.append(image)

processed_images = process_images(sorted(glob.glob(f"{imgs_path}/*.png")), img_size=img_size)

video_inp = []
for i, item in enumerate(processed_images):
    img_np = np.asarray(item['cropped']).copy()
    img_np[..., :3] = (img_np[..., :3] * (img_np[..., 3:] / 255.0)).astype(np.uint8) + (
            (1 - (img_np[..., 3:] / 255.0)) * 255).astype(np.uint8)
    video_inp.append(Image.fromarray(img_np).convert("RGB"))
    item['cropped'] = Image.fromarray(img_np)

    img_np = np.asarray(item['original']).copy()
    img_np[..., :3] = (img_np[..., :3] * (img_np[..., 3:] / 255.0)).astype(np.uint8) + (
            (1 - (img_np[..., 3:] / 255.0)) * 255).astype(np.uint8)
    item['original'] = Image.fromarray(img_np)
    # video_inp.append(Image.fromarray(img_np).convert("RGB"))

width, height = video_inp[0].size

cap_id = cap_id.replace('/','_')
traj = traj.replace('/','_')

# export_to_video([image]+[image_last]+video_inp[:num_frames], f"input_vid_gengs_{traj}.mp4", fps=8)
export_to_video([video_inp[0]]+video_inp[:num_frames], f"outputs/input_cropped_{cap_id[-5:]}_{traj}_{infer_img}.mp4", fps=8)


# img_list = sorted(glob.glob(f"{imgs_path}/*.png"))
# imgs_path = novel_view_replicate_infer("https://replicate.delivery/yhqm/eSrDBDed6EkQIEha3VBfVCcHYRlMwhE6jJBfALGmdQcKQZjOB/trained_model.tar", 'TRAINEDOBJECT', img_list, 'lora_data', controlnet_scale=0.6)

# imgs_path = f"left_71/lora/output_infer"
# imgs_path = f"{traj}/lora/output_infer"

# video_inp_lora = []
# for n, im_path in enumerate(sorted(glob.glob(f"{imgs_path}/*.png"))):
#     image = load_image(im_path)
#     video_inp_lora.append(image)
#
# for n in range(len(video_inp_lora), num_frames):
#     video_inp_lora.append(image)

# model_path = "/media/vahid/DATA/projects/CogVideo/models/CogVideoX-5b-I2V"
model_path = "/media/vahid/DATA/projects/CogVideo/models/CogVideoX1.5-5B-I2V"

# vae = AutoencoderKLCogVicogvideox-lora-v1.5__optimizer_adamw__steps_5000__lr-schedule_cosine_with_restarts__learning-rate_2e-deoX.from_pretrained("THUDM/CogVideoX-5b-I2V", subfolder="vae", torch_dtype=torch.bfloat16)
pipe = CogVideoXFramesToVideoPipeline.from_pretrained(model_path, torch_dtype=torch.bfloat16).to("cuda")
# pipe = CogVideoXVideoToVideoPipeline.from_pretrained("THUDM/CogVideoX-5b-I2V", torch_dtype=torch.bfloat16).to("cuda")
lora_path = "/media/vahid/DATA/projects/cogvideox-factory/runs/"
lora_path += "cogvideox-lora-v1.5__optimizer_adamw__steps_6000__lr-schedule_cosine_with_restarts__learning-rate_2e-4/checkpoint-5800"
lora_rank = 256
lora_alpha = 256
lora_scaling = lora_alpha / lora_rank

pipe.load_lora_weights(lora_path, weight_name="pytorch_lora_weights.safetensors", adapter_name="test_1")
# pipe.fuse_lora(lora_scale=lora_scaling)
pipe.set_adapters(["test_1"], [lora_scaling])

# # pipe.to("cuda")

pipe.vae.enable_tiling()



# vid_path = '/media/vahid/DATA/projects/cogvideox-factory/assets/tests/videos/sneaker_side.mp4'
# inp_vid = load_video(vid_path)
# video = pipe(image, prompt, num_frames=num_frames, use_dynamic_cfg=True)

video = pipe(video_inp[0], frames=video_inp[:num_frames], prompt=prompt, negative_prompt=negative_prompt, num_frames=num_frames,
             height=height, width=width, use_dynamic_cfg=True, num_inference_steps=50, use_noise_condition=use_noise_condition).frames[0]


vid_out = []
for v in video:
    vid_out.append(v)
export_to_video(vid_out, f"outputs/output_cropped_{cap_id[-5:]}_{traj}_{infer_img}.mp4", fps=8)

vid_orig_out = []
dir_name = f'outputs/output_cropped_{cap_id[-5:]}_{traj}_{infer_img}'
os.makedirs(dir_name, exist_ok=True)
for i, item in enumerate(processed_images[:len(vid_out)]):
    processed_cropped_img = vid_out[i]
    video_inp.append(item['cropped'])
    original_img = item['original'].convert("RGB")
    offset = item['offset']
    pasted_img = paste_cropped_back(original_img, processed_cropped_img, offset)
    vid_orig_out.append(pasted_img)
    img_name = f'{dir_name}/{i:03d}.jpg'
    pasted_img.save(img_name)

export_to_video(vid_orig_out, f"outputs/output_original_{cap_id[-5:]}_{traj}_{infer_img}.mp4", fps=8)
