import os, glob, cv2, time

import numpy as np
from PIL import Image
import torch
from diffusers import CogVideoXFramesToVideoPipeline
from diffusers.utils import export_to_video, load_image, load_video


def add_noise_to_frame(frame, noise_type="blur", amplitude=10):
    """
    Adds noise to a single frame.

    Args:
        frame (numpy.ndarray): Input frame.
        noise_type (str): Type of noise ('gaussian' or 'blur').
        amplitude (float): Amplitude of the noise.

    Returns:
        numpy.ndarray: Frame with added noise.
    """
    if noise_type == "gaussian":
        noise = np.random.normal(0, amplitude, frame.shape).astype(np.uint8)
        noisy_frame = cv2.add(frame, noise)
    elif noise_type == "blur":
        kernel_size = max(1, int(amplitude))  # Ensure kernel size is at least 1
        if kernel_size % 2 == 0:
            kernel_size += 1  # Kernel size must be odd for cv2.GaussianBlur
        noisy_frame = cv2.GaussianBlur(frame, (kernel_size, kernel_size), 0)
    else:
        raise ValueError("Invalid noise_type. Choose 'gaussian' or 'blur'.")
    return noisy_frame


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


def crop_box(image, left, top, right, bottom):
    """Crops a PIL image to the target dimensions (multiples of 8) using center crop."""
    img_width, img_height = image.size

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
    left_box = 100000
    upper_box = 100000
    right_box = 0
    lower_box = 0

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
            left_box = min(left_box, left)
            upper_box = min(upper_box, upper)
            right_box = max(right_box, right)
            lower_box = max(lower_box, lower)

    # If any image is not transparent, assume the entire image is needed.
    if max_bbox_width == 0 or max_bbox_height == 0:
        max_bbox_width = max(img.width for img in images) if images else 0
        max_bbox_height = max(img.height for img in images) if images else 0

    ### create a margin to be sure all of the obejct is inside
    max_bbox_width *= 1.15
    max_bbox_height *= 1.15

    # Calculate target size.
    target_width = int(round(max_bbox_width) / 2) * 2
    target_height = int(round(max_bbox_height) / 2) * 2

    result = []
    img_size = max(target_width, target_height)
    for image in images:
        # cropped_img, offset = crop_to_multiple_of_8(image.resize((img_size, img_size)), target_width, target_height)
        cropped_img, offset = crop_box(image.resize((img_size, img_size)), left_box, upper_box, right_box, lower_box)
        result.append({"original": image, "cropped": cropped_img, "offset": offset})

    return result



num_frames = 9
skip = 3
use_noise_condition = False
infer_img = f'{num_frames}_{skip}_gs'
last_mode = False
if last_mode:
    infer_img += '_last'


# # traj = 'interpolate_11_to_54'
# # cap_id = 'prod_jfajardo_8qToo88NrhSxVKPGQ10a_yellow_potted_plant_360_cuNr4aNczbYk3gHqIU6g'
# # prompt_obj = 'A nice yellow potted plant'
#
# traj = 'interpolate_123_to_74'
# cap_id = 'prod_danielmiles_grEuokLbbZS9Nbapi8Jx_American_football_helmet_360_pSa79FIWoBJQ4HlyCl3e'
# prompt_obj = 'A American football helmet'
#
#
# prompt = f"{prompt_obj} in the video while camera trajectory is rotating around it, High quality, ultrarealistic detail and breath-taking movie-like camera shot."
# negative_prompt = 'blurry, low-quality'
#
# imgs_path = f'/media/vahid/DATA/data/animl_data/generated_video_data_interpolated_processed_/{cap_id}/video_gen_data/{traj}'
# # input_frame_list = sorted(glob.glob(f"{imgs_path}/grm/*.png"))
# cond_frame_list = sorted(glob.glob(f"{imgs_path}/gs/*.png"))
#
# cond_frame_list = cond_frame_list[::skip][:num_frames]
# # input_frame_list = input_frame_list[::skip][:num_frames]
#
# input_frame_list[0] = cond_frame_list[0]
# input_frame_list[-1] = cond_frame_list[-1]

# traj = 'interpolate_69_to_102'
# cap_id = 'prod_cuadra-jose0480_L3mwRbuOV0l32SZ2XDHo_sneaker_on_branded_box_360_FROlggrPWW0hwOyl79vF'
# prompt_obj = 'A sneaker'

# traj = 'interpolate_114_to_81'
# cap_id = 'prod_daneman124_scMvFiaPcEt3ok7on7Gg_anime_themed_packaging_360_TBXFLdZCAeC7TLw2bz5R'
# prompt_obj = 'A anime packaging'


# traj = 'interpolate_74_to_83'
# cap_id = 'prod_Wethenew_FFQqngSfrG1oyil7gT4s_Air_Jordan_1_High_Chicago_Lost_and_Founf_rOUey9qofolH3Nj1cpij'
# prompt_obj = 'A Air Jordan 1 High Chicago Lost and Foun'


# # traj = 'interpolate_125_to_51'
# # cap_id = 'prod_98qv7fsbbx_3Q0sWHqukKWh57dZXtxf_bag_of_crepe_mix_360_hDkepu8tUbK5nbViNxji'
# # prompt_obj = 'A bag of crepe mix'
#
# traj = 'interpolate_3_to_67'
# cap_id = 'prod_talia-graphicg_FcwnS8QqKaoiXxaaM1IO_Pac-Man_arcade_game_model_360_ZYvxAtp8CKidTE3TjJXN'
# prompt_obj = 'A Pac-Man arcade game model'
#
# # traj = 'interpolate_29_to_12'
# # cap_id = 'prod_UserTests_TpYfxf0PS1SiYhOkzgY3_decorative_indoor_plant_360_Quszez6fQc2iknWsL2M8'
# # prompt_obj = 'A indoor plant'
#
#
# prompt = f"{prompt_obj} in the video while camera trajectory is rotating around it, High quality, ultrarealistic detail and breath-taking movie-like camera shot."
# negative_prompt = 'blurry, low-quality'
#
# imgs_path = f'/media/vahid/DATA/data/animl_data/generated_video_data_interpolated_processed_/{cap_id}/video_gen_data/{traj}'
# # input_frame_list = sorted(glob.glob(f"{imgs_path}/grm/*.png"))
# input_frame_list = sorted(glob.glob(f"{imgs_path}/gs/*.png"))

traj = '0/bottom'
cap_id = 'dev/superuser_z7v9nureXRtGqa2aJl9K/Bouteille_de_vin_sur_table_quick_180_XdiO7Z4LmiEreHwF7xTr'
prompt_obj = 'A bottle of vine'

prompt = f"{prompt_obj} in the video while camera trajectory is rotating around it, High quality, ultrarealistic detail and breath-taking movie-like camera shot."
negative_prompt = 'blurry, low-quality'
imgs_path = f'/media/vahid/DATA/data/animl_data/trainings/{cap_id}/novel_views/novel_views/{traj}'
input_frame_list = sorted(glob.glob(f"{imgs_path}/gs_main/*.png"))


cap_id = cap_id.replace('/','_')
traj = traj.replace('/','_')

input_frame_list = input_frame_list[::skip][:num_frames]


# path_frame_first = f'{imgs_path}/gs/001.png'
# input_frame_list[0] = path_frame_first

# if last_mode:
#     last_frame_name = input_frame_list[-1].split('/')[-1]
#     path_frame_last = f'{imgs_path}/gs/{last_frame_name}'
#     input_frame_list[num_frames-1] = path_frame_last

img_size = 1024
bucket_size = np.asarray([768, 960, 1280])
processed_images = process_images(input_frame_list[:num_frames], img_size=img_size)
width, height = processed_images[0]['cropped'].width, processed_images[0]['cropped'].height
inp_w, inp_h = bucket_size[np.argmin(np.abs(bucket_size - width))], bucket_size[np.argmin(np.abs(bucket_size - height))]
inp_w, inp_h = 1280, 960

video_inp = []
video_inp_orig = []

for i, item in enumerate(processed_images):
    img_np = np.asarray(item['cropped']).copy()
    img_np[..., :3] = (img_np[..., :3] * (img_np[..., 3:] / 255.0)).astype(np.uint8) + (
            (1 - (img_np[..., 3:] / 255.0)) * 255).astype(np.uint8)
    # if i > 0:
    #     if i != len(processed_images) - 1:
    #         img_np = add_noise_to_frame(img_np, noise_type='blur', amplitude=50)
    #     else:
    #         if not last_mode:
    #             img_np = add_noise_to_frame(img_np, noise_type='blur', amplitude=50)

    img_pil = Image.fromarray(img_np).convert("RGB")
    video_inp_orig.append(img_pil)
    img_pil = img_pil.resize((inp_w, inp_h))

    video_inp.append(img_pil)
    item['cropped'] = Image.fromarray(img_np)

    img_np = np.asarray(item['original']).copy()
    img_np[..., :3] = (img_np[..., :3] * (img_np[..., 3:] / 255.0)).astype(np.uint8) + (
            (1 - (img_np[..., 3:] / 255.0)) * 255).astype(np.uint8)
    item['original'] = Image.fromarray(img_np)
    # video_inp.append(Image.fromarray(img_np).convert("RGB"))


fol_name = f'{cap_id[-5:]}_{traj}_{infer_img}'
dir_name = f'outputs_v15/{fol_name}/input'
os.makedirs(dir_name, exist_ok=True)
dir_name_orig = f'outputs_v15/{fol_name}/input_orig'
os.makedirs(dir_name_orig, exist_ok=True)
vid_out = []
for i, v in enumerate(video_inp):
    img_name = f'{dir_name}/{i:03d}.jpg'
    v.save(img_name)
    video_inp_orig[i].save(f'{dir_name_orig}/{i:03d}.jpg')

# export_to_video([image]+[image_last]+video_inp[:num_frames], f"input_vid_gengs_{traj}.mp4", fps=8)
export_to_video(video_inp[:num_frames], f"{dir_name}/input.mp4", fps=8)


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

model_path = "/media/vahid/DATA/projects/CogVideo/models/CogVideoX-5b-I2V"
# model_path = "/media/vahid/DATA/projects/CogVideo/models/CogVideoX1.5-5B-I2V"
model_path = f'{model_path}_guide'

# vae = AutoencoderKLCogVicogvideox-lora-v1.5__optimizer_adamw__steps_5000__lr-schedule_cosine_with_restarts__learning-rate_2e-deoX.from_pretrained("THUDM/CogVideoX-5b-I2V", subfolder="vae", torch_dtype=torch.bfloat16)
print('loading video model ...')
st = time.time()
pipe = CogVideoXFramesToVideoPipeline.from_pretrained(model_path, torch_dtype=torch.bfloat16).to("cuda")
# pipe = CogVideoXVideoToVideoPipeline.from_pretrained("THUDM/CogVideoX-5b-I2V", torch_dtype=torch.bfloat16).to("cuda")
lora_path = "/media/vahid/DATA/projects/cogvideox-factory/runs/"

if last_mode:
    lora_path += "cogvideox-lora-v1.5_last_optimizer_adamw__steps_4500__lr-schedule_cosine_with_restarts__learning-rate_2e-4/checkpoint-2000"
else:
    lora_path += "cogvideox-lora_v1-upscale__steps_9000__learning-rate_1e-5/checkpoint-750"



lora_scaling = 1.0

print('finished in ', time.time()-st)
st = time.time()
print('loading lora weights ...')
pipe.load_lora_weights(lora_path, weight_name="pytorch_lora_weights.safetensors", adapter_name="test_1")
pipe.set_adapters(["test_1"], [lora_scaling])
pipe.vae.enable_tiling()
print('finished in ', time.time()-st)
st = time.time()
print('running inference ...')

if last_mode:
    video = pipe([video_inp[0],video_inp[-1]], frames=video_inp[:num_frames], prompt=prompt, negative_prompt=negative_prompt, num_frames=num_frames,
                 height=video_inp[0].height, width=video_inp[0].width, use_dynamic_cfg=True, num_inference_steps=50, use_noise_condition=use_noise_condition).frames[0]
else:
    video = pipe(video_inp[0], frames=video_inp[:num_frames], prompt=prompt, negative_prompt=negative_prompt, num_frames=num_frames,
                 height=video_inp[0].height, width=video_inp[0].width, use_dynamic_cfg=True, num_inference_steps=50, use_noise_condition=use_noise_condition).frames[0]

print('finished in ', time.time()-st)
dir_name = f'outputs_v15/{fol_name}/output_raw'
os.makedirs(dir_name, exist_ok=True)
vid_out = []
for i, v in enumerate(video):
    vid_out.append(v)
    img_name = f'{dir_name}/{i:03d}.jpg'
    v.save(img_name)
export_to_video(vid_out, f"{dir_name}/raw.mp4", fps=8)

vid_orig_out = []
dir_name = f'outputs_v15/{fol_name}/output_orig'
os.makedirs(dir_name, exist_ok=True)
for i, item in enumerate(processed_images[:len(vid_out)]):
    processed_cropped_img = vid_out[i].resize((width, height))
    processed_cropped_img.save(f'{dir_name}/{i:03d}.jpg')
    video_inp.append(item['cropped'])
    original_img = item['original'].convert("RGB")
    offset = item['offset']
    pasted_img = paste_cropped_back(original_img, processed_cropped_img, offset)
    vid_orig_out.append(pasted_img)

export_to_video(vid_orig_out, f"outputs_v15/output_original_{cap_id[-5:]}_{traj}_{infer_img}.mp4", fps=8)
