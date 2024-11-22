import os, glob

import torch
from diffusers import CogVideoXFramesToVideoPipeline
from diffusers.utils import export_to_video, load_image, load_video

from diffusers import AutoencoderKLCogVideoX, CogVideoXVideoToVideoPipeline, CogVideoXTransformer3DModel, CogVideoXDPMScheduler
from transformers import T5EncoderModel
num_frames = 17
use_noise_condition = False
traj = 'left_179'
prompt = "A nice sneaker. High quality, ultrarealistic detail and breath-taking movie-like camera shot."
negative_prompt = 'blurry, low-quality'
# imgs_path = '/media/vahid/DATA/data/animl_data/generated_video_data_processed/prod_akmeso2k11_uzvgktub4XaXSPBHebTp_white_and_green_sneakers_quick_180_wNoyIxBvatRwV1gdEQMr/video_gen_data/left_70/grm'
# img_path = '/media/vahid/DATA/data/animl_data/generated_video_data_processed/prod_akmeso2k11_uzvgktub4XaXSPBHebTp_white_and_green_sneakers_quick_180_wNoyIxBvatRwV1gdEQMr/video_gen_data/left_70/gs/000.png'
# img_path = '/media/vahid/DATA/data/animl_data/trainings/prod/UserTests_TpYfxf0PS1SiYhOkzgY3/Chaussure_de_sport_verte_360_LxBa6OKOzA08RauwfyHt/gengs/inference_images/002.png'
imgs_path = f'/media/vahid/DATA/data/animl_data/generated_video_data_processed/prod_zenithattireglobal_nLe5o4J96EsRRUXOGefe_pair_of_beige_sneakers_360_ovHorTi2EvXgUifx39Jy/video_gen_data/{traj}/gengs'
img_path = f'/media/vahid/DATA/data/animl_data/generated_video_data_processed/prod_zenithattireglobal_nLe5o4J96EsRRUXOGefe_pair_of_beige_sneakers_360_ovHorTi2EvXgUifx39Jy/video_gen_data/{traj}/gs/000.png'
img_path_last = f'/media/vahid/DATA/data/animl_data/generated_video_data_processed/prod_zenithattireglobal_nLe5o4J96EsRRUXOGefe_pair_of_beige_sneakers_360_ovHorTi2EvXgUifx39Jy/video_gen_data/{traj}/gs/{num_frames-1:03d}.png'



video_inp = []
for n, im_path in enumerate(sorted(glob.glob(f"{imgs_path}/*.png"))):
    image = load_image(im_path)
    video_inp.append(image)

for n in range(len(video_inp), num_frames):
    video_inp.append(image)

image = load_image(img_path).resize((512,512))
image_last = load_image(img_path_last).resize((512,512))

export_to_video([image]+[image_last]+video_inp[:num_frames], f"input_vid_gengs_{traj}.mp4", fps=8)


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

# vae = AutoencoderKLCogVideoX.from_pretrained("THUDM/CogVideoX-5b-I2V", subfolder="vae", torch_dtype=torch.bfloat16)
pipe = CogVideoXFramesToVideoPipeline.from_pretrained("/media/vahid/DATA/projects/CogVideo/models/CogVideoX-5b-I2V", torch_dtype=torch.bfloat16).to("cuda")
# pipe = CogVideoXVideoToVideoPipeline.from_pretrained("THUDM/CogVideoX-5b-I2V", torch_dtype=torch.bfloat16).to("cuda")
lora_path = "/media/vahid/DATA/projects/cogvideox-factory/runs/cogvideox-lora__optimizer_adamw__steps_4500__lr-schedule_cosine_with_restarts__learning-rate_2e-4/checkpoint-2000"
lora_rank = 256
lora_alpha = 256
lora_scaling = lora_alpha / lora_rank
pipe.load_lora_weights(lora_path, weight_name="pytorch_lora_weights.safetensors", adapter_name="test_1")
# pipe.fuse_lora(lora_scale=lora_scaling)
pipe.set_adapters(["test_1"], [lora_scaling])

# # pipe.to("cuda")

# pipe.vae.enable_tiling()



# vid_path = '/media/vahid/DATA/projects/cogvideox-factory/assets/tests/videos/sneaker_side.mp4'
# inp_vid = load_video(vid_path)
# video = pipe(image, prompt, num_frames=num_frames, use_dynamic_cfg=True)

video = pipe(image, frames=video_inp[:num_frames], prompt=prompt, negative_prompt=negative_prompt,
                          use_dynamic_cfg=True, num_inference_steps=50, use_noise_condition=use_noise_condition)
export_to_video(video.frames[0][:30], f"output_vid_gengs_{traj}_g5_50_6_lora_cond.mp4", fps=8)

# video = pipe(image, frames=video_inp_lora[:num_frames], prompt=prompt, negative_prompt=negative_prompt,
#                           use_dynamic_cfg=True, num_inference_steps=50, use_noise_condition=True)
# export_to_video(video.frames[0][:30], f"output_vid_lora_{traj}_g5_50_6_lora_test.mp4", fps=8)