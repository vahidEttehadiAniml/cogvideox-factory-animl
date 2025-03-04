import os, glob, cv2
import torch
from diffusers import CogVideoXFramesToVideoPipeline



# model_path = "THUDM/CogVideoX-5b-I2V"
model_path = "/media/vahid/DATA/projects/cogvideox-factory/ckpt/CogVideoX-5b-I2V_guide"

save_path = "/media/vahid/DATA/projects/cogvideox-factory/ckpt/CogVideoX-5b-I2V_lastup"

# model_path = "/media/vahid/DATA/projects/CogVideo/models/CogVideoX1.5-5B-I2V"
# save_path = f"{model_path}_last"

pipe = CogVideoXFramesToVideoPipeline.from_pretrained(model_path, torch_dtype=torch.bfloat16).to("cuda")

lora_path = "/media/vahid/DATA/projects/cogvideox-factory/runs/cogvideox-lora-v1_guide_upscale_49__steps_10000__learning-rate_5e-5/checkpoint-1500"
lora_name = "pytorch_lora_weights.safetensors"

lora_scaling = 1.0

pipe.load_lora_weights(lora_path, weight_name=lora_name, adapter_name="guide")
# pipe.set_adapters(["guide"], [lora_scaling])
pipe.fuse_lora(adapter_names=["guide"], lora_scale=lora_scaling)
pipe.unload_lora_weights()

pipe.save_pretrained(f"{save_path}")