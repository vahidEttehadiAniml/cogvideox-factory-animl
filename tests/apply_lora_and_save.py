import os, glob, cv2
import torch
from diffusers import CogVideoXFramesToVideoPipeline



model_path = "THUDM/CogVideoX-5b-I2V"
save_path = "/data/cogvideox-factory-animl/ckpt/CogVideoX-5b-I2V_guide"

# model_path = "/media/vahid/DATA/projects/CogVideo/models/CogVideoX1.5-5B-I2V"
# save_path = f"{model_path}_last"

pipe = CogVideoXFramesToVideoPipeline.from_pretrained(model_path, torch_dtype=torch.bfloat16)

lora_path = "/data/cogvideox-factory-animl/loras_weights"
lora_name = "v1_guide_49f_2750.safetensors"

lora_scaling = 1.0

pipe.load_lora_weights(lora_path, weight_name=lora_name, adapter_name="guide")
pipe.set_adapters(["guide"], [lora_scaling])
pipe.fuse_lora(adapter_names=["guide"], lora_scale=lora_scaling)
pipe.unload_lora_weights()

pipe.save_pretrained(f"{save_path}")