import os, glob, cv2
import torch
from diffusers import CogVideoXFramesToVideoPipeline



model_path = "THUDM/CogVideoX1.5-5B-I2V"
save_path = "/data/cogvideox-factory-animl/ckpt/CogVideoX1.5-5B-I2V_guide"

pipe = CogVideoXFramesToVideoPipeline.from_pretrained(model_path, torch_dtype=torch.bfloat16).to('cuda')

lora_path = "/data/cogvideox-factory-animl/runs/cogvideox-lora_v1-fast__steps_9000__learning-rate_1e-5/checkpoint-9500"
lora_name = "pytorch_lora_weights.safetensors"

lora_scaling = 1.0

pipe.load_lora_weights(lora_path, weight_name=lora_name, adapter_name="test_1")
pipe.set_adapters(["test_1"], [lora_scaling])
pipe.fuse_lora(adapter_names=["test_1"], lora_scale=lora_scaling)
pipe.unload_lora_weights()

pipe.save_pretrained(f"{save_path}")