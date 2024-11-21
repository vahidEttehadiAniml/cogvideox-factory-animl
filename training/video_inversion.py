import os
import imageio
import tempfile
import numpy as np
from PIL import Image
from typing import Union

import torch
import torchvision

from tqdm import tqdm
from einops import rearrange


@torch.no_grad()
def init_prompt(prompt, pipeline):
    uncond_input = pipeline.tokenizer(
        [""], padding="max_length", max_length=pipeline.tokenizer.model_max_length,
        return_tensors="pt"
    )
    uncond_embeddings = pipeline.text_encoder(uncond_input.input_ids.to(pipeline.device))[0]
    text_input = pipeline.tokenizer(
        [prompt],
        padding="max_length",
        max_length=pipeline.tokenizer.model_max_length,
        truncation=True,
        return_tensors="pt",
    )
    text_embeddings = pipeline.text_encoder(text_input.input_ids.to(pipeline.device))[0]
    context = torch.cat([uncond_embeddings, text_embeddings])

    return context


def next_step(model_output: Union[torch.FloatTensor, np.ndarray], timestep: int,
              sample: Union[torch.FloatTensor, np.ndarray], ddim_scheduler):
    timestep, next_timestep = min(
        timestep - ddim_scheduler.config.num_train_timesteps // ddim_scheduler.num_inference_steps, 999), timestep
    alpha_prod_t = ddim_scheduler.alphas_cumprod[timestep] if timestep >= 0 else ddim_scheduler.final_alpha_cumprod
    alpha_prod_t_next = ddim_scheduler.alphas_cumprod[next_timestep]
    beta_prod_t = 1 - alpha_prod_t
    next_original_sample = (sample - beta_prod_t ** 0.5 * model_output) / alpha_prod_t ** 0.5
    next_sample_direction = (1 - alpha_prod_t_next) ** 0.5 * model_output
    next_sample = alpha_prod_t_next ** 0.5 * next_original_sample + next_sample_direction
    return next_sample


def get_noise_pred_single(latents, t, context, transformer, image_rotary_emb, attention_kwargs):
    bs = latents.shape[0]  # (b*f, c, h, w) or (b, c, f, h, w)
    if bs != context.shape[0]:
        context = context.repeat(bs, 1, 1)  # (b*f, len, dim)
    noise_pred = transformer(
        hidden_states=latents,
        encoder_hidden_states=context,
        timestep=t,
        image_rotary_emb=image_rotary_emb,
        attention_kwargs=attention_kwargs,
        return_dict=False,
    )[0]

    return noise_pred


@torch.no_grad()
def ddim_loop(pipeline, ddim_scheduler, latent, num_inv_steps, prompt):
    context = init_prompt(prompt, pipeline)
    uncond_embeddings, cond_embeddings = context.chunk(2)
    all_latent = [latent]
    latent = latent.clone().detach()
    height, width = latent.shape[-2] * 8, latent.shape[-1] * 8
    image_rotary_emb = (
        pipeline._prepare_rotary_positional_embeddings(height, width, latent.size(1), pipeline._execution_device)
        if pipeline.transformer.config.use_rotary_positional_embeddings
        else None
    )
    for i in tqdm(range(num_inv_steps)):
        t = ddim_scheduler.timesteps[len(ddim_scheduler.timesteps) - i - 1]
        noise_pred = get_noise_pred_single(latent, t, cond_embeddings, pipeline.unet, image_rotary_emb, attention_kwargs=None)
        latent = next_step(noise_pred, t, latent, ddim_scheduler)
        all_latent.append(latent)
    return all_latent


@torch.no_grad()
def ddim_inversion(pipeline, ddim_scheduler, video_latent, num_inv_steps, prompt=""):
    ddim_latents = ddim_loop(pipeline, ddim_scheduler, video_latent, num_inv_steps, prompt)
    return ddim_latents



def video_inversion(pipeline, prompts_emb, video_latents, num_inv_steps):
    # Sample noise that will be added to the latents
    noise = torch.randn_like(video_latents)
    batch_size, num_frames, num_channels, height, width = video_latents.shape
    device = pipeline._execution_device
    prompt_embeds = prompts_emb.to(device)

    timesteps = torch.Tensor([num_inv_steps]*batch_size, dtype=torch.int64).to(device)

    # Prepare rotary embeds
    image_rotary_emb = (
        prepare_rotary_positional_embeddings(
            height=height * VAE_SCALE_FACTOR_SPATIAL,
            width=width * VAE_SCALE_FACTOR_SPATIAL,
            num_frames=num_frames,
            vae_scale_factor_spatial=VAE_SCALE_FACTOR_SPATIAL,
            patch_size=model_config.patch_size,
            attention_head_dim=model_config.attention_head_dim,
            device=accelerator.device,
        )
        if model_config.use_rotary_positional_embeddings
        else None
    )

    # Add noise to the model input according to the noise magnitude at each timestep
    # (this is the forward diffusion process)
    noisy_video_latents = scheduler.add_noise(video_latents, noise, timesteps)
    noisy_model_input = torch.cat([noisy_video_latents, image_latents], dim=2)

    # Predict the noise residual
    model_output = transformer(
        hidden_states=noisy_model_input,
        encoder_hidden_states=prompt_embeds,
        timestep=timesteps,
        image_rotary_emb=image_rotary_emb,
        return_dict=False,
    )[0]

    model_pred = scheduler.get_velocity(model_output, noisy_video_latents, timesteps)