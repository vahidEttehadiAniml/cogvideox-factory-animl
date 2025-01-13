#!/usr/bin/env python3

import argparse
import functools
import json
import os
import pathlib
import queue
import traceback
import uuid
from concurrent.futures import ThreadPoolExecutor
from typing import Any, Dict, List, Optional, Union

import torch
import torch.distributed as dist
from diffusers import AutoencoderKLCogVideoX
from diffusers.training_utils import set_seed
from diffusers.utils import export_to_video, get_logger
from torch.utils.data import DataLoader
from torchvision import transforms
from tqdm import tqdm
from transformers import T5EncoderModel, T5Tokenizer


import decord  # isort:skip

from training.cogvideox.dataset import BucketSampler, VideoDatasetWithResizing, VideoDatasetWithResizeAndRectangleCrop  # isort:skip


decord.bridge.set_bridge("torch")

logger = get_logger(__name__)

DTYPE_MAPPING = {
    "fp32": torch.float32,
    "fp16": torch.float16,
    "bf16": torch.bfloat16,
}

def check_height(x: Any) -> int:
    x = int(x)
    if x % 16 != 0:
        raise argparse.ArgumentTypeError(
            f"`--height_buckets` must be divisible by 16, but got {x} which does not fit criteria."
        )
    return x


def check_width(x: Any) -> int:
    x = int(x)
    if x % 16 != 0:
        raise argparse.ArgumentTypeError(
            f"`--width_buckets` must be divisible by 16, but got {x} which does not fit criteria."
        )
    return x


def check_frames(x: Any) -> int:
    x = int(x)
    if x % 4 != 0 and x % 4 != 1:
        raise argparse.ArgumentTypeError(
            f"`--frames_buckets` must be of form `4 * k` or `4 * k + 1`, but got {x} which does not fit criteria."
        )
    return x


def get_args() -> Dict[str, Any]:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--model_id",
        type=str,
        default="THUDM/CogVideoX-2b",
        help="Hugging Face model ID to use for tokenizer, text encoder and VAE.",
    )
    parser.add_argument("--data_root", type=str, required=True, help="Path to where training data is located.")
    parser.add_argument(
        "--dataset_file", type=str, default=None, help="Path to CSV file containing metadata about training data."
    )
    parser.add_argument(
        "--caption_column",
        type=str,
        default="caption",
        help="If using a CSV file via the `--dataset_file` argument, this should be the name of the column containing the captions. If using the folder structure format for data loading, this should be the name of the file containing line-separated captions (the file should be located in `--data_root`).",
    )
    parser.add_argument(
        "--video_column",
        type=str,
        default="video",
        help="If using a CSV file via the `--dataset_file` argument, this should be the name of the column containing the video paths. If using the folder structure format for data loading, this should be the name of the file containing line-separated video paths (the file should be located in `--data_root`).",
    )
    parser.add_argument(
        "--id_token",
        type=str,
        default=None,
        help="Identifier token appended to the start of each prompt if provided.",
    )
    parser.add_argument(
        "--height_buckets",
        nargs="+",
        type=check_height,
        default=[256, 320, 384, 480, 512, 576, 720, 768, 960, 1024, 1280, 1536],
    )
    parser.add_argument(
        "--width_buckets",
        nargs="+",
        type=check_width,
        default=[256, 320, 384, 480, 512, 576, 720, 768, 960, 1024, 1280, 1536],
    )
    parser.add_argument(
        "--frame_buckets",
        nargs="+",
        type=check_frames,
        default=[49],
    )
    parser.add_argument(
        "--random_flip",
        type=float,
        default=None,
        help="If random horizontal flip augmentation is to be used, this should be the flip probability.",
    )
    parser.add_argument(
        "--dataloader_num_workers",
        type=int,
        default=0,
        help="Number of subprocesses to use for data loading. 0 means that the data will be loaded in the main process.",
    )
    parser.add_argument(
        "--pin_memory",
        action="store_true",
        help="Whether or not to use the pinned memory setting in pytorch dataloader.",
    )
    parser.add_argument(
        "--video_reshape_mode",
        type=str,
        default=None,
        help="All input videos are reshaped to this mode. Choose between ['center', 'random', 'none']",
    )
    parser.add_argument(
        "--save_image_latents",
        action="store_true",
        help="Whether or not to encode and store image latents, which are required for image-to-video finetuning. The image latents are the first frame of input videos encoded with the VAE.",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        required=True,
        help="Path to output directory where preprocessed videos/latents/embeddings will be saved.",
    )
    parser.add_argument("--max_num_frames", type=int, default=49, help="Maximum number of frames in output video.")
    parser.add_argument(
        "--max_sequence_length", type=int, default=226, help="Max sequence length of prompt embeddings."
    )
    parser.add_argument("--target_fps", type=int, default=8, help="Frame rate of output videos.")
    parser.add_argument(
        "--save_latents_and_embeddings",
        action="store_true",
        help="Whether to encode videos/captions to latents/embeddings and save them in pytorch serializable format.",
    )
    parser.add_argument(
        "--use_slicing",
        action="store_true",
        help="Whether to enable sliced encoding/decoding in the VAE. Only used if `--save_latents_and_embeddings` is also used.",
    )
    parser.add_argument(
        "--use_tiling",
        action="store_true",
        help="Whether to enable tiled encoding/decoding in the VAE. Only used if `--save_latents_and_embeddings` is also used.",
    )
    parser.add_argument("--batch_size", type=int, default=1, help="Number of videos to process at once in the VAE.")
    parser.add_argument(
        "--num_decode_threads",
        type=int,
        default=0,
        help="Number of decoding threads for `decord` to use. The default `0` means to automatically determine required number of threads.",
    )
    parser.add_argument(
        "--dtype",
        type=str,
        choices=["fp32", "fp16", "bf16"],
        default="fp32",
        help="Data type to use when generating latents and prompt embeddings.",
    )
    parser.add_argument("--seed", type=int, default=42, help="Seed for reproducibility.")
    parser.add_argument(
        "--num_artifact_workers", type=int, default=4, help="Number of worker threads for serializing artifacts."
    )
    return parser.parse_args()


def _get_t5_prompt_embeds(
    tokenizer: T5Tokenizer,
    text_encoder: T5EncoderModel,
    prompt: Union[str, List[str]],
    num_videos_per_prompt: int = 1,
    max_sequence_length: int = 226,
    device: Optional[torch.device] = None,
    dtype: Optional[torch.dtype] = None,
    text_input_ids=None,
):
    prompt = [prompt] if isinstance(prompt, str) else prompt
    batch_size = len(prompt)

    if tokenizer is not None:
        text_inputs = tokenizer(
            prompt,
            padding="max_length",
            max_length=max_sequence_length,
            truncation=True,
            add_special_tokens=True,
            return_tensors="pt",
        )
        text_input_ids = text_inputs.input_ids
    else:
        if text_input_ids is None:
            raise ValueError("`text_input_ids` must be provided when the tokenizer is not specified.")

    prompt_embeds = text_encoder(text_input_ids.to(device))[0]
    prompt_embeds = prompt_embeds.to(dtype=dtype, device=device)

    # duplicate text embeddings for each generation per prompt, using mps friendly method
    _, seq_len, _ = prompt_embeds.shape
    prompt_embeds = prompt_embeds.repeat(1, num_videos_per_prompt, 1)
    prompt_embeds = prompt_embeds.view(batch_size * num_videos_per_prompt, seq_len, -1)

    return prompt_embeds


def encode_prompt(
    tokenizer: T5Tokenizer,
    text_encoder: T5EncoderModel,
    prompt: Union[str, List[str]],
    num_videos_per_prompt: int = 1,
    max_sequence_length: int = 226,
    device: Optional[torch.device] = None,
    dtype: Optional[torch.dtype] = None,
    text_input_ids=None,
):
    prompt = [prompt] if isinstance(prompt, str) else prompt
    prompt_embeds = _get_t5_prompt_embeds(
        tokenizer,
        text_encoder,
        prompt=prompt,
        num_videos_per_prompt=num_videos_per_prompt,
        max_sequence_length=max_sequence_length,
        device=device,
        dtype=dtype,
        text_input_ids=text_input_ids,
    )
    return prompt_embeds


def compute_prompt_embeddings(
    tokenizer: T5Tokenizer,
    text_encoder: T5EncoderModel,
    prompts: List[str],
    max_sequence_length: int,
    device: torch.device,
    dtype: torch.dtype,
    requires_grad: bool = False,
):
    if requires_grad:
        prompt_embeds = encode_prompt(
            tokenizer,
            text_encoder,
            prompts,
            num_videos_per_prompt=1,
            max_sequence_length=max_sequence_length,
            device=device,
            dtype=dtype,
        )
    else:
        with torch.no_grad():
            prompt_embeds = encode_prompt(
                tokenizer,
                text_encoder,
                prompts,
                num_videos_per_prompt=1,
                max_sequence_length=max_sequence_length,
                device=device,
                dtype=dtype,
            )
    return prompt_embeds


to_pil_image = transforms.ToPILImage(mode="RGB")


def save_image(image: torch.Tensor, path: pathlib.Path) -> None:
    image = to_pil_image(image)
    image.save(path)


def save_video(video: torch.Tensor, path: pathlib.Path, fps: int = 8) -> None:
    video = [to_pil_image(frame) for frame in video]
    export_to_video(video, path, fps=fps)


def save_prompt(prompt: str, path: pathlib.Path) -> None:
    with open(path, "w", encoding="utf-8") as file:
        file.write(prompt)


def save_metadata(metadata: Dict[str, Any], path: pathlib.Path) -> None:
    with open(path, "w", encoding="utf-8") as file:
        file.write(json.dumps(metadata))



@torch.no_grad()
def main():
    args = get_args()
    set_seed(args.seed)

    output_dir = pathlib.Path(args.output_dir)
    tmp_dir = output_dir.joinpath("tmp")

    output_dir.mkdir(parents=True, exist_ok=True)
    tmp_dir.mkdir(parents=True, exist_ok=True)

    # Initialize distributed processing
    if "LOCAL_RANK" in os.environ:
        local_rank = int(os.environ["LOCAL_RANK"])
        torch.cuda.set_device(local_rank)
        dist.init_process_group(backend="nccl")
        world_size = dist.get_world_size()
        rank = dist.get_rank()
    else:
        # Single GPU
        local_rank = 0
        world_size = 1
        rank = 0
        torch.cuda.set_device(rank)

    # Create folders for saving intermediates
    images_dir = output_dir.joinpath("images")
    image_latents_dir = output_dir.joinpath("image_latents")
    videos_dir = output_dir.joinpath("videos")
    video_latents_dir = output_dir.joinpath("video_latents")
    prompts_dir = output_dir.joinpath("prompts")
    prompt_embeds_dir = output_dir.joinpath("prompt_embeds")

    # Create directories
    images_dir.mkdir(parents=True, exist_ok=True)
    image_latents_dir.mkdir(parents=True, exist_ok=True)
    videos_dir.mkdir(parents=True, exist_ok=True)
    video_latents_dir.mkdir(parents=True, exist_ok=True)
    prompts_dir.mkdir(parents=True, exist_ok=True)
    prompt_embeds_dir.mkdir(parents=True, exist_ok=True)

    weight_dtype = DTYPE_MAPPING[args.dtype]
    target_fps = args.target_fps

    # 1. Dataset
    dataset_init_kwargs = {
        "data_root": args.data_root,
        "dataset_file": args.dataset_file,
        "caption_column": args.caption_column,
        "video_column": args.video_column,
        "max_num_frames": args.max_num_frames,
        "id_token": args.id_token,
        "height_buckets": args.height_buckets,
        "width_buckets": args.width_buckets,
        "frame_buckets": args.frame_buckets,
        "load_tensors": False,
        "random_flip": args.random_flip,
        "image_to_video": args.save_image_latents,
    }
    if args.video_reshape_mode is None:
        dataset = VideoDatasetWithResizing(**dataset_init_kwargs)
    else:
        dataset = VideoDatasetWithResizeAndRectangleCrop(
            video_reshape_mode=args.video_reshape_mode, **dataset_init_kwargs
        )

    original_dataset_size = len(dataset)

    # Split data among GPUs
    if world_size > 1:
        samples_per_gpu = original_dataset_size // world_size
        start_index = rank * samples_per_gpu
        end_index = start_index + samples_per_gpu
        if rank == world_size - 1:
            end_index = original_dataset_size  # Make sure the last GPU gets the remaining data

        # Slice the data
        dataset.prompts = dataset.prompts[start_index:end_index]
        dataset.video_paths = dataset.video_paths[start_index:end_index]
    else:
        pass

    rank_dataset_size = len(dataset)

    # 2. Dataloader
    def collate_fn(data):
        prompts = [x["prompt"] for x in data[0]]

        images = None
        if args.save_image_latents:
            images = [x["image"] for x in data[0]]
            images = torch.stack(images).to(dtype=weight_dtype, non_blocking=True)

        videos = [x["video"] for x in data[0]]
        videos = torch.stack(videos).to(dtype=weight_dtype, non_blocking=True)

        return {
            "images": images,
            "videos": videos,
            "prompts": prompts,
        }

    dataloader = DataLoader(
        dataset,
        batch_size=1,
        sampler=BucketSampler(dataset, batch_size=args.batch_size, shuffle=True, drop_last=False),
        collate_fn=collate_fn,
        num_workers=args.dataloader_num_workers,
        pin_memory=args.pin_memory,
    )

    # 3. Prepare models
    device = f"cuda:{rank}"

    if args.save_latents_and_embeddings:
        tokenizer = T5Tokenizer.from_pretrained(args.model_id, subfolder="tokenizer")
        text_encoder = T5EncoderModel.from_pretrained(
            args.model_id, subfolder="text_encoder", torch_dtype=weight_dtype
        )
        text_encoder = text_encoder.to(device)

        vae = AutoencoderKLCogVideoX.from_pretrained(args.model_id, subfolder="vae", torch_dtype=weight_dtype)
        vae = vae.to(device)

        if args.use_slicing:
            vae.enable_slicing()
        if args.use_tiling:
            vae.enable_tiling()

    # Tracks processed files for creating final JSONL
    processed_files = []

    # 4. Compute latents and embeddings and save
    if rank == 0:
        iterator = tqdm(
            dataloader, desc="Encoding", total=(rank_dataset_size + args.batch_size - 1) // args.batch_size
        )
    else:
        iterator = dataloader

    for step, batch in enumerate(iterator):
        try:
            images = None
            image_latents = None
            video_latents = None
            prompt_embeds = None

            if args.save_image_latents:
                images = batch["images"].to(device, non_blocking=True)
                images = images.permute(0, 2, 1, 3, 4)  # [B, C, F, H, W]

            videos = batch["videos"].to(device, non_blocking=True)
            videos = videos.permute(0, 2, 1, 3, 4)  # [B, C, F, H, W]

            prompts = batch["prompts"]

            # Encode videos & images
            if args.save_latents_and_embeddings:
                if args.use_slicing:
                    if args.save_image_latents:
                        encoded_slices = [vae._encode(image_slice) for image_slice in images.split(1)]
                        image_latents = torch.cat(encoded_slices)
                        image_latents = image_latents.to(memory_format=torch.contiguous_format, dtype=weight_dtype)

                    encoded_slices = [vae._encode(video_slice) for video_slice in videos.split(1)]
                    video_latents = torch.cat(encoded_slices)

                else:
                    if args.save_image_latents:
                        image_latents = vae._encode(images)
                        image_latents = image_latents.to(memory_format=torch.contiguous_format, dtype=weight_dtype)

                    video_latents = vae._encode(videos)

                video_latents = video_latents.to(memory_format=torch.contiguous_format, dtype=weight_dtype)

                # Encode prompts
                prompt_embeds = compute_prompt_embeddings(
                    tokenizer,
                    text_encoder,
                    prompts,
                    args.max_sequence_length,
                    device,
                    weight_dtype,
                    requires_grad=False,
                )

            if images is not None:
                images = (images.permute(0, 2, 1, 3, 4) + 1) / 2

            videos = (videos.permute(0, 2, 1, 3, 4) + 1) / 2

            # Save each data point individually
            for i in range(len(prompts)):
                # Generate unique filename
                filename = uuid.uuid4()
                filename_str = str(filename)

                # Save prompt
                with open(prompts_dir / f"{filename_str}.txt", "w", encoding="utf-8") as f:
                    f.write(prompts[i])

                # Save prompt embeddings
                if prompt_embeds is not None:
                    torch.save(prompt_embeds[i], prompt_embeds_dir / f"{filename_str}.pt")

                # Save videos
                if videos is not None:
                    video_tensor = videos[i]
                    video_frames = [to_pil_image(frame) for frame in video_tensor]
                    export_to_video(video_frames, videos_dir / f"{filename_str}.mp4", fps=target_fps)

                    # Save video latents
                    if video_latents is not None:
                        torch.save(video_latents[i], video_latents_dir / f"{filename_str}.pt")

                # Save images
                if images is not None and args.save_image_latents:
                    image_tensor = images[i]
                    save_image(image_tensor[0], images_dir / f"{filename_str}.png")

                    # Save image latents
                    if image_latents is not None:
                        torch.save(image_latents[i], image_latents_dir / f"{filename_str}.pt")

                # Track processed files for final JSONL
                processed_files.append({
                    "prompt": prompts[i],
                    "prompt_embed": f"prompt_embeds/{filename_str}.pt" if prompt_embeds is not None else None,
                    "image": f"images/{filename_str}.png" if images is not None else None,
                    "image_latent": f"image_latents/{filename_str}.pt" if image_latents is not None else None,
                    "video": f"videos/{filename_str}.mp4",
                    "video_latent": f"video_latents/{filename_str}.pt" if video_latents is not None else None,
                })

                # After processing each data point, create/update a running JSONL
                with open(output_dir / "data.jsonl", "a", encoding="utf-8") as f:
                    json.dump(processed_files[-1], f)
                    f.write("\n")

        except Exception:
            print("-------------------------")
            print(f"An exception occurred while processing data: {rank=}, {world_size=}, {step=}")
            traceback.print_exc()
            print("-------------------------")

    # 5. Complete distributed processing
    if world_size > 1:
        dist.barrier()
        dist.destroy_process_group()

    if rank == 0:
        # Create prompts.txt
        prompts_txt = output_dir.joinpath("prompts.txt")
        with open(prompts_txt, "w") as file:
            for item in processed_files:
                file.write(f"{item['prompt']}\n")

        # Create videos.txt
        videos_txt = output_dir.joinpath("videos.txt")
        with open(videos_txt, "w") as file:
            for item in processed_files:
                file.write(f"{item['video']}\n")
        print(f"Completed preprocessing. All files saved to `{output_dir.as_posix()}`")

if __name__ == "__main__":
    main()