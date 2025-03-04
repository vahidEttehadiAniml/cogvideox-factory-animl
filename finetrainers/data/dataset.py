import pathlib
import random
from typing import Any, Dict, List, Optional, Tuple

import datasets
import datasets.data_files
import datasets.distributed
import datasets.exceptions
import huggingface_hub
import huggingface_hub.errors
import numpy as np
import PIL.Image
import torch
import torch.distributed.checkpoint.stateful
from diffusers.utils import load_image, load_video
from huggingface_hub import list_repo_files, repo_exists, snapshot_download
from tqdm.auto import tqdm

from .. import constants
from .. import functional as FF
from ..logging import get_logger
from . import utils


import decord  # isort:skip

decord.bridge.set_bridge("torch")

logger = get_logger()


MAX_PRECOMPUTABLE_ITEMS_LIMIT = 1024


class ImageCaptionFilePairDataset(torch.utils.data.IterableDataset, torch.distributed.checkpoint.stateful.Stateful):
    def __init__(self, root: str, infinite: bool = False) -> None:
        super().__init__()

        self.root = pathlib.Path(root)
        self.infinite = infinite

        data = []
        caption_files = sorted(utils.find_files(self.root.as_posix(), "*.txt", depth=0))
        for caption_file in caption_files:
            data_file = self._find_data_file(caption_file)
            if data_file:
                data.append(
                    {
                        "caption": (self.root / caption_file).as_posix(),
                        "image": (self.root / data_file).as_posix(),
                    }
                )

        data = datasets.Dataset.from_list(data)
        data = data.cast_column("image", datasets.Image(mode="RGB"))

        self._data = data.to_iterable_dataset()
        self._sample_index = 0
        self._precomputable_once = len(data) <= MAX_PRECOMPUTABLE_ITEMS_LIMIT

    def _get_data_iter(self):
        if self._sample_index == 0:
            return iter(self._data)

        if isinstance(self._data, datasets.Dataset) and self._sample_index >= len(self._data):
            return iter([])

        return iter(self._data.skip(self._sample_index))

    def __iter__(self):
        while True:
            for sample in self._get_data_iter():
                self._sample_index += 1
                sample["caption"] = _read_caption_from_file(sample["caption"])
                sample["image"] = _preprocess_image(sample["image"])
                yield sample

            if not self.infinite:
                logger.warning(f"Dataset ({self.__class__.__name__}={self.root}) has run out of data")
                break
            else:
                self._sample_index = 0

    def load_state_dict(self, state_dict):
        self._sample_index = state_dict["sample_index"]

    def state_dict(self):
        return {"sample_index": self._sample_index}

    def _find_data_file(self, caption_file: str) -> str:
        caption_file = pathlib.Path(caption_file)
        data_file = None
        found_data = 0

        for extension in constants.SUPPORTED_IMAGE_FILE_EXTENSIONS:
            image_filename = caption_file.with_suffix(f".{extension}")
            if image_filename.exists():
                found_data += 1
                data_file = image_filename

        if found_data == 0:
            return False
        elif found_data > 1:
            raise ValueError(
                f"Multiple data files found for caption file {caption_file}. Please ensure there is only one data "
                f"file per caption file. The following extensions are supported:\n"
                f"  - Images: {constants.SUPPORTED_IMAGE_FILE_EXTENSIONS}\n"
            )

        return data_file.as_posix()


class VideoCaptionFilePairDataset(torch.utils.data.IterableDataset, torch.distributed.checkpoint.stateful.Stateful):
    def __init__(self, root: str, infinite: bool = False) -> None:
        super().__init__()

        self.root = pathlib.Path(root)
        self.infinite = infinite

        data = []
        caption_files = sorted(utils.find_files(self.root.as_posix(), "*.txt", depth=0))
        for caption_file in caption_files:
            data_file = self._find_data_file(caption_file)
            if data_file:
                data.append(
                    {
                        "caption": (self.root / caption_file).as_posix(),
                        "video": (self.root / data_file).as_posix(),
                    }
                )

        data = datasets.Dataset.from_list(data)
        data = data.cast_column("video", datasets.Video())

        self._data = data.to_iterable_dataset()
        self._sample_index = 0
        self._precomputable_once = len(data) <= MAX_PRECOMPUTABLE_ITEMS_LIMIT

    def _get_data_iter(self):
        if self._sample_index == 0:
            return iter(self._data)
        return iter(self._data.skip(self._sample_index))

    def __iter__(self):
        while True:
            for sample in self._get_data_iter():
                self._sample_index += 1
                sample["caption"] = _read_caption_from_file(sample["caption"])
                sample["video"] = _preprocess_video(sample["video"])
                yield sample

            if not self.infinite:
                logger.warning(f"Dataset ({self.__class__.__name__}={self.root}) has run out of data")
                break
            else:
                self._sample_index = 0

    def load_state_dict(self, state_dict):
        self._sample_index = state_dict["sample_index"]

    def state_dict(self):
        return {"sample_index": self._sample_index}

    def _find_data_file(self, caption_file: str) -> str:
        caption_file = pathlib.Path(caption_file)
        data_file = None
        found_data = 0

        for extension in constants.SUPPORTED_VIDEO_FILE_EXTENSIONS:
            video_filename = caption_file.with_suffix(f".{extension}")
            if video_filename.exists():
                found_data += 1
                data_file = video_filename

        if found_data == 0:
            return False
        elif found_data > 1:
            raise ValueError(
                f"Multiple data files found for caption file {caption_file}. Please ensure there is only one data "
                f"file per caption file. The following extensions are supported:\n"
                f"  - Videos: {constants.SUPPORTED_VIDEO_FILE_EXTENSIONS}\n"
            )

        return data_file.as_posix()


class ImageFileCaptionFileListDataset(
    torch.utils.data.IterableDataset, torch.distributed.checkpoint.stateful.Stateful
):
    def __init__(self, root: str, infinite: bool = False) -> None:
        super().__init__()

        VALID_CAPTION_FILES = ["caption.txt", "captions.txt", "prompt.txt", "prompts.txt"]
        VALID_IMAGE_FILES = ["image.txt", "images.txt"]

        self.root = pathlib.Path(root)
        self.infinite = infinite

        data = []
        existing_caption_files = [file for file in VALID_CAPTION_FILES if (self.root / file).exists()]
        existing_image_files = [file for file in VALID_IMAGE_FILES if (self.root / file).exists()]

        if len(existing_caption_files) == 0:
            raise FileNotFoundError(
                f"No caption file found in {self.root}. Must have exactly one of {VALID_CAPTION_FILES}"
            )
        if len(existing_image_files) == 0:
            raise FileNotFoundError(
                f"No image file found in {self.root}. Must have exactly one of {VALID_IMAGE_FILES}"
            )
        if len(existing_caption_files) > 1:
            raise ValueError(
                f"Multiple caption files found in {self.root}. Must have exactly one of {VALID_CAPTION_FILES}"
            )
        if len(existing_image_files) > 1:
            raise ValueError(
                f"Multiple image files found in {self.root}. Must have exactly one of {VALID_IMAGE_FILES}"
            )

        caption_file = existing_caption_files[0]
        image_file = existing_image_files[0]

        with open((self.root / caption_file).as_posix(), "r") as f:
            captions = f.read().splitlines()
        with open((self.root / image_file).as_posix(), "r") as f:
            images = f.read().splitlines()
            images = [(self.root / image).as_posix() for image in images]

        if len(captions) != len(images):
            raise ValueError(f"Number of captions ({len(captions)}) must match number of images ({len(images)})")

        for caption, image in zip(captions, images):
            data.append({"caption": caption, "image": image})

        data = datasets.Dataset.from_list(data)
        data = data.cast_column("image", datasets.Image(mode="RGB"))

        self._data = data.to_iterable_dataset()
        self._sample_index = 0
        self._precomputable_once = len(data) <= MAX_PRECOMPUTABLE_ITEMS_LIMIT

    def _get_data_iter(self):
        if self._sample_index == 0:
            return iter(self._data)
        return iter(self._data.skip(self._sample_index))

    def __iter__(self):
        while True:
            for sample in self._get_data_iter():
                self._sample_index += 1
                sample["image"] = _preprocess_image(sample["image"])
                yield sample

            if not self.infinite:
                logger.warning(f"Dataset ({self.__class__.__name__}={self.root}) has run out of data")
                break
            else:
                self._sample_index = 0

    def load_state_dict(self, state_dict):
        self._sample_index = state_dict["sample_index"]

    def state_dict(self):
        return {"sample_index": self._sample_index}


class VideoFileCaptionFileListDataset(
    torch.utils.data.IterableDataset, torch.distributed.checkpoint.stateful.Stateful
):
    def __init__(self, root: str, infinite: bool = False) -> None:
        super().__init__()

        VALID_CAPTION_FILES = ["caption.txt", "captions.txt", "prompt.txt", "prompts.txt"]
        VALID_VIDEO_FILES = ["video.txt", "videos.txt"]

        self.root = pathlib.Path(root)
        self.infinite = infinite

        data = []
        existing_caption_files = [file for file in VALID_CAPTION_FILES if (self.root / file).exists()]
        existing_video_files = [file for file in VALID_VIDEO_FILES if (self.root / file).exists()]

        if len(existing_caption_files) == 0:
            raise FileNotFoundError(
                f"No caption file found in {self.root}. Must have exactly one of {VALID_CAPTION_FILES}"
            )
        if len(existing_video_files) == 0:
            raise FileNotFoundError(
                f"No video file found in {self.root}. Must have exactly one of {VALID_VIDEO_FILES}"
            )
        if len(existing_caption_files) > 1:
            raise ValueError(
                f"Multiple caption files found in {self.root}. Must have exactly one of {VALID_CAPTION_FILES}"
            )
        if len(existing_video_files) > 1:
            raise ValueError(
                f"Multiple video files found in {self.root}. Must have exactly one of {VALID_VIDEO_FILES}"
            )

        caption_file = existing_caption_files[0]
        video_file = existing_video_files[0]

        with open((self.root / caption_file).as_posix(), "r") as f:
            captions = f.read().splitlines()
        with open((self.root / video_file).as_posix(), "r") as f:
            videos = f.read().splitlines()
            videos = [(self.root / video).as_posix() for video in videos]

        if len(captions) != len(videos):
            raise ValueError(f"Number of captions ({len(captions)}) must match number of videos ({len(videos)})")

        for caption, video in zip(captions, videos):
            data.append({"caption": caption, "video": video})

        data = datasets.Dataset.from_list(data)
        data = data.cast_column("video", datasets.Video())

        self._data = data.to_iterable_dataset()
        self._sample_index = 0
        self._precomputable_once = len(data) <= MAX_PRECOMPUTABLE_ITEMS_LIMIT

    def _get_data_iter(self):
        if self._sample_index == 0:
            return iter(self._data)
        return iter(self._data.skip(self._sample_index))

    def __iter__(self):
        while True:
            for sample in self._get_data_iter():
                self._sample_index += 1
                sample["video"] = _preprocess_video(sample["video"])
                yield sample

            if not self.infinite:
                logger.warning(f"Dataset ({self.__class__.__name__}={self.root}) has run out of data")
                break
            else:
                self._sample_index = 0

    def load_state_dict(self, state_dict):
        self._sample_index = state_dict["sample_index"]

    def state_dict(self):
        return {"sample_index": self._sample_index}


class ImageFolderDataset(torch.utils.data.IterableDataset, torch.distributed.checkpoint.stateful.Stateful):
    def __init__(self, root: str, infinite: bool = False) -> None:
        super().__init__()

        self.root = pathlib.Path(root)
        self.infinite = infinite

        data = datasets.load_dataset("imagefolder", data_dir=self.root.as_posix(), split="train")

        self._data = data.to_iterable_dataset()
        self._sample_index = 0
        self._precomputable_once = len(data) <= MAX_PRECOMPUTABLE_ITEMS_LIMIT

    def _get_data_iter(self):
        if self._sample_index == 0:
            return iter(self._data)
        return iter(self._data.skip(self._sample_index))

    def __iter__(self):
        while True:
            for sample in self._get_data_iter():
                self._sample_index += 1
                sample["image"] = _preprocess_image(sample["image"])
                yield sample

            if not self.infinite:
                logger.warning(f"Dataset ({self.__class__.__name__}={self.root}) has run out of data")
                break
            else:
                self._sample_index = 0

    def load_state_dict(self, state_dict):
        self._sample_index = state_dict["sample_index"]

    def state_dict(self):
        return {"sample_index": self._sample_index}


class VideoFolderDataset(torch.utils.data.IterableDataset, torch.distributed.checkpoint.stateful.Stateful):
    def __init__(self, root: str, infinite: bool = False) -> None:
        super().__init__()

        self.root = pathlib.Path(root)
        self.infinite = infinite

        data = datasets.load_dataset("videofolder", data_dir=self.root.as_posix(), split="train")

        self._data = data.to_iterable_dataset()
        self._sample_index = 0
        self._precomputable_once = len(data) <= MAX_PRECOMPUTABLE_ITEMS_LIMIT

    def _get_data_iter(self):
        if self._sample_index == 0:
            return iter(self._data)
        return iter(self._data.skip(self._sample_index))

    def __iter__(self):
        while True:
            for sample in self._get_data_iter():
                self._sample_index += 1
                sample["video"] = _preprocess_video(sample["video"])
                yield sample

            if not self.infinite:
                logger.warning(f"Dataset ({self.__class__.__name__}={self.root}) has run out of data")
                break
            else:
                self._sample_index = 0

    def load_state_dict(self, state_dict):
        self._sample_index = state_dict["sample_index"]

    def state_dict(self):
        return {"sample_index": self._sample_index}


class ImageWebDataset(torch.utils.data.IterableDataset, torch.distributed.checkpoint.stateful.Stateful):
    def __init__(self, dataset_name: str, infinite: bool = False) -> None:
        super().__init__()

        self.dataset_name = dataset_name
        self.infinite = infinite

        data = datasets.load_dataset(dataset_name, split="train", streaming=True)
        data = data.rename_column("txt", "caption")
        for column_name in constants.SUPPORTED_IMAGE_FILE_EXTENSIONS:
            if column_name in data.column_names:
                data = data.cast_column(column_name, datasets.Image(mode="RGB"))
                data = data.rename_column(column_name, "image")

        self._data = data
        self._sample_index = 0
        self._precomputable_once = False

    def _get_data_iter(self):
        if self._sample_index == 0:
            return iter(self._data)
        return iter(self._data.skip(self._sample_index))

    def __iter__(self):
        while True:
            for sample in self._get_data_iter():
                self._sample_index += 1
                yield sample

            if not self.infinite:
                logger.warning(f"Dataset {self.dataset_name} has run out of data")
                break
            else:
                # Reset offset for the next iteration
                self._sample_index = 0
                logger.warning(f"Dataset {self.dataset_name} is being re-looped")

    def load_state_dict(self, state_dict):
        self._sample_index = state_dict["sample_index"]

    def state_dict(self):
        return {"sample_index": self._sample_index}


class VideoWebDataset(torch.utils.data.IterableDataset, torch.distributed.checkpoint.stateful.Stateful):
    def __init__(self, dataset_name: str, infinite: bool = False) -> None:
        super().__init__()

        self.dataset_name = dataset_name
        self.infinite = infinite

        data = datasets.load_dataset(dataset_name, split="train", streaming=True)
        data = data.rename_column("txt", "caption")
        for column_name in constants.SUPPORTED_VIDEO_FILE_EXTENSIONS:
            if column_name in data.column_names:
                data = data.cast_column(column_name, datasets.Video())
                data = data.rename_column(column_name, "video")

        self._data = data
        self._sample_index = 0
        self._precomputable_once = False

    def _get_data_iter(self):
        if self._sample_index == 0:
            return iter(self._data)
        return iter(self._data.skip(self._sample_index))

    def __iter__(self):
        while True:
            for sample in self._get_data_iter():
                self._sample_index += 1
                yield sample

            if not self.infinite:
                logger.warning(f"Dataset {self.dataset_name} has run out of data")
                break
            else:
                # Reset offset for the next iteration
                self._sample_index = 0
                logger.warning(f"Dataset {self.dataset_name} is being re-looped")

    def load_state_dict(self, state_dict):
        self._sample_index = state_dict["sample_index"]

    def state_dict(self):
        return {"sample_index": self._sample_index}


class ValidationDataset(torch.utils.data.IterableDataset):
    def __init__(self, filename: str):
        super().__init__()

        self.filename = pathlib.Path(filename)

        if not self.filename.exists():
            raise FileNotFoundError(f"File {self.filename.as_posix()} does not exist")

        if self.filename.suffix == ".csv":
            data = datasets.load_dataset("csv", data_files=self.filename.as_posix(), split="train")
        elif self.filename.suffix == ".json":
            data = datasets.load_dataset("json", data_files=self.filename.as_posix(), split="train", field="data")
        elif self.filename.suffix == ".parquet":
            data = datasets.load_dataset("parquet", data_files=self.filename.as_posix(), split="train")
        elif self.filename.suffix == ".arrow":
            data = datasets.load_dataset("arrow", data_files=self.filename.as_posix(), split="train")
        else:
            _SUPPORTED_FILE_FORMATS = [".csv", ".json", ".parquet", ".arrow"]
            raise ValueError(
                f"Unsupported file format {self.filename.suffix} for validation dataset. Supported formats are: {_SUPPORTED_FILE_FORMATS}"
            )

        self._data = data.to_iterable_dataset()

    def __iter__(self):
        for sample in self._data:
            # For consistency reasons, we mandate that "caption" is always present in the validation dataset.
            # However, since the model specifications use "prompt", we create an alias here.
            sample["prompt"] = sample["caption"]

            # Load image or video if the path is provided
            # TODO(aryan): need to handle custom columns here for control conditions
            sample["image"] = None
            sample["video"] = None

            if sample.get("image_path", None) is not None:
                image_path = pathlib.Path(sample["image_path"])
                if not image_path.is_file():
                    logger.warning(f"Image file {image_path.as_posix()} does not exist.")
                else:
                    sample["image"] = load_image(sample["image_path"])

            if sample.get("video_path", None) is not None:
                video_path = pathlib.Path(sample["video_path"])
                if not video_path.is_file():
                    logger.warning(f"Video file {video_path.as_posix()} does not exist.")
                else:
                    sample["video"] = load_video(sample["video_path"])

            sample = {k: v for k, v in sample.items() if v is not None}
            yield sample


class IterableDatasetPreprocessingWrapper(
    torch.utils.data.IterableDataset, torch.distributed.checkpoint.stateful.Stateful
):
    def __init__(
        self,
        dataset: torch.utils.data.IterableDataset,
        dataset_type: str,
        id_token: Optional[str] = None,
        image_resolution_buckets: List[Tuple[int, int]] = None,
        video_resolution_buckets: List[Tuple[int, int, int]] = None,
        reshape_mode: str = "bicubic",
        remove_common_llm_caption_prefixes: bool = False,
        **kwargs,
    ):
        super().__init__()

        self.dataset = dataset
        self.dataset_type = dataset_type
        self.id_token = id_token
        self.image_resolution_buckets = image_resolution_buckets
        self.video_resolution_buckets = video_resolution_buckets
        self.reshape_mode = reshape_mode
        self.remove_common_llm_caption_prefixes = remove_common_llm_caption_prefixes

        logger.info(
            f"Initializing IterableDatasetPreprocessingWrapper for the dataset with the following configuration:\n"
            f"  - Dataset Type: {dataset_type}\n"
            f"  - ID Token: {id_token}\n"
            f"  - Image Resolution Buckets: {image_resolution_buckets}\n"
            f"  - Video Resolution Buckets: {video_resolution_buckets}\n"
            f"  - Reshape Mode: {reshape_mode}\n"
            f"  - Remove Common LLM Caption Prefixes: {remove_common_llm_caption_prefixes}\n"
        )

    def __iter__(self):
        logger.info("Starting IterableDatasetPreprocessingWrapper for the dataset")
        for sample in iter(self.dataset):
            if self.dataset_type == "image":
                if self.image_resolution_buckets:
                    sample["image"] = FF.resize_to_nearest_bucket_image(
                        sample["image"], self.image_resolution_buckets, self.reshape_mode
                    )
            elif self.dataset_type == "video":
                if self.video_resolution_buckets:
                    sample["video"], _first_frame_only = FF.resize_to_nearest_bucket_video(
                        sample["video"], self.video_resolution_buckets, self.reshape_mode
                    )
                    if _first_frame_only:
                        msg = (
                            "The number of frames in the video is less than the minimum bucket size "
                            "specified. The first frame is being used as a single frame video. This "
                            "message is logged at the first occurence and for every 128th occurence "
                            "after that."
                        )
                        logger.log_freq("WARNING", "BUCKET_TEMPORAL_SIZE_UNAVAILABLE", msg, frequency=128)
                        sample["video"] = sample["video"][0]

            if self.remove_common_llm_caption_prefixes:
                sample["caption"] = FF.remove_prefix(sample["caption"], constants.COMMON_LLM_START_PHRASES)

            if self.id_token is not None:
                sample["caption"] = f"{self.id_token} {sample['caption']}"

            yield sample

    def load_state_dict(self, state_dict):
        self.dataset.load_state_dict(state_dict["dataset"])

    def state_dict(self):
        return {"dataset": self.dataset.state_dict()}


class IterableCombinedDataset(torch.utils.data.IterableDataset, torch.distributed.checkpoint.stateful.Stateful):
    def __init__(self, datasets: List[torch.utils.data.IterableDataset], buffer_size: int, shuffle: bool = False):
        super().__init__()

        self.datasets = datasets
        self.buffer_size = buffer_size
        self.shuffle = shuffle

        logger.info(
            f"Initializing IterableCombinedDataset with the following configuration:\n"
            f"  - Number of Datasets: {len(datasets)}\n"
            f"  - Buffer Size: {buffer_size}\n"
            f"  - Shuffle: {shuffle}\n"
        )

    def __iter__(self):
        logger.info(f"Starting IterableCombinedDataset with {len(self.datasets)} datasets")
        iterators = [iter(dataset) for dataset in self.datasets]
        buffer = []
        per_iter = max(1, self.buffer_size // len(iterators))

        for index, it in enumerate(iterators):
            for _ in tqdm(range(per_iter), desc=f"Filling buffer from data iterator {index}"):
                try:
                    buffer.append((it, next(it)))
                except StopIteration:
                    continue

        while len(buffer) > 0:
            idx = 0
            if self.shuffle:
                idx = random.randint(0, len(buffer) - 1)
            current_it, sample = buffer.pop(idx)
            yield sample
            try:
                buffer.append((current_it, next(current_it)))
            except StopIteration:
                pass

    def load_state_dict(self, state_dict):
        for dataset, dataset_state_dict in zip(self.datasets, state_dict["datasets"]):
            dataset.load_state_dict(dataset_state_dict)

    def state_dict(self):
        return {"datasets": [dataset.state_dict() for dataset in self.datasets]}


# TODO(aryan): maybe write a test for this
def initialize_dataset(
    dataset_name_or_root: str, dataset_type: str = "video", streaming: bool = True, infinite: bool = False
) -> torch.utils.data.IterableDataset:
    # 1. If there is a metadata.json or metadata.jsonl or metadata.csv file, we use the
    #    ImageFolderDataset or VideoFolderDataset class respectively.
    # 2. If there is a list of .txt files sharing the same name with image extensions, we
    #    use the ImageCaptionFileDataset class.
    # 3. If there is a list of .txt files sharing the same name with video extensions, we
    #    use the VideoCaptionFileDataset class.
    # 4. If there is a dataset name, we use the ImageWebDataset or VideoWebDataset class.
    assert dataset_type in ["image", "video"]

    try:
        does_repo_exist_on_hub = repo_exists(dataset_name_or_root, repo_type="dataset")
    except huggingface_hub.errors.HFValidationError:
        does_repo_exist_on_hub = False

    if does_repo_exist_on_hub:
        return _initialize_hub_dataset(dataset_name_or_root, dataset_type, infinite)
    else:
        return _initialize_local_dataset(dataset_name_or_root, dataset_type, infinite)


def combine_datasets(
    datasets: List[torch.utils.data.IterableDataset], buffer_size: int, shuffle: bool = False
) -> torch.utils.data.IterableDataset:
    return IterableCombinedDataset(datasets=datasets, buffer_size=buffer_size, shuffle=shuffle)


def wrap_iterable_dataset_for_preprocessing(
    dataset: torch.utils.data.IterableDataset, dataset_type: str, config: Dict[str, Any]
) -> torch.utils.data.IterableDataset:
    return IterableDatasetPreprocessingWrapper(dataset, dataset_type, **config)


def _read_caption_from_file(filename: str) -> str:
    with open(filename, "r") as f:
        return f.read().strip()


def _preprocess_image(image: PIL.Image.Image) -> torch.Tensor:
    image = image.convert("RGB")
    image = np.array(image).astype(np.float32)
    image = torch.from_numpy(image)
    image = image.permute(2, 0, 1).contiguous() / 127.5 - 1.0
    return image


def _preprocess_video(video: decord.VideoReader) -> torch.Tensor:
    video = video.get_batch(list(range(len(video))))
    video = video.permute(0, 3, 1, 2).contiguous()
    video = video.float() / 127.5 - 1.0
    return video


def _initialize_local_dataset(dataset_name_or_root: str, dataset_type: str, infinite: bool = False):
    root = pathlib.Path(dataset_name_or_root)
    supported_metadata_files = ["metadata.json", "metadata.jsonl", "metadata.csv"]
    metadata_files = [root / metadata_file for metadata_file in supported_metadata_files]
    metadata_files = [metadata_file for metadata_file in metadata_files if metadata_file.exists()]

    dataset = None
    if len(metadata_files) == 0:
        raise ValueError(
            f"No metadata file found. Please ensure there is a metadata file named one of: {supported_metadata_files}."
        )
    elif len(metadata_files) > 1:
        raise ValueError("Found multiple metadata files. Please ensure there is only one metadata file.")
    else:
        if dataset_type == "image":
            dataset = ImageFolderDataset(root.as_posix(), infinite=infinite)
        else:
            dataset = VideoFolderDataset(root.as_posix(), infinite=infinite)

    if dataset is None:
        if dataset_type == "image":
            dataset = ImageCaptionFilePairDataset(root.as_posix(), infinite=infinite)
        else:
            dataset = VideoCaptionFilePairDataset(root.as_posix(), infinite=infinite)

    return dataset


def _initialize_hub_dataset(dataset_name: str, dataset_type: str, infinite: bool = False):
    _common_caption_files = ["prompt.txt", "prompts.txt", "caption.txt", "captions.txt"]
    _common_video_files = ["video.txt", "videos.txt"]
    _common_image_files = ["image.txt", "images.txt"]

    repo_file_list = list_repo_files(dataset_name, repo_type="dataset")
    has_caption_files = any(file in repo_file_list for file in _common_caption_files)
    has_video_files = any(file in repo_file_list for file in _common_video_files)
    has_image_files = any(file in repo_file_list for file in _common_image_files)

    if has_caption_files and (has_video_files or has_image_files):
        return _initialize_file_dataset(dataset_name, dataset_type, infinite)
    else:
        return _initialize_webdataset(dataset_name, dataset_type, infinite)


def _initialize_file_dataset(
    dataset_name: str, dataset_type: str, infinite: bool = False
) -> torch.utils.data.IterableDataset:
    logger.info(f"Downloading dataset {dataset_name} from the HF Hub")
    dataset_root = snapshot_download(dataset_name, repo_type="dataset")
    if dataset_type == "image":
        return ImageFileCaptionFileListDataset(dataset_root, infinite=infinite)
    else:
        return VideoFileCaptionFileListDataset(dataset_root, infinite=infinite)


def _initialize_webdataset(
    dataset_name: str, dataset_type: str, infinite: bool = False
) -> torch.utils.data.IterableDataset:
    logger.info(f"Streaming webdataset {dataset_name} from the HF Hub")
    if dataset_type == "image":
        return ImageWebDataset(dataset_name, infinite=infinite)
    else:
        return VideoWebDataset(dataset_name, infinite=infinite)
