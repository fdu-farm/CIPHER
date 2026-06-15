import os
from PIL import Image

import torch
import torch.nn.functional as F
from torch.utils.data import Dataset
from torchvision.transforms import Compose, Resize, Normalize, InterpolationMode, ToTensor


class SquarePad:
    """Pad a tensor image to be square."""

    def __call__(self, image):
        height, width = image.shape[-2], image.shape[-1]
        max_wh = max(width, height)

        pad_w = max_wh - width
        pad_h = max_wh - height

        pad_left = pad_w // 2
        pad_right = pad_w - pad_left
        pad_top = pad_h // 2
        pad_bottom = pad_h - pad_top

        return F.pad(image, (pad_left, pad_right, pad_top, pad_bottom), "constant", 0)


class CIPHERFineTuningImageDirectoryDataset(Dataset):
    """
    Dataset for diffusion fine-tuning with paired image and prompt files.

    In the CIPHER setting, prompts should encode both disease semantics and
    sensitive-attribute metadata so downstream causal interventions can swap or
    suppress those tokens explicitly.

    Args:
        image_dir_path (str): Path to the directory containing image files.
        text_dir_path (str): Path to the directory containing text prompt files.
        tokenizer (callable): Tokenizer function from Hugging Face transformers.
        data_filter_file (str, optional): Path to a file containing a list of image stems
            to include. Each line should be an image stem (e.g., 'image001').
        size (int, optional): Target size for the smaller edge after padding (square).
    """

    def __init__(
        self,
        image_dir_path,
        text_dir_path,
        tokenizer,
        data_filter_file=None,
        size=512,
    ):
        if not os.path.isdir(image_dir_path):
            raise ValueError(f"Image directory does not exist: {image_dir_path}")
        if not os.path.isdir(text_dir_path):
            raise ValueError(f"Text directory does not exist: {text_dir_path}")

        self.image_dir_path = image_dir_path
        self.text_dir_path = text_dir_path
        self.tokenizer = tokenizer

        self.image_transforms = Compose(
            [
                ToTensor(),
                SquarePad(),
                Resize(size, interpolation=InterpolationMode.BILINEAR),
                Normalize([0.5], [0.5]),
            ]
        )

        allowed_exts = (".jpg", ".jpeg", ".png")
        all_image_files = sorted(
            [
                f
                for f in os.listdir(image_dir_path)
                if f.lower().endswith(allowed_exts)
            ]
        )

        self.samples = []
        for image_filename in all_image_files:
            image_stem = os.path.splitext(image_filename)[0]
            image_path = os.path.join(image_dir_path, image_filename)
            text_path = os.path.join(text_dir_path, image_stem + ".txt")

            if os.path.exists(text_path):
                self.samples.append((image_path, text_path, image_stem))
            else:
                print(
                    f"Warning: No corresponding text file found for image: {image_filename}. Skipping."
                )

        self.data_filter = None
        if data_filter_file is not None:
            if not os.path.isfile(data_filter_file):
                raise ValueError(f"Data filter file does not exist: {data_filter_file}")

            self.data_filter = set()
            with open(data_filter_file, "r") as file:
                for line in file:
                    stem = os.path.splitext(line.strip())[0]
                    if stem:
                        self.data_filter.add(stem)

            print(f"Length of data filter: {len(self.data_filter)}")
            self.samples = [
                (img_p, txt_p, stem)
                for img_p, txt_p, stem in self.samples
                if stem in self.data_filter
            ]
            print(f"Dataset size after filter: {len(self.samples)}")
        else:
            print("No data filter provided.")

        if not self.samples:
            raise ValueError(
                "No valid image-text pairs found after initialization/filtering. Check paths and filter."
            )

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        image_path, text_path, image_stem = self.samples[idx]

        with Image.open(image_path) as img:
            image = img.convert("RGB")

        sample = {}
        sample["pixel_values"] = self.image_transforms(image)

        with open(text_path, "r", encoding="utf-8") as f:
            prompt = f.read().strip()

        prompt_tokenized = self.tokenizer(
            prompt,
            padding="max_length",
            truncation=True,
            max_length=self.tokenizer.model_max_length,
            return_tensors="pt",
        )
        sample["input_ids"] = prompt_tokenized.input_ids.squeeze()
        sample["attention_mask"] = prompt_tokenized.attention_mask.squeeze()
        sample["loss_weights"] = torch.ones(1, dtype=torch.float32).squeeze()

        return sample
