import os
import pandas as pd
from PIL import Image
from torch.utils.data import Dataset
from typing import Union
from dataclasses import dataclass
import torchvision.transforms as transforms
import numpy as np
import torch

@dataclass
class ImageSize:
    width: int
    height: int

default_tokenizer_kwargs = dict(
    padding="max_length",
    max_length=77,
    truncation=True,
    add_special_tokens=True,
    return_tensors="pt"
)

default_image_transforms =  transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]),
])

def convert_pil_mask_to_tensor(mask):
    arr = np.array(mask)
    arr = arr.astype(np.float32) / 255.0
    return torch.tensor(arr).unsqueeze(0).unsqueeze(0)

class ImageCaptionDataset(Dataset):
    def __init__(
        self,
        image_folder: str,
        csv_filename: str,
        mask_folder: Union[str, None] = None,
        validate_csv: bool = True,
        size: Union[ImageSize, None] = None,
    ):
        super().__init__()
        assert os.path.exists(image_folder), f"Invalid image_folder: {image_folder}"
        assert os.path.exists(csv_filename), f"Invalid csv_filename: {csv_filename}"
        df = pd.read_csv(csv_filename)
        assert (
            "caption" in list(df.columns)
        ), f"Expected column: 'caption' to exist but got columns: {list(df.columns)}"
        assert (
            "image_path" in list(df.columns)
        ), f"Expected column: 'image_path' to exist but got columns: {list(df.columns)}"

        self.csv_filename = csv_filename
        self.captions = df.caption.values
        self.image_path = df.image_path.values
        self.size = size
        self.image_folder = image_folder
        self.mask_folder = mask_folder


        if mask_folder is not None:
            assert (
            "mask_path" in list(df.columns)
            ), f"Expected column: 'mask_path' to exist but got columns: {list(df.columns)}"
            self.mask_path = df.mask_path
        else:
            self.mask_path = None

        if validate_csv:
            self.validate_csv()

    def validate_csv(self):
        for idx in range(len(self.image_path)):
            filename = os.path.join(self.image_folder, self.image_path[idx])
            assert os.path.exists(
                filename
            ), f"Invalid image path: {filename}\nPlease check your CSV file: {self.csv_filename}"
        
        if self.mask_path is not None:
            for idx in range(len(self.mask_path)):
                filename = os.path.join(self.image_folder, self.image_path[idx])
                assert os.path.exists(
                    filename
                ), f"Invalid mask_path: {filename}\nPlease check your CSV file: {self.csv_filename}"

    def __getitem__(self, idx: int) -> dict:
        filename = os.path.join(self.image_folder, self.image_path[idx])
        image = Image.open(filename)
        if self.size is not None:
            # inspired by dataset_and_utils.py -> prepare_image
            image = image.resize(
                (self.size.width, self.size.height),
                resample=Image.BICUBIC,
                reducing_gap=1,
            )
        image = image.convert("RGB")
        caption = self.captions[idx]

        if self.mask_path is not None:
            image_width, image_height = image.size
            mask_filename = os.path.join(self.mask_folder, self.mask_path[idx])
            mask = Image.open(mask_filename)
            mask = mask.convert("L")
            mask = mask.resize(
                (image_width, image_height),
                resample=Image.BICUBIC,
                reducing_gap=1,
            )
        else:
            mask  = None

        return {"image": image, "caption": caption, "mask": mask}

    def __len__(self) -> int:
        return len(self.captions)


class PreprocessedDataset(Dataset):
    def __init__(
        self,
        image_caption_dataset: ImageCaptionDataset,
        tokenizers: list,
        vae,
        text_encoders: Union[list, None] = None,
        cache: bool = True,
        tokenizer_kwargs: dict = default_tokenizer_kwargs,
        scale_vae_latents: bool = True
    ):
        self.image_caption_dataset=image_caption_dataset
        self.tokenizers=tokenizers
        self.vae=vae
        self.text_encoders=text_encoders
        self.cache=cache
        self.tokenizer_kwargs=tokenizer_kwargs
        self.scale_vae_latents=scale_vae_latents        

    def __getitem__(self, idx: int):
        
        data = self.image_caption_dataset[idx]
        image, caption, mask = data["image"], data["caption"], data["mask"]

        tokenized_captions = []

        for tokenizer in self.tokenizers:
            tokenized_text = tokenizer(
                caption,
                **self.tokenizer_kwargs
            ).input_ids.squeeze()
            tokenized_captions.append(
                tokenized_text
            )

        image_tensor = default_image_transforms(image).unsqueeze(0).to(self.vae.device, dtype = self.vae.dtype)

        if mask is not None:
            mask = convert_pil_mask_to_tensor(mask)

        # raise ValueError(image_tensor.mean(), image_tensor.var())
        vae_latent = self.vae.encode(image_tensor).latent_dist.sample()
        if self.scale_vae_latents:
            vae_latent = vae_latent * self.vae.config.scaling_factor

        return {
            "tokenized_captions": tokenized_captions,
            "vae_latent": vae_latent.squeeze(),
            "mask": mask
        }
