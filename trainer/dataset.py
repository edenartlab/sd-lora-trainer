import os
import torch
import numpy as np
import pandas as pd
import PIL
from PIL import Image
from torch.utils.data import Dataset
from typing import Tuple, Dict

def prepare_image(
    pil_image: PIL.Image.Image, w: int = 512, h: int = 512, pipe=None,
) -> torch.Tensor:
    pil_image = pil_image.resize((w, h), resample=Image.BICUBIC, reducing_gap=1)
    image = pipe.image_processor.preprocess(pil_image)
    return image


def prepare_mask(
    pil_image: PIL.Image.Image, w: int = 512, h: int = 512
) -> torch.Tensor:
    pil_image = pil_image.resize((w, h), resample=Image.BICUBIC, reducing_gap=1)
    arr = np.array(pil_image.convert("L"))
    arr = arr.astype(np.float32) / 255.0
    arr = np.expand_dims(arr, 0)
    image = torch.from_numpy(arr).unsqueeze(0)
    return image


class PreprocessedDataset(Dataset):
    def __init__(
        self,
        data_dir: str,
        pipe,
        tokenizer_1,
        tokenizer_2,
        vae_encoder,
        text_encoder_1=None,
        text_encoder_2=None,
        do_cache: bool = False,
        size: int = 512,
        text_dropout: float = 0.0,
        aspect_ratio_bucketing: bool = False,
        train_batch_size: int = None,# required for aspect_ratio_bucketing
        substitute_caption_map: Dict[str, str] = {},
    ):
        super().__init__()
        self.data_dir = data_dir
        self.csv_path = os.path.join(data_dir, "captions.csv")
        self.data = pd.read_csv(self.csv_path)

        self.caption = self.data["caption"]
        # make it lowercase
        self.caption = self.caption.str.lower()
        for key, value in substitute_caption_map.items():
            self.caption = self.caption.str.replace(key.lower(), value)

        self.image_path = self.data["image_path"]

        if "mask_path" not in self.data.columns:
            self.mask_path = None
        else:
            self.mask_path = self.data["mask_path"]

        self.pipe = pipe

        self.text_encoder_1 = text_encoder_1
        self.text_encoder_2 = text_encoder_2

        self.tokenizer_1 = tokenizer_1
        self.tokenizer_2 = tokenizer_2

        self.vae_encoder = vae_encoder
        self.vae_scaling_factor = self.vae_encoder.config.scaling_factor
        self.text_dropout = text_dropout
        self.size = size

        if do_cache:
            self.vae_latents = []
            self.tokens_tuple = []
            self.masks = []
            self.do_cache = True

            print("Caching latents, masks and token_IDs...\n")

            for idx in range(len(self.data)):
                token, vae_latent, mask = self._process(idx)
                self.vae_latents.append(vae_latent)
                self.tokens_tuple.append(token)
                self.masks.append(mask)

            print(f"\nCached latents masks and token_IDs for {len(self.vae_latents)} images.")
            del self.vae_encoder

        else:
            self.do_cache = False

        if aspect_ratio_bucketing:
            print("Using aspect ratio bucketing.")
            assert train_batch_size is not None, f"Please also provide a `train_batch_size` when you have set `aspect_ratio_bucketing == True`"
            from .utils.aspect_ratio_bucketing import BucketManager
            aspect_ratios = {}
            for idx in range(len(self.data)):
                aspect_ratios[idx] = Image.open(os.path.join(self.data_dir, self.image_path[idx])).size

            print(aspect_ratios)
            self.bucket_manager = BucketManager(
                aspect_ratios = aspect_ratios,
                bsz = train_batch_size,
                debug=True
            )
        else:
            print("Not using aspect ratio bucketing.")
            self.bucket_manager = None

    def get_aspect_ratio_bucketed_batch(self):
        assert self.bucket_manager is not None, f"Expected self.bucket_manager to not be None! In order to get an aspect ratio bucketed batch, please set aspect_ratio_bucketing = True and set a value for train_batch_size when doing __init__()"
        indices, resolution = self.bucket_manager.get_batch()

        print(f"Got bucket batch: {indices}, resolution: {resolution}")

        tok1, tok2, vae_latents, masks = [], [], [], []
        
        for idx in indices:

            if  self.tokenizer_2 is None:
                t1, v, m = self.__getitem__(idx = idx, bucketing_resolution=resolution)
            else:
                (t1, t2), v, m = self.__getitem__(idx = idx, bucketing_resolution=resolution)
                tok2.append(t2.unsqueeze(0))

            tok1.append(t1.unsqueeze(0))
            vae_latents.append(v.unsqueeze(0))
            masks.append(m.unsqueeze(0))

        tok1 = torch.cat(tok1, dim = 0)
        if  self.tokenizer_2 is None:
            pass
        else:
            tok2 = torch.cat(tok2, dim = 0)
        vae_latents = torch.cat(vae_latents, dim = 0)
        masks = torch.cat(masks, dim = 0)

        if self.tokenizer_2 is None:
            return (tok1, None), vae_latents, masks
        else:
            return (tok1, tok2), vae_latents, masks

    def __len__(self) -> int:
        return len(self.data)

    @torch.no_grad()
    def _process(
        self, idx: int, bucketing_resolution: tuple = None
    ) -> Tuple[Tuple[torch.Tensor, torch.Tensor], torch.Tensor, torch.Tensor]:
        image_path = self.image_path[idx]
        image_path = os.path.join(self.data_dir, image_path)
        image = PIL.Image.open(image_path).convert("RGB")
        if bucketing_resolution is None:
            image = prepare_image(image, w = self.size, h = self.size, pipe = self.pipe).to(
                dtype=self.vae_encoder.dtype, device=self.vae_encoder.device
            )
        else:
            image = prepare_image(image, w = bucketing_resolution[0], h = bucketing_resolution[1], pipe = self.pipe).to(
                dtype=self.vae_encoder.dtype, device=self.vae_encoder.device
            )

        caption = self.caption[idx]
        print(caption)

        # tokenizer_1
        ti1 = self.tokenizer_1(
            caption,
            padding="max_length",
            max_length=self.tokenizer_1.model_max_length,
            truncation=True,
            add_special_tokens=True,
            return_tensors="pt",
        ).input_ids.squeeze()

        if self.tokenizer_2 is None:
            ti2 = None
        else:
            ti2 = self.tokenizer_2(
                caption,
                padding="max_length",
                max_length=self.tokenizer_2.model_max_length,
                truncation=True,
                add_special_tokens=True,
                return_tensors="pt",
            ).input_ids.squeeze()

        vae_latent = self.vae_encoder.encode(image).latent_dist
        dummy_vae_latent = vae_latent.sample()

        if self.mask_path is None:
            mask = torch.ones_like(
                dummy_vae_latent, dtype=self.vae_encoder.dtype, device=self.vae_encoder.device
            )

        else:
            mask_path = self.mask_path[idx]
            mask_path = os.path.join(self.data_dir, mask_path)
            mask = PIL.Image.open(mask_path)
            mask = prepare_mask(mask, self.size, self.size).to(
                dtype=self.vae_encoder.dtype, device=self.vae_encoder.device
            )
            
            mask_dtype = mask.dtype
            mask = mask.float()
            mask = torch.nn.functional.interpolate(
                mask, size=(dummy_vae_latent.shape[-2], dummy_vae_latent.shape[-1]), mode="nearest"
            )
            mask = mask.to(dtype=mask_dtype)
            mask = mask.repeat(1, dummy_vae_latent.shape[1], 1, 1)

        assert len(mask.shape) == 4 and len(dummy_vae_latent.shape) == 4

        if ti2 is None: # sd15
            return [ti1], vae_latent, mask.squeeze()
        else: # sdxl
            return [ti1, ti2], vae_latent, mask.squeeze()

    def __getitem__(
        self, idx: int, bucketing_resolution:tuple = None
    ) -> Tuple[Tuple[torch.Tensor, torch.Tensor], torch.Tensor, torch.Tensor]:

        if self.do_cache:
            vae_latent = self.vae_latents[idx].sample() * self.vae_scaling_factor
            return self.tokens_tuple[idx], vae_latent.squeeze(), self.masks[idx]
        else: # This code pathway has not been tested in a long time and might be broken
            tokens, vae_latent, mask = self._process(idx, bucketing_resolution=bucketing_resolution)
            vae_latent = vae_latent.sample() * self.vae_scaling_factor
            return tokens, vae_latent.squeeze(), mask


