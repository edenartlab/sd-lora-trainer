import os
import shutil
import tarfile
import json
import time
import random
import torch
import numpy as np
import pandas as pd

from cog import BasePredictor, BaseModel, File, Input, Path as cogPath
from dotenv import load_dotenv
from main import train
from typing import Iterator, Optional

from trainer.preprocess import preprocess
from trainer.config import TrainingConfig
from trainer.utils.io import clean_filename
from trainer.utils.utils import seed_everything

DEBUG_MODE = False
XANDER_EXPERIMENT = False

load_dotenv()

os.environ["TORCH_HOME"] = "/src/.torch"
os.environ["TRANSFORMERS_CACHE"] = "/src/.huggingface/"
os.environ["DIFFUSERS_CACHE"] = "/src/.huggingface/"
os.environ["HF_HOME"] = "/src/.huggingface/"


class CogOutput(BaseModel):
    files: Optional[list[cogPath]] = []
    name: Optional[str] = None
    thumbnails: Optional[list[cogPath]] = []
    attributes: Optional[dict] = None
    progress: Optional[float] = None
    isFinal: bool = False

class Predictor(BasePredictor):

    GENERATOR_OUTPUT_TYPE = Path if DEBUG_MODE else CogOutput

    def setup(self):
        print("cog:setup")

    def predict(
        self, 
        name: str = Input(
            description="Name of new LORA concept",
            default="unnamed"
        ),
        lora_training_urls: str = Input(
            description="Training images for new LORA concept (can be image urls or an url to a .zip file of images)"
        ),
        concept_mode: str = Input(
            description="What are you trying to learn?",
            choices=["style", "face", "object"],
            default="style",
        ),
        sd_model_version: str = Input(
            description="SDXL gives much better LoRa's if you just need static images. If you want to make AnimateDiff animations, train an SD15 lora.",
            choices=["sdxl", "sd15"],
            default="sdxl",
        ),
        max_train_steps: int = Input(
            description="Number of training steps. Increasing this usually leads to overfitting, only viable if you have > 100 training imgs. For faces you may want to reduce to eg 300",
            default=400
        ),
        resolution: int = Input(
            description="Square pixel resolution which your images will be resized to for training, highly recommended: 512 or 640",
            default=512
        ),
        train_batch_size: int = Input(
            description="Batch size (per device) for training (dont increase unless running on a BIG GPU)",
            default=4
        ),
        unet_lr: float = Input(
            description="final learning rate of unet (after warmup), increasing this usually leads to strong overfitting",
            default=0.001
        ),
        ti_lr: float = Input(
            description="Learning rate for training textual inversion embeddings. Don't alter unless you know what you're doing.",
            default=0.001
        ),
        lora_rank: int = Input(
            description="Rank of LoRA embeddings for the unet.",
            default=16
        ),
        use_dora: bool = Input(
            description="Use Dora instead of LoRa",
            default=False,
        ),
        n_tokens: int = Input(
            description="How many new tokens to train (highly recommended to leave this at 2)",
            ge=1, le=3, default=2
        ),
        seed: int = Input(
            description="Random seed for reproducible training. Leave empty to use a random seed",
            default=None,
        ),

    ) -> Iterator[GENERATOR_OUTPUT_TYPE]:

        """
        lambda training speed (SDXL):
        bs=2: 3.5 imgs/s, 1.8 batches/s
        bs=3: 5.1 imgs/s
        bs=4: 6.0 imgs/s,
        bs=6: 8.0 imgs/s,
        """

        debug = False

        print("cog:predict starting new training job...")
        if not debug:
            yield CogOutput(name=name, progress=0.0)
        
        config = TrainingConfig(
            name=name,
            lora_training_urls=lora_training_urls,
            concept_mode=concept_mode,
            sd_model_version=sd_model_version,
            seed=seed,
            resolution=resolution,
            train_batch_size=train_batch_size,
            max_train_steps=max_train_steps,
            checkpointing_steps=10000,
            ti_lr=ti_lr,
            unet_lr=unet_lr,
            lora_rank=lora_rank,
            use_dora=use_dora,
            caption_model="blip",
            n_tokens=n_tokens,
            verbose=True,
            debug=debug,
        )
        
        train_generator = train(config=config)
        print(f"Debug: {debug}")

        while True:
            try:
                progress_f = next(train_generator)
                if not debug:
                    yield CogOutput(name=name, progress=np.round(progress_f, 2))
            except StopIteration as e:
                config, output_save_dir = e.value  # Capture the return value
                break

        validation_grid_img_path = os.path.join(output_save_dir, "validation_grid.jpg")
        out_path = f"{clean_filename(name)}_eden_concept_lora_{int(time.time())}.tar"
        directory = cogPath(output_save_dir)

        with tarfile.open(out_path, "w") as tar:
            print("Adding files to tar...")
            for file_path in directory.rglob("*"):
                print(file_path)
                arcname = file_path.relative_to(directory)
                tar.add(file_path, arcname=arcname)
            
            # Add instructions README:
            tar.add("instructions_README.md", arcname="README.md")
            tar.add("comfyUI_workflow_lora_txt2img.json", arcname="comfyUI_workflow_lora_txt2img.json")
            if sd_model_version == "sd15":
                tar.add("comfyUI_workflow_lora_adiff.json", arcname="comfyUI_workflow_lora_adiff.json")

        attributes = {}
        attributes['grid_prompts'] = config.training_attributes["validation_prompts"]
        attributes['job_time_seconds'] = config.job_time

        print(f"LORA training finished in {config.job_time:.1f} seconds")
        print(f"Returning {out_path}")

        if DEBUG_MODE or debug:
            yield cogPath(out_path)
        else:
            yield CogOutput(files=[cogPath(out_path)], name=name, thumbnails=[cogPath(validation_grid_img_path)], attributes=config.dict(), isFinal=True, progress=1.0)