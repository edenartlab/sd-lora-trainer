
import os
import shutil
import tarfile
import json
import time
import random
import torch
import numpy as np
import pandas as pd

from dotenv import load_dotenv
from main import train

from trainer.preprocess import preprocess
from trainer.models import pretrained_models
from trainer.config import TrainingConfig
from trainer.utils.io import clean_filename
from trainer.utils.utils import seed_everything

class Eden_LoRa_trainer:
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                    "training_images_folder_path": ("STRING", {"default": "."}),
                    "lora_name": ("STRING", {"default": ""}),
                    "sd_model_version": (["sdxl", "sd15"], ),
                    "seed": ("INT", {"default": 0, "min": 0, "max": 100000}),
                    "resolution": ("INT", {"default": 512, "min": 256, "max": 768}),
                    "train_batch_size": ("INT", {"default": 4, "min": 1, "max": 8}),
                    "max_train_steps":  ("INT", {"default": 400, "min": 50, "max": 1000}),
                    "ti_lr":   ("FLOAT", {"default": 0.001, "min": 0.0001, "max": 0.01, "step": 0.0001}),
                    "unet_lr": ("FLOAT", {"default": 0.001, "min": 0.0001, "max": 0.01, "step": 0.0001}),
                    "lora_rank": ("INT", {"default": 16, "min": 1, "max": 64}),
                    "use_dora": ("BOOLEAN", {"default": False}),
                    "n_tokens": ("INT", {"default": 2, "min": 1, "max": 3}),
                }
        }

    CATEGORY = "Eden 🌱"
    RETURN_TYPES = ("STRING",)
    FUNCTION = "train_lora"

    def train_lora(self, training_images_folder_path,
            lora_name = "",
            concept_mode = "style",
            sd_model_version = "sdxl",
            seed = 0,
            resolution = 521,
            train_batch_size = 4,
            max_train_steps = 400,
            ti_lr = 0.001,
            unet_lr = 0.001,
            lora_rank = 16,
            use_dora = False,
            n_tokens = 2
            ):
        
        print("Starting new training job...")

        config = TrainingConfig(
            name="test",
            lora_training_urls=training_images_folder_path,
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
            debug=True,
        )
        
        with torch.inference_mode(False):
            train_generator = train(config=config)
            while True:
                try:
                    progress_f = next(train_generator)
                except StopIteration as e:
                    config, output_save_dir = e.value  # Capture the return value
                    break

        validation_grid_img_path = os.path.join(output_save_dir, "validation_grid.jpg")
        out_path = f"{clean_filename(lora_name)}_eden_concept_lora_{int(time.time())}.tar"
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

        return (out_path,)