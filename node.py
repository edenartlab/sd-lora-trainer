import os
import tarfile
import json
import time
import torch
import numpy as np
from PIL import Image

from main import train
from trainer.config import TrainingConfig, model_paths
from trainer.utils.io import clean_filename

import folder_paths
import comfy.utils

class Eden_LoRa_trainer:
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                    "training_images_folder_path": ("STRING", {"default": "."}),
                    "ckpt_name": (folder_paths.get_filename_list("checkpoints"), ),
                    "lora_name": ("STRING", {"default": "Eden_Token_LoRa"}),
                    "mode": (["style", "face", "object"], ),
                    "training_resolution": ("INT", {"default": 512, "min": 256, "max": 1024}),
                    "train_batch_size": ("INT", {"default": 4, "min": 1, "max": 8}),
                    "max_train_steps":  ("INT", {"default": 300, "min": 10, "max": 10000}),
                    "ti_lr":   ("FLOAT", {"default": 0.001, "min": 0.0, "max": 0.005, "step": 0.0001}),
                    "unet_lr": ("FLOAT", {"default": 0.0005, "min": 0.0, "max": 0.005, "step": 0.0001}),
                    "lora_rank": ("INT", {"default": 16, "min": 1, "max": 64}),
                    "disable_ti": ("BOOLEAN", {"default": False}),
                    "n_tokens": ("INT", {"default": 3, "min": 1, "max": 5}),
                    "save_checkpoint_every_n_steps": ("INT", {"default": 200, "min": 10, "max": 10000}),
                    "sample_imgs_lora_scale": ("FLOAT", {"default": 0.7, "min": 0.0, "max": 1.25}),
                    "plot_training_graphs_on_disk": ("BOOLEAN", {"default": False}),
                    "seed": ("INT", {"default": 0, "min": 0, "max": 100000}),
                }
        }

    CATEGORY = "Eden 🌱"
    RETURN_TYPES = ("IMAGE", "STRING", "STRING", "STRING")
    RETURN_NAMES = ("sample_images", "lora_path", "embedding_path", "final_msg")
    FUNCTION = "train_lora"

    def train_lora(self, 
            training_images_folder_path,
            ckpt_name,
            lora_name,
            mode,
            training_resolution,
            train_batch_size,
            max_train_steps ,
            ti_lr,
            unet_lr,
            lora_rank,
            disable_ti,
            n_tokens,
            plot_training_graphs_on_disk,
            save_checkpoint_every_n_steps,
            sample_imgs_lora_scale,
            seed,
            ):
        
        print("Starting new training job...")

        # Overwrite hardcoded paths to point to comfyUI folders:
        model_paths.set_path("CLIP", os.path.join(folder_paths.models_dir, "clipseg"))
        model_paths.set_path("FLORENCE", os.path.join(folder_paths.models_dir, "LLM"))
        model_paths.set_path("BLIP", os.path.join(folder_paths.models_dir, "blip"))
        model_paths.set_path("SR", os.path.join(folder_paths.models_dir, "upscale_models"))
        model_paths.set_path("SD", os.path.join(folder_paths.models_dir, "checkpoints"))

        ckpt_path = folder_paths.get_full_path("checkpoints", ckpt_name)

        config = TrainingConfig(
            name=lora_name,
            output_dir="output",
            lora_training_urls=training_images_folder_path,
            concept_mode=mode,
            ckpt_path=ckpt_path,
            seed=seed,
            resolution=training_resolution,
            train_batch_size=train_batch_size,
            max_train_steps=max_train_steps,
            checkpointing_steps=save_checkpoint_every_n_steps,
            sample_imgs_lora_scale=sample_imgs_lora_scale,
            ti_lr=ti_lr,
            unet_lr=unet_lr,
            lora_rank=lora_rank,
            use_dora=False,
            caption_model="blip",
            disable_ti=disable_ti,
            n_tokens=n_tokens,
            verbose=True,
            debug=plot_training_graphs_on_disk,
        )

        pbar = comfy.utils.ProgressBar(100)

        with torch.inference_mode(False):
            train_generator = train(config=config)
            while True:
                try:
                    progress_f = next(train_generator)
                    pbar.update_absolute(progress_f * 100)
                except StopIteration as e:
                    config, output_save_dir = e.value  # Capture the return value
                    break

        attributes = {}
        attributes['grid_prompts'] = config.training_attributes["validation_prompts"]
        attributes['job_time_seconds'] = config.job_time

        print(f"LORA training node finished in {config.job_time:.1f} seconds")
        print("---------- Made with love by Eden.art 🌱 ----------")
        
        # safetensors paths:
        paths = [os.path.join(output_save_dir, f) for f in os.listdir(output_save_dir) if f.endswith(".safetensors")]

        # find the index of the path containing "_embeddings.safetensors":
        for i, path in enumerate(paths):
            if "_embeddings.safetensors" in path:
                embedding_path = path
            else:
                lora_path = path

        # Load the grid images:
        grid_images = []
        grid_dir = os.path.dirname(output_save_dir)
        for f in os.listdir(grid_dir):
            if "validation_grid" in f:
                grid_image = Image.open(os.path.join(grid_dir, f))
                grid_image = np.array(grid_image).astype(np.float32) / 255.0
                grid_image = torch.from_numpy(grid_image)
                grid_images.append(grid_image)
        
        grid_images = torch.stack(grid_images)

        # Make sure that grid_images always has 4 dimensions:
        if len(grid_images.shape) == 3:
            grid_images = grid_images.unsqueeze(0)

        final_msg = f"LoRa trained in {config.job_time/60:.1f} minutes. Files saved at {output_save_dir}"

        return (grid_images, lora_path, embedding_path, final_msg)