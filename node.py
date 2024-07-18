import os
import tarfile
import json
import time
import torch

from main import train
from trainer.config import TrainingConfig, model_paths
from trainer.utils.io import clean_filename

import folder_paths

class Eden_LoRa_trainer:
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                    "training_images_folder_path": ("STRING", {"default": "."}),
                    "ckpt_name": (folder_paths.get_filename_list("checkpoints"), ),
                    "lora_name": ("STRING", {"default": "Eden_LoRa"}),
                    "mode": (["style", "face", "object"], ),
                    "resolution": ("INT", {"default": 512, "min": 256, "max": 768}),
                    "train_batch_size": ("INT", {"default": 4, "min": 1, "max": 8}),
                    "max_train_steps":  ("INT", {"default": 400, "min": 50, "max": 1000}),
                    "ti_lr":   ("FLOAT", {"default": 0.001, "min": 0.0001, "max": 0.01, "step": 0.0001}),
                    "unet_lr": ("FLOAT", {"default": 0.001, "min": 0.0001, "max": 0.01, "step": 0.0001}),
                    "lora_rank": ("INT", {"default": 16, "min": 1, "max": 64}),
                    "use_dora": ("BOOLEAN", {"default": False}),
                    "n_tokens": ("INT", {"default": 2, "min": 1, "max": 3}),
                    "debug_mode": ("BOOLEAN", {"default": False}),
                    "checkpointing_steps": ("INT", {"default": 200, "min": 100, "max": 2000}),
                    "seed": ("INT", {"default": 0, "min": 0, "max": 100000}),
                }
        }

    CATEGORY = "Eden 🌱"
    RETURN_TYPES = ("STRING", "STRING", "FLOAT")
    RETURN_NAMES = ("lora_path", "embedding_path", "total_runtime_s")
    FUNCTION = "train_lora"

    def train_lora(self, 
            training_images_folder_path,
            ckpt_name,
            lora_name = "eden_lora",
            mode = "style",
            seed = 0,
            resolution = 521,
            train_batch_size = 4,
            max_train_steps = 400,
            ti_lr = 0.001,
            unet_lr = 0.001,
            lora_rank = 16,
            use_dora = False,
            n_tokens = 2,
            debug_mode = False,
            checkpointing_steps = 1000,
            ):
        
        print("Starting new training job...")

        # Overwrite hardcoded paths to point to comfyUI folders:
        model_paths.set_path("CLIP", os.path.join(folder_paths.models_dir, "clipseg"))
        model_paths.set_path("BLIP", os.path.join(folder_paths.models_dir, "blip"))
        model_paths.set_path("SR", os.path.join(folder_paths.models_dir, "upscale_models"))
        model_paths.set_path("SD", os.path.join(folder_paths.models_dir, "checkpoints"))

        ckpt_path = folder_paths.get_full_path("checkpoints", ckpt_name)

        config = TrainingConfig(
            name=lora_name,
            lora_training_urls=training_images_folder_path,
            concept_mode=mode,
            ckpt_path=ckpt_path,
            seed=seed,
            resolution=resolution,
            train_batch_size=train_batch_size,
            max_train_steps=max_train_steps,
            checkpointing_steps=checkpointing_steps,
            ti_lr=ti_lr,
            unet_lr=unet_lr,
            lora_rank=lora_rank,
            use_dora=use_dora,
            caption_model="blip",
            n_tokens=n_tokens,
            verbose=True,
            debug=debug_mode,
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
        directory = cogPath(output_save_dir)

        attributes = {}
        attributes['grid_prompts'] = config.training_attributes["validation_prompts"]
        attributes['job_time_seconds'] = config.job_time

        print(f"LORA training finished in {config.job_time:.1f} seconds")
        print(f"Returning {out_path}")

        lora_path = os.path.join(output_save_dir, config.name + ".safetensors")
        embedding_path = os.path.join(output_save_dir, config.name + "_embeddings.safetensors")

        return (lora_path, embedding_path, config.job_time)