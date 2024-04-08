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
from trainer.models import pretrained_models
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


"""

if XANDER_EXPERIMENT:
    # overwrite some settings for experimentation:
    lora_param_scaler = 0.1
    l1_penalty = 0.2
    prodigy_d_coef = 0.2
    ti_lr = 1e-3
    lora_rank = 24

    lora_training_urls = "https://storage.googleapis.com/public-assets-xander/A_workbox/lora_training_sets/plantoid_5.zip"
    concept_mode = "object"
    mask_target_prompts = ""
    left_right_flip_augmentation = True

    output_dir1 = os.path.join(out_root_dir, run_name + "_xander")
    input_dir1, n_imgs1, trigger_text1, segmentation_prompt1, captions1 = preprocess(
        output_dir1,
        concept_mode,
        input_zip_path=lora_training_urls,
        caption_text=caption_prefix,
        mask_target_prompts=mask_target_prompts,
        target_size=resolution,
        crop_based_on_salience=crop_based_on_salience,
        use_face_detection_instead=use_face_detection_instead,
        temp=clipseg_temperature,
        left_right_flip_augmentation=left_right_flip_augmentation,
        augment_imgs_up_to_n = augment_imgs_up_to_n,
        seed = seed,
        caption_model = caption_model
    )

    lora_training_urls = "https://storage.googleapis.com/public-assets-xander/A_workbox/lora_training_sets/gene_5.zip"
    concept_mode = "face"
    mask_target_prompts = "face"
    left_right_flip_augmentation = False

    output_dir2 = os.path.join(out_root_dir, run_name + "_gene")
    input_dir2, n_imgs2, trigger_text2, segmentation_prompt2, captions2 = preprocess(
        output_dir2,
        concept_mode,
        input_zip_path=lora_training_urls,
        caption_text=caption_prefix,
        mask_target_prompts=mask_target_prompts,
        target_size=resolution,
        crop_based_on_salience=crop_based_on_salience,
        use_face_detection_instead=use_face_detection_instead,
        temp=clipseg_temperature,
        left_right_flip_augmentation=left_right_flip_augmentation,
        augment_imgs_up_to_n = augment_imgs_up_to_n,
        seed = seed,
    )
    
    # Merge the two preprocessing steps:
    n_imgs = n_imgs1 + n_imgs2
    captions = captions1 + captions2
    trigger_text = trigger_text1
    segmentation_prompt = segmentation_prompt1

    # Create merged outdir:
    output_dir = os.path.join(out_root_dir, run_name + "_combined")
    input_dir  = os.path.join(output_dir, "images_out")
    os.makedirs(input_dir, exist_ok=True)

    # Merge the two preprocessed datasets:
    merge_datasets(input_dir1, input_dir2, input_dir, token_dict.keys())

"""

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
        lora_training_urls: str = Input(
            description="Training images for new LORA concept (can be image urls or a .zip file of images)"
        ),
        concept_mode: str = Input(
            description=" 'face' / 'style' / 'object' (default)",
            default="object",
        ),
        sd_model_version: str = Input(
            description=" 'sdxl' / 'sd15' ",
            default="sdxl",
        ),
        name: str = Input(
            description="Name of new LORA concept",
            default="unnamed"
        ),
        seed: int = Input(
            description="Random seed for reproducible training. Leave empty to use a random seed",
            default=None,
        ),
        resolution: int = Input(
            description="Square pixel resolution which your images will be resized to for training, recommended [512-640]",
            default=512,
        ),
        train_batch_size: int = Input(
            description="Batch size (per device) for training",
            default=4,
        ),
        max_train_steps: int = Input(
            description="Number of training steps.",
            default=600,
        ),
        checkpointing_steps: int = Input(
            description="Number of steps between saving checkpoints. Set to very very high number to disable checkpointing, because you don't need intermediate checkpoints.",
            default=10000,
        ),
        is_lora: bool = Input(
            description="Whether to use LoRA training. If set to False, will use full fine tuning",
            default=True,
        ),
        prodigy_d_coef: float = Input(
            description="Multiplier for internal learning rate of Prodigy optimizer",
            default=0.5,
        ),
        ti_lr: float = Input(
            description="Learning rate for training textual inversion embeddings. Don't alter unless you know what you're doing.",
            default=1e-3,
        ),
        ti_weight_decay: float = Input(
            description="weight decay for textual inversion embeddings. Don't alter unless you know what you're doing.",
            default=3e-4,
        ),
        lora_weight_decay: float = Input(
            description="weight decay for lora parameters. Don't alter unless you know what you're doing.",
            default=0.002,
        ),
        l1_penalty: float = Input(
            description="Sparsity penalty for the LoRA matrices, possibly improves merge-ability and generalization",
            default=0.1,
        ),
        lora_rank: int = Input(
            description="Rank of LoRA embeddings. For faces 5 is good, for complex concepts / styles you can try 8 or 12",
            default=12,
        ),
        caption_model: str = Input(
            description="Which captioning model to use. ['gpt4-v', 'blip'] are supported right now",
            default="blip",
        ),
        n_tokens: int = Input(
            description="How many new tokens to inject per concept",
            default=2,
        ),
        verbose: bool = Input(description="verbose output", default=True),
        debug: bool = Input(
            description="For debugging locally only (dont activate this on replicate)",
            default=False,
        ),
        off_ratio_power: float = Input(
            description="How strongly to correct the embedding std vs the avg-std (0=off, 0.05=weak, 0.1=standard)",
            default=0.05,
        ),

    ) -> Iterator[GENERATOR_OUTPUT_TYPE]:

        """
        lambda @1024 training speed (SDXL):
        bs=2: 3.5 imgs/s, 1.8 batches/s
        bs=3: 5.1 imgs/s
        bs=4: 6.0 imgs/s,
        bs=6: 8.0 imgs/s,
        """

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
            checkpointing_steps=checkpointing_steps,
            is_lora=is_lora,
            prodigy_d_coef=prodigy_d_coef,
            ti_lr=ti_lr,
            ti_weight_decay=ti_weight_decay,
            lora_weight_decay=lora_weight_decay,
            l1_penalty=l1_penalty,
            lora_rank=lora_rank,
            caption_model=caption_model,
            n_tokens=n_tokens,
            verbose=verbose,
            debug=debug,
            off_ratio_power=off_ratio_power
        )
        
        train_generator = train(config=config)

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

        attributes = {}
        attributes['grid_prompts'] = config.training_attributes["validation_prompts"]
        attributes['job_time_seconds'] = config.job_time

        print(f"LORA training finished in {config.job_time:.1f} seconds")
        print(f"Returning {out_path}")

        if DEBUG_MODE or debug:
            yield cogPath(out_path)
        else:
            # clear the output_directory to avoid running out of space on the machine:
            #shutil.rmtree(output_dir)
            yield CogOutput(files=[cogPath(out_path)], name=name, thumbnails=[cogPath(validation_grid_img_path)], attributes=config.dict(), isFinal=True, progress=1.0)