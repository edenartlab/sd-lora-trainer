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
from preprocess import preprocess
from trainer_pti import main
from typing import Iterator, Optional
from trainer.utils.io import clean_filename

from trainer.utils.seed import seed_everything
from trainer.utils.config_modification import post_process_args
from trainer.utils.tokens import obtain_inserting_list_tokens
from trainer.models import pretrained_models
from trainer.config import TrainingConfig

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
            description="Training images for new LORA concept (can be image urls or a .zip file of images)", 
            default=None
        ),
        concept_mode: str = Input(
            description=" 'face' / 'style' / 'object' (default)",
            default="object",
        ),
        sd_model_version: str = Input(
            description=" 'sdxl' / 'sd15' ",
            default="sdxl",
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

        start_time = time.time()

        config = TrainingConfig(
            output_dir="path/to/output",
            name="my_training",
            lora_training_urls="https://example.com/lora",
            concept_mode="face",
            sd_model_version="sdxl",
            # Add other parameters as needed
        )

        out_root_dir = "lora_models"

        if seed is None:
            seed = np.random.randint(0, 2**32 - 1)

        # Try to make the training reproducible:
        seed_everything(seed = seed)

        print(f"cog:predict:train_lora:{concept_mode}")

        if not debug:
            yield CogOutput(name=name, progress=0.0)

        # Initialize pretrained_model dictionary
        pretrained_model = pretrained_models[sd_model_version]

        # hardcoded for now:
        token_list = [f"TOK:{n_tokens}"]

        inserting_list_tokens, token_dict = obtain_inserting_list_tokens(token_list=token_list)


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

        else: # normal, single token run:
            
            output_dir = os.path.join(out_root_dir, run_name)

        if not debug:
            yield CogOutput(name=name, progress=0.05)  

        config = TrainingConfig(
            name=name,
            pretrained_model=pretrained_model,
            lora_training_urls=lora_training_urls,
            concept_mode=concept_mode,
            sd_model_version=sd_model_version,
            seed=seed,
            resolution=resolution,
            train_batch_size=train_batch_size,
            num_train_epochs=num_train_epochs,
            max_train_steps=max_train_steps,
            checkpointing_steps=checkpointing_steps,
            gradient_accumulation_steps=gradient_accumulation_steps,
            is_lora=is_lora,
            prodigy_d_coef=prodigy_d_coef,
            ti_lr=ti_lr,
            ti_weight_decay=ti_weight_decay,
            lora_weight_decay=lora_weight_decay,
            l1_penalty=l1_penalty,
            lora_param_scaler=lora_param_scaler,
            snr_gamma=snr_gamma,
            lora_rank=lora_rank,
            caption_prefix=caption_prefix,
            caption_model=caption_model,
            left_right_flip_augmentation=left_right_flip_augmentation,
            augment_imgs_up_to_n=augment_imgs_up_to_n,
            n_tokens=n_tokens,
            mask_target_prompts=mask_target_prompts,
            crop_based_on_salience=crop_based_on_salience,
            use_face_detection_instead=use_face_detection_instead,
            clipseg_temperature=clipseg_temperature,
            verbose=verbose,
            run_name=run_name,
            debug=debug,
            hard_pivot=hard_pivot,
            off_ratio_power=off_ratio_power,
            allow_tf32 = True,
            weight_type="bf16",
            inserting_list_tokens=inserting_list_tokens,
            token_dict=token_dict,
            device="cuda:0",
            output_dir=output_dir,
            scale_lr=False,
            crops_coords_top_left_h = 0,
            crops_coords_top_left_w = 0,
            do_cache = True,
            unet_learning_rate = 1.0,
            lr_scheduler = "constant",
            lr_warmup_steps = 50,
            lr_num_cycles = 1,
            lr_power = 1.0,
            dataloader_num_workers = 0,
        )

        config.save_as_json(
            os.path.join(output_dir, "training_args.json")
        )

        train_generator = main(
            config=config
        )

        while True:
            try:
                progress_f = next(train_generator)
                if not debug:
                    yield CogOutput(name=name, progress=np.round(progress_f, 2))
            except StopIteration as e:
                output_save_dir, validation_prompts = e.value  # Capture the return value
                break

        # save final training_args:
        final_args_dict_path = os.path.join(output_dir, "training_args.json")
        config.save_as_json(
            final_args_dict_path
        )
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
        attributes['grid_prompts'] = validation_prompts
        runtime = time.time() - start_time
        attributes['job_time_seconds'] = runtime

        print(f"LORA training finished in {runtime:.1f} seconds")
        print(f"Returning {out_path}")

        if DEBUG_MODE or debug:
            yield cogPath(out_path)
        else:
            # clear the output_directory to avoid running out of space on the machine:
            #shutil.rmtree(output_dir)
            yield CogOutput(files=[cogPath(out_path)], name=name, thumbnails=[cogPath(validation_grid_img_path)], attributes=config.dict(), isFinal=True, progress=1.0)