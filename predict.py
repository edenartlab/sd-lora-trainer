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
from io_utils import clean_filename

from trainer.utils.seed import seed_everything
from trainer.utils.download import download_weights
from trainer.utils.config_modification import modify_args_based_on_concept_mode
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
            description="Square pixel resolution which your images will be resized to for training recommended [768-1024]",
            default=960,
        ),
        train_batch_size: int = Input(
            description="Batch size (per device) for training",
            default=4,
        ),
        num_train_epochs: int = Input(
            description="Number of epochs to loop through your training dataset",
            default=10000,
        ),
        max_train_steps: int = Input(
            description="Number of individual training steps. Takes precedence over num_train_epochs",
            default=600,
        ),
        checkpointing_steps: int = Input(
            description="Number of steps between saving checkpoints. Set to very very high number to disable checkpointing, because you don't need one.",
            default=10000,
        ),
        gradient_accumulation_steps: int = Input(
             description="Number of training steps to accumulate before a backward pass. Effective batch size = gradient_accumulation_steps * batch_size",
             default=1,
         ),
        is_lora: bool = Input(
            description="Whether to use LoRA training. If set to False, will use Full fine tuning",
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
        lora_param_scaler: float = Input(
            description="Multiplier for the starting weights of the lora matrices",
            default=0.5,
        ),
        snr_gamma: float = Input(
            description="see https://arxiv.org/pdf/2303.09556.pdf, set to None to disable snr training",
            default=5.0,
        ),
        lora_rank: int = Input(
            description="Rank of LoRA embeddings. For faces 5 is good, for complex concepts / styles you can try 8 or 12",
            default=12,
        ),
        caption_prefix: str = Input(
            description="Prefix text prepended to automatic captioning. Must contain the 'TOK'. Example is 'a photo of TOK, '.  If empty, chatgpt will take care of this automatically",
            default="",
        ),
        caption_model: str = Input(
            description="Which captioning model to use. ['gpt4-v', 'blip'] are supported right now",
            default="blip",
        ),
        left_right_flip_augmentation: bool = Input(
            description="Add left-right flipped version of each img to the training data, recommended for most cases. If you are learning a face, you prob want to disable this",
            default=True,
        ),
        augment_imgs_up_to_n: int = Input(
            description="Apply data augmentation (no lr-flipping) until there are n training samples (0 disables augmentation completely)",
            default=20,
        ),
        n_tokens: int = Input(
            description="How many new tokens to inject per concept",
            default=2,
        ),
        mask_target_prompts: str = Input(
            description="Prompt that describes most important part of the image, will be used for CLIP-segmentation. For example, if you are learning a person 'face' would be a good segmentation prompt",
            default=None,
        ),
        crop_based_on_salience: bool = Input(
            description="If you want to crop the image to `target_size` based on the important parts of the image, set this to True. If you want to crop the image based on face detection, set this to False",
            default=True,
        ),
        use_face_detection_instead: bool = Input(
            description="If you want to use face detection instead of CLIPSeg for masking. For face applications, we recommend using this option.",
            default=False,
        ),
        clipseg_temperature: float = Input(
            description="How blurry you want the CLIPSeg mask to be. We recommend this value be something between `0.5` to `1.0`. If you want to have more sharp mask (but thus more errorful), you can decrease this value.",
            default=0.7,
        ),
        verbose: bool = Input(description="verbose output", default=True),
        run_name: str = Input(
            description="Subdirectory where all files will be saved",
            default=str(int(time.time())),
        ),
        debug: bool = Input(
            description="for debugging locally only (dont activate this on replicate)",
            default=False,
        ),
        hard_pivot: bool = Input(
            description="Use hard freeze for ti_lr. If set to False, will use soft transition of learning rates",
            default=False,
        ),
        off_ratio_power: float = Input(
            description="How strongly to correct the embedding std vs the avg-std (0=off, 0.05=weak, 0.1=standard)",
            default=0.1,
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
        out_root_dir = "lora_models"

        if seed is None:
            seed = np.random.randint(0, 2**32 - 1)

        # Try to make the training reproducible:
        seed_everything(seed = seed)
        (
            concept_mode, left_right_flip_augmentation, mask_target_prompts, clipseg_temperature, l1_penalty
        ) = modify_args_based_on_concept_mode(
            concept_mode=concept_mode,
            left_right_flip_augmentation=left_right_flip_augmentation,
            mask_target_prompts=mask_target_prompts,
            clipseg_temperature=clipseg_temperature,
            l1_penalty=l1_penalty
        )

        print(f"cog:predict:train_lora:{concept_mode}")

        if not debug:
            yield CogOutput(name=name, progress=0.0)

        # Initialize pretrained_model dictionary
        pretrained_model = pretrained_models[sd_model_version]

        # Download the weights if they don't exist locally
        if not os.path.exists(pretrained_model['path']):
            download_weights(pretrained_model['url'], pretrained_model['path'])
        
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
            input_dir, n_imgs, trigger_text, segmentation_prompt, captions = preprocess(
                working_directory=output_dir,
                concept_mode=concept_mode,
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
            num_training_images=n_imgs, 
            trigger_text=trigger_text,  
            segmentation_prompt=segmentation_prompt,
            allow_tf32 = True,
            mixed_precision="bf16",
            instance_data_dir=os.path.join(input_dir, "captions.csv"),
            inserting_list_tokens=inserting_list_tokens,
            token_dict=token_dict,
            device="cuda:0",
            training_captions=captions[:50]
        )

        config.save_as_json(
            os.path.join(output_dir, "training_args.json")
        )

        train_generator = main(
            output_dir=output_dir,
            resolution=resolution,
            train_batch_size=train_batch_size,
            num_train_epochs=num_train_epochs,
            max_train_steps=max_train_steps,
            l1_penalty=l1_penalty,
            prodigy_d_coef=prodigy_d_coef,
            ti_lr=ti_lr,
            ti_weight_decay=ti_weight_decay,
            snr_gamma=snr_gamma,
            lora_weight_decay=lora_weight_decay,
            checkpointing_steps=checkpointing_steps,
            scale_lr=False,
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