import random
import json
import os
import gc
import torch
from ..dataset_and_utils import load_models
from .lora import patch_pipe_with_lora, prepare_prompt_for_lora
from ..val_prompts import val_prompts
from diffusers import EulerDiscreteScheduler
from PIL import Image

def make_validation_img_grid(img_folder):
    """

    find all the .jpg imgs in img_folder (template = *.jpg)
    if >=4 validation imgs, create a 2x2 grid of them
    otherwise just return the first validation img

    """
    
    # Find all validation images
    validation_imgs = sorted([f for f in os.listdir(img_folder) if f.endswith(".jpg")])

    if len(validation_imgs) < 4:
        # If less than 4 validation images, return path of the first one
        return os.path.join(img_folder, validation_imgs[0])
    else:
        # If >= 4 validation images, create 2x2 grid
        imgs = [Image.open(os.path.join(img_folder, img)) for img in validation_imgs[:4]]

        # Assuming all images are the same size, get dimensions of first image
        width, height = imgs[0].size

        # Create an empty image with 2x2 grid size
        grid_img = Image.new("RGB", (2 * width, 2 * height))

        # Paste the images into the grid
        for i in range(2):
            for j in range(2):
                grid_img.paste(imgs.pop(0), (i * width, j * height))

        # Save the new image
        grid_img_path = os.path.join(img_folder, "validation_grid.jpg")
        grid_img.save(grid_img_path)

        return grid_img_path

@torch.no_grad()
def render_images(training_pipeline, render_size, lora_path, train_step, seed, is_lora, pretrained_model, lora_scale = 0.7, n_steps = 25, n_imgs = 4, device = "cuda:0"):

    random.seed(seed)
    
    with open(os.path.join(lora_path, "training_args.json"), "r") as f:
        training_args = json.load(f)
        concept_mode = training_args["concept_mode"]

    if concept_mode == "style":
        validation_prompts_raw = random.sample(val_prompts['style'], n_imgs)
        validation_prompts_raw[0] = ''

    elif concept_mode == "face":
        validation_prompts_raw = random.sample(val_prompts['face'], n_imgs)
        validation_prompts_raw[0] = '<concept>'
    else:
        validation_prompts_raw = random.sample(val_prompts['object'], n_imgs)
        validation_prompts_raw[0] = '<concept>'


    reload_entire_pipeline = False
    if reload_entire_pipeline: # reload the entire pipeline from disk and load in the lora module
        print(f"Reloading entire pipeline from disk..")
        gc.collect()
        torch.cuda.empty_cache()

        (pipeline,
            tokenizer_one,
            tokenizer_two,
            noise_scheduler,
            text_encoder_one,
            text_encoder_two,
            vae,
            unet) = load_models(pretrained_model, device, torch.float16)

        pipeline = pipeline.to(device)
        pipeline = patch_pipe_with_lora(pipeline, lora_path)

    else:
        print(f"Re-using training pipeline for inference, just swapping the scheduler..")
        pipeline = training_pipeline
        training_scheduler = pipeline.scheduler
    
    pipeline.scheduler = EulerDiscreteScheduler.from_config(pipeline.scheduler.config)
    validation_prompts = [prepare_prompt_for_lora(prompt, lora_path) for prompt in validation_prompts_raw]
    generator = torch.Generator(device=device).manual_seed(0)
    pipeline_args = {
                "negative_prompt": "nude, naked, poorly drawn face, ugly, tiling, out of frame, extra limbs, disfigured, deformed body, blurry, blurred, watermark, text, grainy, signature, cut off, draft", 
                "num_inference_steps": n_steps,
                "guidance_scale": 7,
                "height": render_size[0],
                "width": render_size[1],
                }

    if is_lora > 0:
        cross_attention_kwargs = {"scale": lora_scale}
    else:
        cross_attention_kwargs = None

    for i in range(n_imgs):
        pipeline_args["prompt"] = validation_prompts[i]
        print(f"Rendering validation img with prompt: {validation_prompts[i]}")
        image = pipeline(**pipeline_args, generator=generator, cross_attention_kwargs = cross_attention_kwargs).images[0]
        image.save(os.path.join(lora_path, f"img_{train_step:04d}_{i}.jpg"), format="JPEG", quality=95)

    # create img_grid:
    img_grid_path = make_validation_img_grid(lora_path)

    if not reload_entire_pipeline: # restore the training scheduler
        pipeline.scheduler = training_scheduler

    return validation_prompts_raw
