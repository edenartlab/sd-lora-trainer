import torch
import os
import random
import shutil
import json
import gc
from diffusers import EulerDiscreteScheduler
from .val_prompts import val_prompts
from ..models import load_models
from .lora import patch_pipe_with_lora
from .prompt import prepare_prompt_for_lora
from .io import make_validation_img_grid

@torch.no_grad()
def render_images(pipe, render_size, lora_path, train_step, seed, is_lora, pretrained_model, trigger_text: str, lora_scale = 0.7, n_steps = 25, n_imgs = 4, device = "cuda:0", verbose: bool = True):
    training_scheduler = pipe.scheduler
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

        (pipe,
            tokenizer_one,
            tokenizer_two,
            noise_scheduler,
            text_encoder_one,
            text_encoder_two,
            vae,
            unet) = load_models(pretrained_model, device, torch.float16)

        pipe = pipe.to(device)
        pipe = patch_pipe_with_lora(pipe, lora_path)
    else:
        print(f"Re-using training pipeline for inference, just swapping the scheduler..")
        
    pipe.scheduler = EulerDiscreteScheduler.from_config(pipe.scheduler.config, timestep_spacing="trailing")
    validation_prompts = [prepare_prompt_for_lora(prompt, lora_path, verbose=verbose, trigger_text=trigger_text) for prompt in validation_prompts_raw]
    generator = torch.Generator(device=device).manual_seed(0)
    pipeline_args = {
                "negative_prompt": "nude, naked, poorly drawn face, ugly, tiling, out of frame, extra limbs, disfigured, deformed body, blurry, blurred, watermark, text, grainy, signature, cut off, draft", 
                "num_inference_steps": n_steps,
                "guidance_scale": 8,
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
        image = pipe(**pipeline_args, generator=generator, cross_attention_kwargs = cross_attention_kwargs).images[0]
        image.save(os.path.join(lora_path, f"img_{train_step:04d}_{i}.jpg"), format="JPEG", quality=95)

    img_grid_path = make_validation_img_grid(lora_path)
    # Copy the grid image to the parent directory for easier comparison:
    grid_img_path = os.path.join(lora_path, "validation_grid.jpg")
    shutil.copy(grid_img_path, os.path.join(os.path.dirname(lora_path), f"validation_grid_{train_step:04d}.jpg"))
    pipe.scheduler = training_scheduler

    return validation_prompts_raw



