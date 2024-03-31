from trainer.utils.models import load_models, pretrained_models
from trainer.utils.lora import patch_pipe_with_lora
from trainer.utils.val_prompts import val_prompts
from trainer.utils.prompt import prepare_prompt_for_lora
from trainer.utils.io import make_validation_img_grid
from trainer.dataset_and_utils import seed_everything
from diffusers import EulerDiscreteScheduler

import torch
from huggingface_hub import hf_hub_download
import os, json, random, time

if __name__ == "__main__":
    pretrained_model = pretrained_models['sdxl']
    lora_path = 'lora_models/clipx_tiny_test---sdxl_style_lora/checkpoints/checkpoint-500'
    lora_scale = 0.75

    seed = 0
    render_size = (1024, 1024)  # W,H
    n_imgs = 4
    n_steps = 30
    guidance_scale = 8

    use_lightning = False

    output_dir = f'test_images/{lora_path.split("/")[-1]}'
    os.makedirs(output_dir, exist_ok=True)

    seed_everything(seed)

    (pipe,
        tokenizer_one,
        tokenizer_two,
        noise_scheduler,
        text_encoder_one,
        text_encoder_two,
        vae,
        unet) = load_models(pretrained_model, 'cuda', torch.float16)

    if use_lightning:
        repo = "ByteDance/SDXL-Lightning"
        ckpt = "sdxl_lightning_8step_lora.safetensors" # Use the correct ckpt for your step setting!
        pipe.load_lora_weights(hf_hub_download(repo, ckpt))
        pipe.fuse_lora()
        n_steps = 8
        guidance_scale=1

    pipe = patch_pipe_with_lora(pipe, lora_path, lora_scale=lora_scale)

    with open(os.path.join(lora_path, "training_args.json"), "r") as f:
        training_args = json.load(f)
        concept_mode = training_args["concept_mode"]
        trigger_text = training_args["training_attributes"]["trigger_text"]

    if concept_mode == "style":
        validation_prompts_raw = random.sample(val_prompts['style'], n_imgs)
        validation_prompts_raw[0] = ''
    elif concept_mode == "face":
        validation_prompts_raw = random.sample(val_prompts['face'], n_imgs)
        validation_prompts_raw[0] = '<concept>'
    else:
        validation_prompts_raw = random.sample(val_prompts['object'], n_imgs)
        validation_prompts_raw[0] = '<concept>'

    validation_prompts = [prepare_prompt_for_lora(prompt, lora_path, verbose=1, trigger_text=trigger_text) for prompt in validation_prompts_raw]

    pipe.scheduler = EulerDiscreteScheduler.from_config(pipe.scheduler.config, timestep_spacing="trailing")
    generator = torch.Generator(device='cuda').manual_seed(seed)
    pipeline_args = {
                "negative_prompt": "nude, naked, poorly drawn face, ugly, tiling, out of frame, extra limbs, disfigured, deformed body, blurry, blurred, watermark, text, grainy, signature, cut off, draft", 
                "num_inference_steps": n_steps,
                "guidance_scale": guidance_scale,
                "height": render_size[0],
                "width": render_size[1],
                }

    #cross_attention_kwargs = {"scale": lora_scale}
    cross_attention_kwargs = None
    kwargs_str = 'none' if cross_attention_kwargs is None else f'kwargs'

    for i in range(n_imgs):
        pipeline_args["prompt"] = validation_prompts[i]
        print(f"Rendering test img with prompt: {validation_prompts[i]}")
        image = pipe(**pipeline_args, generator=generator, cross_attention_kwargs = cross_attention_kwargs).images[0]
        image.save(os.path.join(output_dir, f"img_seed_{seed}_{i}_lora_scale_{lora_scale:.2f}_{kwargs_str}_{int(time.time())}.jpg"), format="JPEG", quality=95)
