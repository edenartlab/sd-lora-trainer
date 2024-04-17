from diffusers import DDPMScheduler, EulerDiscreteScheduler, StableDiffusionPipeline, StableDiffusionXLPipeline
from peft import PeftModel
import numpy as np
import torch
from huggingface_hub import hf_hub_download
import os, json, random, time, sys

sys.path.append('.')
sys.path.append('..')
from trainer.models import load_models, pretrained_models
from trainer.lora import patch_pipe_with_lora
from trainer.utils.val_prompts import val_prompts
from trainer.utils.io import make_validation_img_grid
from trainer.utils.utils import seed_everything, pick_best_gpu_id
from trainer.inference import encode_prompt_advanced
from trainer.checkpoint import load_checkpoint

if __name__ == "__main__":

    pretrained_model = pretrained_models['sdxl']
    lora_path      = 'lora_models/xander_ti--16_21-53-50-sdxl_face_lora_512_0.8_gpt4-v/checkpoints/checkpoint-0'
    lora_scales    = np.linspace(0.5, 0.7, 3)
    render_size    = (768, 768)  # H,W
    n_imgs         = 4
    n_loops        = 1

    n_steps        = 35
    guidance_scale = 7.5
    seed           = 0
    use_lightning  = 0

    #####################################################################################

    output_dir = f'rendered_images/{lora_path.split("/")[-1]}'
    os.makedirs(output_dir, exist_ok=True)

    seed_everything(seed)
    pick_best_gpu_id()

    pipe = load_checkpoint(
        pretrained_model_version="sdxl",
        pretrained_model_path=pretrained_models["sdxl"]["path"],
        checkpoint_folder=lora_path,
        is_lora=True,
        device="cuda:0"
    )

    if use_lightning:
        repo = "ByteDance/SDXL-Lightning"
        ckpt = "sdxl_lightning_8step_lora.safetensors" # Use the correct ckpt for your step setting!
        pipe.load_lora_weights(hf_hub_download(repo, ckpt))
        pipe.fuse_lora()
        n_steps = 8
        guidance_scale=1.5

    with open(os.path.join(lora_path, "training_args.json"), "r") as f:
        training_args = json.load(f)

    if training_args["concept_mode"] == "style":
        validation_prompts_raw = random.choices(val_prompts['style'], k=n_imgs)
    elif training_args["concept_mode"] == "face":
        validation_prompts_raw = random.choices(val_prompts['face'], k=n_imgs)
    else:
        validation_prompts_raw = random.choices(val_prompts['object'], k=n_imgs)

    validation_prompts_raw = [
            "actress in TOK at a gala",
            "TOK, facy dinner party",
            "woman in TOK and jacket",
            "smiling woman in TOK at a party",
            "jennifer jones on TOK at music awards"
    ]

    negative_prompt = "nude, naked, poorly drawn face, ugly, tiling, out of frame, extra limbs, disfigured, deformed body, blurry, blurred, watermark, text, grainy, signature, cut off, draft"
    pipeline_args = {
                "num_inference_steps": n_steps,
                "guidance_scale": guidance_scale,
                "height": render_size[0],
                "width": render_size[1],
                }
    for jj in range(n_loops):
        for i in range(len(validation_prompts_raw)):
            for lora_scale in lora_scales:
                seed += 1
                #pipe = load_model(pretrained_model)
                #pipe.unet = PeftModel.from_pretrained(model = pipe.unet, model_id = lora_path, adapter_name = 'eden_lora')
                pipe = patch_pipe_with_lora(pipe, lora_path, lora_scale=lora_scale)
                generator = torch.Generator(device='cuda').manual_seed(seed)

                c, uc, pc, puc = encode_prompt_advanced(pipe, lora_path, validation_prompts_raw[i], negative_prompt, lora_scale, guidance_scale, concept_mode = training_args["concept_mode"])

                pipeline_args['prompt_embeds'] = c
                pipeline_args['negative_prompt_embeds'] = uc
                if pretrained_model['version'] == 'sdxl':
                    pipeline_args['pooled_prompt_embeds'] = pc
                    pipeline_args['negative_pooled_prompt_embeds'] = puc

                image = pipe(**pipeline_args, generator=generator).images[0]
                image.save(os.path.join(output_dir, f"{validation_prompts_raw[i][:40]}_seed_{seed}_{i}_lora_scale_{lora_scale:.2f}_{int(time.time())}.jpg"), format="JPEG", quality=95)

        seed += 1