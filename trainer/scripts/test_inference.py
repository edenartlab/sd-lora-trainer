from diffusers import DDPMScheduler, EulerDiscreteScheduler, StableDiffusionPipeline, StableDiffusionXLPipeline
from peft import PeftModel
import numpy as np
import torch
from huggingface_hub import hf_hub_download
import os, json, random, time, sys

sys.path.append('..')
from trainer.models import load_models, pretrained_models
from trainer.lora import patch_pipe_with_lora
from trainer.utils.val_prompts import val_prompts
from trainer.utils.io import make_validation_img_grid
from trainer.utils.utils import seed_everything, pick_best_gpu_id
from trainer.utils.inference import encode_prompt_advanced

def load_model(pretrained_model):
    if pretrained_model['version'] == "sd15":
        pipe = StableDiffusionPipeline.from_single_file(
            pretrained_model['path'], torch_dtype=torch.float16, use_safetensors=True)
    else:
        pipe = StableDiffusionXLPipeline.from_single_file(
            pretrained_model['path'], torch_dtype=torch.float16, use_safetensors=True)

    pipe = pipe.to('cuda', dtype=torch.float16)
    pipe.scheduler = EulerDiscreteScheduler.from_config(pipe.scheduler.config) #, timestep_spacing="trailing")

    return pipe

if __name__ == "__main__":

    pretrained_model = pretrained_models['sdxl']
    lora_path      = '/data/xander/Projects/cog/GitHub_repos/diffusion_trainer2/lora_models/xander--07_19-19-35-sdxl_face_dora/checkpoints/checkpoint-360'
    lora_scales    = np.linspace(0.4, 0.6, 3)
    render_size    = (1024, 256+1024)  # H,W
    n_imgs         = 10
    n_loops        = 4

    n_steps        = 35
    guidance_scale = 7.5
    seed           = 3
    use_lightning  = 1

    #####################################################################################

    output_dir = f'rendered_images4_lightning/{lora_path.split("/")[-1]}'
    os.makedirs(output_dir, exist_ok=True)

    seed_everything(seed)
    pick_best_gpu_id()

    pipe = load_model(pretrained_model)
    pipe.unet = PeftModel.from_pretrained(model = pipe.unet, model_id = lora_path, adapter_name = 'eden_lora')

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
        'an elderly <concept>, old man',
        'a photo of <concept> as a young boy',
        'a painting of elderly <concept>, impressionism, stunning composition',
        'a painting of <concept> as a young kid, playing in the garden',
        'A digital illustration of <concept> as a child, exploring a mystical forest, vibrant colors',
        'An old-fashioned portrait of elderly <concept>, sitting in a cozy armchair, reading a book',
        'A watercolor painting of young <concept>, running through a meadow with a kite, bright and joyful',
        'A sketch of <concept> as a teenager, gazing at the stars, dreamy and contemplative',
        'A vintage photograph of elderly <concept>, wearing a classic hat, exuding wisdom and elegance',
        'A cartoon drawing of young <concept>, having a playful snowball fight, full of energy and laughter',
        'A surrealist painting of <concept> as a child, riding a fantastical creature, imaginative and whimsical',
        'A black and white photo of elderly <concept>, sitting on a park bench, reflecting on lifes journey',
        'A futuristic hologram of elderly <concept>, floating in a space station, surrounded by stars and galaxies'
        'A clay animation of baby <concept>, crawling in a magical garden with talking flowers and dancing insects',
        'An abstract painting of ancient <concept>, merging with the roots of an ancient tree, symbolizing wisdom and eternity',
        'A neon-lit digital art piece of toddler <concept>, playing with holographic toys in a cyberpunk playground',
        'A mosaic of centenarian <concept>, composed of thousands of tiny images from their life, telling a story of a century',
        'A pop art portrait of young <concept>, riding a colorful unicorn, surrounded by rainbows and candy clouds',
        'A steampunk illustration of elderly <concept>, inventing a time machine, surrounded by gears and steam',
        'A fantasy drawing of newborn <concept>, cradled in the arms of a gentle giant, in a land of giants and mythical creatures'

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
                pipe = load_model(pretrained_model)
                pipe.unet = PeftModel.from_pretrained(model = pipe.unet, model_id = lora_path, adapter_name = 'eden_lora')
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