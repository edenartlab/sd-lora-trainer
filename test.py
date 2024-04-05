from trainer.models import load_models, pretrained_models
from trainer.utils.lora import patch_pipe_with_lora, blend_conditions
from trainer.utils.val_prompts import val_prompts
from trainer.utils.prompt import prepare_prompt_for_lora
from trainer.utils.io import make_validation_img_grid
from trainer.dataset_and_utils import pick_best_gpu_id
from trainer.utils.seed import seed_everything
from diffusers import EulerDiscreteScheduler

import torch
from huggingface_hub import hf_hub_download
import os, json, random, time
import matplotlib.pyplot as plt

if __name__ == "__main__":

    pretrained_model = pretrained_models['sdxl']

    seed = 1
    render_size = (512, 512)  # H,W
    n_imgs = 24
    n_steps = 30
    guidance_scale = 8

    orig_size = 1024
    target_size = 1024

    #####################################################################################

    output_dir = f'test_images/resolution_exp_512'
    os.makedirs(output_dir, exist_ok=True)

    seed_everything(seed)
    gpu_id = pick_best_gpu_id()

    (pipe,
        tokenizer_one,
        tokenizer_two,
        noise_scheduler,
        text_encoder_one,
        text_encoder_two,
        vae,
        unet) = load_models(pretrained_model, f'cuda:{gpu_id}', torch.float16)

    validation_prompts = random.choices(val_prompts['style'], k=n_imgs)

    pipe.scheduler = EulerDiscreteScheduler.from_config(pipe.scheduler.config, timestep_spacing="trailing")
    generator = torch.Generator(device='cuda').manual_seed(seed)
    negative_prompt = "nude, naked, poorly drawn face, ugly, tiling, out of frame, extra limbs, disfigured, deformed body, blurry, blurred, watermark, text, grainy, signature, cut off, draft"
    pipeline_args = {
                "num_inference_steps": n_steps,
                "guidance_scale": guidance_scale,
                "height": render_size[0],
                "width": render_size[1],
                }

    cs, pcs = [], []
    for i in range(n_imgs):
        embeds = pipe.encode_prompt(
                    validation_prompts[i],
                    f'cuda:{gpu_id}',
                    1,
                    True,
                    negative_prompt
                )

        c, uc, pc, puc = embeds
        cs.append(c)
        pcs.append(pc)

    cs = torch.stack(cs).squeeze()
    pcs = torch.stack(pcs).squeeze()

    cs_norms = torch.norm(cs, dim=-1)
    pcs_norms = torch.norm(pcs, dim=-1)
    print(cs_norms.shape)
    print(pcs_norms.shape)

    for i in range(5):
        print('------------------------')
        prompt_token_norms = cs_norms[i].cpu().numpy()
        plt.figure(figsize=(10, 5))
        plt.plot(prompt_token_norms[2:])
        plt.title(f'{prompt_token_norms[0]} {prompt_token_norms[1]} {validation_prompts[i]}')
        plt.ylim(20,40)
        plt.savefig(f'norms_{i}.png')
        plt.close()

    for i in range(n_imgs):
        pipeline_args["prompt"] = validation_prompts[i]
        pipeline_args["negative_prompt"] = negative_prompt
        print(f"Rendering test img with prompt: {validation_prompts[i]}")
        image = pipe(**pipeline_args, generator=generator,
                    original_size = (orig_size, orig_size),
                    target_size = (target_size, target_size),
                    ).images[0]
        image.save(os.path.join(output_dir, f"img_seed_{seed}_{i}_{orig_size}_{target_size}_{int(time.time())}.jpg"), format="JPEG", quality=95)
