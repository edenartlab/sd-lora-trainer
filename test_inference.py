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


def blend_conditions(embeds1, embeds2, lora_scale, 
        token_scale_power = 0.5,  # adjusts the curve of the interpolation
        min_token_scale   = 0.5,  # minimum token scale (corresponds to lora_scale = 0)
        verbose = True,
        ):
        
    """
    using lora_scale (or token_scale), apply linear interpolation between two sets of embeddings
    """

    c1, uc1, pc1, puc1 = embeds1
    c2, uc2, pc2, puc2 = embeds2

    token_scale = lora_scale ** token_scale_power
    # rescale the [0,1] range to [min_token_scale, 1] range:
    token_scale = min_token_scale + (1 - min_token_scale) * token_scale

    if verbose:
        print(f"Setting token_scale to {token_scale:.2f} (lora_scale = {lora_scale}, power = {token_scale_power})")
        print('-------------------------')
    try:
        c   = (1 - token_scale) * c1   + token_scale * c2
        pc  = (1 - token_scale) * pc1  + token_scale * pc2
        uc  = (1 - token_scale) * uc1  + token_scale * uc2
        puc = (1 - token_scale) * puc1 + token_scale * puc2
    except:
        print(f"Error in blending conditions, reverting to c2, uc2, pc2, puc2")
        token_scale = 1.0
        c   = c2
        pc  = pc2
        uc  = uc2
        puc = puc2

    return (c, uc, pc, puc), token_scale








if __name__ == "__main__":

    pretrained_model = pretrained_models['sdxl']
    lora_path = 'lora_models/clipx_tiny_test---sdxl_style_lora/checkpoints/checkpoint-500'
    lora_scale = 0.75
    modulate_token_strength = True

    seed = 0
    render_size = (1024, 1024)  # W,H
    n_imgs = 4
    n_steps = 30
    guidance_scale = 8

    use_lightning = False
    
    #####################################################################################

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
    elif concept_mode == "face":
        validation_prompts_raw = random.sample(val_prompts['face'], n_imgs)
    else:
        validation_prompts_raw = random.sample(val_prompts['object'], n_imgs)

    validation_prompts = [prepare_prompt_for_lora(prompt, lora_path, verbose=1, trigger_text=trigger_text) for prompt in validation_prompts_raw]

    pipe.scheduler = EulerDiscreteScheduler.from_config(pipe.scheduler.config, timestep_spacing="trailing")
    generator = torch.Generator(device='cuda').manual_seed(seed)
    negative_prompt = "nude, naked, poorly drawn face, ugly, tiling, out of frame, extra limbs, disfigured, deformed body, blurry, blurred, watermark, text, grainy, signature, cut off, draft"
    pipeline_args = {
                "num_inference_steps": n_steps,
                "guidance_scale": guidance_scale,
                "height": render_size[0],
                "width": render_size[1],
                }

    #cross_attention_kwargs = {"scale": lora_scale}
    cross_attention_kwargs = None
    kwargs_str = 'none' if cross_attention_kwargs is None else f'kwargs'

    for i in range(n_imgs):

        if modulate_token_strength:
            embeds = pipe.encode_prompt(
                validation_prompts[i],
                do_classifier_free_guidance=guidance_scale > 1,
                negative_prompt=negative_prompt)

            zero_embeds = pipe.encode_prompt(
                validation_prompts_raw[i],
                do_classifier_free_guidance=guidance_scale > 1,
                negative_prompt=negative_prompt)

            embeds, token_scale = blend_conditions(zero_embeds, embeds, lora_scale)
            c, uc, pc, puc = embeds

            pipeline_args['prompt_embeds'] = c
            pipeline_args['negative_prompt_embeds'] = uc

            if pretrained_model['version'] == 'sdxl':
                pipeline_args['pooled_prompt_embeds'] = pc
                pipeline_args['negative_pooled_prompt_embeds'] = puc
        else:
            pipeline_args["prompt"] = validation_prompts[i]
            pipeline_args["negative_prompt"] = negative_prompt
            token_scale = 1.0

        print(f"Rendering test img with prompt: {validation_prompts[i]}")
        image = pipe(**pipeline_args, generator=generator, cross_attention_kwargs = cross_attention_kwargs).images[0]
        image.save(os.path.join(output_dir, f"img_seed_{seed}_{i}_tok_scale_{token_scale:.2f}_lora_scale_{lora_scale:.2f}_{kwargs_str}_{int(time.time())}.jpg"), format="JPEG", quality=95)
