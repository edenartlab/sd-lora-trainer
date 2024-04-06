import torch
import os
import random
import shutil
import json
import gc
import re
from diffusers import EulerDiscreteScheduler
from .val_prompts import val_prompts
from ..models import load_models
from .lora import patch_pipe_with_lora
from .io import make_validation_img_grid

def replace_in_string(s, replacements):
    while True:
        replaced = False
        for target, replacement in replacements.items():
            new_s = re.sub(target, replacement, s, flags=re.IGNORECASE)
            if new_s != s:
                s = new_s
                replaced = True
        if not replaced:
            break
    return s

def fix_prompt(prompt):
    # Fix common mistakes
    fix_replacements = {
        r",,": ",",
        r"\s\s+": " ",  # Replaces one or more whitespace characters with a single space
        r"\s\.": ".",
        r"\s,": ","
    }
    return replace_in_string(prompt, fix_replacements)

def prepare_prompt_for_lora(prompt, lora_path, interpolation=False, verbose=True):
    """
    This function is rather ugly, but implements a custom token-replacement policy we adopted at Eden:
    Basically you trigger the lora with a token "TOK" or "<concept>", and then this token gets replaced with the actual learned tokens
    """

    if "_no_token" in lora_path:
        return prompt
        
    orig_prompt = prompt

    # Helper function to read JSON
    def read_json_from_path(path):
        with open(path, "r") as f:
            return json.load(f)

    # Check existence of "special_params.json"
    if not os.path.exists(os.path.join(lora_path, "special_params.json")):
        raise ValueError("This concept is from an old lora trainer that was deprecated. Please retrain your concept for better results!")

    token_map = read_json_from_path(os.path.join(lora_path, "special_params.json"))
    training_args = read_json_from_path(os.path.join(lora_path, "training_args.json"))

    trigger_text = training_args["training_attributes"]["trigger_text"]
    
    try:
        lora_name = str(training_args["name"])
    except: # fallback for old loras that dont have the name field:
        lora_name = "concept"

    lora_name_encapsulated = "<" + lora_name + ">"

    try:
        mode = training_args["concept_mode"]
    except KeyError:
        try:
            mode = training_args["mode"]
        except KeyError:
            mode = "object"

    # Handle different modes
    if mode != "style":
        replacements = {
            "<concept>": trigger_text,
            "<concepts>": trigger_text + "'s",
            lora_name_encapsulated: trigger_text,
            lora_name_encapsulated.lower(): trigger_text,
            lora_name: trigger_text,
            lora_name.lower(): trigger_text,
        }
        prompt = replace_in_string(prompt, replacements)
        if trigger_text not in prompt:
            prompt = trigger_text + ", " + prompt
    else:
        style_replacements = {
            "in the style of <concept>": "in the style of TOK",
            f"in the style of {lora_name_encapsulated}": "in the style of TOK",
            f"in the style of {lora_name_encapsulated.lower()}": "in the style of TOK",
            f"in the style of {lora_name}": "in the style of TOK",
            f"in the style of {lora_name.lower()}": "in the style of TOK"
        }
        prompt = replace_in_string(prompt, style_replacements)
        if "in the style of TOK" not in prompt:
            prompt = "in the style of TOK, " + prompt
        
    # Final cleanup
    prompt = replace_in_string(prompt, {"<concept>": "TOK", lora_name_encapsulated: "TOK"})

    if interpolation and mode != "style":
        prompt = "TOK, " + prompt

    # Replace tokens based on token map
    prompt = replace_in_string(prompt, token_map)
    prompt = fix_prompt(prompt)

    if verbose:
        print('-------------------------')
        print("Adjusted prompt for LORA:")
        print(orig_prompt)
        print('-- to:')
        print(prompt)
        print('-------------------------')

    return prompt

def get_conditioning_signals(config, pipe, token_indices, text_encoders, weight_dtype):

    if config.sd_model_version == 'sdxl':
        if len(token_indices) == 1:
            token_indices = (token_indices[0], None)

        # tokens to text embeds
        prompt_embeds_list = []
        for tok, text_encoder in zip(token_indices, text_encoders):
            if tok is None:
                continue

            prompt_embeds_out = text_encoder(
                tok.to(text_encoder.device),
                output_hidden_states=True,
            )

            pooled_prompt_embeds = prompt_embeds_out[0]
            prompt_embeds = prompt_embeds_out.hidden_states[-2]
            bs_embed, seq_len, _ = prompt_embeds.shape
            prompt_embeds = prompt_embeds.view(bs_embed, seq_len, -1)
            prompt_embeds_list.append(prompt_embeds)

        prompt_embeds = torch.concat(prompt_embeds_list, dim=-1)
        pooled_prompt_embeds = pooled_prompt_embeds.view(bs_embed, -1)

        # Create Spatial-dimensional conditions.
        original_size = (config.resolution, config.resolution)
        target_size   = (config.resolution, config.resolution)

        # I dont understand why, but I get better results hardcoding these values...
        original_size = (1024, 1024)
        #target_size   = (768, 768)

        crops_coords_top_left = (config.crops_coords_top_left_h, config.crops_coords_top_left_w)
        #add_time_ids = list(original_size + crops_coords_top_left + target_size)
        #add_time_ids = torch.tensor([add_time_ids])
        
        if pipe.text_encoder_2 is None:
            text_encoder_projection_dim = int(pooled_prompt_embeds.shape[-1])
        else:
            text_encoder_projection_dim = pipe.text_encoder_2.config.projection_dim

        add_time_ids = pipe._get_add_time_ids(
            original_size,
            crops_coords_top_left,
            target_size,
            dtype=prompt_embeds.dtype,
            text_encoder_projection_dim=text_encoder_projection_dim,
        )

        add_time_ids = add_time_ids.to(config.device, dtype=prompt_embeds.dtype).repeat(
            bs_embed, 1
        )

    elif config.sd_model_version == 'sd15':
        prompt_embeds = text_encoders[0](
                    token_indices[0].to(text_encoders[0].device),
                    output_hidden_states=True,
                )[0]
        pooled_prompt_embeds, add_time_ids = None, None

    return prompt_embeds, pooled_prompt_embeds, add_time_ids

def blend_conditions(embeds1, embeds2, lora_scale,
        token_scale_power = 0.4,  # adjusts the curve of the interpolation
        min_token_scale   = 0.5,  # minimum token scale (corresponds to lora_scale = 0)
        token_scale       = None,
        verbose = 1,
        ):
        
    """
    using lora_scale, apply linear interpolation between two sets of embeddings
    """

    c1, uc1, pc1, puc1 = embeds1
    c2, uc2, pc2, puc2 = embeds2

    if token_scale is None: # compute the token_scale based on lora_scale:
        token_scale = lora_scale ** token_scale_power
        # rescale the [0,1] range to [min_token_scale, 1] range:
        token_scale = min_token_scale + (1 - min_token_scale) * token_scale

    if verbose:
        print(f"Setting token_scale to {token_scale:.2f} (lora_scale = {lora_scale:.2f}, power = {token_scale_power})")
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


def encode_prompt_advanced(pipe, lora_path, prompt, negative_prompt, lora_scale, guidance_scale):
    """
    Helper function to encode the lora_prompt (containing a trained token) and a zero prompt (without the token)
    This allows interpolating the strength of the trained token in the final image.
    """

    lora_prompt = prepare_prompt_for_lora(prompt, lora_path, verbose=1)
    zero_prompt = prompt.replace('<concept>', "")
    zero_prompt = fix_prompt(zero_prompt)

    print(f'Embedding lora prompt: {lora_prompt}')
    embeds = pipe.encode_prompt(
        lora_prompt,
        do_classifier_free_guidance=guidance_scale > 1,
        negative_prompt=negative_prompt)

    print(f"Embedding zero prompt: {zero_prompt}")
    zero_embeds = pipe.encode_prompt(
        zero_prompt,
        do_classifier_free_guidance=guidance_scale > 1,
        negative_prompt=negative_prompt)

    embeds, token_scale = blend_conditions(zero_embeds, embeds, lora_scale)

    return embeds


@torch.no_grad()
def render_images(pipe, render_size, lora_path, train_step, seed, is_lora, pretrained_model, lora_scale,
        trigger_text: str, 
        n_steps = 25, 
        n_imgs = 4, 
        device = "cuda:0", 
        verbose: bool = True
    ):
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
        pipe.vae = pipe.vae.to(device).to(pipe.unet.dtype)
        
    pipe.scheduler = EulerDiscreteScheduler.from_config(pipe.scheduler.config, timestep_spacing="trailing")
    generator = torch.Generator(device=device).manual_seed(0)
    negative_prompt = "nude, naked, poorly drawn face, ugly, tiling, out of frame, extra limbs, disfigured, deformed body, blurry, blurred, watermark, text, grainy, signature, cut off, draft"
    pipeline_args = {
                "num_inference_steps": n_steps,
                "guidance_scale": 8,
                "height": render_size[0],
                "width": render_size[1],
                }

    for i in range(n_imgs):
        print(f"Rendering validation img with prompt: {validation_prompts_raw[i]}")
        c, uc, pc, puc = encode_prompt_advanced(pipe, lora_path, validation_prompts_raw[i], negative_prompt, lora_scale, guidance_scale = 8)

        pipeline_args['prompt_embeds'] = c
        pipeline_args['negative_prompt_embeds'] = uc
        if pretrained_model['version'] == 'sdxl':
            pipeline_args['pooled_prompt_embeds'] = pc
            pipeline_args['negative_pooled_prompt_embeds'] = puc

        image = pipe(**pipeline_args, generator=generator).images[0]
        image.save(os.path.join(lora_path, f"img_{train_step:04d}_{i}.jpg"), format="JPEG", quality=95)

    img_grid_path = make_validation_img_grid(lora_path)
    # Copy the grid image to the parent directory for easier comparison:
    grid_img_path = os.path.join(lora_path, "validation_grid.jpg")
    shutil.copy(grid_img_path, os.path.join(os.path.dirname(lora_path), f"validation_grid_{train_step:04d}.jpg"))
    pipe.scheduler = training_scheduler

    pipe.vae = pipe.vae.to('cpu')
    gc.collect()
    torch.cuda.empty_cache()

    return validation_prompts_raw

@torch.no_grad()
def render_images_eval(
    concept_mode: str,
    output_folder: str,
    render_size: tuple,
    lora_path: str,
    seed: int,
    is_lora: bool,
    pretrained_model: str,
    trigger_text: str,
    lora_scale=0.7,
    n_steps=25,
    n_imgs=4,
    device="cuda:0",
    verbose: bool = True,
):
    random.seed(seed)
    assert os.path.exists(output_folder), f'Invalid folder: {output_folder}'

    if concept_mode == "style":
        validation_prompts_raw = random.sample(val_prompts["style"], n_imgs)
        validation_prompts_raw[0] = ""
    elif concept_mode == "face":
        validation_prompts_raw = random.sample(val_prompts["face"], n_imgs)
        validation_prompts_raw[0] = "<concept>"
    else:
        validation_prompts_raw = random.sample(val_prompts["object"], n_imgs)
        validation_prompts_raw[0] = "<concept>"
   
    print(f"Reloading entire pipeline from disk for eval...")
    gc.collect()
    torch.cuda.empty_cache()

    (
        pipe,
        tokenizer_one,
        tokenizer_two,
        noise_scheduler,
        text_encoder_one,
        text_encoder_two,
        vae,
        unet,
    ) = load_models(pretrained_model, device, torch.float16)

    pipe = pipe.to(device)
    pipe = patch_pipe_with_lora(pipe, lora_path)
    
    pipe.scheduler = EulerDiscreteScheduler.from_config(
        pipe.scheduler.config, timestep_spacing="trailing"
    )
    validation_prompts = [
        prepare_prompt_for_lora(
            prompt, lora_path, verbose=verbose, trigger_text=trigger_text
        )
        for prompt in validation_prompts_raw
    ]
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
    
    filenames = []
    for i in range(n_imgs):
        pipeline_args["prompt"] = validation_prompts[i]
        print(f"Rendering validation img with prompt: {validation_prompts[i]}")
        image = pipe(
            **pipeline_args,
            generator=generator,
            cross_attention_kwargs=cross_attention_kwargs,
        ).images[0]
        filename = os.path.join(output_folder, f"{i}.jpg")
        image.save(
            filename,
            format="JPEG",
            quality=95,
        )
        filenames.append(filename)

    return filenames, validation_prompts
