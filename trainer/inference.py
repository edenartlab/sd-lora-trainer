import torch
import os
import random
import shutil
import json
import gc
import re
from diffusers import EulerDiscreteScheduler

from trainer.utils.val_prompts import val_prompts
from trainer.utils.utils import fix_prompt, replace_in_string
from trainer.models import load_models
from .checkpoint import load_checkpoint
from .lora import patch_pipe_with_lora

from diffusers import (
    DDPMScheduler,
    EulerDiscreteScheduler,
    StableDiffusionPipeline,
    StableDiffusionXLPipeline,
)


def load_model(pretrained_model: dict):
    if pretrained_model["version"] == "sd15":
        pipe = StableDiffusionPipeline.from_single_file(
            pretrained_model["path"], torch_dtype=torch.float16, use_safetensors=True
        )
    else:
        pipe = StableDiffusionXLPipeline.from_single_file(
            pretrained_model["path"], torch_dtype=torch.float16, use_safetensors=True
        )

    pipe = pipe.to("cuda", dtype=torch.float16)
    pipe.scheduler = EulerDiscreteScheduler.from_config(
        pipe.scheduler.config
    )  # , timestep_spacing="trailing")

    return pipe


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
        raise ValueError(
            "This concept is from an old lora trainer that was deprecated. Please retrain your concept for better results!"
        )

    token_map = read_json_from_path(os.path.join(lora_path, "special_params.json"))
    training_args = read_json_from_path(os.path.join(lora_path, "training_args.json"))

    trigger_text = training_args["training_attributes"]["trigger_text"]

    try:
        lora_name = str(training_args["name"])
    except:  # fallback for old loras that dont have the name field:
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
            f"in the style of {lora_name.lower()}": "in the style of TOK",
        }
        prompt = replace_in_string(prompt, style_replacements)
        if "in the style of TOK" not in prompt:
            prompt = "in the style of TOK, " + prompt

    # Final cleanup
    prompt = replace_in_string(
        prompt, {"<concept>": "TOK", lora_name_encapsulated: "TOK"}
    )

    if interpolation and mode != "style":
        prompt = "TOK, " + prompt

    # Replace tokens based on token map
    prompt = replace_in_string(prompt, token_map)
    prompt = fix_prompt(prompt)

    if verbose:
        print("-------------------------")
        print("Adjusted prompt for LORA:")
        print(orig_prompt)
        print("-- to:")
        print(prompt)
        print("-------------------------")

    return prompt


def get_conditioning_signals(config, pipe, captions):
    conditioning_signals = pipe.encode_prompt(
        prompt=captions,
        device=pipe.unet.device,
        num_images_per_prompt=1,
        do_classifier_free_guidance=True,
        negative_prompt=None,
        clip_skip=None,
    )

    try:  # sd15
        prompt_embeds, negative_prompt_embeds = conditioning_signals
        pooled_prompt_embeds, add_time_ids = None, None

    except:  # sdxl
        (
            prompt_embeds,
            negative_prompt_embeds,
            pooled_prompt_embeds,
            negative_pooled_prompt_embeds,
        ) = conditioning_signals

        # Create Spatial-dimensional conditions.
        # I dont understand why, but I get better results hardcoding the original_size values...
        # original_size = (config.resolution, config.resolution)
        original_size = (1024, 1024)
        target_size = (config.resolution, config.resolution)

        crops_coords_top_left = (
            config.crops_coords_top_left_h,
            config.crops_coords_top_left_w,
        )

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
            prompt_embeds.shape[0], 1
        )

    return prompt_embeds, pooled_prompt_embeds, add_time_ids


def blend_conditions(
    embeds1,
    embeds2,
    lora_scale,
    token_scale_power=0.4,  # adjusts the curve of the interpolation
    min_token_scale=0.5,  # minimum token scale (corresponds to lora_scale = 0)
    token_scale=None,
    verbose=1,
):
    """
    using lora_scale, apply linear interpolation between two sets of embeddings
    """
    try:  # sdxl:
        c1, uc1, pc1, puc1 = embeds1
        c2, uc2, pc2, puc2 = embeds2
    except:  # sd15:
        c1, uc1 = embeds1
        c2, uc2 = embeds2
        pc1, pc2, puc1, puc2 = None, None, None, None

    if token_scale is None:  # compute the token_scale based on lora_scale:
        token_scale = lora_scale**token_scale_power
        # rescale the [0,1] range to [min_token_scale, 1] range:
        token_scale = min_token_scale + (1 - min_token_scale) * token_scale

    if verbose:
        print(
            f"Setting token_scale to {token_scale:.2f} (lora_scale = {lora_scale:.2f}, power = {token_scale_power})"
        )

    try:
        c = (1 - token_scale) * c1 + token_scale * c2
        uc = (1 - token_scale) * uc1 + token_scale * uc2
        try:
            pc = (1 - token_scale) * pc1 + token_scale * pc2
            puc = (1 - token_scale) * puc1 + token_scale * puc2
        except:
            pc, puc = None, None

        embeds = (c, uc, pc, puc)
    except:
        print(
            f"Error in blending conditions for toking interpolation, falling back to embeds2"
        )
        token_scale = 1.0
        embeds = (c2, uc2, pc2, puc2)

    return embeds, token_scale


def encode_prompt_advanced(
    pipe,
    lora_path,
    prompt,
    negative_prompt,
    lora_scale,
    guidance_scale,
    token_scale=None,
    concept_mode=None,
):
    """
    Helper function to encode the lora_prompt (containing a trained token) and a zero prompt (without the token)
    This allows interpolating the strength of the trained token in the final image.
    """

    lora_prompt = prepare_prompt_for_lora(prompt, lora_path, verbose=1)
    if concept_mode == "face":
        replace_str = "person"
    elif concept_mode == "object":
        replace_str = "object"
    else:
        replace_str = ""

    zero_prompt = prompt.replace("<concept>", replace_str)
    zero_prompt = fix_prompt(zero_prompt)

    print(f"Embedding lora prompt: {lora_prompt}")
    print(f"Embedding zero prompt: {zero_prompt}")

    try:  # sdxl:
        embeds = pipe.encode_prompt(
            lora_prompt,
            do_classifier_free_guidance=guidance_scale > 1,
            negative_prompt=negative_prompt,
        )

        zero_embeds = pipe.encode_prompt(
            zero_prompt,
            do_classifier_free_guidance=guidance_scale > 1,
            negative_prompt=negative_prompt,
        )

    except:  # sd15:
        embeds = pipe.encode_prompt(lora_prompt, pipe.device, 1, True, negative_prompt)

        zero_embeds = pipe.encode_prompt(
            zero_prompt, pipe.device, 1, True, negative_prompt
        )

    embeds, token_scale = blend_conditions(
        zero_embeds, embeds, lora_scale, token_scale=token_scale
    )

    return embeds


@torch.no_grad()
def render_images(
    render_size,
    lora_path,
    train_step,
    seed,
    is_lora,
    pretrained_model,
    lora_scale,
    n_steps=25,
    n_imgs=4,
    device="cuda:0",
    pipe = None,
    checkpoint_folder: str = None,
):
    if checkpoint_folder is not None:
        assert pipe is None, f"Expected either one of checkpoint_folder or pipe to be None. But got: checkpoint_folder: {checkpoint_folder} and pipe is not None"
    
    if pipe is not None:
        assert checkpoint_folder is None, f"Expected either one of checkpoint_folder or pipe to be None. But got pipe is NOT None checkpoint_folder is: {checkpoint_folder}"
    
    random.seed(seed)

    with open(os.path.join(lora_path, "training_args.json"), "r") as f:
        training_args = json.load(f)
        concept_mode = training_args["concept_mode"]

    if concept_mode == "style":
        validation_prompts_raw = random.sample(val_prompts["style"], n_imgs)
        validation_prompts_raw[0] = ""

    elif concept_mode == "face":
        validation_prompts_raw = random.sample(val_prompts["face"], n_imgs)
        validation_prompts_raw[0] = "<concept>"
    else:
        validation_prompts_raw = random.sample(val_prompts["object"], n_imgs)
        validation_prompts_raw[0] = "<concept>"

    if (
        checkpoint_folder is not None
    ):  # reload the entire pipeline from disk and load in the lora module
        print(f"Reloading checkpoint from disk: {checkpoint_folder}")
        gc.collect()
        torch.cuda.empty_cache()
        
        pipe = load_checkpoint(
            pretrained_model_version=pretrained_model["version"],
            pretrained_model_path=pretrained_model["path"],
            checkpoint_folder=checkpoint_folder,
            is_lora=is_lora,
            device=device,
        )

    else:
        assert pipe is not None
        training_scheduler = pipe.scheduler
        print(f"Using existing model for inference")
        print(
            f"Re-using training pipeline for inference, just swapping the scheduler.."
        )
        pipe.vae = pipe.vae.to(device).to(pipe.unet.dtype)

    pipe.scheduler = EulerDiscreteScheduler.from_config(
        pipe.scheduler.config, timestep_spacing="trailing"
    )
    generator = torch.Generator(device=device).manual_seed(seed)
    negative_prompt = "nude, naked, poorly drawn face, ugly, tiling, out of frame, extra limbs, disfigured, deformed body, blurry, blurred, watermark, text, grainy, signature, cut off, draft"
    pipeline_args = {
        "num_inference_steps": n_steps,
        "guidance_scale": 8,
        "width": render_size[0],
        "height": render_size[1],
    }

    for i in range(n_imgs):
        print(f"Rendering validation img with prompt: {validation_prompts_raw[i]}")
        c, uc, pc, puc = encode_prompt_advanced(
            pipe,
            lora_path,
            validation_prompts_raw[i],
            negative_prompt,
            lora_scale,
            guidance_scale=8,
            concept_mode=concept_mode,
        )

        pipeline_args["prompt_embeds"] = c
        pipeline_args["negative_prompt_embeds"] = uc
        if pretrained_model["version"] == "sdxl":
            pipeline_args["pooled_prompt_embeds"] = pc
            pipeline_args["negative_pooled_prompt_embeds"] = puc

        image = pipe(**pipeline_args, generator=generator).images[0]
        image.save(
            os.path.join(lora_path, f"img_{train_step:04d}_{i}.jpg"),
            format="JPEG",
            quality=95,
        )

    if checkpoint_folder is None:
        pipe.scheduler = training_scheduler
        pipe.vae = pipe.vae.to("cpu")

    gc.collect()
    torch.cuda.empty_cache()

    return validation_prompts_raw


@torch.no_grad()
def render_images_eval(
    concept_mode: str,
    output_folder: str,
    render_size: tuple,
    checkpoint_folder: str,
    seed: int,
    is_lora: bool,
    pretrained_model: dict,
    trigger_text: str,
    lora_scale=0.7,
    n_steps=25,
    n_imgs=4,
    device="cuda:0",
    verbose: bool = True,
):
    random.seed(seed)
    assert os.path.exists(output_folder), f"Invalid folder: {output_folder}"

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

    pipe = load_checkpoint(
        pretrained_model_version=pretrained_model["version"],
        pretrained_model_path=pretrained_model["path"],
        checkpoint_folder=checkpoint_folder,
        is_lora=is_lora,
        device=device,
    )

    pipe.scheduler = EulerDiscreteScheduler.from_config(
        pipe.scheduler.config, timestep_spacing="trailing"
    )
    generator = torch.Generator(device=device).manual_seed(seed)
    negative_prompt = "nude, naked, poorly drawn face, ugly, tiling, out of frame, extra limbs, disfigured, deformed body, blurry, blurred, watermark, text, grainy, signature, cut off, draft"
    pipeline_args = {
        "num_inference_steps": n_steps,
        "guidance_scale": 8,
        "height": render_size[0],
        "width": render_size[1],
    }

    filenames = []
    for i in range(n_imgs):
        print(f"Rendering validation img with prompt: {validation_prompts_raw[i]}")
        c, uc, pc, puc = encode_prompt_advanced(
            pipe,
            checkpoint_folder,
            validation_prompts_raw[i],
            negative_prompt,
            lora_scale,
            guidance_scale=8,
            concept_mode=concept_mode,
        )

        pipeline_args["prompt_embeds"] = c
        pipeline_args["negative_prompt_embeds"] = uc
        if pretrained_model["version"] == "sdxl":
            pipeline_args["pooled_prompt_embeds"] = pc
            pipeline_args["negative_pooled_prompt_embeds"] = puc

        image = pipe(**pipeline_args, generator=generator).images[0]
        filename = os.path.join(output_folder, f"{i}.jpg")
        image.save(
            filename,
            format="JPEG",
            quality=95,
        )
        filenames.append(filename)

    return filenames, validation_prompts_raw
