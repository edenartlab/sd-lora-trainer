import fnmatch
import json
import math
import os
import sys
import random
import time
import shutil
import gc
import numpy as np
from typing import List, Optional

import torch
import torch.utils.checkpoint
import torch.nn.functional as F

from diffusers.models.attention_processor import LoRAAttnProcessor2_0
from peft import LoraConfig, get_peft_model, PeftModel
from diffusers.optimization import get_scheduler
from diffusers import EulerDiscreteScheduler
from safetensors.torch import save_file
from tqdm import tqdm

from dataset_and_utils import (
    PreprocessedDataset,
    TokenEmbeddingsHandler,
    load_models,
    unet_attn_processors_state_dict
)

from io_utils import make_validation_img_grid
from safetensors.torch import load_file
import matplotlib.pyplot as plt

def compute_snr(noise_scheduler, timesteps):
    """
    Computes SNR as per
    https://github.com/TiankaiHang/Min-SNR-Diffusion-Training/blob/521b624bd70c67cee4bdf49225915f5945a872e3/guided_diffusion/gaussian_diffusion.py#L847-L849
    """
    alphas_cumprod = noise_scheduler.alphas_cumprod
    sqrt_alphas_cumprod = alphas_cumprod**0.5
    sqrt_one_minus_alphas_cumprod = (1.0 - alphas_cumprod) ** 0.5

    # Expand the tensors.
    # Adapted from https://github.com/TiankaiHang/Min-SNR-Diffusion-Training/blob/521b624bd70c67cee4bdf49225915f5945a872e3/guided_diffusion/gaussian_diffusion.py#L1026
    sqrt_alphas_cumprod = sqrt_alphas_cumprod.to(device=timesteps.device)[timesteps].float()
    while len(sqrt_alphas_cumprod.shape) < len(timesteps.shape):
        sqrt_alphas_cumprod = sqrt_alphas_cumprod[..., None]
    alpha = sqrt_alphas_cumprod.expand(timesteps.shape)

    sqrt_one_minus_alphas_cumprod = sqrt_one_minus_alphas_cumprod.to(device=timesteps.device)[timesteps].float()
    while len(sqrt_one_minus_alphas_cumprod.shape) < len(timesteps.shape):
        sqrt_one_minus_alphas_cumprod = sqrt_one_minus_alphas_cumprod[..., None]
    sigma = sqrt_one_minus_alphas_cumprod.expand(timesteps.shape)

    # Compute SNR.
    snr = (alpha / sigma) ** 2
    return snr

def plot_torch_hist(parameters, epoch, checkpoint_dir, name, bins=100, min_val=-1, max_val=1, ymax_f = 0.75):
    # Flatten and concatenate all parameters into a single tensor
    all_params = torch.cat([p.data.view(-1) for p in parameters])

    # Convert to CPU for plotting
    all_params_cpu = all_params.cpu().float().numpy()

    # Plot histogram
    plt.figure()
    plt.hist(all_params_cpu, bins=bins, density=False)
    plt.ylim(0, ymax_f * len(all_params_cpu))
    plt.xlim(min_val, max_val)
    plt.xlabel('Weight Value')
    plt.ylabel('Count')
    plt.title(f'Epoch {epoch} {name} Histogram (std = {np.std(all_params_cpu):.4f})')
    plt.savefig(f"{checkpoint_dir}/{name}_histogram_{epoch:04d}.png")
    plt.close()

# plot the learning rates:
def plot_lrs(lora_lrs, ti_lrs, save_path='learning_rates.png'):
    plt.figure()
    plt.plot(range(len(lora_lrs)), lora_lrs, label='LoRA LR')
    plt.plot(range(len(lora_lrs)), ti_lrs, label='TI LR')
    plt.yscale('log')  # Set y-axis to log scale
    plt.ylim(1e-6, 3e-3)
    plt.xlabel('Step')
    plt.ylabel('Learning Rate')
    plt.title('Learning Rate Curves')
    plt.legend()
    plt.savefig(save_path)
    plt.close()

from scipy.signal import savgol_filter
def plot_loss(losses, save_path='losses.png', window_length=31, polyorder=3):
    if len(losses) < window_length:
        return
    
    smoothed_losses = savgol_filter(losses, window_length, polyorder)
    
    plt.figure()
    plt.plot(losses, label='Actual Losses')
    plt.plot(smoothed_losses, label='Smoothed Losses', color='red')
    # plt.yscale('log')  # Uncomment if log scale is desired
    plt.xlabel('Step')
    plt.ylabel('Training Loss')
    plt.legend()
    plt.savefig(save_path)
    plt.close()

def patch_pipe_with_lora(pipe, lora_path):

    pipe.unet = PeftModel.from_pretrained(pipe.unet, lora_path)
    pipe.unet.merge_adapter()
    
    # Load the textual_inversion token embeddings into the pipeline:
    try: #SDXL
        handler = TokenEmbeddingsHandler([pipe.text_encoder, pipe.text_encoder_2], [pipe.tokenizer, pipe.tokenizer_2])
    except: #SD15
        handler = TokenEmbeddingsHandler([pipe.text_encoder, None], [pipe.tokenizer, None])

    embeddings_path = [f for f in os.listdir(lora_path) if f.endswith("embeddings.safetensors")][0]
    handler.load_embeddings(os.path.join(lora_path, embeddings_path))

    return pipe

def get_avg_lr(optimizer):
    # Calculate the weighted average effective learning rate
    total_lr = 0
    total_params = 0
    for group in optimizer.param_groups:
        d = group['d']
        lr = group['lr']
        bias_correction = 1  # Default value
        if group['use_bias_correction']:
            beta1, beta2 = group['betas']
            k = group['k']
            bias_correction = ((1 - beta2**(k+1))**0.5) / (1 - beta1**(k+1))

        effective_lr = d * lr * bias_correction

        # Count the number of parameters in this group
        num_params = sum(p.numel() for p in group['params'] if p.requires_grad)
        total_lr += effective_lr * num_params
        total_params += num_params

    if total_params == 0:
        return 0.0
    else: return total_lr / total_params


import re

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

def print_trainable_parameters(model):
    trainable_params = 0
    all_param = 0
    for _, param in model.named_parameters():
        all_param += param.numel()
        if param.requires_grad:
            trainable_params += param.numel()
    line_delimiter = "#" * 50
    print(line_delimiter)
    print(
        f"Trainable params: {trainable_params/1000000:.1f}M || All params: {all_param/1000000:.1f}M || trainable = {100 * trainable_params / all_param:.2f}%"
    )
    print(line_delimiter)

def prepare_prompt_for_lora(prompt, lora_path, interpolation=False, verbose=True):
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
    
    try:
        lora_name = str(training_args["name"])
    except: # fallback for old loras that dont have the name field:
        return training_args["trigger_text"] + ", " + prompt

    lora_name_encapsulated = "<" + lora_name + ">"
    trigger_text = training_args["trigger_text"]

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

    # Fix common mistakes
    fix_replacements = {
        r",,": ",",
        r"\s\s+": " ",  # Replaces one or more whitespace characters with a single space
        r"\s\.": ".",
        r"\s,": ","
    }
    prompt = replace_in_string(prompt, fix_replacements)

    if verbose:
        print('-------------------------')
        print("Adjusted prompt for LORA:")
        print(orig_prompt)
        print('-- to:')
        print(prompt)
        print('-------------------------')

    return prompt

from val_prompts import val_prompts
@torch.no_grad()
def render_images(training_pipeline, render_size, lora_path, train_step, seed, is_lora, pretrained_model, lora_scale = 0.7, n_steps = 25, n_imgs = 4, debug = False, device = "cuda:0"):

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


    reload_entire_pipeline = True
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

def save(output_dir, global_step, unet, embedding_handler, token_dict, args_dict, seed, is_lora, unet_lora_parameters, unet_param_to_optimize_names):
    """
    Save the LORA model to output_dir, optionally with some example images

    """
    print(f"Saving checkpoint at step.. {global_step}")
    os.makedirs(output_dir, exist_ok=True)

    args_dict["n_training_steps"] = global_step
    args_dict["total_n_imgs_seen"] = global_step * args_dict["train_batch_size"]

    if not is_lora:
        lora_tensors = {
            name: param
            for name, param in unet.named_parameters()
            if name in unet_param_to_optimize_names
        }
        save_file(lora_tensors, f"{output_dir}/unet.safetensors",)
    elif len(unet_lora_parameters) > 0:
        unet.save_pretrained(save_directory = output_dir)

    try:
        concept_name = args_dict["name"].lower()
    except:
        concept_name = "eden_concept_lora"

    # Make sure all weird delimiter characters are removed from concept_name before using it as a filepath:
    concept_name = concept_name.replace(" ", "_").replace("/", "_").replace("\\", "_").replace(":", "_").replace("*", "_").replace("?", "_").replace("\"", "_").replace("<", "_").replace(">", "_").replace("|", "_")

    embedding_handler.save_embeddings(f"{output_dir}/{concept_name}_embeddings.safetensors",)

    with open(f"{output_dir}/special_params.json", "w") as f:
        json.dump(token_dict, f)
    with open(f"{output_dir}/training_args.json", "w") as f:
        json.dump(args_dict, f, indent=4)

def main(
    pretrained_model,
    instance_data_dir: Optional[str] = "./dataset/zeke/captions.csv",
    output_dir: str = "lora_output",
    seed: Optional[int] = random.randint(0, 2**32 - 1),
    resolution: int = 768,
    crops_coords_top_left_h: int = 0,
    crops_coords_top_left_w: int = 0,
    train_batch_size: int = 1,
    do_cache: bool = True,
    num_train_epochs: int = 10000,
    max_train_steps: Optional[int] = None,
    checkpointing_steps: int = 500000,  # default to no checkpoints
    gradient_accumulation_steps: int = 1,  # todo
    unet_learning_rate: float = 1.0,
    ti_lr: float = 3e-4,
    lora_lr: float = 1.0,
    prodigy_d_coef: float = 0.33,
    l1_penalty: float = 0.0,
    lora_weight_decay: float = 0.005,
    ti_weight_decay: float = 0.001,
    scale_lr: bool = False,
    lr_scheduler: str = "constant",
    lr_warmup_steps: int = 50,
    lr_num_cycles: int = 1,
    lr_power: float = 1.0,
    snr_gamma: float = 5.0,
    dataloader_num_workers: int = 0,
    allow_tf32: bool = True,
    mixed_precision: Optional[str] = "bf16",
    device: str = "cuda:0",
    token_dict: dict = {"TOKEN": "<s0>"},
    inserting_list_tokens: List[str] = ["<s0>"],
    verbose: bool = True,
    is_lora: bool = True,
    lora_rank: int = 8,
    args_dict: dict = {},
    debug: bool = False,
    hard_pivot: bool = True,
    off_ratio_power: float = 0.1,
) -> None:
    if allow_tf32:
        torch.backends.cuda.matmul.allow_tf32 = True

    print("Using seed", seed)
    torch.manual_seed(seed)

    weight_dtype = torch.float32
    if mixed_precision == "fp16":
        weight_dtype = torch.float16
    elif mixed_precision == "bf16":
        weight_dtype = torch.bfloat16

    print(f"Loading models with weight_dtype: {weight_dtype}")

    if scale_lr:
        unet_learning_rate = (
            unet_learning_rate * gradient_accumulation_steps * train_batch_size
        )

    (   
        pipe,
        tokenizer_one,
        tokenizer_two,
        noise_scheduler,
        text_encoder_one,
        text_encoder_two,
        vae,
        unet,
    ) = load_models(pretrained_model, device, weight_dtype)

    # Initialize new tokens for training.
    embedding_handler = TokenEmbeddingsHandler(
        [text_encoder_one, text_encoder_two], [tokenizer_one, tokenizer_two]
    )
    
    #starting_toks = ["person", "face"]
    starting_toks = None
    embedding_handler.initialize_new_tokens(inserting_toks=inserting_list_tokens, starting_toks=starting_toks, seed=seed)
    text_encoders = [text_encoder_one, text_encoder_two]

    unet_param_to_optimize = []
    text_encoder_parameters = []
    for text_encoder in text_encoders:
        if text_encoder is not  None:
            for name, param in text_encoder.named_parameters():
                if "token_embedding" in name:
                    param.requires_grad = True
                    text_encoder_parameters.append(param)
                else:
                    param.requires_grad = False

    unet_param_to_optimize_names = []
    unet_lora_parameters = []

    if not is_lora:
        WHITELIST_PATTERNS = [
            # "*.attn*.weight",
            # "*ff*.weight",
            "*"
        ]
        BLACKLIST_PATTERNS = ["*.norm*.weight", "*time*"]
        for name, param in unet.named_parameters():
            if any(
                fnmatch.fnmatch(name, pattern) for pattern in WHITELIST_PATTERNS
            ) and not any(
                fnmatch.fnmatch(name, pattern) for pattern in BLACKLIST_PATTERNS
            ):
                param.requires_grad_(True)
                unet_param_to_optimize_names.append(name)
                print(f"Training: {name}")
            else:
                param.requires_grad_(False)

        # Optimizer creation
        params_to_optimize = [
            {
                "params": text_encoder_parameters,
                "lr": ti_lr,
                "weight_decay": ti_weight_decay,
            },
        ]

        params_to_optimize_prodigy = [
            {
                "params": unet_param_to_optimize,
                "lr": unet_learning_rate,
                "weight_decay": lora_weight_decay,
            },
        ]

    else:
        
        # Do lora-training instead.
        unet.requires_grad_(False)
        # https://huggingface.co/docs/peft/main/en/developer_guides/lora#rank-stabilized-lora
        unet_lora_config = LoraConfig(
            r=lora_rank,
            lora_alpha=lora_rank,
            init_lora_weights="gaussian",
            target_modules=["to_k", "to_q", "to_v", "to_out.0"],
            #use_rslora=True,
            use_dora=True,
        )
        #unet.add_adapter(unet_lora_config)
        
        unet = get_peft_model(unet, unet_lora_config)
        print_trainable_parameters(unet)

        unet_lora_parameters = list(filter(lambda p: p.requires_grad, unet.parameters()))

        params_to_optimize = [
            {
                "params": text_encoder_parameters,
                "lr": ti_lr,
                "weight_decay": ti_weight_decay,
            },
        ]

        params_to_optimize_prodigy = [
            {
                "params": unet_lora_parameters,
                "lr": 1.0,
                "weight_decay": lora_weight_decay,
            },
        ]
    
    optimizer_type = "prodigy" # hardcode for now

    if optimizer_type != "prodigy":
        optimizer = torch.optim.AdamW(
            params_to_optimize,
            weight_decay=0.0, # this wd doesn't matter, I think
        )
    else:        
        try:
            import prodigyopt
        except ImportError:
            raise ImportError("To use Prodigy, please install the prodigyopt library: `pip install prodigyopt`")

        # Note: the specific settings of Prodigy seem to matter A LOT
        optimizer_prod = prodigyopt.Prodigy(
                        params_to_optimize_prodigy,
                        d_coef = prodigy_d_coef,
                        lr=1.0,
                        decouple=True,
                        use_bias_correction=True,
                        safeguard_warmup=True,
                        weight_decay=lora_weight_decay,
                        betas=(0.9, 0.99),
                        growth_rate=1.025,  # this slows down the lr_rampup
                    )
        
        optimizer = torch.optim.AdamW(
            params_to_optimize,
            weight_decay=ti_weight_decay,
        )
        
    train_dataset = PreprocessedDataset(
        instance_data_dir,
        tokenizer_one,
        tokenizer_two,
        vae,
        do_cache=True,
        substitute_caption_map=token_dict,
    )

    print(f"# PTI : Loaded dataset, do_cache: {do_cache}")
    train_dataloader = torch.utils.data.DataLoader(
        train_dataset,
        batch_size=train_batch_size,
        shuffle=True,
        num_workers=dataloader_num_workers,
    )

    num_update_steps_per_epoch = math.ceil(
        len(train_dataloader) / gradient_accumulation_steps
    )
    if max_train_steps is None:
        max_train_steps = num_train_epochs * num_update_steps_per_epoch

    lr_scheduler = get_scheduler(
        lr_scheduler,
        optimizer=optimizer,
        num_warmup_steps=lr_warmup_steps * gradient_accumulation_steps,
        num_training_steps=max_train_steps * gradient_accumulation_steps,
        num_cycles=lr_num_cycles,
        power=lr_power,
    )

    num_update_steps_per_epoch = math.ceil(
        len(train_dataloader) / gradient_accumulation_steps
    )
    num_train_epochs = math.ceil(max_train_steps / num_update_steps_per_epoch)

    total_batch_size = train_batch_size * gradient_accumulation_steps

    if verbose:
        print(f"# PTI :  Running training ")
        print(f"# PTI :  Num examples = {len(train_dataset)}")
        print(f"# PTI :  Num batches each epoch = {len(train_dataloader)}")
        print(f"# PTI :  Num Epochs = {num_train_epochs}")
        print(f"# PTI :  Instantaneous batch size per device = {train_batch_size}")
        print(
            f"  Total train batch size (w. parallel, distributed & accumulation) = {total_batch_size}"
        )
        print(f"# PTI :  Gradient Accumulation steps = {gradient_accumulation_steps}")
        print(f"# PTI :  Total optimization steps = {max_train_steps}")

    global_step = 0
    first_epoch = 0
    last_save_step = 0

    progress_bar = tqdm(range(global_step, max_train_steps), position=0, leave=True)
    checkpoint_dir = os.path.join(str(output_dir), "checkpoints")
    if os.path.exists(checkpoint_dir):
        shutil.rmtree(checkpoint_dir)
    os.makedirs(f"{checkpoint_dir}")

    # Experimental TODO: warmup the token embeddings using CLIP-similarity optimization
    #embedding_handler.pre_optimize_token_embeddings(train_dataset)
    
    ti_lrs, lora_lrs = [], []
    losses = []
    start_time, images_done = time.time(), 0

    for epoch in range(first_epoch, num_train_epochs):
        unet.train()
        progress_bar.set_description(f"# PTI :step: {global_step}, epoch: {epoch}")

        for step, batch in enumerate(train_dataloader):
            progress_bar.update(1)

            if hard_pivot:
                if epoch >= num_train_epochs // 2:
                    if optimizer is not None:
                        print("----------------------")
                        print("# PTI :  Pivot halfway")
                        print("----------------------")
                        # remove text encoder parameters from the optimizer
                        optimizer.param_groups = None
                        # remove the optimizer state corresponding to text_encoder_parameters
                        for param in text_encoder_parameters:
                            if param in optimizer.state:
                                del optimizer.state[param]
                        optimizer = None

            else: # Update learning rates gradually:
                finegrained_epoch = epoch + step / len(train_dataloader)
                completion_f = finegrained_epoch / num_train_epochs
                # param_groups[1] goes from ti_lr to 0.0 over the course of training
                optimizer.param_groups[0]['lr'] = ti_lr * (1 - completion_f) ** 2.0

            
            try: #sdxl
                (tok1, tok2), vae_latent, mask = batch
            except: #sd15
                tok1, vae_latent, mask = batch
                tok2 = None

            vae_latent = vae_latent.to(weight_dtype)

            # tokens to text embeds
            prompt_embeds_list = []
            for tok, text_encoder in zip((tok1, tok2), text_encoders):
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
            original_size = (resolution, resolution)
            target_size   = (resolution, resolution)
            crops_coords_top_left = (crops_coords_top_left_h, crops_coords_top_left_w)
            add_time_ids = list(original_size + crops_coords_top_left + target_size)
            add_time_ids = torch.tensor([add_time_ids])
            add_time_ids = add_time_ids.to(device, dtype=prompt_embeds.dtype).repeat(
                bs_embed, 1
            )

            # Sample noise that we'll add to the latents
            noise = torch.randn_like(vae_latent)
            bsz = vae_latent.shape[0]

            timesteps = torch.randint(
                0,
                noise_scheduler.config.num_train_timesteps,
                (bsz,),
                device=vae_latent.device,
            ).long()

            noisy_model_input = noise_scheduler.add_noise(vae_latent, noise, timesteps)

            noise_sigma = 0.0
            if noise_sigma > 0.0: # experimental: apply random noise to the conditioning vectors as a form of regularization
                prompt_embeds[0,1:-2,:] += torch.randn_like(prompt_embeds[0,1:-2,:]) * noise_sigma

            # Predict the noise residual
            model_pred = unet(
                noisy_model_input,
                timesteps,
                prompt_embeds,
                added_cond_kwargs={"text_embeds": pooled_prompt_embeds, "time_ids": add_time_ids},
            ).sample

            # Get the unet prediction target depending on the prediction type:
            if noise_scheduler.config.prediction_type == "epsilon":
                target = noise
            elif noise_scheduler.config.prediction_type == "v_prediction":
                target = noise_scheduler.get_velocity(model_input, noise, timesteps)
            else:
                raise ValueError(f"Unknown prediction type {noise_scheduler.config.prediction_type}")

            # Compute the loss:
            if snr_gamma is None:
                loss = (model_pred - target).pow(2) * mask

                # modulate loss by the inverse of the mask's mean value
                mean_mask_values = mask.mean(dim=list(range(1, len(loss.shape))))
                mean_mask_values = mean_mask_values / mean_mask_values.mean()
                loss = loss.mean(dim=list(range(1, len(loss.shape)))) / mean_mask_values

                # Average the normalized errors across the batch
                loss = loss.mean()

            else:
                # Compute loss-weights as per Section 3.4 of https://arxiv.org/abs/2303.09556.
                # Since we predict the noise instead of x_0, the original formulation is slightly changed.
                # This is discussed in Section 4.2 of the same paper.
                snr = compute_snr(noise_scheduler, timesteps)
                base_weight = (
                    torch.stack([snr, snr_gamma * torch.ones_like(timesteps)], dim=1).min(dim=1)[0] / snr
                )
                if noise_scheduler.config.prediction_type == "v_prediction":
                    # Velocity objective needs to be floored to an SNR weight of one.
                    mse_loss_weights = base_weight + 1
                else:
                    # Epsilon and sample both use the same loss weights.
                    mse_loss_weights = base_weight

                mse_loss_weights = mse_loss_weights / mse_loss_weights.mean()
                loss = (model_pred - target).pow(2) * mask
                loss = loss.mean(dim=list(range(1, len(loss.shape)))) * mse_loss_weights

                if 1: # modulate loss by the inverse of the mask's mean value
                    mean_mask_values = mask.mean(dim=list(range(1, len(loss.shape))))
                    mean_mask_values = mean_mask_values / mean_mask_values.mean()
                    loss = loss.mean(dim=list(range(1, len(loss.shape)))) / mean_mask_values

                loss = loss.mean()

            if l1_penalty > 0.0:
                # Compute normalized L1 norm (mean of abs sum) of all lora parameters:
                l1_norm = sum(p.abs().sum() for p in unet_lora_parameters) / sum(p.numel() for p in unet_lora_parameters)
                loss += l1_penalty * l1_norm

            losses.append(loss.item())

            loss = loss / gradient_accumulation_steps
            loss.backward()

            '''
            apart from the usual gradient accumulation steps,
            we also do a backward pass after computing the last forward pass in the epoch (last_batch == True)
            this is to make sure that we're not missing out on any data 
            '''
            last_batch = (step + 1 == len(train_dataloader))
            if (step + 1) % gradient_accumulation_steps == 0 or last_batch:
                if optimizer is not None:
                    optimizer.step()
                    optimizer.zero_grad()
                
                optimizer_prod.step()
                optimizer_prod.zero_grad()

                # after every optimizer step, we reset the non-trainable embeddings to the original embeddings
                embedding_handler.retract_embeddings(print_stds = (global_step % 50 == 0))
                embedding_handler.fix_embedding_std(off_ratio_power)
            
            # Track the learning rates for final plotting:
            lora_lrs.append(get_avg_lr(optimizer_prod))
            try:
                ti_lrs.append(optimizer.param_groups[0]['lr'])
            except:
                ti_lrs.append(0.0)

            # Print some statistics:
            if (global_step % checkpointing_steps == 0):
                output_save_dir = f"{checkpoint_dir}/checkpoint-{global_step}"
                save(output_save_dir, global_step, unet, embedding_handler, token_dict, args_dict, seed, is_lora, unet_lora_parameters, unet_param_to_optimize_names)
                last_save_step = global_step

                if debug:
                    validation_prompts = render_images(pipe, target_size, output_save_dir, global_step, seed, is_lora, pretrained_model, n_imgs = 4, debug=debug)
                    gc.collect()
                    torch.cuda.empty_cache()
                    token_embeddings = embedding_handler.get_trainable_embeddings()
                    for i, token_embeddings_i in enumerate(token_embeddings):
                        plot_torch_hist(token_embeddings_i[0], global_step, output_dir, f"embeddings_weights_token_0_{i}", min_val=-0.05, max_val=0.05, ymax_f = 0.05)
                        plot_torch_hist(token_embeddings_i[1], global_step, output_dir, f"embeddings_weights_token_1_{i}", min_val=-0.05, max_val=0.05, ymax_f = 0.05)
                    
                    embedding_handler.print_token_info()
                    plot_torch_hist(unet_lora_parameters, global_step, output_dir, "lora_weights", min_val=-0.3, max_val=0.3, ymax_f = 0.05)
                    plot_loss(losses, save_path=f'{output_dir}/losses.png')
                    plot_lrs(lora_lrs, ti_lrs, save_path=f'{output_dir}/learning_rates.png')
            
            images_done += train_batch_size
            global_step += 1

            if global_step % 100 == 0:
                print(f" ---- avg training fps: {images_done / (time.time() - start_time):.2f}", end="\r")

            if global_step % (max_train_steps//20) == 0:
                progress = (global_step / max_train_steps) + 0.05
                yield np.min((progress, 1.0))


    # final_save
    if (global_step - last_save_step) > 51:
        output_save_dir = f"{checkpoint_dir}/checkpoint-{global_step}"
    else:
        output_save_dir = f"{checkpoint_dir}/checkpoint-{last_save_step}"

    if debug:
        plot_loss(losses, save_path=f'{output_dir}/losses.png')
        plot_lrs(lora_lrs, ti_lrs, save_path=f'{output_dir}/learning_rates.png')
        plot_torch_hist(unet_lora_parameters, global_step, output_dir, "lora_weights", min_val=-0.3, max_val=0.3, ymax_f = 0.05)
        plot_torch_hist(embedding_handler.get_trainable_embeddings(), global_step, output_dir, "embeddings_weights", min_val=-0.05, max_val=0.05, ymax_f = 0.05)      

    if not os.path.exists(output_save_dir):
        save(output_save_dir, global_step, unet, embedding_handler, token_dict, args_dict, seed, is_lora, unet_lora_parameters, unet_param_to_optimize_names)
        validation_prompts = render_images(pipe, target_size, output_save_dir, global_step, seed, is_lora, pretrained_model, n_imgs = 4, n_steps = 35, debug=debug)
    else:
        print(f"Skipping final save, {output_save_dir} already exists")

    del unet
    del vae
    del text_encoder_one
    del text_encoder_two
    del tokenizer_one
    del tokenizer_two
    del embedding_handler
    del pipe
    gc.collect()
    torch.cuda.empty_cache()

    with open(f"{output_save_dir}/training_args.json", "w") as f:
        args_dict["grid_prompts"] = validation_prompts
        json.dump(args_dict, f, indent=4)

    return output_save_dir, validation_prompts


if __name__ == "__main__":
    main()