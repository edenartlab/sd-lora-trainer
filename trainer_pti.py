# Bootstrapped from Huggingface diffuser's code.
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

from diffusers.models.attention_processor import LoRAAttnProcessor, LoRAAttnProcessor2_0
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
from diffusers import StableDiffusionPipeline, StableDiffusionXLPipeline
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

def plot_loss(losses, save_path='losses.png'):
    plt.figure()
    plt.plot(losses)
    #plt.yscale('log')  # Set y-axis to log scale
    plt.xlabel('Step')
    plt.ylabel('Training Loss')
    plt.savefig(save_path)
    plt.close()

    

def patch_pipe_with_lora(pipe, lora_path):

    with open(os.path.join(lora_path, "training_args.json"), "r") as f:
        training_args  = json.load(f)
        lora_rank      = training_args["lora_rank"]
        try:
            concept_name = training_args["name"].lower()
        except:
            concept_name = "eden_concept_lora"

        # Make sure all weird delimiter characters are removed from concept_name before using it as a filepath:
        concept_name = concept_name.replace(" ", "_").replace("/", "_").replace("\\", "_").replace(":", "_").replace("*", "_").replace("?", "_").replace("\"", "_").replace("<", "_").replace(">", "_").replace("|", "_")

    unet = pipe.unet
    lora_safetensors_path = os.path.join(lora_path, f"{concept_name}_lora.safetensors")

    if os.path.exists(lora_safetensors_path):
        tensors = load_file(lora_safetensors_path)
        unet_lora_attn_procs = {}
        for name, attn_processor in unet.attn_processors.items():
            cross_attention_dim = (
                None
                if name.endswith("attn1.processor")
                else unet.config.cross_attention_dim
            )
            if name.startswith("mid_block"):
                hidden_size = unet.config.block_out_channels[-1]
            elif name.startswith("up_blocks"):
                block_id = int(name[len("up_blocks.")])
                hidden_size = list(reversed(unet.config.block_out_channels))[
                    block_id
                ]
            elif name.startswith("down_blocks"):
                block_id = int(name[len("down_blocks.")])
                hidden_size = unet.config.block_out_channels[block_id]

            module = LoRAAttnProcessor2_0(
                hidden_size=hidden_size,
                cross_attention_dim=cross_attention_dim,
                rank=lora_rank,
            )
            unet_lora_attn_procs[name] = module.to("cuda")

        unet.set_attn_processor(unet_lora_attn_procs)

    else:
        unet_path = os.path.join(lora_path, "unet.safetensors")
        tensors = load_file(unet_path)

    unet.load_state_dict(tensors, strict=False)
    try: #SDXL
        handler = TokenEmbeddingsHandler([pipe.text_encoder, pipe.text_encoder_2], [pipe.tokenizer, pipe.tokenizer_2])
    except: #SD15
        handler = TokenEmbeddingsHandler([pipe.text_encoder, None], [pipe.tokenizer, None])

    handler.load_embeddings(os.path.join(lora_path, f"{concept_name}_embeddings.safetensors"))
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

    print(f"lora name: {lora_name}")
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
    print(f"lora mode: {mode}")
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


@torch.no_grad()
def render_images(lora_path, train_step, seed, is_lora, pretrained_model, lora_scale = 0.7, n_imgs = 4, debug = False, device = "cuda:0"):

    random.seed(seed)

    with open(os.path.join(lora_path, "training_args.json"), "r") as f:
        training_args = json.load(f)
        concept_mode = training_args["concept_mode"]

    if concept_mode == "style":
        validation_prompts = [
                'a beautiful mountainous landscape, boulders, fresh water stream, setting sun',
                'the stunning skyline of New York City',
                'fruit hanging from a tree, highly detailed texture, soil, rain, drops, photo realistic, surrealism, highly detailed, 8k macrophotography',
                'the Taj Mahal, stunning wallpaper',
                'A majestic tree rooted in circuits, leaves shimmering with data streams, stands as a beacon where the digital dawn caresses the fog-laden, binary soil—a symphony of pixels and chlorophyll.',
                'A beautiful octopus, with swirling tendrils and a pulsating heart of fiery opal hues, hovers ethereally against a starry void, sculpted through a meticulous flame-working technique.',
                'a stunning image of an aston martin sportscar',
                'the streets of new york city, traffic lights, skyline, setting sun',
                'An ethereal, levitating monolith backlit by a supernova sky, casting iridescent light on the ice-spiked Martian terrain. Neo-futurism, Dali surrealism, wide-angle lens, chiaroscuro lighting.',
                'a portrait of a beautiful young woman',
                'a luminous white lotus blossom floats on rippling waters, green petals',
                'the all seeing eye made of golden feathers, surrounded by waterfall, photorealistic, ethereal aesthetics, powerful',
                'A luminescent glass butterfly, wings shimmering elegantly, depicts deftly the fragility yet adamantine spirit of nature. It encapsulates Atari honkaku technique, glowing embers inside capturing the sun as it gracefully clenches lifes sweet unpredictability',
                'Glass Roots: A luminescent glass sculpture of a fully bloomed rose emerges from a broken marble pedestal, natures resilience triumphant amidst the decay. Shadows cast by a dim overhead spotlight. Delicate veins intertwine the transparent petals, illuminating from within, symbolizing fragilitys steely core.',
                'Eternal Arbor Description: A colossal, life-size tapestry hangs majestically in a dimly lit chamber. Its profound serenity contrasts the grand spectacle it unfolds. Hundreds of intricately woven stitches meticulously portray a towering, ancient oak tree, its knotted branches embracing the heavens. ',
                'In the heart of an ancient forest, a massive projection illuminates the darkness. A lone figure, a majestic mythical creature made of shimmering gold, materializes, casting a radiant glow amidst the towering trees. intricate geometric surfaces encasing an expanse of flora and fauna,',
                'The Silent Of Silicon, a digital deer rendered in hyper-realistic 3D, eyes glowing in binary code, comfortably resting amidst rich motherboard-green foliage, accented under crisply fluorescent, simulated LED dawn.',
                'owl made up of geometric shapes, contours of glowing plasma, black background, dramatic, full picture, ultra high res, octane',
                'A twisting creature of reflective dragonglass swirling above a scorched field amidst a large clearing in a dark forest',
                'what do i say to make me exist, oriental mythical beasts, in the golden danish age, in the history of television in the style of light violet and light red, serge najjar, playful and whimsical, associated press photo, afrofuturism-inspired, alasdair mclellan, electronic media',
                'A towering, rusted iron monolith emerges from a desolate cityscape, piercing the horizon with audacious defiance. Amidst contrasting patches of verdant, natures forgotten touch yearns for connection, provoking intense introspection and tumultuous emotions. vibrant splatters of chaotic paint epitom',
                'A humanoid figure with a luminous, translucent body floats in a vast, ethereal digital landscape. Strands of brilliant, iridescent code rain down, intertwining with the figure. a blend of human features and intricate circuitry, hinting at the merging of organic and digital existence',
                'In the heart of a dense digital forest, a majestic, crystalline unicorn rises. Its translucent, pixelated mane seamlessly transitions into the vibrant greens and golds of the surrounding floating circuit board leaves. Soft moonlight filters through the gaps, creating a breathtaking',
                'Silver mushroom with gem spots emerging from water',
                'Binary Love: A heart-shaped composition made up of glowing binary code, symbolizing the merging of human emotion and technology, incredible digital art, cyberpunk, neon colors, glitch effects, 3D octane render, HD',
                'A labyrinthine maze representing the search for answers and understanding, Abstract expressionism, muted color palette, heavy brushstrokes, textured surfaces, somber atmosphere, symbolic elements',
                'A solitary tree standing tall amidst a sea of buildings, Urban nature photography, vibrant colors, juxtaposition of natural elements with urban landscapes, play of light and shadow, storytelling through compositions',
                ]
        validation_prompts = random.sample(validation_prompts, n_imgs)
        validation_prompts[0] = ''

    elif concept_mode == "face":
        validation_prompts = [
                "an intricate wood carving of <concept> in a historic temple",
                '<concept> as pixel art, 8-bit video game style',
                'painting of <concept> by Vincent van Gogh',
                '<concept> as a superhero, wearing a cape',
                '<concept> as a statue made of marble',
                '<concept> as a character in a noir graphic novel, under a rain-soaked streetlamp',
                'stop motion animation of <concept> using clay, Wallace and Gromit style',
                '<concept> portrayed in a famous renaissance painting, replacing Mona Lisas face',
                'a photo of <concept> attending the Oscars, walking down the red carpet with sunglasses',
                '<concept> as a pop vinyl figure, complete with oversized head and small body',
                '<concept> as a retro holographic sticker, shimmering in bright colors',
                '<concept> as a bobblehead on a car dashboard, nodding incessantly',
                "<concept> captured in a snow globe, complete with intricate details",
                "a photo of <concept> climbing mount Everest in the snow, alpinism",
                "<concept> as an action figure superhero, lego toy, toy story",
                'a photo of a massive statue of <concept> in the middle of the city',
                'a masterful oil painting portraying <concept> with vibrant colors, brushstrokes and textures',
                'a vibrant low-poly artwork of <concept>, rendered in SVG, vector graphics',
                '<concept>, polaroid photograph',
                'a huge <concept> sand sculpture on a sunny beach, made of sand',
                '<concept> immortalized as an exquisite marble statue with masterful chiseling, swirling marble patterns and textures',
                ]
        validation_prompts = random.sample(validation_prompts, n_imgs)
        validation_prompts[0] = '<concept>'
    else:
        validation_prompts = [
                "an intricate wood carving of <concept> in a historic temple",
                "<concept> captured in a snow globe, complete with intricate details",
                "<concept> as a retro holographic sticker, shimmering in bright colors",
                "a painting of <concept> by Vincent van Gogh, impressionism, oil painting, vibrant colors, texture",
                "<concept> as an action figure superhero, lego toy, toy story",
                'a photo of a massive statue of <concept> in the middle of the city',
                'a masterful oil painting portraying <concept> with vibrant colors, thick brushstrokes, abstract, surrealism',
                'an intricate origami paper sculpture of <concept>',
                'a vibrant low-poly artwork of <concept>, rendered in SVG, vector graphics',
                'an artistic polaroid photograph of <concept>, vintage',
                '<concept> immortalized as an exquisite marble statue with masterful chiseling, swirling marble patterns and textures',
                'a colorful and dynamic <concept> mural sprawling across the side of a building in a city pulsing with life',
                "<concept> transformed into a stained glass window, casting vibrant colors in the light",
                "A whimsical papier-mâché sculpture of <concept>, bursting with color and whimsy.",
                "<concept> as a futuristic neon sign, glowing vividly in the night, cyberpunk, vaporwave",
                "A detailed graphite pencil sketch of <concept>, showcasing shadows and depth, pencil drawing, grayscale",
                "<concept> reimagined as a detailed mechanical model, complete with moving parts, metal, gears, steampunk",
                "A vibrant pixel art representation of <concept>, classic 8-bit video game",
                "A breathtaking ice sculpture of <concept>, carved with precision and clarity, ice carving, frozen",
        ]
        validation_prompts = random.sample(validation_prompts, n_imgs)
        validation_prompts[0] = '<concept>'

    torch.cuda.empty_cache()
    print(f"Loading inference pipeline from {pretrained_model['path']}...")

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
    pipeline.scheduler = EulerDiscreteScheduler.from_config(pipeline.scheduler.config)

    validation_prompts_raw = validation_prompts
    validation_prompts = [prepare_prompt_for_lora(prompt, lora_path) for prompt in validation_prompts]
    generator = torch.Generator(device=device).manual_seed(0)
    pipeline_args = {
                "negative_prompt": "nude, naked, poorly drawn face, ugly, tiling, out of frame, extra limbs, disfigured, deformed body, blurry, blurred, watermark, text, grainy, signature, cut off, draft", 
                "num_inference_steps": 35,
                "guidance_scale": 7,
                }

    if is_lora > 0:
        cross_attention_kwargs = {"scale": lora_scale}
    else:
        cross_attention_kwargs = None

    #with torch.cuda.amp.autocast():
    for i in range(n_imgs):
        pipeline_args["prompt"] = validation_prompts[i]
        print(f"Rendering validation img with prompt: {validation_prompts[i]}")
        image = pipeline(**pipeline_args, generator=generator, cross_attention_kwargs = cross_attention_kwargs).images[0]
        image.save(os.path.join(lora_path, f"img_{train_step:04d}_{i}.jpg"), format="JPEG", quality=95)

    # create img_grid:
    img_grid_path = make_validation_img_grid(lora_path)

    del pipeline
    gc.collect()
    torch.cuda.empty_cache()

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
        lora_tensors = unet_attn_processors_state_dict(unet)
    else:
        lora_tensors = {}
    try:
        concept_name = args_dict["name"].lower()
    except:
        concept_name = "eden_concept_lora"

    # Make sure all weird delimiter characters are removed from concept_name before using it as a filepath:
    concept_name = concept_name.replace(" ", "_").replace("/", "_").replace("\\", "_").replace(":", "_").replace("*", "_").replace("?", "_").replace("\"", "_").replace("<", "_").replace(">", "_").replace("|", "_")

    save_file(lora_tensors, f"{output_dir}/{concept_name}_lora.safetensors")
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
    ) = load_models(pretrained_model, device, weight_dtype, keep_vae_float32 = True)

    # Initialize new tokens for training.
    embedding_handler = TokenEmbeddingsHandler(
        [text_encoder_one, text_encoder_two], [tokenizer_one, tokenizer_two]
    )
    
    #starting_toks = ["person", "face"]
    starting_toks = None
    embedding_handler.initialize_new_tokens(inserting_toks=inserting_list_tokens, starting_toks=starting_toks, seed=seed)
    
    #if debug:
    #    embedding_handler.plot_token_embeddings(["man", "face", "woman", "foot", "born"], output_folder = output_dir)

    text_encoders = [text_encoder_one, text_encoder_two]

    unet_param_to_optimize = []
    # fine tune only attn weights

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
        ]  # TODO : make this a parameter
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
        unet_lora_attn_procs = {}

        for name, attn_processor in unet.attn_processors.items():
            cross_attention_dim = (
                None
                if name.endswith("attn1.processor")
                else unet.config.cross_attention_dim
            )
            if name.startswith("mid_block"):
                hidden_size = unet.config.block_out_channels[-1]
            elif name.startswith("up_blocks"):
                block_id = int(name[len("up_blocks.")])
                hidden_size = list(reversed(unet.config.block_out_channels))[block_id]
            elif name.startswith("down_blocks"):
                block_id = int(name[len("down_blocks.")])
                hidden_size = unet.config.block_out_channels[block_id]

            module = LoRAAttnProcessor2_0(
                hidden_size=hidden_size,
                cross_attention_dim=cross_attention_dim,
                rank=lora_rank,
            )

            # scale all the parameters inside the lora module by lora_param_scaler
            for param in module.parameters():
                param.data = param.data * args_dict['lora_param_scaler']

            unet_lora_attn_procs[name] = module
            module.to(device)
            unet_lora_parameters.extend(module.parameters())

        unet.set_attn_processor(unet_lora_attn_procs)

        print("Creating optimizer with:")
        print(f"lora_lr: {lora_lr}, lora_weight_decay: {lora_weight_decay}")
        print(f"ti_lr: {ti_lr}, ti_weight_decay: {ti_weight_decay}")

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

        print("Instantiating prodigy optimizer!")
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
        vae.float(),
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

    # Experimental: warmup the token embeddings using CLIP-similarity:
    #embedding_handler.pre_optimize_token_embeddings(train_dataset)
    
    ti_lrs, lora_lrs = [], []
    
    # Count the total number of lora parameters
    total_n_lora_params = sum(p.numel() for p in unet_lora_parameters)

    #output_save_dir = f"{checkpoint_dir}/checkpoint-{global_step}"
    #save(output_save_dir, global_step, unet, embedding_handler, token_dict, args_dict, seed, is_lora, unet_param_to_optimize_names)
    losses = []

    start_time, images_done = time.time(), 0

    for epoch in range(first_epoch, num_train_epochs):
        unet.train()

        for step, batch in enumerate(train_dataloader):

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

            progress_bar.update(1)
            progress_bar.set_description(f"# PTI :step: {global_step}, epoch: {epoch}")
            #progress_bar.refresh()
            
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
            target_size = (resolution, resolution)
            crops_coords_top_left = (crops_coords_top_left_h, crops_coords_top_left_w)
            add_time_ids = list(original_size + crops_coords_top_left + target_size)
            add_time_ids = torch.tensor([add_time_ids])

            add_time_ids = add_time_ids.to(device, dtype=prompt_embeds.dtype).repeat(
                bs_embed, 1
            )

            added_kw = {"text_embeds": pooled_prompt_embeds, "time_ids": add_time_ids}

            # Sample noise that we'll add to the latents
            noise = torch.randn_like(vae_latent)
            bsz = vae_latent.shape[0]

            timesteps = torch.randint(
                0,
                noise_scheduler.config.num_train_timesteps,
                (bsz,),
                device=vae_latent.device,
            )
            timesteps = timesteps.long()

            noisy_model_input = noise_scheduler.add_noise(vae_latent, noise, timesteps)

            noise_sigma = 0.0
            if noise_sigma > 0.0: # apply random noise to the conditioning vectors:
                prompt_embeds[0,1:-2,:] += torch.randn_like(prompt_embeds[0,1:-2,:]) * noise_sigma

            # Predict the noise residual
            model_pred = unet(
                noisy_model_input,
                timesteps,
                prompt_embeds,
                added_cond_kwargs=added_kw,
            ).sample

            # Compute the loss:
            if snr_gamma is None:
                loss = (model_pred - noise).pow(2) * mask

                if 1: # modulate loss by the inverse of the mask's mean value
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
                loss = (model_pred - noise).pow(2) * mask
                loss = loss.mean(dim=list(range(1, len(loss.shape)))) * mse_loss_weights

                if 1: # modulate loss by the inverse of the mask's mean value
                    mean_mask_values = mask.mean(dim=list(range(1, len(loss.shape))))
                    mean_mask_values = mean_mask_values / mean_mask_values.mean()
                    loss = loss.mean(dim=list(range(1, len(loss.shape)))) / mean_mask_values

                loss = loss.mean()

            if l1_penalty > 0.0:
                # Compute normalized L1 norm (mean of abs sum) of all lora parameters:
                l1_norm = sum(p.abs().sum() for p in unet_lora_parameters) / total_n_lora_params
                #print(f"Loss: {loss.item():.6f}, L1-penalty: {(l1_penalty * l1_norm).item():.6f}")
                loss += l1_penalty * l1_norm

            if global_step % 100 == 0 and debug:
                embedding_handler.print_token_info()

            if global_step % (max_train_steps//3) == 0 and debug:
                plot_torch_hist(unet_lora_parameters, global_step, output_dir, "lora_weights", min_val=-0.3, max_val=0.3, ymax_f = 0.05)
                
                token_embeddings = embedding_handler.get_trainable_embeddings()
                for i, token_embeddings_i in enumerate(token_embeddings):
                    plot_torch_hist(token_embeddings_i[0], global_step, output_dir, f"embeddings_weights_token_0_{i}", min_val=-0.05, max_val=0.05, ymax_f = 0.05)
                    plot_torch_hist(token_embeddings_i[1], global_step, output_dir, f"embeddings_weights_token_1_{i}", min_val=-0.05, max_val=0.05, ymax_f = 0.05)
                
                plot_loss(losses, save_path=f'{output_dir}/losses.png')
                plot_lrs(lora_lrs, ti_lrs, save_path=f'{output_dir}/learning_rates.png')

            losses.append(loss.item())
            loss.backward()

            if optimizer is not None:
                optimizer.step()
                optimizer.zero_grad()
            
            optimizer_prod.step()
            optimizer_prod.zero_grad()

            # every step, we reset the non-trainable embeddings to the original embeddings
            embedding_handler.retract_embeddings(print_stds = (global_step % 50 == 0))
            embedding_handler.fix_embedding_std(off_ratio_power)
            
            # Track the learning rates for final plotting:
            try:
                ti_lrs.append(optimizer.param_groups[0]['lr'])
            except:
                ti_lrs.append(0.0)

            lora_lrs.append(get_avg_lr(optimizer_prod))

            # Print some statistics:
            if (global_step % checkpointing_steps == 0) and (global_step > 0):
                output_save_dir = f"{checkpoint_dir}/checkpoint-{global_step}"
                save(output_save_dir, global_step, unet, embedding_handler, token_dict, args_dict, seed, is_lora, unet_lora_parameters, unet_param_to_optimize_names)
                last_save_step = global_step
                if debug:
                    plot_loss(losses, save_path=f'{output_dir}/losses.png')
                    plot_lrs(lora_lrs, ti_lrs, save_path=f'{output_dir}/learning_rates.png')
                    validation_prompts = render_images(output_save_dir, global_step, seed, is_lora, pretrained_model, n_imgs = 4, debug=debug)
            
            images_done += train_batch_size
            global_step += 1

            if global_step % 100 == 0:
                print(f" ---- avg training fps: {images_done / (time.time() - start_time):.2f}", end="\r")

            if global_step % (max_train_steps//20) == 0:
                progress = (global_step / max_train_steps) + 0.05
                yield np.min((progress, 1.0))


    # final_save
    if (global_step - last_save_step) > 101:
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
    else:
        print(f"Skipping final save, {output_save_dir} already exists")

    # clear the model cache and save grid imgs:
    del unet
    del vae
    del text_encoder_one
    del text_encoder_two
    del tokenizer_one
    del tokenizer_two
    del embedding_handler
    gc.collect()
    torch.cuda.empty_cache()

    validation_prompts = render_images(output_save_dir, global_step, seed, is_lora, pretrained_model, n_imgs = 4, debug=debug)
    
    with open(f"{output_save_dir}/training_args.json", "w") as f:
        args_dict["grid_prompts"] = validation_prompts
        json.dump(args_dict, f, indent=4)

    return output_save_dir, validation_prompts


if __name__ == "__main__":
    main()
