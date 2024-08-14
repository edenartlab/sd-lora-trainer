import os
import time
import subprocess
import torch
from diffusers import AutoencoderKL, DDPMScheduler, EulerDiscreteScheduler, UNet2DConditionModel, StableDiffusionPipeline, StableDiffusionXLPipeline

def load_models(pretrained_model, device, weight_dtype = torch.float16):
    # check if the model is already downloaded:
    if not os.path.exists(pretrained_model['path']):
        download_weights(pretrained_model['url'], pretrained_model['path'])

    tokenizer_two, text_encoder_two = None, None
    print(f"Loading model weights from {os.path.abspath(pretrained_model['path'])} with dtype: {weight_dtype}...")
    
    try:
        print("Loading as SDXL model...")
        pipe = StableDiffusionXLPipeline.from_single_file(
            pretrained_model['path'], torch_dtype=weight_dtype, use_safetensors=True)
        sd_model_version = "sdxl"
        tokenizer_two = pipe.tokenizer_2
        text_encoder_two = pipe.text_encoder_2
        text_encoder_two.requires_grad_(False)
        text_encoder_two.to(device, dtype=weight_dtype)
    except:
        print("Loading as SD15 model...")
        pipe = StableDiffusionPipeline.from_single_file(
            pretrained_model['path'], torch_dtype=weight_dtype, use_safetensors=True)
        sd_model_version = "sd15"

    print(f"Loaded {sd_model_version} model!")
    pipe = pipe.to(device, dtype=weight_dtype)
    noise_scheduler = DDPMScheduler.from_config(pipe.scheduler.config)

    vae = pipe.vae
    unet = pipe.unet
    tokenizer_one = pipe.tokenizer
    text_encoder_one = pipe.text_encoder

    vae.requires_grad_(False)
    vae.to(device, dtype=weight_dtype)
    unet.to(device, dtype=weight_dtype)
    text_encoder_one.requires_grad_(False)
    text_encoder_one.to(device, dtype=weight_dtype)

    return (
        pipe,
        tokenizer_one,
        tokenizer_two,
        noise_scheduler,
        text_encoder_one,
        text_encoder_two,
        vae,
        unet,
    ), sd_model_version

def download_weights(url, dest):
    start = time.time()
    print("downloading url: ", url)
    print("downloading to: ", dest, '...')

    # Make sure the destination directory exists
    dest_dir = os.path.dirname(dest)
    if not os.path.exists(dest_dir):
        os.makedirs(dest_dir)

    try:
        subprocess.check_call(["wget", "-q", "-O", dest, url])
    except subprocess.CalledProcessError as e:
        print("Error occurred while downloading:")
        print("Exit status:", e.returncode)
        print("Output:", e.output)
    except Exception as e:
        print("An unexpected error occurred:", e)

    print(f"Downloading {url} took {time.time() - start} seconds")


def print_trainable_parameters(model, model_name = ''):
    trainable_params = 0
    all_param = 0
    for name, param in model.named_parameters():
        all_param += param.numel()
        if param.requires_grad and "token_embedding" not in name:
            trainable_params += param.numel()
    line_delimiter = "#" * 80
    print(line_delimiter)
    print(
        f"Trainable {model_name} params: {trainable_params/1000000:.1f}M || All params: {all_param/1000000:.1f}M || trainable = {100 * trainable_params / all_param:.2f}%"
    )
    print(line_delimiter)