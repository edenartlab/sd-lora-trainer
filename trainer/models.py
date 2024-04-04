import torch
from diffusers import AutoencoderKL, DDPMScheduler, EulerDiscreteScheduler, UNet2DConditionModel, StableDiffusionPipeline, StableDiffusionXLPipeline


############################################################################################################


SDXL_MODEL_CACHE = "./models/juggernaut_v6.safetensors"
SDXL_URL         = "https://edenartlab-lfs.s3.amazonaws.com/models/checkpoints/juggernautXL_v6.safetensors"

#SDXL_MODEL_CACHE = "./models/Juggernaut-XL_v9_RunDiffusionPhoto_v2.safetensors"
#SDXL_URL         = "https://huggingface.co/RunDiffusion/Juggernaut-XL-v9/resolve/main/Juggernaut-XL_v9_RunDiffusionPhoto_v2.safetensors"

#SD15_MODEL_CACHE = "./models/juggernaut_reborn.safetensors"
#SD15_URL         = "https://edenartlab-lfs.s3.amazonaws.com/models/checkpoints/juggernaut_reborn.safetensors"

SD15_MODEL_CACHE = "./models/DreamShaper_6.31_BakedVae.safetensors"
SD15_URL         = "https://huggingface.co/Lykon/DreamShaper/resolve/main/DreamShaper_6.31_BakedVae.safetensors"

#SD15_MODEL_CACHE = "./models/v1-5-pruned.safetensors"
#SD15_URL         = "https://huggingface.co/Lykon/DreamShaper/resolve/main/DreamShaper_6.31_BakedVae.safetensors"


pretrained_models = {
    "sdxl": {"path": SDXL_MODEL_CACHE, "url": SDXL_URL, "version": "sdxl"},
    "sd15": {"path": SD15_MODEL_CACHE, "url": SD15_URL, "version": "sd15"}
}

############################################################################################################



def load_models(pretrained_model, device, weight_dtype = torch.float16, keep_vae_float32 = False):
    if not isinstance(pretrained_model, dict) or 'path' not in pretrained_model or 'version' not in pretrained_model:
        raise ValueError("pretrained_model must be a dict with 'path' and 'version' keys")

    print(f"Loading model weights from {pretrained_model['path']} with dtype: {weight_dtype}...")

    if pretrained_model['version'] == "sd15":
        pipe = StableDiffusionPipeline.from_single_file(
            pretrained_model['path'], torch_dtype=weight_dtype, use_safetensors=True)
    else:
        pipe = StableDiffusionXLPipeline.from_single_file(
            pretrained_model['path'], torch_dtype=weight_dtype, use_safetensors=True)

    pipe = pipe.to(device, dtype=weight_dtype)
    noise_scheduler = DDPMScheduler.from_config(pipe.scheduler.config)

    vae = pipe.vae
    unet = pipe.unet
    tokenizer_one = pipe.tokenizer
    text_encoder_one = pipe.text_encoder

    vae.requires_grad_(False)
    text_encoder_one.requires_grad_(False)

    text_encoder_one.to(device, dtype=weight_dtype)
    unet.to(device, dtype=weight_dtype)
    if keep_vae_float32:
        vae.to(device, dtype=torch.float32)
    else:
        vae.to(device, dtype=weight_dtype)
        if weight_dtype != torch.float32:
            print(f"Warning: VAE will be loaded as {weight_dtype}, this is fine for inference but might not be for training..")

    tokenizer_two = text_encoder_two = None
    if pretrained_model['version'] == "sdxl":
        tokenizer_two = pipe.tokenizer_2
        text_encoder_two = pipe.text_encoder_2
        text_encoder_two.requires_grad_(False)
        text_encoder_two.to(device, dtype=weight_dtype)
        
    return (
        pipe,
        tokenizer_one,
        tokenizer_two,
        noise_scheduler,
        text_encoder_one,
        text_encoder_two,
        vae,
        unet,
    )



def print_trainable_parameters(model, name = ''):
    trainable_params = 0
    all_param = 0
    for _, param in model.named_parameters():
        all_param += param.numel()
        if param.requires_grad:
            trainable_params += param.numel()
    line_delimiter = "#" * 70
    print('\n', line_delimiter)
    print(
        f"Trainable {name} params: {trainable_params/1000000:.1f}M || All params: {all_param/1000000:.1f}M || trainable = {100 * trainable_params / all_param:.2f}%"
    )
    print(line_delimiter, '\n')