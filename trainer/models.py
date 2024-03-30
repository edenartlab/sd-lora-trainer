SDXL_MODEL_CACHE = "./models/juggernaut_v6.safetensors"
SDXL_URL         = "https://edenartlab-lfs.s3.amazonaws.com/models/checkpoints/juggernautXL_v6.safetensors"

SDXL_MODEL_CACHE = "./models/Juggernaut-XL_v9_RunDiffusionPhoto_v2.safetensors"
SDXL_URL         = "https://huggingface.co/RunDiffusion/Juggernaut-XL-v9/resolve/main/Juggernaut-XL_v9_RunDiffusionPhoto_v2.safetensors"

SD15_MODEL_CACHE = "./models/juggernaut_reborn.safetensors"
SD15_URL         = "https://edenartlab-lfs.s3.amazonaws.com/models/checkpoints/juggernaut_reborn.safetensors"

SD15_MODEL_CACHE = "./models/DreamShaper_6.31_BakedVae.safetensors"
SD15_URL         = "https://huggingface.co/Lykon/DreamShaper/resolve/main/DreamShaper_6.31_BakedVae.safetensors"

pretrained_models = {
    "sdxl": {"path": SDXL_MODEL_CACHE, "url": SDXL_URL, "version": "sdxl"},
    "sd15": {"path": SD15_MODEL_CACHE, "url": SD15_URL, "version": "sd15"}
}