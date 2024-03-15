SDXL_MODEL_CACHE = "./models/juggernaut_v6.safetensors"
SDXL_URL         = "https://edenartlab-lfs.s3.amazonaws.com/models/checkpoints/juggernautXL_v6.safetensors"

SD15_MODEL_CACHE = "./models/juggernaut_reborn.safetensors"
SD15_URL         = "https://edenartlab-lfs.s3.amazonaws.com/models/checkpoints/juggernaut_reborn.safetensors"

pretrained_models = {
    "sdxl": {"path": SDXL_MODEL_CACHE, "url": SDXL_URL, "version": "sdxl"},
    "sd15": {"path": SD15_MODEL_CACHE, "url": SD15_URL, "version": "sd15"}
}