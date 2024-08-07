# Set GPU ID to run these jobs on:
GPU_ID="device=3"

cog predict --gpus $GPU_ID \
    -i name="xander_sdxl_cog" \
    -i lora_training_urls="https://storage.googleapis.com/public-assets-xander/A_workbox/lora_training_sets/xander_big.zip" \
    -i concept_mode="face" \
    -i sd_model_version="sdxl" \
    -i max_train_steps="360" \
    -i caption_model="blip" \
    -i debug="False" \
    -i seed="0"