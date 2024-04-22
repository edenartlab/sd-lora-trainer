# Set GPU ID to run these jobs on:
GPU_ID="device=2"

cog predict --gpus $GPU_ID \
    -i name="clipx_sdxl_cog" \
    -i caption_prefix="in the style of TOK, " \
    -i concept_mode="style" \
    -i sd_model_version="sdxl" \
    -i max_train_steps="400" \
    -i checkpointing_steps="500" \
    -i debug="False" \
    -i lora_training_urls="https://storage.googleapis.com/public-assets-xander/A_workbox/lora_training_sets/clipx_tiny.zip" \
    -i seed="0"