# Set GPU ID to run these jobs on:
GPU_ID="device=3"

cog predict --gpus $GPU_ID \
    -i name="xander_sdxl_cog" \
    -i lora_training_urls="https://storage.googleapis.com/public-assets-xander/A_workbox/lora_training_sets/clipx_tiny.zip" \
    -i concept_mode="style" \
    -i sd_model_version="sdxl" \
    -i max_train_steps="300" \
    -i n_sample_imgs="8" \
    -i debug="True" \
    -i seed="0"