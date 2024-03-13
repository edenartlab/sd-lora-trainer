cog predict --gpus device=2\
    -i run_name="clipx_sdxl" \
    -i caption_prefix="in the style of TOK, " \
    -i concept_mode="style" \
    -i train_batch_size="4" \
    -i sd_model_version="sdxl" \
    -i max_train_steps="400" \
    -i checkpointing_steps="100" \
    -i debug="True" \
    -i lora_training_urls="https://storage.googleapis.com/public-assets-xander/A_workbox/lora_training_sets/clipx_tiny.zip" \
    -i seed="0"

cog predict \
    -i run_name="clipx_sd15" \
    -i caption_prefix="in the style of TOK, " \
    -i concept_mode="style" \
    -i train_batch_size="4" \
    -i sd_model_version="sd15" \
    -i max_train_steps="400" \
    -i checkpointing_steps="100" \
    -i debug="True" \
    -i lora_training_urls="https://storage.googleapis.com/public-assets-xander/A_workbox/lora_training_sets/clipx_tiny.zip" \
    -i seed="0"