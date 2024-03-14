from trainer import TrainerConfig, Trainer
from preprocess import preprocess
import os

out_root_dir = "./lora_models"
run_name = "test"
output_dir = os.path.join(out_root_dir, run_name)

input_dir, n_imgs, trigger_text, segmentation_prompt, captions = preprocess(
    output_dir,
    concept_mode = "style",
    input_zip_path="https://storage.googleapis.com/public-assets-xander/A_workbox/lora_training_sets/clipx_tiny.zip" ,
    caption_text="in the style of TOK, ",
    mask_target_prompts=None,
    target_size=860,
    crop_based_on_salience=True,
    use_face_detection_instead=False,
    temp=0.7,
    left_right_flip_augmentation=True,
    augment_imgs_up_to_n = 20,
    seed = 1,
)


config = TrainerConfig(
    pretrained_model = {
        "path": "models/juggernaut_v6.safetensors",
        "version": "sdxl"
    },
    optimizer_name = "prodigy",
    instance_data_dir = os.path.join(input_dir, "captions.csv"),
    output_dir = "lora_models/clip_sdxl",
    seed= 0,
    resolution= 768,
    crops_coords_top_left_h = 0,
    crops_coords_top_left_w = 0,
    train_batch_size = 1,
    do_cache = True,
    num_train_epochs = 10000,
    max_train_steps = 400,
    checkpointing_steps = 20, # default to no checkpoints
    gradient_accumulation_steps = 1, # todo
    unet_learning_rate = 1.0,
    ti_lr = 3e-4,
    lora_lr = 1.0,
    prodigy_d_coef = 0.33,
    l1_penalty = 0.0,
    lora_weight_decay = 0.005,
    ti_weight_decay = 0.001,
    scale_lr = False,
    lr_scheduler = "constant",
    lr_warmup_steps = 50,
    lr_num_cycles = 1,
    lr_power = 1.0,
    snr_gamma = 5.0,
    dataloader_num_workers = 0,
    allow_tf32 = True,
    mixed_precision = "bf16",
    device = "cuda:0",
    token_dict = {"TOKEN": "<s0>"},
    inserting_list_tokens = ["<s0>"],
    verbose = True,
    is_lora = True,
    lora_rank = 8,
    args_dict = {},
    debug = False,
    hard_pivot = True,
    off_ratio_power = 0.1,
    concept_mode="style"
)

trainer = Trainer(
    config=config
)
trainer.train()
print("DONE")