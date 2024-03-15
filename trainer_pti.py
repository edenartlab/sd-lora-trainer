from trainer import TrainerConfig, Trainer
from preprocess import preprocess
import os
from io_utils import MODEL_DICT

out_root_dir = "./lora_models"
run_name     = "clipx_ti_only"
concept_mode = "style"

output_dir = os.path.join(out_root_dir, run_name)

input_dir, n_imgs, trigger_text, segmentation_prompt, captions = preprocess(
    output_dir,
    concept_mode = concept_mode,
    input_zip_path = "https://storage.googleapis.com/public-assets-xander/A_workbox/lora_training_sets/clipx_tiny.zip" ,
    caption_text="in the style of TOK, ",
    mask_target_prompts=None,
    target_size=960,
    crop_based_on_salience=True,
    use_face_detection_instead=False,
    temp=0.7,
    left_right_flip_augmentation=True,
    augment_imgs_up_to_n = 20,
    seed = 0,
)

print('-------------------------------------------')
print(f"Trigger text: {trigger_text}")
print(f'n_imgs: {n_imgs}')
print(f'concept_mode: {concept_mode}')
print('-------------------------------------------')


config = (TrainerConfig)(
    pretrained_model = MODEL_DICT['sdxl'],
    name='unnamed',
    concept_mode=concept_mode,
    trigger_text=trigger_text,
    instance_data_dir = os.path.join(input_dir, "captions.csv"),
    output_dir = output_dir,
    resolution= 960,
    train_batch_size = 4,
    max_train_steps = 420,
    checkpointing_steps = 140, 
    gradient_accumulation_steps = 1,
    textual_inversion_lr = 1e-3,
    textual_inversion_weight_decay = 3e-4,
    lora_weight_decay = 0.002,
    prodigy_d_coef = 0.5,
    l1_penalty = 0.1,
    snr_gamma = 5.0,
    mixed_precision = "fp32",
    token_dict = {"TOK": "<s0><s1>"},
    inserting_list_tokens = ["<s0>","<s1>"],
    is_lora = True,
    lora_rank = 12,
    lora_alpha = 12,
    hard_pivot = False,
    off_ratio_power = 0.1,
    args_dict = {},
    debug = True,
    seed = 0
)

trainer = Trainer(config)
trainer.train()
print("DONE")