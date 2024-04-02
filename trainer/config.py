from typing import Union, List, Dict
from pydantic import BaseModel
import json
from typing import Literal
# Parse the model:
from trainer.models import pretrained_models

class TrainingConfig(BaseModel):
    output_dir: str
    name: str = "unnamed"
    lora_training_urls: str
    concept_mode: Literal["face", "style", "object"]
    sd_model_version: Literal["sdxl", "sd15"]
    pretrained_model: dict = None
    seed: Union[int, None] = None
    resolution: int = 512
    validation_img_size: List[int] = [1024, 1024]
    train_batch_size: int = 4
    num_train_epochs: int = 10000
    max_train_steps: int = 600
    checkpointing_steps: int = 10000
    gradient_accumulation_steps: int = 1
    is_lora: bool = True
    clip_grad_norm: float = -1
    prodigy_d_coef: float = 0.5
    ti_lr: float = 1e-3
    ti_weight_decay: float = 3e-4
    lora_weight_decay: float = 0.002
    l1_penalty: float = 0.1
    noise_offset: float = 0.05
    snr_gamma: float = 5.0
    lora_rank: int = 12
    use_dora: bool = False
    caption_prefix: str = ""
    caption_model: Literal["gpt4-v", "blip"] = "blip"
    left_right_flip_augmentation: bool = True
    augment_imgs_up_to_n: int = 20
    mask_target_prompts: Union[None, str] = None
    crop_based_on_salience: bool = True
    use_face_detection_instead: bool = False
    clipseg_temperature: float = 0.7
    n_sample_imgs: int = 4
    verbose: bool = False
    run_name: str = "default_run_name"
    debug: bool = False
    hard_pivot: bool = False
    off_ratio_power: float = False
    allow_tf32: bool = True
    mixed_precision: Literal["fp16", "bf16", "fp32"] = "bf16"
    n_tokens: int = 2
    inserting_list_tokens: List[str] = ["<s0>","<s1>"]
    token_dict: dict = {"TOK": "<s0><s1>"}
    device: str = "cuda:0"
    crops_coords_top_left_h: int = 0
    crops_coords_top_left_w: int = 0
    do_cache: bool = True
    unet_learning_rate: float = 1.0
    lr_scheduler: str = "constant"
    lr_warmup_steps: int = 50
    lr_num_cycles: int = 1
    lr_power: float = 1.0
    dataloader_num_workers: int = 0
    training_attributes: dict = {}
    aspect_ratio_bucketing: bool = False

    def save_as_json(self, file_path: str) -> None:
        with open(file_path, 'w') as f:
            json.dump(self.dict(), f, indent=4)

    @classmethod
    def from_json(cls, file_path: str):
        with open(file_path, 'r') as f:
            config_data = json.load(f)

        # Parse the model:
        from trainer.models import pretrained_models
        config_data["pretrained_model"] = pretrained_models[config_data["sd_model_version"]]

        # add some metrics to the foldername:
        lora_str = "dora" if config_data["use_dora"] else "lora"
        config_data["output_dir"] = config_data["output_dir"] + f"---{config_data['sd_model_version']}_{config_data['concept_mode']}_{lora_str}"

        return cls(**config_data)