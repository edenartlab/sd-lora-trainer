from typing import Union, List, Dict
from pydantic import BaseModel
import json
from typing import Literal

class TrainingConfig(BaseModel, extra = "forbid"):
    output_dir: str
    pretrained_model: Dict[str, str]
    name: str = "unnamed"
    lora_training_urls: str
    concept_mode: Literal["face", "style", "object"]
    sd_model_version: Literal["sdxl", "sd15"]
    seed: Union[int, None] = None
    resolution: int = 960
    train_batch_size: int = 4
    num_train_epochs: int = 10_000
    max_train_steps: int = 600
    checkpointing_steps: int
    gradient_accumulation_steps: int = 1
    is_lora: bool = True
    prodigy_d_coef: float = 0.5
    ti_lr: float = 1e-3
    ti_weight_decay: float = 3e-4
    lora_weight_decay: float = 0.002
    l1_penalty: float = 0.1
    lora_param_scaler: float = 0.5
    snr_gamma: float = 5.0
    lora_rank: int = 12
    caption_prefix: str = ""
    caption_model: Literal["gpt4-v", "blip"] = "blip"
    left_right_flip_augmentation: bool = True
    augment_imgs_up_to_n: int = 20
    n_tokens: int = 2
    mask_target_prompts: Union[None, str]
    crop_based_on_salience: bool = True
    use_face_detection_instead: bool = False
    clipseg_temperature: float = 0.7
    verbose: bool = False
    run_name: str = "default_run_name"
    debug: bool = False
    hard_pivot: bool = False
    off_ratio_power: float = False
    num_training_images: int
    trigger_text: str
    segmentation_prompt: str
    training_captions: List[str]
    allow_tf32: bool = True
    mixed_precision: Literal["fp16", "bf16", "fp32"] = "bf16"
    instance_data_dir: str
    inserting_list_tokens: List[str] = ["<s0>"]
    token_dict: dict = {"TOKEN": "<s0>"}
    device: str = "cuda:0"
    scale_lr: bool
    crops_coords_top_left_h: int = 0
    crops_coords_top_left_w: int = 0
    do_cache: bool = True
    unet_learning_rate: float = 1.0
    lr_scheduler: str = "constant"
    lr_warmup_steps: int = 50
    lr_num_cycles: int = 1
    lr_power: float = 1.0
    dataloader_num_workers: int = 0

    def save_as_json(self, file_path: str) -> None:
        with open(file_path, 'w') as f:
            json.dump(self.dict(), f, indent=4)

    @classmethod
    def from_json(cls, file_path: str):
        with open(file_path, 'r') as f:
            config_data = json.load(f)
        return cls(**config_data)