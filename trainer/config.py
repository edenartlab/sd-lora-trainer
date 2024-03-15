from typing import Optional, List, Dict, Any
from pydantic import BaseModel, Field
import random
import json
from typing import Literal
import torch

precision_map = {
    "fp16": torch.float16,
    "bf16": torch.bfloat16,
    "fp32": torch.float32
}

class TrainerConfig(BaseModel, extra = "forbid"):
    pretrained_model: Dict[str, str] # should be a dict with keys "path" and "version"
    name: str='unnamed',
    trigger_text: str='a photo of TOK, ',
    instance_data_dir: str = "./dataset/zeke/captions.csv"
    concept_mode: Literal["face", "concept", "object", "style"]
    output_dir: str = "lora_output"
    seed: Optional[int] = Field(default_factory=lambda: random.randint(0, 2**32 - 1))
    resolution: int = 960
    crops_coords_top_left_h: int = 0
    crops_coords_top_left_w: int = 0
    train_batch_size: int = 1
    train_dataset_cache: bool = True
    num_train_epochs: int = 10000
    max_train_steps: Optional[int] = None
    checkpointing_steps: int = 500000
    gradient_accumulation_steps: int = 1
    unet_learning_rate: float = 1.0
    textual_inversion_lr: float = 1e-3
    textual_inversion_weight_decay: float = 3e-4
    prodigy_d_coef: float = 0.5,
    l1_penalty: float = 0.0
    lora_weight_decay: float = 0.005
    scale_lr_based_on_grad_acc: bool = False
    lr_scheduler_name: str = "constant"
    lr_warmup_steps: int = 50
    lr_num_cycles: int = 1
    lr_power: float = 1.0
    snr_gamma: float = 5.0
    dataloader_num_workers: int = 0
    allow_tf32: bool = True
    precision: Literal["bf16", "fp16", "fp32"] = "bf16"
    optimizer_name: Literal["prodigy", "adamw"] = "prodigy"
    device: str = "cuda"
    token_dict: Dict[str, str] = {"TOK": "<s0><s1>"}
    inserting_list_tokens: List[str] = ["<s0><s1>"]
    verbose: bool = True
    is_lora: bool = True
    lora_rank: int = 12
    lora_alpha: int = 12
    args_dict: Dict[str, Any] = {}
    debug: bool = False
    hard_pivot: bool = True
    off_ratio_power: float = 0.1

    def save_as_json(self, file_path: str) -> None:
        with open(file_path, 'w') as f:
            json.dump(self.dict(), f, indent=4)
