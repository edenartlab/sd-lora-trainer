from peft import LoraConfig, get_peft_model
import torch

def get_text_encoder_lora_parameters(text_encoder, lora_rank, lora_alpha_multiplier, use_dora: bool):
    text_encoder_lora_config = LoraConfig(
        r=lora_rank,
        lora_alpha=lora_rank * lora_alpha_multiplier,
        init_lora_weights="gaussian",
        target_modules=["k_proj", "q_proj", "v_proj", "out_proj"],
        use_dora=use_dora,
    )
    text_encoder_peft_model = get_peft_model(text_encoder, text_encoder_lora_config)
    text_encoder_lora_params = list(filter(lambda p: p.requires_grad, text_encoder_peft_model.parameters()))

    return text_encoder_peft_model, text_encoder_lora_params

def get_optimizer_and_peft_models_text_encoder_lora(
    text_encoders: list, 
    lora_rank: int, 
    lora_alpha_multiplier: float, 
    use_dora: bool, 
    optimizer_name: str,
    lora_lr: float,
    weight_decay: float
):
    text_encoder_lora_parameters = []
    text_encoder_peft_models = []
    for text_encoder in text_encoders:
        if text_encoder is not None:
            text_encoder_peft_model, text_encoder_lora_params = get_text_encoder_lora_parameters(
                text_encoder=text_encoder,
                lora_rank=lora_rank,
                lora_alpha_multiplier=lora_alpha_multiplier,
                use_dora=use_dora
            )
            text_encoder_lora_parameters.extend(text_encoder_lora_params)
            text_encoder_peft_models.append(text_encoder_peft_model)

    if optimizer_name == "adamw":
        optimizer_text_encoder_lora = torch.optim.AdamW(
                text_encoder_lora_parameters, 
                lr =  lora_lr,
                weight_decay=weight_decay if not use_dora else 0.0
            )
    else:
        raise NotImplementedError(f"Text encoder LoRA finetuning is not yet implemented for optimizer: {optimizer_name}")

    return optimizer_text_encoder_lora, text_encoder_peft_models


class OptimizerCollection:
    def __init__(
        self,
        optimizer_unet = None,
        optimizer_textual_inversion = None,
        optimizer_text_encoder_lora = None
    ):
        """
        run operations on all the relevant optimizers with a single function call
        """
        self.optimizer_unet=optimizer_unet
        self.optimizer_textual_inversion=optimizer_textual_inversion
        self.optimizer_text_encoder_lora=optimizer_text_encoder_lora
        
        self.optimizers = [
            self.optimizer_unet,
            self.optimizer_textual_inversion,
            self.optimizer_text_encoder_lora
        ]

    def zero_grad(self):
        for optimizer in self.optimizers:
            if optimizer is not None:
                optimizer.zero_grad()
    
    def step(self):
        for optimizer in self.optimizers:
            if optimizer is not None:
                optimizer.step()