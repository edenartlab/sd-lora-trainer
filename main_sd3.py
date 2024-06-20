"""
todos:
[x] - load pretrained:
    - [x] pipe
    - [x] tokenizers
    - [x] unet
    - [x] vae
    - [x] noise scheduler
[x] - initial token embeddings handler with their respective tokenizers
[x] - init new tokens: ["<s0>","<s1>"]
[] - [optional] init lora params for text encoders
[x] - get textual inversion params and it's corresponding optimizer
[x] - either do full finetuning of transformer or init lora params
[] - init PreprocessedDataset object
[] - init OptimizerCollection containing all optimizers
[] - [debug step] visualize a random token embedding
[] - do training. Save checkpoint during and after training
"""
import copy
from transformers import (
    CLIPTokenizer, T5TokenizerFast, PretrainedConfig
)
from diffusers import (
    StableDiffusion3Pipeline,
    FlowMatchEulerDiscreteScheduler,
    SD3Transformer2DModel,
    AutoencoderKL
)

## shared components with the other trainer (sdxl/sd15)
from trainer.embedding_handler import TokenEmbeddingsHandler
from trainer.loss import (
    ConditioningRegularizer
)
from trainer.optimizer import (
    OptimizerCollection, 
    get_optimizer_and_peft_models_text_encoder_lora, 
    get_textual_inversion_optimizer,
    get_unet_lora_parameters,
    get_unet_optimizer
)
from trainer.optimizer import count_trainable_params
from peft import LoraConfig, get_peft_model

def load_sd3_tokenizers():
    # Load the tokenizers
    tokenizer_one = CLIPTokenizer.from_pretrained(
        "stabilityai/stable-diffusion-3-medium-diffusers",
        subfolder="tokenizer",
        revision=None,
    )
    tokenizer_two = CLIPTokenizer.from_pretrained(
        "stabilityai/stable-diffusion-3-medium-diffusers",
        subfolder="tokenizer_2",
        revision=None,
    )
    tokenizer_three = T5TokenizerFast.from_pretrained(
        "stabilityai/stable-diffusion-3-medium-diffusers",
        subfolder="tokenizer_3",
        revision=None,
    )

    return tokenizer_one, tokenizer_two, tokenizer_three

def import_model_class_from_model_name_or_path(
    pretrained_model_name_or_path: str, revision: str, subfolder: str = "text_encoder"
):
    text_encoder_config = PretrainedConfig.from_pretrained(
        pretrained_model_name_or_path, subfolder=subfolder, revision=revision
    )
    model_class = text_encoder_config.architectures[0]
    if model_class == "CLIPTextModelWithProjection":
        from transformers import CLIPTextModelWithProjection

        return CLIPTextModelWithProjection
    elif model_class == "T5EncoderModel":
        from transformers import T5EncoderModel

        return T5EncoderModel
    else:
        raise ValueError(f"{model_class} is not supported.")
    
def load_text_encoders(
    class_one, 
    class_two, 
    class_three, 
    variant = None
):
    """
    "Variant of the model files of the pretrained model identifier from huggingface.co/models, 'e.g.' fp16",
    """
    text_encoder_one = class_one.from_pretrained(
        "stabilityai/stable-diffusion-3-medium-diffusers", subfolder="text_encoder", 
        revision=None, 
        variant=variant
    )
    text_encoder_two = class_two.from_pretrained(
        "stabilityai/stable-diffusion-3-medium-diffusers", subfolder="text_encoder_2", 
        revision=None, 
        variant=variant
    )
    text_encoder_three = class_three.from_pretrained(
        "stabilityai/stable-diffusion-3-medium-diffusers", subfolder="text_encoder_3", 
        revision=None, 
        variant=variant
    )
    return text_encoder_one, text_encoder_two, text_encoder_three

def load_sd3_text_encoders():
    text_encoder_cls_one = import_model_class_from_model_name_or_path(
        "stabilityai/stable-diffusion-3-medium-diffusers", 
        revision = None
    )
    text_encoder_cls_two = import_model_class_from_model_name_or_path(
        "stabilityai/stable-diffusion-3-medium-diffusers", 
        revision = None, 
        subfolder="text_encoder_2"
    )
    text_encoder_cls_three = import_model_class_from_model_name_or_path(
        "stabilityai/stable-diffusion-3-medium-diffusers", 
        revision = None, 
        subfolder="text_encoder_3"
    )
    return load_text_encoders(
        class_one = text_encoder_cls_one, 
        class_two=text_encoder_cls_two, 
        class_three=text_encoder_cls_three
    )

def load_sd3_noise_scheduler():
    noise_scheduler = FlowMatchEulerDiscreteScheduler.from_pretrained(
        "stabilityai/stable-diffusion-3-medium-diffusers", 
        subfolder="scheduler"
    )
    noise_scheduler_copy = copy.deepcopy(noise_scheduler)
    return noise_scheduler_copy

def load_sd3_transformer():
    transformer = SD3Transformer2DModel.from_pretrained(
        "stabilityai/stable-diffusion-3-medium-diffusers", subfolder="transformer", 
        revision=None, 
        variant=None
    )
    return transformer

def load_sd3_vae():
    vae = AutoencoderKL.from_pretrained(
        "stabilityai/stable-diffusion-3-medium-diffusers",
        subfolder="vae",
        revision=None, 
        variant=None
    )
    return vae

def freeze_all_gradients(models: list):
    for model in models:
        model.requires_grad_(False)

def main():
    # 1. Load tokenizers
    tokenizer_one, tokenizer_two, tokenizer_three = load_sd3_tokenizers()
    
    # 2. Load text encoders
    text_encoder_one, text_encoder_two, text_encoder_three = load_sd3_text_encoders()

    pipeline = StableDiffusion3Pipeline.from_pretrained(
        "stabilityai/stable-diffusion-3-medium-diffusers"
    )

    # 3. load noise scheduler
    noise_scheduler = load_sd3_noise_scheduler()

    # 4. extract transformer
    transformer = load_sd3_transformer()

    # 5. load vae
    vae = load_sd3_vae()

    # 6. freeze all grads
    freeze_all_gradients(
        models = [
            transformer,
            vae,
            text_encoder_one,
            text_encoder_two,
            text_encoder_three,
        ]
    )

    text_encoders = [text_encoder_one, text_encoder_two, text_encoder_three]
    # initialize token embedding handler
    # Initialize new tokens for training.
    embedding_handler = TokenEmbeddingsHandler(
        text_encoders = text_encoders, 
        tokenizers = [tokenizer_one, tokenizer_two, tokenizer_three]
    )

    embedding_handler.initialize_new_tokens(
        inserting_toks=["<s0>","<s1>"],
        starting_toks=None, 
        seed=0
    )
    # Experimental TODO: warmup the token embeddings using CLIP-similarity optimization

    from trainer.config import TrainingConfig
    config= TrainingConfig(
        lora_training_urls = "none",
        concept_mode = "object",
        sd_model_version = "sd3",
        training_attributes = {
            "gpt_description": "A banana with a face"
        },
        token_warmup_steps = 0,
        is_lora = True,
        lora_rank = 4
    )

    embedding_handler.make_embeddings_trainable()
    embedding_handler.token_regularizer = ConditioningRegularizer(
        config, 
        embedding_handler
    )

    embedding_handler.pre_optimize_token_embeddings(
        config, 
        pipe = pipeline
    )
    ## TODO: [optional] init lora params for text encoders

    # get textual inversion params and it's optimizer
    optimizer_ti, textual_inversion_params = get_textual_inversion_optimizer(
        text_encoders=text_encoders,
        textual_inversion_lr=config.ti_lr,
        textual_inversion_weight_decay=config.ti_weight_decay,
        optimizer_name=config.ti_optimizer ## hardcoded
    )

    # either go full finetuning or lora on transformer
    if not config.is_lora: # This code pathway has not been tested in a long while
        print(f"Doing full fine-tuning on the U-Net")
        transformer.requires_grad_(True)
        transformer_trainable_params = transformer.parameters()
    else:
        transformer_lora_config = LoraConfig(
            r=config.lora_rank,
            lora_alpha=config.lora_rank,
            init_lora_weights="gaussian",
            target_modules=["to_k", "to_q", "to_v", "to_out.0"],
        )
        transformer = get_peft_model(model = transformer, peft_config = transformer_lora_config)
        transformer_trainable_params = [
            x for x in transformer.parameters() if x.requires_grad
        ]
    
    print(f"config.is_lora: {config.is_lora} params: {count_trainable_params(transformer)}")
    
    

if __name__ == "__main__":
    main()