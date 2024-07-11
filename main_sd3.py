import os
import math
import copy
from transformers import (
    CLIPTokenizer, T5TokenizerFast, PretrainedConfig
)
from diffusers import (
    StableDiffusion3Pipeline,
    FlowMatchEulerDiscreteScheduler,
)

## shared components with the other trainer (sdxl/sd15)
from trainer.embedding_handler import TokenEmbeddingsHandler
from trainer.loss import (
    ConditioningRegularizer
)
from trainer.optimizer import count_trainable_params
from peft import LoraConfig, get_peft_model
from typing import Iterable
import prodigyopt
import torch
from trainer.preprocess import preprocess
from trainer.dataset import PreprocessedDataset
import argparse
from trainer.config import TrainingConfig
from tqdm import tqdm
import wandb
from peft.utils import get_peft_model_state_dict
import bitsandbytes as bnb

TRAIN_TRANSFORMER = True
TRAIN_TEXTUAL_INVERSION = False

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

def load_sd3_noise_scheduler():
    noise_scheduler = FlowMatchEulerDiscreteScheduler.from_pretrained(
        "stabilityai/stable-diffusion-3-medium-diffusers", 
        subfolder="scheduler"
    )
    noise_scheduler_copy = copy.deepcopy(noise_scheduler)
    return noise_scheduler_copy

def freeze_all_gradients(models: list):
    for model in models:
        model.requires_grad_(False)

def get_transformer_optimizer(
    prodigy_d_coef: float,
    prodigy_growth_factor: float,
    lora_weight_decay: float,
    use_dora: bool,
    transformer_trainable_params: Iterable,
    optimizer_name="prodigy",
    lr = 1e-4
):
    if optimizer_name == "adamw":
        optimizer = torch.optim.AdamW(transformer_trainable_params, lr = lr, weight_decay=lora_weight_decay if not use_dora else 0.0)
    
    elif optimizer_name == "adamw_8bit":
         optimizer = bnb.optim.AdamW8bit(transformer_trainable_params, lr = lr, weight_decay=lora_weight_decay)
    elif optimizer_name == "prodigy":
        # Note: the specific settings of Prodigy seem to matter A LOT
        optimizer = prodigyopt.Prodigy(
            transformer_trainable_params,
            d_coef = prodigy_d_coef,
            lr=1.0, ## the lr arg is ignored for the prodigy optimizer
            decouple=True,
            use_bias_correction=True,
            safeguard_warmup=True,
            weight_decay=lora_weight_decay if not use_dora else 0.0,
            betas=(0.9, 0.99),
            growth_rate=prodigy_growth_factor  # lower values make the lr go up slower (1.01 is for 1k step runs, 1.02 is for 500 step runs)
        )
    else:
        raise NotImplementedError(f"Invalid optimizer_name for unet: {optimizer_name}")
    
    print(f"Created {optimizer_name} optimizer for transformer!")
    return optimizer

def get_sigmas(timesteps, noise_scheduler, device, n_dim=4, dtype=torch.float32):
    sigmas = noise_scheduler.sigmas.to(device=device, dtype=dtype)
    schedule_timesteps = noise_scheduler.timesteps.to(device)
    timesteps = timesteps.to(device)
    step_indices = [(schedule_timesteps == t).nonzero().item() for t in timesteps]

    sigma = sigmas[step_indices].flatten()
    while len(sigma.shape) < n_dim:
        sigma = sigma.unsqueeze(-1)
    return sigma

def _encode_prompt_with_t5(
    text_encoder,
    tokenizer,
    prompt=None,
    num_images_per_prompt=1,
    device=None,
):
    prompt = [prompt] if isinstance(prompt, str) else prompt
    batch_size = len(prompt)

    text_inputs = tokenizer(
        prompt,
        padding="max_length",
        max_length=77,
        truncation=True,
        add_special_tokens=True,
        return_tensors="pt",
    )
    text_input_ids = text_inputs.input_ids
    prompt_embeds = text_encoder(text_input_ids.to(device))[0]

    dtype = text_encoder.dtype
    prompt_embeds = prompt_embeds.to(dtype=dtype, device=device)

    _, seq_len, _ = prompt_embeds.shape

    # duplicate text embeddings and attention mask for each generation per prompt, using mps friendly method
    prompt_embeds = prompt_embeds.repeat(1, num_images_per_prompt, 1)
    prompt_embeds = prompt_embeds.view(batch_size * num_images_per_prompt, seq_len, -1)

    return prompt_embeds


def _encode_prompt_with_clip(
    text_encoder,
    tokenizer,
    prompt: str,
    device=None,
    num_images_per_prompt: int = 1,
):
    prompt = [prompt] if isinstance(prompt, str) else prompt
    batch_size = len(prompt)

    text_inputs = tokenizer(
        prompt,
        padding="max_length",
        max_length=77,
        truncation=True,
        return_tensors="pt",
    )

    text_input_ids = text_inputs.input_ids
    prompt_embeds = text_encoder(text_input_ids.to(device), output_hidden_states=True)

    pooled_prompt_embeds = prompt_embeds[0]
    prompt_embeds = prompt_embeds.hidden_states[-2]
    prompt_embeds = prompt_embeds.to(dtype=text_encoder.dtype, device=device)

    _, seq_len, _ = prompt_embeds.shape
    # duplicate text embeddings for each generation per prompt, using mps friendly method
    prompt_embeds = prompt_embeds.repeat(1, num_images_per_prompt, 1)
    prompt_embeds = prompt_embeds.view(batch_size * num_images_per_prompt, seq_len, -1)

    return prompt_embeds, pooled_prompt_embeds

def encode_prompt(
    text_encoders,
    tokenizers,
    prompt: str,
    device=None,
    num_images_per_prompt: int = 1,
    textual_inversion_prompt_embeds = None,
    textual_inversion_prompt_embeds_2 =None
):
    prompt = [prompt] if isinstance(prompt, str) else prompt

    clip_tokenizers = tokenizers[:2]
    clip_text_encoders = text_encoders[:2]

    clip_prompt_embeds_list = []
    clip_pooled_prompt_embeds_list = []
    for tokenizer, text_encoder in zip(clip_tokenizers, clip_text_encoders):
        prompt_embeds, pooled_prompt_embeds = _encode_prompt_with_clip(
            text_encoder=text_encoder,
            tokenizer=tokenizer,
            prompt=prompt,
            device=device if device is not None else text_encoder.device,
            num_images_per_prompt=num_images_per_prompt,
        )
        if (textual_inversion_prompt_embeds is not None) and (textual_inversion_prompt_embeds_2 is not None):
            pass
        else:
            clip_prompt_embeds_list.append(prompt_embeds)
        clip_pooled_prompt_embeds_list.append(pooled_prompt_embeds)
    
    if (textual_inversion_prompt_embeds is not None) and (textual_inversion_prompt_embeds_2 is not None):
        clip_prompt_embeds_list = [
            textual_inversion_prompt_embeds,
            textual_inversion_prompt_embeds_2
        ]
    clip_prompt_embeds = torch.cat(clip_prompt_embeds_list, dim=-1)
    pooled_prompt_embeds = torch.cat(clip_pooled_prompt_embeds_list, dim=-1)

    with torch.no_grad():
        t5_prompt_embed = _encode_prompt_with_t5(
            text_encoders[-1],
            tokenizers[-1],
            prompt=prompt,
            num_images_per_prompt=num_images_per_prompt,
            device=device if device is not None else text_encoders[-1].device,
        )
    clip_prompt_embeds = torch.nn.functional.pad(
        clip_prompt_embeds, (0, t5_prompt_embed.shape[-1] - clip_prompt_embeds.shape[-1])
    )
    prompt_embeds = torch.cat([clip_prompt_embeds, t5_prompt_embed], dim=-2)

    return prompt_embeds, pooled_prompt_embeds

def compute_text_embeddings(prompt, text_encoders, tokenizers, device, textual_inversion_prompt_embeds = None, textual_inversion_prompt_embeds_2 = None):
    # with torch.no_grad():
    prompt_embeds, pooled_prompt_embeds = encode_prompt(text_encoders, tokenizers, prompt, textual_inversion_prompt_embeds=textual_inversion_prompt_embeds, textual_inversion_prompt_embeds_2=textual_inversion_prompt_embeds_2)
    prompt_embeds = prompt_embeds.to(device)
    pooled_prompt_embeds = pooled_prompt_embeds.to(device)
    return prompt_embeds, pooled_prompt_embeds

def compute_gradient_norms(trainable_params: list):
    gradient_norms = []
    for param in trainable_params:
        if param.grad is not None:
            gradient_norm = param.grad.norm().item()
            gradient_norms.append(
                gradient_norm
            )
    assert len(gradient_norms)> 0
    return gradient_norms

def save_transformer_lora_checkpoint(transformer, folder):
    os.system(
        f"mkdir -p {folder}"
    )
    transformer_lora_layers_to_save = get_peft_model_state_dict(transformer)
    StableDiffusion3Pipeline.save_lora_weights(
        folder, 
        transformer_lora_layers=transformer_lora_layers_to_save
    )
    print(f"Saved transformer lora checkpoint here:{folder}")

class AllOptimizers:
    def __init__(
        self,
        optimizer_dict
    ):  
        assert isinstance(optimizer_dict, dict)
        self.optimizers = optimizer_dict
    
    def zero_grad(self):
        for key in self.optimizers.keys():
            if self.optimizers[key] is not None:
                self.optimizers[key].zero_grad()

    def step(self):
        for key in self.optimizers.keys():
            if self.optimizers[key] is not None:
                self.optimizers[key].step()

def find_surrounding_text(input_string, trigger_text):
    start_index = input_string.find(trigger_text)
    if start_index == -1:
        return None, None

    end_index = start_index + len(trigger_text)

    text_before = input_string[:start_index] if start_index > 0 else None
    text_after = input_string[end_index:] if end_index < len(input_string) else None

    return text_before, text_after

class TextualInversion:
    def __init__(
        self,
        embedding_module: callable,
        tokenizer: callable,
        trigger_text: str = "TOK",
        num_tokens: int = 2,
        device = "cuda:0",
        embedding_size: int = None,
        initial_embed_string = None
    ):
        self.embedding_module = embedding_module
        self.tokenizer = tokenizer
        self.trigger_text = trigger_text
        self.num_tokens = num_tokens
        self.device = device
        self.embedding_size = embedding_size
        assert self.num_tokens>0
        if embedding_size is not None:
            assert embedding_size>0
            self.embedding_size = embedding_size
        else:
            foo = self.tokenize_and_embed(
                text = "hello world"
            )
            self.embedding_size = foo.shape[-1]
            print(f"Auto-determined embedding_size to be: {self.embedding_size}")

        if initial_embed_string is None:
            embed_tensor = torch.randn(self.num_tokens, self.embedding_size).to(self.device)
        else:
            """
            The starting point of the TI token is the embedding corresponding to initial_embed_string
            """
            # [0,1:-1,:] -> 0 means: remove the first (batch) dim, 1:-1 means skip the START and END tokens
            with torch.no_grad():
                embed_tensor = self.tokenize_and_embed(text = initial_embed_string, padded = False)[0,1:-1,:]

            assert embed_tensor.shape[0] == num_tokens, f"Expected embed_tensor to have {num_tokens} tokens but got: {embed_tensor.shape[0]}. Try changing the num_tokens arg to {embed_tensor.shape[0]} to ignore this error."
        embed_tensor.requires_grad = True
        self.params = torch.nn.Parameter(embed_tensor)

    def tokenize_and_embed(self, text: str, padded = False):

        if padded:
            token_ids = self.tokenizer.encode(
                text, 
                return_tensors = "pt",
                padding="max_length",
                max_length=77,
                truncation=True,
            ).to(self.device)
        else:
            token_ids = self.tokenizer.encode(
                text, 
                return_tensors = "pt",
            ).to(self.device)
        return self.embedding_module(token_ids)

    def compute_text_embeddings(self, text: str, padded_length = None):

        if self.trigger_text in text:
            text_before, text_after = find_surrounding_text(
                input_string=text, 
                trigger_text=self.trigger_text
            )

            all_embeddings = []

            if text_before is not None:
                # [:, :-1, :] means skip the END token
                all_embeddings.append(self.tokenize_and_embed(text_before, padded = False)[:, :-1, :].to(self.params.device))
            
            all_embeddings.append(self.params.unsqueeze(0))

            if text_after is not None:
                # [:, 1:, :] means skip the START token
                all_embeddings.append(self.tokenize_and_embed(text_after, padded = False)[:, 1:, :].to(self.params.device))
            all_embeddings = torch.cat(all_embeddings, dim = 1)

            if padded_length is None:
                return all_embeddings
            else:
                current_length = all_embeddings.shape[1]
                if current_length < padded_length:
                    padding_embeds = torch.cat(
                        [
                            all_embeddings[:,-1:,:]
                            for i in range(padded_length-current_length)
                        ],
                        dim = 1
                    )
                    padded_embeddings = torch.cat(
                        [
                            all_embeddings,
                            padding_embeds
                        ],
                        dim = 1
                    )
                    assert padded_embeddings.shape[1] == padded_length
                    return padded_embeddings
        else:
            return self.tokenize_and_embed(text)

def get_textual_inversion_prompt_embeds(
    textual_inversion: TextualInversion,
    textual_inversion_2: TextualInversion,
    prompts: list,
    text_encoders: list,
    tokenizers: list,
    device: str
):
    textual_inversion_prompt_embeds_list =  [
        textual_inversion.compute_text_embeddings(
            text = prompt, 
            padded_length=77
        )
        for prompt in prompts
    ]
    textual_inversion_prompt_embeds = torch.cat(
        textual_inversion_prompt_embeds_list,
        dim = 0
    )
    textual_inversion_prompt_embeds_list_2 = [
        textual_inversion_2.compute_text_embeddings(
            text = prompt,
            padded_length = 77
        )
        for prompt in prompts
    ]
    textual_inversion_prompt_embeds_2 = torch.cat(
        textual_inversion_prompt_embeds_list_2,
        dim = 0
    )

    prompt_embeds, pooled_prompt_embeds = compute_text_embeddings(
        prompt = prompts, 
        text_encoders = text_encoders, 
        tokenizers = tokenizers,
        device=device,
        textual_inversion_prompt_embeds=textual_inversion_prompt_embeds,
        textual_inversion_prompt_embeds_2=textual_inversion_prompt_embeds_2
    )
    prompt_embeds = prompt_embeds.to(dtype=pooled_prompt_embeds.dtype)
    return prompt_embeds, pooled_prompt_embeds


def main(config: TrainingConfig, wandb_log = False, output_dir = None):
    TRAIN_TEXTUAL_INVERSION =  True if config.ti_lr != None else False
    
    device = "cuda:0"
    inference_device = "cuda:1"
    # 1. Load tokenizers
    tokenizer_one, tokenizer_two, tokenizer_three = load_sd3_tokenizers()
    tokenizers = [tokenizer_one, tokenizer_two, tokenizer_three]

    pipeline = StableDiffusion3Pipeline.from_pretrained(
        "stabilityai/stable-diffusion-3-medium-diffusers",
        torch_dtype=torch.bfloat16
    )

    text_encoder_one, text_encoder_two, text_encoder_three = pipeline.text_encoder.to(device), pipeline.text_encoder_2.to(device), pipeline.text_encoder_3.to(device)

    if TRAIN_TEXTUAL_INVERSION:
        """
        Revamping textual inversion
        """
        textual_inversion = TextualInversion(
            embedding_module=pipeline.text_encoder.text_model.embeddings,
            tokenizer=pipeline.tokenizer,
            trigger_text = "<s0><s1>, ",
            num_tokens = 2,
            initial_embed_string = "Belgian Man"
        )
        textual_inversion_2 = TextualInversion(
            embedding_module=pipeline.text_encoder_2.text_model.embeddings,
            tokenizer=pipeline.tokenizer_2,
            trigger_text = "<s0><s1>, ",
            num_tokens = 2,
            initial_embed_string = "Belgian Man"
        )
        textual_inversion_params = [textual_inversion.params, textual_inversion_2.params]
        optimizer_textual_inversion = torch.optim.SGD(
            textual_inversion_params,
            lr = 1e-4,
            weight_decay = 0.0
        )

    # 3. load noise scheduler
    noise_scheduler = load_sd3_noise_scheduler()

    # 4. extract transformer
    # transformer = load_sd3_transformer().to(device)
    transformer = pipeline.transformer.to(device)

    # 5. load vae
    vae = pipeline.vae.to(device)

    # 6. freeze all grads
    freeze_all_gradients(
        models = [
            transformer,
            vae,
            # text_encoder_one,
            # text_encoder_two,
            text_encoder_three,
        ]
    )

    text_encoders = [text_encoder_one, text_encoder_two, text_encoder_three]

    """
    override some config params because we're recycling an sdxl config here
    """
    config.ti_lr_warmup_steps = 200
    # config.is_lora = True
    config.token_warmup_steps = 0
    # config.lora_rank = 8
    config.sd_model_version = "sd3"
    # change default weighting scheme https://github.com/huggingface/diffusers/commit/a1d55e14baa051a8ec0c02949c0c27c1e6b21379
    weighting_scheme = "sigma_sqrt"
    logit_mean = 0.0
    logit_std = 1.0

    """
    for the sweep
    """
    if output_dir is not None:
        config.output_dir = output_dir

    # config.token_dict = {}
    config, input_dir = preprocess(
        config,
        working_directory=config.output_dir,
        concept_mode=config.concept_mode,
        input_zip_path=config.lora_training_urls,
        caption_text=config.caption_prefix,
        mask_target_prompts=config.mask_target_prompts,
        target_size=config.resolution,
        crop_based_on_salience=config.crop_based_on_salience,
        use_face_detection_instead=config.use_face_detection_instead,
        left_right_flip_augmentation=config.left_right_flip_augmentation,
        augment_imgs_up_to_n = config.augment_imgs_up_to_n,
        caption_model = config.caption_model,
        seed = config.seed,
    )

    checkpoints_folder = os.path.join(
        config.output_dir,
        "checkpoints"
    )
    
    inference_prompts = [
        "<s0><s1>, A man is eating popcorn while holding a knife", 
        "<s0><s1>, A man is taking a selfie in space",
        "<s0><s1>, Gentleman with a moustache dressed up as santa",
        "<s0><s1>, An 8 bit pixel art portrait of a man",
        "<s0><s1>, A man as a character within skyrim",
    ]

    if TRAIN_TEXTUAL_INVERSION:
        pass
    else:
        optimizer_textual_inversion = None
    
    # either go full finetuning or lora on transformer

    if not config.is_lora: # This code pathway has not been tested in a long while
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
            x for x in transformer.parameters() if x.requires_grad == True
        ]
    
    
    if TRAIN_TRANSFORMER:
        optimizer_transformer = get_transformer_optimizer(
            prodigy_d_coef=config.prodigy_d_coef,
            prodigy_growth_factor=config.unet_prodigy_growth_factor,
            lora_weight_decay=config.lora_weight_decay,
            use_dora=config.use_dora,
            transformer_trainable_params=transformer_trainable_params,
            optimizer_name=config.unet_optimizer_type,
            lr = config.unet_learning_rate
        )
    else:
        optimizer_transformer = None

    if TRAIN_TRANSFORMER:
        print(f"Transformer trainable params: {count_trainable_params(transformer)}")

    if TRAIN_TEXTUAL_INVERSION:
        print(f"Textual inversion trainable params: {sum([x.numel() for x in textual_inversion_params])}")

    optimizer = AllOptimizers(
        optimizer_dict={
            "transformer": optimizer_transformer,
            "textual_inversion": optimizer_textual_inversion
        }
    )
    train_dataset = PreprocessedDataset(
        input_dir,
        pipeline,
        vae,
        size = config.train_img_size,
        do_cache=config.do_cache,
        substitute_caption_map=config.token_dict,
        aspect_ratio_bucketing=config.aspect_ratio_bucketing,
        train_batch_size=config.train_batch_size
    )
    print(f"train_dataset contains: {len(train_dataset)} samples")

    ## not working right now because the number of tokens in tokenizers[0] does not match the number of tokens in the other tokenizers. Need to fix later in needed.
    # embedding_handler.visualize_random_token_embeddings(os.path.join(config.output_dir, 'ti_embeddings'), n = 10)

    train_dataloader = torch.utils.data.DataLoader(
        train_dataset,
        batch_size=config.train_batch_size,
        shuffle=True,
        num_workers=config.dataloader_num_workers,
    )
    global_step = 0
    num_train_steps = min(
        len(train_dataloader) * config.num_train_epochs,
        config.max_train_steps
    )
    print(f'Will train for {num_train_steps} steps')
    progress_bar = tqdm(
        range(global_step, num_train_steps), 
        position=0, 
        leave=True, 
        desc = "Training model"
    )

    if wandb_log:
        wandb.init(
            project = "eden-concept-trainer-sd3",
            config = config.dict()
        )
    for epoch in range(config.num_train_epochs):
        if config.aspect_ratio_bucketing:
            train_dataset.bucket_manager.start_epoch()
        progress_bar.set_description(f"# Trainer step: {global_step}, epoch: {epoch}")

        for step, batch in enumerate(train_dataloader):
            optimizer.zero_grad()
            # optimizer_ti.zero_grad()
            progress_bar.update(1)
            finegrained_epoch = epoch + step / len(train_dataloader)
            completion_f = finegrained_epoch / config.num_train_epochs

            """
            Scale learning rate of textual inversion params
            """
            if TRAIN_TEXTUAL_INVERSION:
                if config.ti_optimizer != "prodigy": # Update ti_learning rate gradually:
                    optimizer_textual_inversion.param_groups[0]['lr'] = config.ti_lr * (1 - completion_f) ** 2.0
                    # warmup the ti-lr:
                    if config.ti_lr_warmup_steps > 0:
                        warmup_f = min(global_step / config.ti_lr_warmup_steps, 1.0)
                        optimizer_textual_inversion.param_groups[0]['lr'] *= warmup_f
                    if config.freeze_ti_after_completion_f <= completion_f:
                        optimizer_textual_inversion.param_groups[0]['lr'] *= 0

            if not config.aspect_ratio_bucketing:
                captions, vae_latent, mask = batch
            else:
                captions, vae_latent, mask = train_dataset.get_aspect_ratio_bucketed_batch()

            model_input = vae_latent
            prompts = captions
            """
            some hardcoding on the captions just to see whether it works
            """
            # prompts = [
            #     x.replace("tok, ", "") for x in prompts
            # ]
            # prompts = [
            #     x.replace("bananaman", "<s0><s1>") for x in prompts
            # ]
            """
            done with hardcoding
            """
            # print(f"Global step: {global_step} Example prompt: {prompts[0]}")

            if not TRAIN_TEXTUAL_INVERSION:
                prompt_embeds, pooled_prompt_embeds = compute_text_embeddings(
                    prompt = prompts, 
                    text_encoders = text_encoders, 
                    tokenizers = [
                        tokenizer_one, tokenizer_two, tokenizer_three
                    ],
                    device=device
                )
            else:
                prompt_embeds, pooled_prompt_embeds = get_textual_inversion_prompt_embeds(
                    textual_inversion=textual_inversion,
                    textual_inversion_2=textual_inversion_2,
                    prompts = prompts,
                    text_encoders=text_encoders,
                    tokenizers = [tokenizer_one, tokenizer_two, tokenizer_three],
                    device=device
                )

            # Sample noise that we'll add to the latents
            noise = torch.randn_like(model_input)
            bsz = model_input.shape[0]

            # Sample a random timestep for each image
            """
            https://github.com/huggingface/diffusers/pull/8528/files#diff-e9278acb04a0c99638275caa05ddbbe608ad5115f053fcb78d794251b4fdc560
            """
            # indices = torch.randint(0, noise_scheduler_copy.config.num_train_timesteps, (bsz,))
            # for weighting schemes where we sample timesteps non-uniformly
            if weighting_scheme == "logit_normal":
                # See 3.1 in the SD3 paper ($rf/lognorm(0.00,1.00)$).
                u = torch.normal(mean=logit_mean, std=logit_std, size=(bsz,), device="cpu")
                u = torch.nn.functional.sigmoid(u)
            elif weighting_scheme == "mode":
                u = torch.rand(size=(bsz,), device="cpu")
                u = 1 - u - args.mode_scale * (torch.cos(math.pi * u / 2) ** 2 - 1 + u)
            else:
                u = torch.rand(size=(bsz,), device="cpu")

            indices = (u * noise_scheduler.config.num_train_timesteps).long()
            timesteps = noise_scheduler.timesteps[indices].to(device=model_input.device)

            # Add noise according to flow matching.
            sigmas = get_sigmas(
                timesteps=timesteps,
                noise_scheduler=noise_scheduler,
                device = device,
                n_dim=model_input.ndim, 
                dtype=model_input.dtype
            )
            noisy_model_input = sigmas * noise.to(sigmas.device) + (1.0 - sigmas) * model_input.to(sigmas.device)
            # Predict the noise residual
            model_pred = transformer(
                hidden_states=noisy_model_input.to(device),
                timestep=timesteps.to(device),
                encoder_hidden_states=prompt_embeds.to(device),
                pooled_projections=pooled_prompt_embeds.to(device),
                return_dict=False,
            )[0]
            model_pred = model_pred * (-sigmas) + noisy_model_input

            # TODO (kashif, sayakpaul): weighting sceme needs to be experimented with :)
            if weighting_scheme == "sigma_sqrt":
                weighting = (sigmas**-2.0).float()
            elif weighting_scheme == "logit_normal":
                # See 3.1 in the SD3 paper ($rf/lognorm(0.00,1.00)$).
                u = torch.normal(mean=logit_mean, std=logit_std, size=(bsz,), device=device)
                weighting = torch.nn.functional.sigmoid(u)
            elif weighting_scheme == "mode":
                # See sec 3.1 in the SD3 paper (20).
                u = torch.rand(size=(bsz,), device=device)
                weighting = 1 - u - args.mode_scale * (torch.cos(math.pi * u / 2) ** 2 - 1 + u)

            # simplified flow matching aka 0-rectified flow matching loss
            # target = model_input - noise
            target = model_input

            # Compute regular loss.
            loss_term = (
                    weighting.float().to(model_pred.device) * (model_pred.float() - target.float().to(model_pred.device)) ** 2
                )
            """
            apply mask
            """
            assert loss_term.shape == mask.shape

            loss_term = loss_term * mask.to(loss_term.device)

            loss = torch.mean(
                loss_term.reshape(target.shape[0], -1),
                1,
            )
            loss = loss.mean()

            loss.backward()

            if TRAIN_TRANSFORMER:
                torch.nn.utils.clip_grad_norm_(transformer_trainable_params, max_norm=1)
            
            if TRAIN_TEXTUAL_INVERSION:
                torch.nn.utils.clip_grad_norm_([textual_inversion.params, textual_inversion_2.params], max_norm=1)

            if global_step % config.gradient_accumulation_steps == 0:
                optimizer.step()
                # print(f"Performed an optimization step")

            progress_bar.set_postfix(
                {
                    "loss": round(loss.item(), 6)
                }
            )

            transformer_grad_norms = compute_gradient_norms(
                trainable_params=transformer_trainable_params
            )
            global_step += 1


            if wandb_log:
                data = {
                    "loss": loss.item(),
                    "global_step": global_step,
                }
                # data["textual_inversion_lr"] =  optimizer_ti.param_groups[0]['lr']
                if TRAIN_TEXTUAL_INVERSION:
                    data["textual_inversion_lr"] = optimizer_textual_inversion.param_groups[0]['lr']
                    data["textual_inversion_grad_norm"] = textual_inversion.params.grad.norm()
                    data["textual_inversion_2_grad_norm"] = textual_inversion_2.params.grad.norm()

                if TRAIN_TRANSFORMER:
                    data["transformer_lr"] = optimizer_transformer.param_groups[0]['lr'] 
                    data["transformer_grad_norms"] = wandb.Histogram(
                        transformer_grad_norms,
                    )

                wandb.log(
                    data
                )

            if global_step > config.max_train_steps:
                print(f"Reached max steps ({config.max_train_steps}), stopping training!")
                break

            if global_step % config.checkpointing_steps == 0:
                """
                Save intermediate checkpoint
                """                
                # save_transformer_lora_checkpoint(
                #     transformer=transformer,
                #     folder=os.path.join(checkpoints_folder,f"global_step_{global_step}", f"transformer")
                # )
                # temporarily commented out since we don't want to save checkpoints and fill up the storage

                """
                Run inference on a few prompts
                """
                torch.cuda.empty_cache()
                
                if TRAIN_TEXTUAL_INVERSION:
                    prompt_embeds, pooled_prompt_embeds = get_textual_inversion_prompt_embeds(
                        textual_inversion=textual_inversion,
                        textual_inversion_2=textual_inversion_2,
                        prompts = inference_prompts,
                        text_encoders=text_encoders,
                        tokenizers = [tokenizer_one, tokenizer_two, tokenizer_three],
                        device=device
                    )
                    prompt_embeds = prompt_embeds.to(inference_device)
                    pooled_prompt_embeds = pooled_prompt_embeds.to(inference_device)
                    pipeline = pipeline.to(inference_device)

                else:
                    pipeline = pipeline.to(inference_device)

                    prompt_embeds, pooled_prompt_embeds = compute_text_embeddings(
                        prompt = inference_prompts, 
                        text_encoders = text_encoders, 
                        tokenizers = tokenizers,
                        device=inference_device
                    )
                torch.cuda.empty_cache()
                # pipeline.transformer = transformer.to(inference_device)
                result = pipeline(
                    prompt_embeds = prompt_embeds,
                    pooled_prompt_embeds = pooled_prompt_embeds,
                    negative_prompt="",
                    num_inference_steps=28,
                    guidance_scale=7.0,
                    generator = torch.Generator(device=inference_device).manual_seed(0),
                    batch_size = 1
                )

                ## run inference and save images during training
                train_samples_folder = os.path.join(
                    checkpoints_folder,
                    f"global_step_{global_step}",
                    f"generated_samples"
                )
                os.system(
                    f"mkdir -p {train_samples_folder}"
                )
                for index in range(len(result.images)):
                    filename = os.path.join(
                        train_samples_folder,
                        f"global_step_{global_step}_index_{index}.jpg"
                    )
                    result.images[index].save(filename)
                    print(f"Saved: {filename}")

                torch.cuda.empty_cache()
                pipeline = pipeline.to(device)
        
        if global_step > config.max_train_steps:
            print("Reached max steps, stopping training!")
            break
    
    print(f"Training complete. Saving checkpoint...")
    # embedding_handler.save_embeddings(
    #     "sd3_embeddings.safetensors",
    #     txt_encoder_keys = ["1", "2", "3"]
    # )

    """
    Save transformer lora checkpoint
    """
    save_transformer_lora_checkpoint(
        transformer=transformer,
        folder=os.path.join(checkpoints_folder,f"global_step_{global_step}", f"transformer")
    )

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Train a concept')
    parser.add_argument('config_filename', type=str, help='Input JSON configuration file')
    parser.add_argument(
        '--wandb-log', 
        action="store_true", 
        help='enable this arg if you want to log losses to wandb'
    )
    args = parser.parse_args()
    config = TrainingConfig.from_json(file_path=args.config_filename)
    sweep_output_dir = os.path.join(
        "sd3_sweep_outputs",
        os.path.basename(args.config_filename).replace(".json", "")
    )
    main(
        config=config, 
        wandb_log=args.wandb_log,
        # for sweep
        output_dir=sweep_output_dir
    )
    ## cleanup after training run is complete
    os.system(
        f"rm -rf {sweep_output_dir}/images_in"
    )
    os.system(
        f"rm -rf {sweep_output_dir}/images_out"
    )

"""
python3 main_sd3.py training_args_banny.json  --wandb-log
python3 main_sd3.py training_args_face.json
"""