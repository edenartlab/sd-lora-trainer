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
[] - [later] init lora params for text encoders
[x] - get textual inversion params and it's corresponding optimizer
[x] - either do full finetuning of transformer or init lora params
[x] - init optimizer for transformer trainable parameters
[x] - init PreprocessedDataset object
[] - [later] init OptimizerCollection containing all optimizers
[] - [debug step] visualize a random token embedding
[] - do training
    - [x] init train dataloader
    - [] [later] aspect ratio bucketed batch
    - [] [later] update learning rate based if not using the prodigy optimizer
    - [x] dynamic learning rate for textual inversion
    - [x] get either training batch (bucketed or not)
    - [x] training loop without forward or backward pass
    - [x] denoising step from sd3 training code
    - [x] [later] clip grad norms after loss backward
    - [x] wandb log loss
    - [x] loss go down
    - [x] apply mask to denoising loss
[x] - Save checkpoint after training
[] - save checkpoint during training
[x] - inference + visualize examples
[] - use hf accelerate
"""
import os
import math
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

TRAIN_TRANSFORMER = False
TRAIN_TEXTUAL_INVERSION = True

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
    optimizer_name="prodigy"
):
    if optimizer_name == "adamw":
        optimizer = torch.optim.AdamW(transformer_trainable_params, lr = 1e-4, weight_decay=lora_weight_decay if not use_dora else 0.0)
    
    elif optimizer_name == "prodigy":
        # Note: the specific settings of Prodigy seem to matter A LOT
        optimizer = prodigyopt.Prodigy(
            transformer_trainable_params,
            d_coef = prodigy_d_coef,
            lr=1.0,
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
        clip_prompt_embeds_list.append(prompt_embeds)
        clip_pooled_prompt_embeds_list.append(pooled_prompt_embeds)

    clip_prompt_embeds = torch.cat(clip_prompt_embeds_list, dim=-1)
    pooled_prompt_embeds = torch.cat(clip_pooled_prompt_embeds_list, dim=-1)

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

def compute_text_embeddings(prompt, text_encoders, tokenizers, device):
    with torch.no_grad():
        prompt_embeds, pooled_prompt_embeds = encode_prompt(text_encoders, tokenizers, prompt)
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

def main(config: TrainingConfig, wandb_log = False):
    
    device = "cuda:0"
    inference_device = "cuda:1"
    # 1. Load tokenizers
    tokenizer_one, tokenizer_two, tokenizer_three = load_sd3_tokenizers()
    tokenizers = [tokenizer_one, tokenizer_two, tokenizer_three]
    # 2. Load text encoders
    # text_encoder_one, text_encoder_two, text_encoder_three = load_sd3_text_encoders()

    pipeline = StableDiffusion3Pipeline.from_pretrained(
        "stabilityai/stable-diffusion-3-medium-diffusers",
        torch_dtype=torch.bfloat16
    )

    text_encoder_one, text_encoder_two, text_encoder_three = pipeline.text_encoder.to(device), pipeline.text_encoder_2.to(device), pipeline.text_encoder_3.to(device)


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
            # text_encoder_three,
        ]
    )

    text_encoders = [text_encoder_one, text_encoder_two, text_encoder_three]

    # initialize token embedding handler

    if TRAIN_TEXTUAL_INVERSION:
        embedding_handler = TokenEmbeddingsHandler(
            text_encoders = text_encoders, 
            tokenizers = [tokenizer_one, tokenizer_two, tokenizer_three]
        )

        embedding_handler.initialize_new_tokens(
            inserting_toks=["<s0>","<s1>"],
            starting_toks=None, 
            seed=0
        )

    """
    override some config params because we're recycling an sdxl config here
    """
    config.is_lora = True
    config.token_warmup_steps = 0
    config.lora_rank = 8
    config.sd_model_version = "sd3"
    # change default weighting scheme https://github.com/huggingface/diffusers/commit/a1d55e14baa051a8ec0c02949c0c27c1e6b21379
    weighting_scheme = "sigma_sqrt"
    logit_mean = 0.0
    logit_std = 1.0
    # config.token_dict = {}
    # raise AssertionError(config.caption_prefix)
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
    ]

    if TRAIN_TEXTUAL_INVERSION:
        embedding_handler.make_embeddings_trainable()
        embedding_handler.token_regularizer = ConditioningRegularizer(
            config, 
            embedding_handler
        )

        # embedding_handler.pre_optimize_token_embeddings(
        #     config, 
        #     pipe = pipeline
        # )

        # get textual inversion params and it's optimizer
        optimizer_textual_inversion, textual_inversion_params = get_textual_inversion_optimizer(
            text_encoders=text_encoders,
            textual_inversion_lr=config.ti_lr,
            textual_inversion_weight_decay=config.ti_weight_decay,
            optimizer_name=config.ti_optimizer ## hardcoded
        )
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
            optimizer_name=config.unet_optimizer_type
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
            if config.ti_optimizer != "prodigy": # Update ti_learning rate gradually:
                # optimizer_ti.param_groups[0]['lr'] = config.ti_lr * (1 - completion_f) ** 2.0
                # warmup the ti-lr:
                if config.ti_lr_warmup_steps > 0:
                    warmup_f = min(global_step / config.ti_lr_warmup_steps, 1.0)
                    # optimizer_ti.param_groups[0]['lr'] *= warmup_f
                if config.freeze_ti_after_completion_f <= completion_f:
                    pass
                    # optimizer_ti.param_groups[0]['lr'] *= 0

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
            print(f"Global step: {global_step} Example prompt: {prompts[0]}")
            prompt_embeds, pooled_prompt_embeds = compute_text_embeddings(
                prompt = prompts, 
                text_encoders = text_encoders, 
                tokenizers = [
                    tokenizer_one, tokenizer_two, tokenizer_three
                ],
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
                torch.nn.utils.clip_grad_norm_(textual_inversion_params, max_norm=1)

            optimizer.step()

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
                save_transformer_lora_checkpoint(
                    transformer=transformer,
                    folder=os.path.join(checkpoints_folder,f"global_step_{global_step}", f"transformer")
                )

                """
                Run inference on a few prompts
                """
                torch.cuda.empty_cache()
                pipeline = pipeline.to(inference_device)
                pipeline.transformer = transformer.to(inference_device)
                
                prompt_embeds, pooled_prompt_embeds = compute_text_embeddings(
                    prompt = inference_prompts, 
                    text_encoders = text_encoders, 
                    tokenizers = tokenizers,
                    device=inference_device
                )

                result = pipeline(
                    prompt_embeds = prompt_embeds,
                    pooled_prompt_embeds = pooled_prompt_embeds,
                    negative_prompt="",
                    num_inference_steps=28,
                    guidance_scale=7.0,
                    generator = torch.Generator(device="cuda").manual_seed(0),
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
    main(config=config, wandb_log=args.wandb_log)

"""
python3 main_sd3.py training_args_banny.json  --wandb-log
python3 main_sd3.py training_args_face.json
"""