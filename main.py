import math
import os
import time
import shutil
import gc
import numpy as np
import argparse
import itertools
import zipfile
import torch
import torch.utils.checkpoint
from tqdm import tqdm

from trainer.utils.utils import *
from trainer.checkpoint import save_checkpoint
from trainer.embedding_handler import TokenEmbeddingsHandler
from trainer.dataset import PreprocessedDataset
from trainer.config import TrainingConfig
from trainer.models import print_trainable_parameters, load_models
from trainer.loss import compute_diffusion_loss, compute_grad_norm, ConditioningRegularizer, compute_token_attention_loss
from trainer.inference import render_images, get_conditioning_signals
from trainer.preprocess import preprocess
from trainer.utils.io import make_validation_img_grid
from trainer.ti_cross_attn_loss import init_daam_loss, plot_token_attention_loss

from trainer.optimizer import (
    OptimizerCollection, 
    get_optimizer_and_peft_models_text_encoder_lora, 
    get_textual_inversion_optimizer,
    get_unet_lora_parameters,
    get_unet_optimizer
)

def train(config: TrainingConfig):

    seed_everything(config.seed)
    weight_dtype = dtype_map[config.weight_type]

    (   
        pipe,
        tokenizer_one,
        tokenizer_two,
        noise_scheduler,
        text_encoder_one,
        text_encoder_two,
        vae,
        unet,
    ), sd_model_version = load_models(config.pretrained_model, config.device, weight_dtype)

    pipe, daam_loss = init_daam_loss(
        pipeline=pipe
    )

    config.sd_model_version = sd_model_version
    config.pretrained_model["version"] = sd_model_version

    if not config.sample_imgs_lora_scale:
        if config.sd_model_version == "sdxl":
            config.sample_imgs_lora_scale = 0.75
        else:
            config.sample_imgs_lora_scale = 0.85

    if not config.validation_img_size:
        if config.sd_model_version == "sdxl":
            config.validation_img_size = 1024
        else:
            config.validation_img_size = 768

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

    if config.allow_tf32:
        torch.backends.cuda.matmul.allow_tf32 = True

    # Initialize new tokens for training.
    embedding_handler = TokenEmbeddingsHandler(
        text_encoders = [text_encoder_one, text_encoder_two], 
        tokenizers = [tokenizer_one, tokenizer_two]
    )

    embedding_handler.initialize_new_tokens(
        inserting_toks=config.inserting_list_tokens,
        starting_toks=None, 
        seed=config.seed
    )

    # Experimental TODO: warmup the token embeddings using CLIP-similarity optimization
    embedding_handler.make_embeddings_trainable()
    embedding_handler.token_regularizer = ConditioningRegularizer(config, embedding_handler)
    embedding_handler.pre_optimize_token_embeddings(config, pipe)

    # Turn off all gradients for now:
    unet.requires_grad_(False)
    vae.requires_grad_(False)
    text_encoders = embedding_handler.text_encoders
    for txt_encoder in text_encoders:
        if txt_encoder is not None:
            txt_encoder.requires_grad_(False)

    if config.text_encoder_lora_optimizer is not None:
        print("Creating LoRA for text encoder...")
        optimizer_text_encoder_lora , text_encoder_peft_models = get_optimizer_and_peft_models_text_encoder_lora(
            text_encoders=text_encoders,
            lora_rank = config.text_encoder_lora_rank,
            lora_alpha_multiplier = config.lora_alpha_multiplier,
            use_dora = config.use_dora,
            optimizer_name = config.text_encoder_lora_optimizer,
            lora_lr = config.text_encoder_lora_lr,
            weight_decay = config.text_encoder_lora_weight_decay
        )
    else:
        optimizer_text_encoder_lora = None
        text_encoder_peft_models = [None] * len(text_encoders)


    embedding_handler.make_embeddings_trainable()
    if not config.disable_ti:
        optimizer_ti, textual_inversion_params = get_textual_inversion_optimizer(
            text_encoders=text_encoders,
            textual_inversion_lr=config.ti_lr,
            textual_inversion_weight_decay=config.ti_weight_decay,
            optimizer_name=config.ti_optimizer ## hardcoded
        )
    else:
        optimizer_ti = None
        textual_inversion_params = None

    if not config.is_lora: # This code pathway has not been tested in a long while
        print(f"Doing full fine-tuning on the U-Net")
        unet.requires_grad_(True)
        unet_lora_parameters = None
        optimizer_text_encoder_lora = None
        unet_trainable_params = unet.parameters()
    else:
        # Do lora-training instead.
        # https://huggingface.co/docs/peft/main/en/developer_guides/lora#rank-stabilized-lora
        # target_blocks=["block"] for original IP-Adapter
        # target_blocks=["up_blocks.0.attentions.1"] for style blocks only
        # target_blocks = ["up_blocks.0.attentions.1", "down_blocks.2.attentions.1"] # for style+layout blocks

        unet, unet_trainable_params, unet_lora_parameters = get_unet_lora_parameters(
            lora_rank = config.lora_rank,
            lora_alpha_multiplier = config.lora_alpha_multiplier,
            lora_weight_decay=config.lora_weight_decay,
            use_dora = config.use_dora,
            unet=unet,
            pipe=pipe
        )

    if config.unet_lr > 0.0:
        optimizer_unet = get_unet_optimizer(
            prodigy_d_coef=config.prodigy_d_coef,
            prodigy_growth_factor=config.unet_prodigy_growth_factor,
            lora_weight_decay=config.lora_weight_decay,
            use_dora=config.use_dora,
            unet_trainable_params=unet_trainable_params,
            optimizer_name=config.unet_optimizer_type
        )
    else:
        optimizer_unet = None
    
    print_trainable_parameters(unet, model_name = 'unet')
    for i, text_encoder in enumerate(text_encoders):
        if text_encoder is not  None:
            print_trainable_parameters(text_encoder, model_name = f'text_encoder_{i}')

    train_dataset = PreprocessedDataset(
        input_dir,
        pipe,
        vae.float(),
        size = config.train_img_size,
        substitute_caption_map=config.token_dict,
        aspect_ratio_bucketing=config.aspect_ratio_bucketing,
        train_batch_size=config.train_batch_size
    )
    print("Final training captions:")
    print(train_dataset.captions[:40])

    # offload the vae to cpu and release memory:
    vae = vae.to('cpu')
    gc.collect()
    torch.cuda.empty_cache()
    
    train_dataloader = torch.utils.data.DataLoader(
        train_dataset,
        batch_size=config.train_batch_size,
        shuffle=True,
        num_workers=config.dataloader_num_workers
    )

    config.num_train_epochs = int(math.ceil(config.max_train_steps / len(train_dataloader)))
    total_batch_size = config.train_batch_size * config.gradient_accumulation_steps

    print(f"--- Num samples = {len(train_dataset)}")
    print(f"--- Num batches each epoch = {len(train_dataloader)}")
    print(f"--- Num Epochs = {config.num_train_epochs}")
    print(f"--- Instantaneous batch size per device = {config.train_batch_size}")
    print(f"--- Total batch_size (distributed + accumulation) = {total_batch_size}")
    print(f"--- Gradient Accumulation steps = {config.gradient_accumulation_steps}")
    print(f"--- Total optimization steps = {config.max_train_steps}\n", flush = True)

    global_step = 0
    last_save_step = 0

    progress_bar = tqdm(range(global_step, config.max_train_steps), position=0, leave=True)
    checkpoint_dir = os.path.join(str(config.output_dir), "checkpoints")
    if os.path.exists(checkpoint_dir):
        shutil.rmtree(checkpoint_dir)
    os.makedirs(f"{checkpoint_dir}")

    # Data tracking inits:
    start_time, images_done = time.time(), 0
    prompt_embeds_norms = {'main':[], 'reg':[]}
    losses = {'img_loss': [], 'tot_loss': [], 'covariance_tok_reg_loss': [], 'concept_description_loss': [], 'token_std_loss': [], 'token_attention_loss': []}
    grad_norms, token_stds = {'unet': []}, {}
    for i in range(len(text_encoders)):
        grad_norms[f'text_encoder_{i}'] = []
        token_stds[f'text_encoder_{i}'] = {j: [] for j in range(config.n_tokens)}

    # default values for cold (starting) optimizer lr:
    base_unet_lr = 2.0e-4 if (config.is_lora and config.disable_ti) else 5.0e-5
    
    if not config.is_lora:
        base_unet_lr = 1.0e-5

    #######################################################################################################
    
    """
    Storing all optimizers in a single container
    """
    optimizer_collection = OptimizerCollection(
        optimizer_textual_inversion=optimizer_ti,
        optimizer_text_encoders=optimizer_text_encoder_lora,
        optimizer_unet=optimizer_unet,
        debug = config.debug
    )
    optimizers = optimizer_collection.optimizers

    if config.debug:
        embedding_handler.visualize_random_token_embeddings(os.path.join(config.output_dir, 'ti_embeddings'), n = 10)

    for epoch in range(config.num_train_epochs):
        if config.aspect_ratio_bucketing:
            train_dataset.bucket_manager.start_epoch()
        progress_bar.set_description(f"# Trainer step: {global_step}, epoch: {epoch}")

        for step, batch in enumerate(train_dataloader):
            progress_bar.update(1)
            finegrained_epoch = epoch + step / len(train_dataloader)
            completion_f = finegrained_epoch / config.num_train_epochs

            # param_groups[1] goes from ti_lr to 0.0 over the course of training
            if config.ti_optimizer != "prodigy" and optimizers['textual_inversion'] is not None:
                # Apply the exponential learning rate
                optimizers['textual_inversion'].param_groups[0]['lr'] = config.ti_lr * (1 - completion_f) ** 1.7
                # Apply freezing condition
                if completion_f > config.freeze_ti_after_completion_f:
                    optimizers['textual_inversion'].param_groups[0]['lr'] = 0.0

            if optimizers['text_encoders'] is not None:
                optimizers['text_encoders'].param_groups[0]['lr'] = config.text_encoder_lora_lr * (1 - completion_f) ** 2.0

                # warmup the txt-encoder lr:
                if config.txt_encoders_lr_warmup_steps > 0 and optimizers['text_encoders'] is not None:
                    warmup_f = min(global_step / config.txt_encoders_lr_warmup_steps, 1.0)
                    optimizers['text_encoders'].param_groups[0]['lr'] *= warmup_f
            
            if optimizers['unet'] is not None:
                # Calculate the exponential factor
                exp_factor = (config.unet_lr / base_unet_lr) ** (global_step / config.unet_lr_warmup_steps)
                # Apply the exponential learning rate
                optimizers['unet'].param_groups[0]['lr'] = base_unet_lr * exp_factor

                if completion_f < config.freeze_unet_before_completion_f:
                    optimizers['unet'].param_groups[0]['lr'] = 0.0

            if not config.aspect_ratio_bucketing:
                captions, vae_latent, mask = batch
            else:
                captions, vae_latent, mask = train_dataset.get_aspect_ratio_bucketed_batch()

            mask = mask.to(config.device)

            captions = list(captions)
            if config.caption_dropout > 0.0:
                for i in range(len(captions)):
                    if np.random.rand() < config.caption_dropout:
                        captions[i] = config.token_dict["TOK"]

            prompt_embeds, pooled_prompt_embeds, add_time_ids = get_conditioning_signals(
                config, pipe, captions
            )
            
            # Sample noise that we'll add to the latents:
            vae_latent = vae_latent.to(weight_dtype)
            noise = torch.randn_like(vae_latent)

            if config.noise_offset > 0.0:
                # https://www.crosslabs.org//blog/diffusion-with-offset-noise
                noise += config.noise_offset * torch.randn(
                    (noise.shape[0], noise.shape[1], 1, 1), device=noise.device)

            timesteps = torch.randint(
                0,
                noise_scheduler.config.num_train_timesteps,
                (vae_latent.shape[0],),
                device=vae_latent.device,
            ).long()

            noisy_latent = noise_scheduler.add_noise(vae_latent, noise, timesteps)

            # Predict the noise residual
            model_pred = unet(
                noisy_latent,
                timesteps,
                encoder_hidden_states=prompt_embeds,
                timestep_cond=None,
                added_cond_kwargs={"text_embeds": pooled_prompt_embeds, "time_ids": add_time_ids},
                return_dict=False,
            )[0]

            # Compute the loss:
            loss = compute_diffusion_loss(config, model_pred, noise, noisy_latent, mask, noise_scheduler, timesteps)
            losses['img_loss'].append(loss.item())

            if not config.disable_ti:
                token_attention_loss = compute_token_attention_loss(pipe, embedding_handler, captions, mask, daam_loss)
                losses['token_attention_loss'].append(token_attention_loss.item())
                loss = loss + config.token_attention_loss_w * token_attention_loss

            if config.training_attributes["gpt_description"] and config.debug:
                concept_description_loss = embedding_handler.compute_target_prompt_loss(config.training_attributes["gpt_description"], prompt_embeds, pooled_prompt_embeds, config, pipe)
                # Dont apply this loss, just plot it for now:
                loss += 0.0 * concept_description_loss
                losses['concept_description_loss'].append(concept_description_loss.item())

            if config.l1_penalty > 0.0 and unet_lora_parameters:
                # Compute normalized L1 norm (mean of abs sum) of all lora parameters:
                l1_norm = sum(p.abs().sum() for p in unet_lora_parameters) / sum(p.numel() for p in unet_lora_parameters)
                loss += config.l1_penalty * l1_norm

            if optimizers['textual_inversion'] is not None and optimizers['textual_inversion'].param_groups[0]['lr'] > 0.0:
                loss, losses, prompt_embeds_norms = embedding_handler.token_regularizer.apply_regularization(loss, losses, prompt_embeds_norms, prompt_embeds, pipe = pipe)

            losses['tot_loss'].append(loss.item())
            loss = loss / config.gradient_accumulation_steps
            loss.backward()

            last_batch = (step + 1 == len(train_dataloader))
            if (step + 1) % config.gradient_accumulation_steps == 0 or last_batch:

                if optimizers['textual_inversion'] is not None:
                    # zero out the gradients of the non-trained text-encoder embeddings
                    for i, embedding_tensor in enumerate(textual_inversion_params):
                        embedding_tensor.grad.data[:-config.n_tokens, : ] *= 0.

                if config.debug:
                    # Track the average gradient norms:
                    grad_norms['unet'].append(compute_grad_norm(itertools.chain(unet.parameters())).item())
                    for i, text_encoder in enumerate(text_encoders):
                        if text_encoder is not None:
                            text_encoder_norm = compute_grad_norm(itertools.chain(text_encoder.parameters())).item()
                            grad_norms[f'text_encoder_{i}'].append(text_encoder_norm)

                optimizer_collection.step()
                optimizer_collection.zero_grad()

            #############################################################################################################
            
            if config.debug:
                # Track the token embedding stds:
                trainable_embeddings, _ = embedding_handler.get_trainable_embeddings()
                for idx in range(len(text_encoders)):
                    if text_encoders[idx] is not None:
                        embedding_stds = trainable_embeddings[f'txt_encoder_{idx}'].detach().float().std(dim=1)
                        for std_i, std in enumerate(embedding_stds):
                            token_stds[f'text_encoder_{idx}'][std_i].append(embedding_stds[std_i].item())

                if global_step % 50 == 0 and not config.disable_ti and config.debug:
                    img_ratio = config.train_img_size[0] / config.train_img_size[1]
                    plot_token_attention_loss(config.output_dir, pipe, daam_loss, captions, timesteps, token_attention_loss, global_step, img_ratio)

            # Print some statistics:
            if (global_step % config.checkpointing_steps == 0) and (global_step < (config.max_train_steps - 25)) and global_step > 0:
                print(f"\n---- avg training fps: {images_done / (time.time() - start_time):.2f}", end="\r", flush = True)

                output_save_dir = f"{checkpoint_dir}/checkpoint-{global_step}"
                os.makedirs(output_save_dir, exist_ok=True)
                config.save_as_json(
                    os.path.join(output_save_dir, "training_args.json")
                )
                save_checkpoint(
                    output_dir=output_save_dir, 
                    global_step=global_step, 
                    unet=unet, 
                    embedding_handler=embedding_handler, 
                    token_dict=config.token_dict, 
                    is_lora=config.is_lora, 
                    unet_lora_parameters=unet_lora_parameters,
                    name=config.name,
                    text_encoder_peft_models=text_encoder_peft_models,
                    pretrained_model_version=config.pretrained_model["version"]
                )
                last_save_step = global_step

                if config.debug:
                    embedding_handler.print_token_info()
                    if config.is_lora: # plotting this hist for full unet parameters can run OOM
                        plot_torch_hist(unet_lora_parameters, global_step, config.output_dir, "lora_weights", min_val=-0.4, max_val=0.4, ymax_f = 0.08)
                    plot_loss(losses, save_path=f'{config.output_dir}/losses.png')
                    target_std_dict = {f"text_encoder_{idx}_target": embedding_handler.embeddings_settings[f"std_token_embedding_{idx}"].item() for idx in range(len(text_encoders)) if text_encoders[idx] is not None}
                    plot_token_stds(token_stds, save_path=f'{config.output_dir}/token_stds.png', target_value_dict=target_std_dict)
                    plot_grad_norms(grad_norms, save_path=f'{config.output_dir}/grad_norms.png')
                    plot_lrs(optimizer_collection.learning_rate_tracker, save_path=f'{config.output_dir}/learning_rates.png')
                    #plot_curve(prompt_embeds_norms, 'steps', 'norm', 'prompt_embed norms', save_path=f'{config.output_dir}/prompt_embeds_norms.png')

                validation_prompts = render_images(
                    pipe = pipe, 
                    render_size = config.validation_img_size, 
                    lora_path = output_save_dir, 
                    train_step = global_step, 
                    seed = config.seed, 
                    is_lora = config.is_lora, 
                    pretrained_model = config.pretrained_model, 
                    lora_scale = config.sample_imgs_lora_scale,
                    disable_ti = config.disable_ti,
                    prompt_modifier = config.prompt_modifier,
                    n_imgs = config.n_sample_imgs, 
                    device = config.device,
                    checkpoint_folder = None
                )
                img_grid_path = make_validation_img_grid(output_save_dir)
                shutil.copy(img_grid_path, os.path.join(os.path.dirname(output_save_dir), f"validation_grid_{global_step:04d}.jpg"))
                        
                gc.collect()
                torch.cuda.empty_cache()
            
            images_done += config.train_batch_size
            global_step += 1
            
            if global_step % (config.max_train_steps//100) == 0:
                progress = (global_step / config.max_train_steps) + 0.05
                #print_system_info()
                yield np.min((progress, 1.0))

            if global_step > config.max_train_steps:
                print("Reached max steps, stopping training!", flush = True)
                break

    # final_save
    if (global_step - last_save_step) > 26:
        output_save_dir = f"{checkpoint_dir}/checkpoint-{global_step}"
    else:
        output_save_dir = f"{checkpoint_dir}/checkpoint-{last_save_step}"

    if config.debug:
        plot_loss(losses, save_path=f'{config.output_dir}/losses.png')
        target_std_dict = {f"text_encoder_{idx}_target": embedding_handler.embeddings_settings[f"std_token_embedding_{idx}"].item() for idx in range(len(text_encoders)) if text_encoders[idx] is not None}
        plot_token_stds(token_stds, save_path=f'{config.output_dir}/token_stds.png', target_value_dict=target_std_dict)
        plot_lrs(optimizer_collection.learning_rate_tracker, save_path=f'{config.output_dir}/learning_rates.png')
        plot_torch_hist(unet_lora_parameters if config.is_lora else unet.parameters(), global_step, config.output_dir, "lora_weights", min_val=-0.4, max_val=0.4, ymax_f = 0.08)

    if not os.path.exists(output_save_dir):
        os.makedirs(output_save_dir, exist_ok=True)
        config.save_as_json(os.path.join(output_save_dir, "training_args.json"))
        save_checkpoint(
            output_dir=output_save_dir, 
            global_step=global_step, 
            unet=unet, 
            embedding_handler=embedding_handler, 
            token_dict=config.token_dict, 
            is_lora=config.is_lora, 
            unet_lora_parameters=unet_lora_parameters,
            name=config.name,
            pretrained_model_version=config.pretrained_model["version"]
        )
        
        if config.debug and 0:
            # Reload the entire pipe from disk + LoRa:
            pipe_to_use = None
            checkpoint_folder = output_save_dir
            del unet
            del vae
            del text_encoder_one
            del text_encoder_two
            del tokenizer_one
            del tokenizer_two
            del embedding_handler
            del pipe
            del train_dataloader
            del train_dataset
            gc.collect()
            torch.cuda.empty_cache()
        else:
            # Just render images with the active pipe (faster, easier):
            pipe_to_use = pipe
            checkpoint_folder = None

        validation_prompts = render_images(
                pipe = pipe_to_use, 
                render_size=config.validation_img_size, 
                lora_path=output_save_dir, 
                train_step=global_step, 
                seed=config.seed, 
                is_lora=config.is_lora, 
                pretrained_model=config.pretrained_model, 
                lora_scale=config.sample_imgs_lora_scale,
                disable_ti = config.disable_ti,
                prompt_modifier = config.prompt_modifier,
                n_imgs = config.n_sample_imgs, 
                n_steps = 30, 
                device = config.device,
                checkpoint_folder=checkpoint_folder
            )

        img_grid_path = make_validation_img_grid(output_save_dir)
        shutil.copy(img_grid_path, os.path.join(os.path.dirname(output_save_dir), f"validation_grid_{global_step:04d}.jpg"))

    else:
        print(f"Skipping final save, {output_save_dir} already exists")

    if config.debug:
        # Create a zipfile of all the *.py files in the directory
        parent_dir = os.path.dirname(os.path.abspath(__file__))
        zip_file_path = os.path.join(config.output_dir, 'source_code.zip')
        with zipfile.ZipFile(zip_file_path, 'w', zipfile.ZIP_DEFLATED) as zipf:
            zipdir(parent_dir, zipf)

    config.job_time = time.time() - config.start_time
    config.training_attributes["validation_prompts"] = validation_prompts
    config.save_as_json(os.path.join(output_save_dir, "training_args.json"))
    print("Training job complete, saving outputs...", flush = True)
    print("------------------------------------------")

    return config, output_save_dir


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Train a concept')
    parser.add_argument('config_filename', type=str, help='Input JSON configuration file')
    args = parser.parse_args()

    config = TrainingConfig.from_json(file_path=args.config_filename)

    print("Starting new LoRa training run with config:")
    print(config)
    print("------------------------------------------")
    
    for progress in train(config=config):
        print(f"Progress: {(100*progress):.2f}%", end="\r")

    print("Training done :)")
