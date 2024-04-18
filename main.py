import fnmatch
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

import prodigyopt
from peft import LoraConfig, get_peft_model
from typing import Union, Iterable, List, Dict, Tuple, Optional, cast

from trainer.utils.utils import *
from trainer.checkpoint import save_checkpoint
from trainer.embedding_handler import TokenEmbeddingsHandler
from trainer.dataset import PreprocessedDataset
from trainer.config import TrainingConfig
from trainer.models import print_trainable_parameters, load_models
from trainer.loss import *
from trainer.inference import render_images, get_conditioning_signals
from trainer.preprocess import preprocess
from trainer.utils.io import make_validation_img_grid
from trainer.optimizer import (
    OptimizerCollection, 
    get_optimizer_and_peft_models_text_encoder_lora, 
    get_textual_inversion_optimizer,
    get_unet_lora_parameters,
    get_unet_optimizer
)

def train(
    config: TrainingConfig,
):  
    seed_everything(config.seed)

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
        temp=config.clipseg_temperature,
        left_right_flip_augmentation=config.left_right_flip_augmentation,
        augment_imgs_up_to_n = config.augment_imgs_up_to_n,
        caption_model = config.caption_model,
        seed = config.seed,
    )

    if config.allow_tf32:
        torch.backends.cuda.matmul.allow_tf32 = True

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
    ) = load_models(config.pretrained_model, config.device, weight_dtype, keep_vae_float32=0)

    # Initialize new tokens for training.
    embedding_handler = TokenEmbeddingsHandler(
        text_encoders = [text_encoder_one, text_encoder_two], 
        tokenizers = [tokenizer_one, tokenizer_two]
    )

    '''
    initialize 2 new tokens in the embeddings with random initialization
    '''
    embedding_handler.initialize_new_tokens(
        inserting_toks=config.inserting_list_tokens,
        starting_toks=None, 
        seed=config.seed
    )

    unet.requires_grad_(False)
    vae.requires_grad_(False)
    text_encoders = embedding_handler.text_encoders
    

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
        text_encoder_peft_models = None

    optimizer_ti, textual_inversion_params = get_textual_inversion_optimizer(
        text_encoders=text_encoders,
        textual_inversion_lr=config.ti_lr,
        textual_inversion_weight_decay=config.ti_weight_decay,
        optimizer_name=config.ti_optimizer ## hardcoded
    )

    if not config.is_lora: # This code pathway has not been tested in a long while
        print(f"Doing full fine-tuning on the U-Net")
        unet.requires_grad_(True)
        unet_lora_parameters = None
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

    optimizer_unet = get_unet_optimizer(
        prodigy_d_coef=config.prodigy_d_coef,
        lora_weight_decay=config.lora_weight_decay,
        use_dora=config.use_dora,
        unet_trainable_params=unet_trainable_params,
        optimizer_name="prodigy" # hardcode for now
    )
        
    print_trainable_parameters(unet, name = 'unet')
    for i, text_encoder in enumerate(text_encoders):
        if text_encoder is not  None:
            print_trainable_parameters(text_encoder, name = f'text_encoder_{i}')

    train_dataset = PreprocessedDataset(
        input_dir,
        pipe,
        vae.float(),
        size = config.train_img_size,
        do_cache=config.do_cache,
        substitute_caption_map=config.token_dict,
        aspect_ratio_bucketing=config.aspect_ratio_bucketing,
        train_batch_size=config.train_batch_size
    )
    # offload the vae to cpu:
    vae = vae.to('cpu')
    gc.collect()
    torch.cuda.empty_cache()

    print(f"# PTI : Loaded dataset, do_cache: {config.do_cache}")
    train_dataloader = torch.utils.data.DataLoader(
        train_dataset,
        batch_size=config.train_batch_size,
        shuffle=True,
        num_workers=config.dataloader_num_workers,
    )

    num_update_steps_per_epoch = math.ceil(len(train_dataloader) / config.gradient_accumulation_steps)
    num_update_steps_per_epoch = math.ceil(len(train_dataloader))

    if config.max_train_steps is None:
        config.max_train_steps = config.num_train_epochs * num_update_steps_per_epoch

    config.num_train_epochs = math.ceil(config.max_train_steps / num_update_steps_per_epoch)
    total_batch_size = config.train_batch_size * config.gradient_accumulation_steps

    if config.verbose:
        print(f"--- Num samples = {len(train_dataset)}")
        print(f"--- Num batches each epoch = {len(train_dataloader)}")
        print(f"--- Num Epochs = {config.num_train_epochs}")
        print(f"--- Instantaneous batch size per device = {config.train_batch_size}")
        print(f"--- Total batch_size (distributed + accumulation) = {total_batch_size}")
        print(f"--- Gradient Accumulation steps = {config.gradient_accumulation_steps}")
        print(f"--- Total optimization steps = {config.max_train_steps}\n")

    global_step = 0
    last_save_step = 0

    progress_bar = tqdm(range(global_step, config.max_train_steps), position=0, leave=True)
    checkpoint_dir = os.path.join(str(config.output_dir), "checkpoints")
    if os.path.exists(checkpoint_dir):
        shutil.rmtree(checkpoint_dir)
    os.makedirs(f"{checkpoint_dir}")

    # Experimental TODO: warmup the token embeddings using CLIP-similarity optimization
    embedding_handler.pre_optimize_token_embeddings(config)

    # Data tracking inits:
    start_time, images_done = time.time(), 0
    ti_lrs, lora_lrs, prompt_embeds_norms = [], [], {'main':[], 'reg':[]}
    losses = {'img_loss': [], 'tot_loss': []}
    grad_norms, token_stds = {'unet': []}, {}
    for i in range(len(text_encoders)):
        grad_norms[f'text_encoder_{i}'] = []
        token_stds[f'text_encoder_{i}'] = {j: [] for j in range(config.n_tokens)}

    condtioning_regularizer = ConditioningRegularizer(config)

    #######################################################################################################
    
    """
    Storing all optimizers in a single container
    """
    optimizer = OptimizerCollection(
        optimizer_unet=optimizer_unet,
        optimizer_textual_inversion=optimizer_ti,
        optimizer_text_encoder_lora=optimizer_text_encoder_lora
    )

    for epoch in range(config.num_train_epochs):
        if config.aspect_ratio_bucketing:
            train_dataset.bucket_manager.start_epoch()
        progress_bar.set_description(f"# PTI :step: {global_step}, epoch: {epoch}")

        for step, batch in enumerate(train_dataloader):
            progress_bar.update(1)
            if config.hard_pivot:
                if epoch >= config.num_train_epochs // 3:
                    if optimizer.optimizer_textual_inversion is not None:
                        print("Disabling textual_inversion!")
                        # The next line is 100% necessary, otherwise the optimizer will keep updating the embeddings for some reason (even though the optimizer is None)
                        optimizer.optimizer_textual_inversion.param_groups[0]['lr'] = 0.0
                        optimizer.optimizer_textual_inversion = None

            elif config.ti_optimizer != "prodigy": # Update ti_learning rate gradually:
                finegrained_epoch = epoch + step / len(train_dataloader)
                completion_f = finegrained_epoch / config.num_train_epochs
                # param_groups[1] goes from ti_lr to 0.0 over the course of training
                if optimizer.optimizer_textual_inversion is not None:
                    optimizer.optimizer_textual_inversion.param_groups[0]['lr'] = config.ti_lr * (1 - completion_f) ** 2.0

                # warmup the token embedding lr:
                if config.token_embedding_lr_warmup_steps > 0:
                    warmup_f = min(global_step / config.token_embedding_lr_warmup_steps, 1.0)
                    if optimizer.optimizer_textual_inversion is not None:
                        optimizer.optimizer_textual_inversion.param_groups[0]['lr'] *= warmup_f

            if not config.aspect_ratio_bucketing:
                captions, vae_latent, mask = batch
            else:
                captions, vae_latent, mask = train_dataset.get_aspect_ratio_bucketed_batch()

            captions = list(captions)
            #vae_latent = vae_latent.to(pipe.device).to(weight_dtype)
            #mask = mask.to(pipe.device).to(weight_dtype)

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

            if config.l1_penalty > 0.0:
                # Compute normalized L1 norm (mean of abs sum) of all lora parameters:
                l1_norm = sum(p.abs().sum() for p in unet_lora_parameters) / sum(p.numel() for p in unet_lora_parameters)
                loss += config.l1_penalty * l1_norm

            if optimizer.optimizer_textual_inversion is not None and optimizer.optimizer_textual_inversion.param_groups[0]['lr'] > 0.0:
                # Some custom regularization: # TODO test how much these actually help!!
                loss, prompt_embeds_norms = condtioning_regularizer.apply_regularization(loss, prompt_embeds_norms, prompt_embeds, pipe)

            losses['tot_loss'].append(loss.item())
            loss = loss / config.gradient_accumulation_steps
            loss.backward()

            last_batch = (step + 1 == len(train_dataloader))
            if (step + 1) % config.gradient_accumulation_steps == 0 or last_batch:

                if optimizer.optimizer_textual_inversion is not None:
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

                    # after every optimizer step, we do some manual intervention of the embeddings to regularize them:
                    embedding_handler.fix_embedding_std(config.off_ratio_power)

                optimizer.step()
                optimizer.zero_grad()

            #############################################################################################################

            # Track the token embedding stds:
            trainable_embeddings, _ = embedding_handler.get_trainable_embeddings()
            for idx in range(len(text_encoders)):
                if text_encoders[idx] is not None:
                    embedding_stds = torch.stack(trainable_embeddings[f'txt_encoder_{idx}']).detach().float().std(dim=1)
                    for std_i, std in enumerate(embedding_stds):
                        token_stds[f'text_encoder_{idx}'][std_i].append(embedding_stds[std_i].item())

            # Track the learning rates for final plotting:
            lora_lrs.append(get_avg_lr(optimizer_unet))
            try:
                ti_lrs.append(optimizer.optimizer_textual_inversion.param_groups[0]['lr'])
            except:
                ti_lrs.append(0.0)

            # Print some statistics:
            if config.debug and (global_step % config.checkpointing_steps == 0): #and global_step > 0:
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
                    text_encoder_peft_models=text_encoder_peft_models
                )
                last_save_step = global_step

                token_embeddings, trainable_tokens = embedding_handler.get_trainable_embeddings()
                for idx, text_encoder in enumerate(text_encoders):
                    if text_encoder is None:
                        continue
                    n = len(token_embeddings[f'txt_encoder_{idx}'])
                    for i in range(n):
                        token = trainable_tokens[f'txt_encoder_{idx}'][i]
                        # Strip any backslashes from the token name:
                        token = token.replace("/", "_")
                        embedding = token_embeddings[f'txt_encoder_{idx}'][i]
                        plot_torch_hist(embedding, global_step, os.path.join(config.output_dir, 'ti_embeddings') , f"enc_{idx}_tokid_{i}: {token}", min_val=-0.05, max_val=0.05, ymax_f = 0.05, color = 'red')

                embedding_handler.print_token_info()
                plot_torch_hist(unet_lora_parameters if config.is_lora else unet.parameters(), global_step, config.output_dir, "lora_weights", min_val=-0.4, max_val=0.4, ymax_f = 0.08)
                plot_loss(losses, save_path=f'{config.output_dir}/losses.png')
                target_std_dict = {f"txt_encoder_{idx}": embedding_handler.embeddings_settings[f"std_token_embedding_{idx}"].item() for idx in range(len(text_encoders)) if text_encoders[idx] is not None}
                plot_token_stds(token_stds, save_path=f'{config.output_dir}/token_stds.png', target_value_dict=target_std_dict)
                plot_grad_norms(grad_norms, save_path=f'{config.output_dir}/grad_norms.png')
                plot_lrs(lora_lrs, ti_lrs, save_path=f'{config.output_dir}/learning_rates.png')
                plot_curve(prompt_embeds_norms, 'steps', 'norm', 'prompt_embed norms', save_path=f'{config.output_dir}/prompt_embeds_norms.png')
                validation_prompts = render_images(pipe, config.validation_img_size, output_save_dir, global_step, 
                    config.seed, 
                    config.is_lora, 
                    config.pretrained_model, 
                    config.sample_imgs_lora_scale,
                    n_imgs = config.n_sample_imgs, 
                    verbose = config.verbose, 
                    trigger_text = config.training_attributes["trigger_text"],
                    device = config.device
                    )
                img_grid_path = make_validation_img_grid(output_save_dir)
                shutil.copy(img_grid_path, os.path.join(os.path.dirname(output_save_dir), f"validation_grid_{global_step:04d}.jpg"))
                        
                gc.collect()
                torch.cuda.empty_cache()
            
            images_done += config.train_batch_size
            global_step += 1

            if global_step % 100 == 0:
                print(f" ---- avg training fps: {images_done / (time.time() - start_time):.2f}", end="\r")

            if global_step % (config.max_train_steps//20) == 0:
                progress = (global_step / config.max_train_steps) + 0.05
                yield np.min((progress, 1.0))

    # final_save
    if (global_step - last_save_step) > 51:
        output_save_dir = f"{checkpoint_dir}/checkpoint-{global_step}"
    else:
        output_save_dir = f"{checkpoint_dir}/checkpoint-{last_save_step}"

    if config.debug:
        plot_loss(losses, save_path=f'{config.output_dir}/losses.png')
        target_std_dict = {f"txt_encoder_{idx}": embedding_handler.embeddings_settings[f"std_token_embedding_{idx}"].item() for idx in range(len(text_encoders)) if text_encoders[idx] is not None}
        plot_token_stds(token_stds, save_path=f'{config.output_dir}/token_stds.png', target_value_dict=target_std_dict)
        plot_lrs(lora_lrs, ti_lrs, save_path=f'{config.output_dir}/learning_rates.png')
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
            name=config.name
        )
        validation_prompts = render_images(pipe, config.validation_img_size, output_save_dir, global_step, 
            config.seed, 
            config.is_lora, 
            config.pretrained_model, 
            config.sample_imgs_lora_scale,
            n_imgs = config.n_sample_imgs, 
            n_steps = 30, 
            verbose=config.verbose, 
            trigger_text=config.training_attributes["trigger_text"],
            device = config.device
            )
        
        img_grid_path = make_validation_img_grid(output_save_dir)
        shutil.copy(img_grid_path, os.path.join(os.path.dirname(output_save_dir), f"validation_grid_{global_step:04d}.jpg"))

    else:
        print(f"Skipping final save, {output_save_dir} already exists")

    del unet
    del vae
    del text_encoder_one
    del text_encoder_two
    del tokenizer_one
    del tokenizer_two
    del embedding_handler
    del pipe
    gc.collect()
    torch.cuda.empty_cache()

    if config.debug:
        parent_dir = os.path.dirname(os.path.abspath(__file__))
        # Create a zipfile of all the *.py files in the directory
        zip_file_path = os.path.join(config.output_dir, 'source_code.zip')
        with zipfile.ZipFile(zip_file_path, 'w', zipfile.ZIP_DEFLATED) as zipf:
            zipdir(parent_dir, zipf)

    config.job_time = time.time() - config.start_time
    config.training_attributes["validation_prompts"] = validation_prompts
    config.save_as_json(os.path.join(output_save_dir, "training_args.json"))

    return config, output_save_dir



if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Train a concept')
    parser.add_argument("-c", '--config-filename', type=str, help='Input string to be processed')
    args = parser.parse_args()

    config = TrainingConfig.from_json(
        file_path=args.config_filename
    )
    for progress in train(config=config):
        print(f"Progress: {(100*progress):.2f}%", end="\r")

    print("Training done :)")