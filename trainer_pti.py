import fnmatch
import math
import os
import time
import shutil
import gc
import numpy as np
import argparse
import torch
import torch.utils.checkpoint

from peft import LoraConfig, get_peft_model
from diffusers.optimization import get_scheduler
from tqdm import tqdm

from trainer.dataset_and_utils import (
    PreprocessedDataset, 
    plot_torch_hist, 
    plot_loss, 
    plot_lrs,
    load_models
)
from trainer.utils.lora import (
    save_lora,
    TokenEmbeddingsHandler
)

from trainer.utils.dtype import dtype_map
from trainer.config import TrainingConfig
from trainer.utils.model_info import print_trainable_parameters
from trainer.utils.snr import compute_snr
from trainer.utils.training_info import get_avg_lr
from trainer.utils.inference import render_images
from preprocess import preprocess


def main(
    config: TrainingConfig,
):
    input_dir, n_imgs, trigger_text, segmentation_prompt, captions = preprocess(
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
        seed = config.seed,
    )

    instance_data_dir=os.path.join(input_dir, "captions.csv")

    if config.allow_tf32:
        torch.backends.cuda.matmul.allow_tf32 = True

    print("Using seed", config.seed)
    torch.manual_seed(config.seed)

    weight_dtype = dtype_map[config.mixed_precision]

    print(f"Loading models with weight_dtype: {weight_dtype}")

    if config.scale_lr:
        config.unet_learning_rate = (
            config.unet_learning_rate * config.gradient_accumulation_steps * config.train_batch_size
        )

    (   
        pipe,
        tokenizer_one,
        tokenizer_two,
        noise_scheduler,
        text_encoder_one,
        text_encoder_two,
        vae,
        unet,
    ) = load_models(config.pretrained_model, config.device, weight_dtype)

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
        # starting_toks = ["person", "face"],
        starting_toks=None, 
        seed=config.seed
    )
    text_encoders = [text_encoder_one, text_encoder_two]

    text_encoder_parameters = []
    for text_encoder in text_encoders:
        if text_encoder is not  None:
            for name, param in text_encoder.named_parameters():
                if "token_embedding" in name:
                    param.requires_grad = True
                    text_encoder_parameters.append(param)
                else:
                    param.requires_grad = False

    unet_param_to_optimize_names = []
    if not config.is_lora:
        
        WHITELIST_PATTERNS = [
            # "*.attn*.weight",
            # "*ff*.weight",
            "*"
        ]
        BLACKLIST_PATTERNS = ["*.norm*.weight", "*time*"]
        for name, param in unet.named_parameters():
            if any(
                fnmatch.fnmatch(name, pattern) for pattern in WHITELIST_PATTERNS
            ) and not any(
                fnmatch.fnmatch(name, pattern) for pattern in BLACKLIST_PATTERNS
            ):
                param.requires_grad_(True)
                unet_param_to_optimize_names.append(name)
                print(f"Training: {name}")
            else:
                param.requires_grad_(False)

        # Optimizer creation
        params_to_optimize = [
            {
                "params": text_encoder_parameters,
                "lr": config.ti_lr,
                "weight_decay": config.ti_weight_decay,
            },
        ]

        params_to_optimize_prodigy = [
            {
                "params": [],
                "lr": config.unet_learning_rate,
                "weight_decay": config.lora_weight_decay,
            },
        ]

    else:
        
        # Do lora-training instead.
        unet.requires_grad_(False)
        # https://huggingface.co/docs/peft/main/en/developer_guides/lora#rank-stabilized-lora
        unet_lora_config = LoraConfig(
            r=config.lora_rank,
            lora_alpha=config.lora_rank,
            init_lora_weights="gaussian",
            target_modules=["to_k", "to_q", "to_v", "to_out.0"],
            #use_rslora=True,
            use_dora=True,
        )
        #unet.add_adapter(unet_lora_config)
        
        unet = get_peft_model(unet, unet_lora_config)
        print_trainable_parameters(unet, name = 'unet')

        unet_lora_parameters = list(filter(lambda p: p.requires_grad, unet.parameters()))

        params_to_optimize = [
            {
                "params": text_encoder_parameters,
                "lr": config.ti_lr,
                "weight_decay": config.ti_weight_decay,
            },
        ]

        params_to_optimize_prodigy = [
            {
                "params": unet_lora_parameters,
                "lr": 1.0,
                "weight_decay": config.lora_weight_decay,
            },
        ]
    
    optimizer_type = "prodigy" # hardcode for now

    if optimizer_type != "prodigy":
        optimizer = torch.optim.AdamW(
            params_to_optimize,
            weight_decay=0.0, # this wd doesn't matter, I think
        )
    else:        
        try:
            import prodigyopt
        except ImportError:
            raise ImportError("To use Prodigy, please install the prodigyopt library: `pip install prodigyopt`")

        # Note: the specific settings of Prodigy seem to matter A LOT
        optimizer_prod = prodigyopt.Prodigy(
                        params_to_optimize_prodigy,
                        d_coef = config.prodigy_d_coef,
                        lr=1.0,
                        decouple=True,
                        use_bias_correction=True,
                        safeguard_warmup=True,
                        weight_decay=config.lora_weight_decay,
                        betas=(0.9, 0.99),
                        growth_rate=1.025,  # this slows down the lr_rampup
                    )
        
        optimizer = torch.optim.AdamW(
            params_to_optimize,
            weight_decay=config.ti_weight_decay,
        )
        
    train_dataset = PreprocessedDataset(
        instance_data_dir,
        tokenizer_one,
        tokenizer_two,
        vae,
        do_cache=config.do_cache,
        substitute_caption_map=config.token_dict,
    )

    print(f"# PTI : Loaded dataset, do_cache: {config.do_cache}")
    train_dataloader = torch.utils.data.DataLoader(
        train_dataset,
        batch_size=config.train_batch_size,
        shuffle=True,
        num_workers=config.dataloader_num_workers,
    )

    num_update_steps_per_epoch = math.ceil(
        len(train_dataloader) / config.gradient_accumulation_steps
    )
    if config.max_train_steps is None:
        config.max_train_steps = config.num_train_epochs * num_update_steps_per_epoch

    lr_scheduler = get_scheduler(
        config.lr_scheduler,
        optimizer=optimizer,
        num_warmup_steps=config.lr_warmup_steps * config.gradient_accumulation_steps,
        num_training_steps=config.max_train_steps * config.gradient_accumulation_steps,
        num_cycles=config.lr_num_cycles,
        power=config.lr_power,
    )

    num_update_steps_per_epoch = math.ceil(
        len(train_dataloader) / config.gradient_accumulation_steps
    )
    config.num_train_epochs = math.ceil(config.max_train_steps / num_update_steps_per_epoch)

    total_batch_size = config.train_batch_size * config.gradient_accumulation_steps

    if config.verbose:
        print(f"# PTI :  Running training ")
        print(f"# PTI :  Num examples = {len(train_dataset)}")
        print(f"# PTI :  Num batches each epoch = {len(train_dataloader)}")
        print(f"# PTI :  Num Epochs = {config.num_train_epochs}")
        print(f"# PTI :  Instantaneous batch size per device = {config.train_batch_size}")
        print(
            f"  Total train batch size (w. parallel, distributed & accumulation) = {total_batch_size}"
        )
        print(f"# PTI :  Gradient Accumulation steps = {config.gradient_accumulation_steps}")
        print(f"# PTI :  Total optimization steps = {config.max_train_steps}")

    global_step = 0
    first_epoch = 0
    last_save_step = 0

    progress_bar = tqdm(range(global_step, config.max_train_steps), position=0, leave=True)
    checkpoint_dir = os.path.join(str(config.output_dir), "checkpoints")
    if os.path.exists(checkpoint_dir):
        shutil.rmtree(checkpoint_dir)
    
    os.makedirs(f"{checkpoint_dir}")

    # Experimental TODO: warmup the token embeddings using CLIP-similarity optimization
    #embedding_handler.pre_optimize_token_embeddings(train_dataset)
    
    ti_lrs, lora_lrs = [], []
    losses = []
    start_time, images_done = time.time(), 0

    for epoch in range(first_epoch, config.num_train_epochs):
        unet.train()
        progress_bar.set_description(f"# PTI :step: {global_step}, epoch: {epoch}")

        for step, batch in enumerate(train_dataloader):
            progress_bar.update(1)
            if config.hard_pivot:
                if epoch >= config.num_train_epochs // 2:
                    if optimizer is not None:
                        print("----------------------")
                        print("# PTI :  Pivot halfway")
                        print("----------------------")
                        # remove text encoder parameters from the optimizer
                        optimizer.param_groups = None
                        # remove the optimizer state corresponding to text_encoder_parameters
                        for param in text_encoder_parameters:
                            if param in optimizer.state:
                                del optimizer.state[param]
                        optimizer = None

            else: # Update learning rates gradually:
                finegrained_epoch = epoch + step / len(train_dataloader)
                completion_f = finegrained_epoch / config.num_train_epochs
                # param_groups[1] goes from ti_lr to 0.0 over the course of training
                optimizer.param_groups[0]['lr'] = config.ti_lr * (1 - completion_f) ** 2.0

            
            try: #sdxl
                (tok1, tok2), vae_latent, mask = batch
            except: #sd15
                tok1, vae_latent, mask = batch
                tok2 = None

            vae_latent = vae_latent.to(weight_dtype)

            # tokens to text embeds
            prompt_embeds_list = []
            for tok, text_encoder in zip((tok1, tok2), text_encoders):
                if tok is None:
                    continue

                prompt_embeds_out = text_encoder(
                    tok.to(text_encoder.device),
                    output_hidden_states=True,
                )

                pooled_prompt_embeds = prompt_embeds_out[0]
                prompt_embeds = prompt_embeds_out.hidden_states[-2]
                bs_embed, seq_len, _ = prompt_embeds.shape
                prompt_embeds = prompt_embeds.view(bs_embed, seq_len, -1)
                prompt_embeds_list.append(prompt_embeds)

            prompt_embeds = torch.concat(prompt_embeds_list, dim=-1)
            pooled_prompt_embeds = pooled_prompt_embeds.view(bs_embed, -1)

            # Create Spatial-dimensional conditions.
            original_size = (config.resolution, config.resolution)
            target_size   = (config.resolution, config.resolution)
            crops_coords_top_left = (config.crops_coords_top_left_h, config.crops_coords_top_left_w)
            add_time_ids = list(original_size + crops_coords_top_left + target_size)
            add_time_ids = torch.tensor([add_time_ids])
            add_time_ids = add_time_ids.to(config.device, dtype=prompt_embeds.dtype).repeat(
                bs_embed, 1
            )

            # Sample noise that we'll add to the latents:
            noise = torch.randn_like(vae_latent)

            noise_offset = 0.05 # TODO, turn this into an input arg and do a grid search
            if noise_offset > 0.0:
                # https://www.crosslabs.org//blog/diffusion-with-offset-noise
                noise += noise_offset * torch.randn(
                    (noise.shape[0], noise.shape[1], 1, 1), device=noise.device)

            bsz = vae_latent.shape[0]

            timesteps = torch.randint(
                0,
                noise_scheduler.config.num_train_timesteps,
                (bsz,),
                device=vae_latent.device,
            ).long()

            noisy_model_input = noise_scheduler.add_noise(vae_latent, noise, timesteps)

            noise_sigma = 0.0
            if noise_sigma > 0.0: # experimental: apply random noise to the conditioning vectors as a form of regularization
                prompt_embeds[0,1:-2,:] += torch.randn_like(prompt_embeds[0,1:-2,:]) * noise_sigma

            # Predict the noise residual
            model_pred = unet(
                noisy_model_input,
                timesteps,
                prompt_embeds,
                added_cond_kwargs={"text_embeds": pooled_prompt_embeds, "time_ids": add_time_ids},
            ).sample

            # Get the unet prediction target depending on the prediction type:
            if noise_scheduler.config.prediction_type == "epsilon":
                target = noise
            else:
                raise ValueError(f"Unknown prediction type {noise_scheduler.config.prediction_type}")

            # Compute the loss:
            if config.snr_gamma is None:
                loss = (model_pred - target).pow(2) * mask

                # modulate loss by the inverse of the mask's mean value
                mean_mask_values = mask.mean(dim=list(range(1, len(loss.shape))))
                mean_mask_values = mean_mask_values / mean_mask_values.mean()
                loss = loss.mean(dim=list(range(1, len(loss.shape)))) / mean_mask_values

                # Average the normalized errors across the batch
                loss = loss.mean()

            else:
                # Compute loss-weights as per Section 3.4 of https://arxiv.org/abs/2303.09556.
                # Since we predict the noise instead of x_0, the original formulation is slightly changed.
                # This is discussed in Section 4.2 of the same paper.
                snr = compute_snr(noise_scheduler, timesteps)
                base_weight = (
                    torch.stack([snr, config.snr_gamma * torch.ones_like(timesteps)], dim=1).min(dim=1)[0] / snr
                )
                if noise_scheduler.config.prediction_type == "v_prediction":
                    # Velocity objective needs to be floored to an SNR weight of one.
                    mse_loss_weights = base_weight + 1
                else:
                    # Epsilon and sample both use the same loss weights.
                    mse_loss_weights = base_weight

                mse_loss_weights = mse_loss_weights / mse_loss_weights.mean()
                loss = (model_pred - target).pow(2) * mask
                loss = loss.mean(dim=list(range(1, len(loss.shape)))) * mse_loss_weights

                if 1: # modulate loss by the inverse of the mask's mean value
                    mean_mask_values = mask.mean(dim=list(range(1, len(loss.shape))))
                    mean_mask_values = mean_mask_values / mean_mask_values.mean()
                    loss = loss.mean(dim=list(range(1, len(loss.shape)))) / mean_mask_values

                loss = loss.mean()

            if config.l1_penalty > 0.0:
                # Compute normalized L1 norm (mean of abs sum) of all lora parameters:
                l1_norm = sum(p.abs().sum() for p in unet_lora_parameters) / sum(p.numel() for p in unet_lora_parameters)
                loss +=  config.l1_penalty * l1_norm

            losses.append(loss.item())

            loss = loss / config.gradient_accumulation_steps
            loss.backward()

            '''
            apart from the usual gradient accumulation steps,
            we also do a backward pass after computing the last forward pass in the epoch (last_batch == True)
            this is to make sure that we're not missing out on any data 
            '''
            last_batch = (step + 1 == len(train_dataloader))
            if (step + 1) % config.gradient_accumulation_steps == 0 or last_batch:
                if optimizer is not None:

                    for embedding_tensor in text_encoder_parameters:
                        embedding_tensor.grad.data[
                            :-len(config.inserting_list_tokens), 
                            :
                        ] *= 0.
                    optimizer.step()
                    optimizer.zero_grad()
                
                optimizer_prod.step()
                optimizer_prod.zero_grad()

                # after every optimizer step, we reset the non-trainable embeddings to the original embeddings
                embedding_handler.retract_embeddings(print_stds = (global_step % 50 == 0))
                embedding_handler.fix_embedding_std(config.off_ratio_power)
            
            # Track the learning rates for final plotting:
            lora_lrs.append(get_avg_lr(optimizer_prod))
            try:
                ti_lrs.append(optimizer.param_groups[0]['lr'])
            except:
                ti_lrs.append(0.0)

            # Print some statistics:
            if (global_step % config.checkpointing_steps == 0):
                output_save_dir = f"{checkpoint_dir}/checkpoint-{global_step}"
                os.makedirs(output_save_dir, exist_ok=True)
                config.save_as_json(
                    os.path.join(output_save_dir, "training_args.json")
                )
                save_lora(
                    output_dir=output_save_dir, 
                    global_step=global_step, 
                    unet=unet, 
                    embedding_handler=embedding_handler, 
                    token_dict=config.token_dict, 
                    seed=config.seed, 
                    is_lora=config.is_lora, 
                    unet_lora_parameters=unet_lora_parameters,
                    unet_param_to_optimize_names=unet_param_to_optimize_names,
                    name=name
                )
                last_save_step = global_step

                if config.debug:
                    token_embeddings = embedding_handler.get_trainable_embeddings()
                    for i, token_embeddings_i in enumerate(token_embeddings):
                        plot_torch_hist(token_embeddings_i[0], global_step, config.output_dir, f"embeddings_weights_token_0_{i}", min_val=-0.05, max_val=0.05, ymax_f = 0.05)
                        plot_torch_hist(token_embeddings_i[1], global_step, config.output_dir, f"embeddings_weights_token_1_{i}", min_val=-0.05, max_val=0.05, ymax_f = 0.05)
                    
                    embedding_handler.print_token_info()
                    plot_torch_hist(unet_lora_parameters, global_step, config.output_dir, "lora_weights", min_val=-0.3, max_val=0.3, ymax_f = 0.05)
                    plot_loss(losses, save_path=f'{config.output_dir}/losses.png')
                    plot_lrs(lora_lrs, ti_lrs, save_path=f'{config.output_dir}/learning_rates.png')
                    validation_prompts = render_images(pipe, target_size, output_save_dir, global_step, config.seed, config.is_lora, config.pretrained_model, n_imgs = 4, verbose=config.verbose, trigger_text=trigger_text)
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
        plot_lrs(lora_lrs, ti_lrs, save_path=f'{config.output_dir}/learning_rates.png')
        plot_torch_hist(unet_lora_parameters, global_step, config.output_dir, "lora_weights", min_val=-0.3, max_val=0.3, ymax_f = 0.05)
        plot_torch_hist(embedding_handler.get_trainable_embeddings(), global_step, config.output_dir, "embeddings_weights", min_val=-0.05, max_val=0.05, ymax_f = 0.05)      

    if not os.path.exists(output_save_dir):
        os.makedirs(output_save_dir, exist_ok=True)
        config.save_as_json(
            os.path.join(output_save_dir, "training_args.json")
        )
        save_lora(
            output_dir=output_save_dir, 
            global_step=global_step, 
            unet=unet, 
            embedding_handler=embedding_handler, 
            token_dict=config.token_dict, 
            seed=config.seed, 
            is_lora=config.is_lora, 
            unet_lora_parameters=unet_lora_parameters,
            unet_param_to_optimize_names=unet_param_to_optimize_names,
            name=name
        )
        
        validation_prompts = render_images(pipe, target_size, output_save_dir, global_step, config.seed, config.is_lora, config.pretrained_model, n_imgs = 4, n_steps = 35, verbose=config.verbose, trigger_text=trigger_text)
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

    config.save_as_json(
            os.path.join(output_save_dir, "training_args.json")
        )

    return output_save_dir, validation_prompts


if __name__ == "__main__":

    parser = argparse.ArgumentParser(description='Train a concept')
    parser.add_argument("-c", '--config-filename', type=str, help='Input string to be processed')
    args = parser.parse_args()

    config = TrainingConfig.from_json(
        file_path=args.config_filename
    )
    for progress in main(config=config):
        print(f"Progress: {progress}")

    print("Done :)")