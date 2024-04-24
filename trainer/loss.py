import torch
from torch.utils._foreach_utils import _group_tensors_by_device_and_dtype, _has_foreach_support
from trainer.inference import get_conditioning_signals

def compute_snr(noise_scheduler, timesteps):
    """
    Computes SNR as per
    https://github.com/TiankaiHang/Min-SNR-Diffusion-Training/blob/521b624bd70c67cee4bdf49225915f5945a872e3/guided_diffusion/gaussian_diffusion.py#L847-L849
    """
    alphas_cumprod = noise_scheduler.alphas_cumprod
    sqrt_alphas_cumprod = alphas_cumprod**0.5
    sqrt_one_minus_alphas_cumprod = (1.0 - alphas_cumprod) ** 0.5

    # Expand the tensors.
    # Adapted from https://github.com/TiankaiHang/Min-SNR-Diffusion-Training/blob/521b624bd70c67cee4bdf49225915f5945a872e3/guided_diffusion/gaussian_diffusion.py#L1026
    sqrt_alphas_cumprod = sqrt_alphas_cumprod.to(device=timesteps.device)[timesteps].float()
    while len(sqrt_alphas_cumprod.shape) < len(timesteps.shape):
        sqrt_alphas_cumprod = sqrt_alphas_cumprod[..., None]
    alpha = sqrt_alphas_cumprod.expand(timesteps.shape)

    sqrt_one_minus_alphas_cumprod = sqrt_one_minus_alphas_cumprod.to(device=timesteps.device)[timesteps].float()
    while len(sqrt_one_minus_alphas_cumprod.shape) < len(timesteps.shape):
        sqrt_one_minus_alphas_cumprod = sqrt_one_minus_alphas_cumprod[..., None]
    sigma = sqrt_one_minus_alphas_cumprod.expand(timesteps.shape)

    # Compute SNR.
    snr = (alpha / sigma) ** 2
    return snr

def compute_grad_norm(parameters, norm_type = 2.0, foreach = None, error_if_nonfinite = False):
    if isinstance(parameters, torch.Tensor):
        parameters = [parameters]
    grads = [p.grad for p in parameters if p.grad is not None]
    first_device = grads[0].device
    grouped_grads = _group_tensors_by_device_and_dtype([[g.detach() for g in grads]])
    norms = []
    for ((device, _), ([grads], _)) in grouped_grads.items():
        if (foreach is None or foreach) and _has_foreach_support(grads, device=device):
            norms.extend(torch._foreach_norm(grads, norm_type))
        elif foreach:
            raise RuntimeError(f'foreach=True was passed, but can\'t use the foreach API on {device.type} tensors')
        else:
            norms.extend([torch.linalg.vector_norm(g, norm_type) for g in grads])

    total_norm = torch.linalg.vector_norm(torch.stack([norm.to(first_device) for norm in norms]), norm_type)

    return total_norm

def compute_diffusion_loss(config, model_pred, noise, noisy_latent, mask, noise_scheduler, timesteps):
    # Get the unet prediction target depending on the prediction type:
    if noise_scheduler.config.prediction_type == "epsilon":
        target = noise
    elif noise_scheduler.config.prediction_type == "v_prediction":
        print(f"Using velocity prediction!")
        target = noise_scheduler.get_velocity(noisy_latent, noise, timesteps)
    else:
        raise ValueError(f"Unknown prediction type {noise_scheduler.config.prediction_type}")
        
    loss = (model_pred - target).pow(2) * mask

    if config.snr_gamma is None or config.snr_gamma == 0.0:
        # modulate loss by the inverse of the mask's mean value
        mean_mask_values = mask.mean(dim=list(range(1, len(loss.shape))))
        mean_mask_values = mean_mask_values / mean_mask_values.mean()
        loss = loss.mean(dim=list(range(1, len(loss.shape)))) / mean_mask_values
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
        loss = loss.mean(dim=list(range(1, len(loss.shape)))) * mse_loss_weights

        # modulate loss by the inverse of the mask's mean value
        mean_mask_values = mask.mean(dim=list(range(1, len(loss.shape))))
        mean_mask_values = mean_mask_values / mean_mask_values.mean()
        loss = loss.mean(dim=list(range(1, len(loss.shape)))) / mean_mask_values
        loss = loss.mean()

    return loss

class ConditioningRegularizer:
    """
    Regularizes the norms of the prompt_conditioning vectors.
    """

    def __init__(self, config, embedding_handler):
        self.config = config
        self.embedding_handler = embedding_handler
        self.target_norm = 34.5 if config.sd_model_version == 'sdxl' else 27.8
        self.reg_captions = ["a photo of TOK", "TOK", "a photo of TOK next to TOK", "TOK and TOK"]
        self.token_replacement = config.token_dict.get("TOK", "TOK")  # Fallback to "TOK" if not in dict

        self.distribution_regularizers = {}
        idx = 0
        for tokenizer, text_encoder in zip(embedding_handler.tokenizers, embedding_handler.text_encoders):
            if tokenizer is None:
                idx += 1
                continue
            pretrained_token_embeddings = text_encoder.text_model.embeddings.token_embedding.weight.data
            self.distribution_regularizers[f'txt_encoder_{idx}'] = CovarianceLoss(pretrained_token_embeddings)
            idx += 1

    def apply_regularization(self, loss, losses, prompt_embeds_norms, prompt_embeds, pipe=None):
        noise_sigma = 0.0
        if noise_sigma > 0.0: # experimental: apply random noise to the conditioning vectors as a form of regularization
            prompt_embeds[0,1:-2,:] += torch.randn_like(prompt_embeds[0,2:-2,:]) * noise_sigma

        if self.config.cond_reg_w > 0.0:
            reg_loss, regularization_norm_value = self._compute_regularization_loss(prompt_embeds)
            loss += self.config.cond_reg_w * reg_loss
            prompt_embeds_norms['main'].append(regularization_norm_value.item())

        if self.config.tok_cond_reg_w > 0.0 and pipe is not None:
            reg_loss, regularization_norm_value = self._compute_tok_regularization_loss(pipe)
            loss += self.config.tok_cond_reg_w * reg_loss
            prompt_embeds_norms['reg'].append(regularization_norm_value.item())
        
        if self.config.tok_cov_reg_w > 0.0:
            tot_reg_losses = []
            for key, distribution_regularizer in self.distribution_regularizers.items():
                reg_loss = distribution_regularizer.compute_covariance_loss(self.embedding_handler.get_trainable_embeddings()[0][key])
                print(f"Reg loss for {key}: {reg_loss}")
                tot_reg_losses.append(reg_loss)
            
            mean_reg_loss = torch.stack(tot_reg_losses).mean()
            loss += self.config.tok_cov_reg_w * mean_reg_loss
            losses['covariance_tok_reg_loss'].append(mean_reg_loss.item())

        return loss, losses, prompt_embeds_norms

    def _compute_regularization_loss(self, prompt_embeds):
        conditioning_norms = prompt_embeds.norm(dim=-1).mean(dim=0)
        regularization_norm_value = conditioning_norms[2:].mean()
        reg_loss = (regularization_norm_value - self.target_norm).pow(2)
        return reg_loss, regularization_norm_value

    def _compute_tok_regularization_loss(self, pipe):
        reg_captions = [caption.replace("TOK", self.token_replacement) for caption in self.reg_captions]
        reg_prompt_embeds, reg_pooled_prompt_embeds, reg_add_time_ids = get_conditioning_signals(
            self.config, pipe, reg_captions
        )

        reg_conditioning_norms = reg_prompt_embeds.norm(dim=-1).mean(dim=0)
        regularization_norm_value = reg_conditioning_norms[2:].mean()
        reg_loss = (regularization_norm_value - self.target_norm).pow(2)

        return reg_loss, regularization_norm_value

class CovarianceLoss(torch.nn.Module):
    def __init__(self, pretrained_embeddings, dtype=torch.float32):
        super(CovarianceLoss, self).__init__()
        self.dtype = dtype
        self.cov_pretrained = self._calculate_covariance(pretrained_embeddings)

    def _calculate_covariance(self, embeddings):
        embeddings = embeddings.to(self.dtype)
        mean = embeddings.mean(0)
        embeddings_adjusted = embeddings - mean
        covariance = torch.mm(embeddings_adjusted.T, embeddings_adjusted) / (embeddings.size(0) - 1)
        return covariance

    def compute_covariance_loss(self, new_embeddings):
        input_dtype = new_embeddings.dtype
        cov_new = self._calculate_covariance(new_embeddings)
        # Normalizing by the product of the dimensions of the covariance matrix.
        num_features = new_embeddings.size(1)  # Assuming embeddings are of shape [n_samples, n_features]
        scale_factor = num_features * num_features
        loss = torch.norm(self.cov_pretrained - cov_new, p='fro') / scale_factor
        return loss.to(input_dtype)
    



"""

Learned Notes on the token embeddings:
shape = 49410, 768
Computed means of [1, 768] = [0, 0, 0, 0,...]
Computed stds  of [1, 768] = [0.0139, 0.0139, 0.0139, ...]

Computed means of [49410, 1] = [0, 0, 0, 0,...]
Computed stds  of [49410, 1] = [0.0151, 0.0154, 0.0141, ...,  0.0396, 0.0150, 0.0148]

"""


import torch.nn as nn

class DistributionRegularizer:
    def __init__(self, reference_tensors, stats=['mean', 'std', 'norm'], sample_dims=[0]):
        """
        Initialize a regularizer with reference distributions
        Facilitates computing regularization loss for trainable parameters to match the distribution of the reference distributions.
        
        :param reference_tensors: List of torch.Tensors for calculating the statistics.
        :param stats: List of statistics to calculate ('mean', 'std').
        :param sample_dims: Dimensions over which to compute these statistics.
        """
        self.stats = {}
        self.sample_dims = sample_dims
        # Calculate and store the statistics for each stat in stats
        print(f"Computing reference stats from reference tensor of shape: {reference_tensors.shape}...")

        if len(reference_tensors.shape) > 2:
            raise ValueError("Reference tensor should have at most 2 dimensions: [samples x features]")

        for stat in stats:
            if stat == 'mean':
                self.stats['mean'] = reference_tensors.mean(dim=sample_dims, keepdim=True)
                print(f"Computed reference means of shape: {self.stats['mean'].shape}")
                print(self.stats['mean'])
            elif stat == 'std':
                self.stats['std'] = reference_tensors.std(dim=sample_dims, keepdim=True)
                print(f"Computed reference stds of shape: {self.stats['std'].shape}")
                print(self.stats['std'])
            elif stat == 'norm':
                self.stats['norm'] = reference_tensors.norm(dim=-1, keepdim=True)
                print(f"Computed reference norms of shape: {self.stats['norm'].shape}")
                print(self.stats['norm'])
            else:
                raise ValueError(f"Unsupported statistic {stat}")

    def compute_reg_loss(self, list_of_samples):
        """
        Compute the regularization loss by comparing the new_tensor's statistics to the precomputed ones.
        
        :param new_tensor: a list of samples from the trained distribution
        :return: A scalar tensor representing the regularization loss wrt the reference statistics.
        """
        
        loss = 0
        # Calculate the new statistics and compare them to the precomputed ones

        for sample in list_of_samples:
            print(f"Sample shape: {sample.shape}")

            for stat_name, ref_stat in self.stats.items():
                if stat_name == 'mean':
                    new_stat    = new_tensor.mean(dim=self.sample_dims, keepdim=True)
                    reg_penalty = nn.functional.mse_loss(new_stat, ref_stat)
                    loss       += reg_penalty
                    print(f"{stat_name} loss: {reg_penalty.item()}")
                elif stat_name == 'std':
                    new_stat    = new_tensor.std(dim=self.sample_dims, keepdim=True)
                    reg_penalty = nn.functional.mse_loss(new_stat, ref_stat)
                    loss       += reg_penalty
                    print(f"{stat_name} loss: {reg_penalty.item()}")
                elif stat_name == 'norm':
                    new_stat    = new_tensor.norm(dim=-1, keepdim=True)
                    reg_penalty = nn.functional.mse_loss(new_stat, ref_stat)
                    loss       += reg_penalty
                    print(f"{stat_name} loss: {reg_penalty.item()}")

        return loss
