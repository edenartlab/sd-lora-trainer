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

    def __init__(self, config):
        self.config = config
        self.target_norm = 34.5 if config.sd_model_version == 'sdxl' else 27.8
        self.reg_captions = ["a photo of TOK", "TOK", "a photo of TOK next to TOK", "TOK and TOK"]
        self.token_replacement = config.token_dict.get("TOK", "TOK")  # Fallback to "TOK" if not in dict

    def apply_regularization(self, loss, prompt_embeds_norms, prompt_embeds, pipe=None):
        noise_sigma = 0.0
        if noise_sigma > 0.0: # experimental: apply random noise to the conditioning vectors as a form of regularization
            prompt_embeds[0,1:-2,:] += torch.randn_like(prompt_embeds[0,2:-2,:]) * noise_sigma

        if self.config.cond_reg_w > 0.0:
            regularization_loss, regularization_norm_value = self._compute_regularization_loss(prompt_embeds)
            loss += self.config.cond_reg_w * regularization_loss
            prompt_embeds_norms['main'].append(regularization_norm_value.item())

        if self.config.tok_cond_reg_w > 0.0 and pipe is not None:
            regularization_loss, regularization_norm_value = self._compute_tok_regularization_loss(pipe)
            loss += self.config.tok_cond_reg_w * regularization_loss
            prompt_embeds_norms['reg'].append(regularization_norm_value.item())

        return loss, prompt_embeds_norms

    def _compute_regularization_loss(self, prompt_embeds):
        conditioning_norms = prompt_embeds.norm(dim=-1).mean(dim=0)
        regularization_norm_value = conditioning_norms[2:].mean()
        regularization_loss = (regularization_norm_value - self.target_norm).pow(2)
        return regularization_loss, regularization_norm_value

    def _compute_tok_regularization_loss(self, pipe):
        reg_captions = [caption.replace("TOK", self.token_replacement) for caption in self.reg_captions]
        reg_prompt_embeds, reg_pooled_prompt_embeds, reg_add_time_ids = get_conditioning_signals(
            self.config, pipe, reg_captions
        )

        reg_conditioning_norms = reg_prompt_embeds.norm(dim=-1).mean(dim=0)
        regularization_norm_value = reg_conditioning_norms[2:].mean()
        regularization_loss = (regularization_norm_value - self.target_norm).pow(2)

        return regularization_loss, regularization_norm_value