import os
import time
import matplotlib.pyplot as plt
import torch
from torch.utils._foreach_utils import _group_tensors_by_device_and_dtype, _has_foreach_support
from trainer.inference import get_conditioning_signals
from transformers import T5EncoderModel

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
    Regularizes:
        - the norms of the prompt_conditioning vectors
        - the statistics of the token embeddings.
    """

    def __init__(self, config, embedding_handler):
        self.config = config
        self.embedding_handler = embedding_handler
        self.target_norms = {
            "sdxl": 34.5,
            "sd15": 27.8,
            "sd3": 34.5
        }
        print(f'\033[91m[trainer.loss.ConditioningRegularizer] WARNING: Using a magic number: 34.5 for the target norm of sd3. We do not know if this is the ideal value. This might cause bugs or even break training completely.\033[0m')

        self.target_norm = self.target_norms[config.sd_model_version]
        self.reg_captions = ["a photo of TOK", "TOK", "a photo of TOK next to TOK", "TOK and TOK"]
        self.token_replacement = config.token_dict.get("TOK", "TOK")  # Fallback to "TOK" if not in dict

        self.distribution_regularizers = {}
        idx = 0
        for tokenizer, text_encoder in zip(embedding_handler.tokenizers, embedding_handler.text_encoders):
            if tokenizer is None:
                idx += 1
                continue

            if isinstance(text_encoder,  T5EncoderModel):
                pretrained_token_embeddings =  text_encoder.encoder.embed_tokens.weight.data
            else:
                pretrained_token_embeddings = text_encoder.text_model.embeddings.token_embedding.weight.data
            self.distribution_regularizers[f'txt_encoder_{idx}'] = DistributionLoss(pretrained_token_embeddings, outdir = self.config.output_dir if config.debug else None)
            idx += 1

    def apply_regularization(self, loss, losses, prompt_embeds_norms, prompt_embeds, std_loss_w = 0.003, pipe=None):
        noise_sigma = 0.0
        if noise_sigma > 0.0: # experimental: apply random noise to the conditioning vectors as a form of regularization
            prompt_embeds[0,1:-2,:] += torch.randn_like(prompt_embeds[0,2:-2,:]) * noise_sigma

        if self.config.cond_reg_w > 0.0:
            reg_loss, regularization_norm_value = self._compute_regularization_loss(prompt_embeds)
            loss += self.config.cond_reg_w * reg_loss
            if prompt_embeds_norms is not None:
                prompt_embeds_norms['main'].append(regularization_norm_value.item())

        if self.config.tok_cond_reg_w > 0.0 and pipe is not None:
            reg_loss, regularization_norm_value = self._compute_tok_regularization_loss(pipe)
            loss += self.config.tok_cond_reg_w * reg_loss
            if prompt_embeds_norms is not None:
                prompt_embeds_norms['reg'].append(regularization_norm_value.item())
        
        if self.config.tok_cov_reg_w > 0.0:
            tot_reg_losses = []
            for key, distribution_regularizer in self.distribution_regularizers.items():
                reg_loss = distribution_regularizer.compute_covariance_loss(self.embedding_handler.get_trainable_embeddings()[0][key])
                tot_reg_losses.append(reg_loss)
            
            mean_reg_loss = torch.stack(tot_reg_losses).mean()
            loss += self.config.tok_cov_reg_w * mean_reg_loss
            losses['covariance_tok_reg_loss'].append(mean_reg_loss.item())

        if std_loss_w > 0.0:
            tot_std_losses = []
            for key, distribution_regularizer in self.distribution_regularizers.items():
                std_loss = distribution_regularizer.compute_std_loss(self.embedding_handler.get_trainable_embeddings()[0][key])
                tot_std_losses.append(std_loss)
            
            mean_std_loss = torch.stack(tot_std_losses).mean()
            loss += std_loss_w * mean_std_loss
            losses['token_std_loss'].append(mean_std_loss.item())

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


class DistributionLoss(torch.nn.Module):
    """
    Class to simplify the calculation of the covariance loss between the trained token embeddings and the pretrained embeddings.
    """
    def __init__(self, pretrained_embeddings, dtype=torch.float32, outdir = None):
        super(DistributionLoss, self).__init__()
        print(f"Initialized a new DistributionLoss with shape: {pretrained_embeddings.shape}")
        self.dtype = dtype
        self.target_cov   = self._calculate_covariance(pretrained_embeddings)
        self.target_stds  = pretrained_embeddings.std(-1)
        self.target_stds_mean = self.target_stds.mean()
        self.target_stds_var  = self.target_stds.std()**2 / self.target_stds.mean()

        if outdir:
            # Plot a histogram of the stds:
            plt.figure()
            plt.hist(self.target_stds.detach().float().cpu().numpy(), bins=100)
            plt.title(f"stds of tokens (shape = {pretrained_embeddings.shape[0]} x {pretrained_embeddings.shape[1]})")
            plt.xlim(0, 0.02)
            plt.savefig(os.path.join(outdir, f"stds_histogram_{int(time.time()*100)}.png"))

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
        loss = torch.norm(self.target_cov - cov_new, p='fro') / scale_factor
        return loss.to(input_dtype)
    
    def compute_std_loss(self, new_embeddings):
        if new_embeddings.size(1) == 1:
            new_embeddings = new_embeddings.unsqueeze(0)

        deviation_loss = ((self.target_stds_mean - new_embeddings.std(-1))**2 / self.target_stds_var).mean()

        return deviation_loss



#######################################################################
#######################################################################

## Everything below here is experimental stuff not yet fully functional:

import torch
import numpy as np
from torch.distributions import MultivariateNormal, Normal
from torch.distributions.distribution import Distribution

class GaussianKDE(Distribution):
    def __init__(self, X, bw = 0.1):
        """
        X : tensor (n, d)
          `n` points with `d` dimensions to which KDE will be fit
        bw : numeric
          bandwidth for Gaussian kernel
        """
        self.X = X
        self.bw = bw
        self.dims = X.shape[-1]
        self.n = X.shape[0]
        self.mvn = MultivariateNormal(loc=torch.zeros(self.dims),
                                      covariance_matrix=torch.eye(self.dims))

    def sample(self, num_samples):
        idxs = (np.random.uniform(0, 1, num_samples) * self.n).astype(int)
        norm = Normal(loc=self.X[idxs], scale=self.bw)
        return norm.sample()

    def score_samples(self, Y, X=None):
        """Returns the kernel density estimates of each point in `Y`.

        Parameters
        ----------
        Y : tensor (m, d)
          `m` points with `d` dimensions for which the probability density will
          be calculated
        X : tensor (n, d), optional
          `n` points with `d` dimensions to which KDE will be fit. Provided to
          allow batch calculations in `log_prob`. By default, `X` is None and
          all points used to initialize KernelDensityEstimator are included.


        Returns
        -------
        log_probs : tensor (m)
          log probability densities for each of the queried points in `Y`
        """
        if X == None:
            X = self.X
        log_probs = torch.log(
            (self.bw**(-self.dims) *
             torch.exp(self.mvn.log_prob(
                 (X.unsqueeze(1) - Y) / self.bw))).sum(dim=0) / self.n)

        return log_probs

    def log_prob(self, Y):
        """Returns the total log probability of one or more points, `Y`, using
        a Multivariate Normal kernel fit to `X` and scaled using `bw`.

        Parameters
        ----------
        Y : tensor (m, d)
          `m` points with `d` dimensions for which the probability density will
          be calculated

        Returns
        -------
        log_prob : numeric
          total log probability density for the queried points, `Y`
        """

        X_chunks = self.X.split(1000)
        Y_chunks = Y.split(1000)

        log_prob = 0

        for x in X_chunks:
            for y in Y_chunks:
                log_prob += self.score_samples(y, x).sum(dim=0)

        return log_prob
    

class DifferentiableHistogram:
    """
    TODO fix this function
    
    """
    def __init__(self, x, bins=64, min_range=None, max_range=None, bandwidth=0.02):
        self.bins = bins
        self.bandwidth = bandwidth * (x.max() - x.min())
        
        if min_range is None or max_range is None:
            self.min_range = x.min()
            self.max_range = x.max()
        else:
            self.min_range = min_range
            self.max_range = max_range

        # Create bins
        self.bin_edges = torch.linspace(self.min_range, self.max_range, bins + 1).to(x.device)
        self.bin_centers = (self.bin_edges[:-1] + self.bin_edges[1:]) / 2.0
        
        # Compute histogram using Gaussian smoothing with bandwidth
        distances = (x.unsqueeze(1) - self.bin_centers.unsqueeze(0)) / self.bandwidth
        weights = torch.exp(-0.5 * (distances ** 2))
        histogram = weights.sum(dim=0)
        
        # Normalize to form a PDF
        self.pdf = histogram / histogram.sum()

        # Plot the histogram for validation
        plt.figure()
        plt.plot(self.bin_centers.float().cpu().numpy(), self.pdf.float().cpu().numpy())
        plt.title(f"PDF of token embeddings (shape = {x.shape})")
        plt.xlim(0, x.max().item()*1.1)
        plt.savefig(f"pdf_histogram_{int(time.time()*100)}.png")
        plt.close()

    def __call__(self, y):
        """
        Compute the negative log likelihood for a given sample y.
        Arguments:
        - y: Tensor of shape (m,) for which to compute the loss.
        Returns:
        - loss: Scalar representing the negative log likelihood of sample y.
        """
        y_distances = (y.unsqueeze(1) - self.bin_centers.unsqueeze(0)) / self.bandwidth
        y_weights = torch.exp(-0.5 * (y_distances ** 2))
        likelihoods = (self.pdf * y_weights).sum(dim=1)
        
        nll = -torch.log(likelihoods).mean()
        return nll


"""

Learned Notes on the token embeddings:
shape = 49410, 768
Computed means of [1, 768] = [0, 0, 0, 0,...]
Computed stds  of [1, 768] = [0.0139, 0.0139, 0.0139, ...]

Computed means of [49410, 1] = [0, 0, 0, 0,...]
Computed stds  of [49410, 1] = [0.0151, 0.0154, 0.0141, ...,  0.0396, 0.0150, 0.0148]

"""

