import os
from typing import Dict, List, Optional, Tuple

import random
import numpy as np
import pandas as pd
import gc
import PIL
import torch
import torch.utils.checkpoint
from diffusers import AutoencoderKL, DDPMScheduler, EulerDiscreteScheduler, UNet2DConditionModel, StableDiffusionPipeline, StableDiffusionXLPipeline
from PIL import Image
from safetensors import safe_open
from safetensors.torch import save_file
from torch.utils.data import Dataset
from transformers import AutoTokenizer, PretrainedConfig
import torch.nn.functional as F
import matplotlib.pyplot as plt
from .utils.seed import seed_everything

def pick_best_gpu_id():
    try:
        # pick the GPU with the most free memory:
        gpu_ids = [i for i in range(torch.cuda.device_count())]
        print(f"# of visible GPUs: {len(gpu_ids)}")
        gpu_mem = []
        for gpu_id in gpu_ids:
            free_memory, tot_mem = torch.cuda.mem_get_info(device=gpu_id)
            gpu_mem.append(free_memory)
            print("GPU %d: %d MB free" %(gpu_id, free_memory / 1024 / 1024))
        
        if len(gpu_ids) == 0:
            # no GPUs available, use CPU:
            os.environ["CUDA_VISIBLE_DEVICES"] = ""
            return None

        best_gpu_id = gpu_ids[np.argmax(gpu_mem)]
        # set this to be the active GPU:
        os.environ["CUDA_VISIBLE_DEVICES"] = str(best_gpu_id)
        print("Using GPU %d" %best_gpu_id)
        return best_gpu_id
    except Exception as e:
        print(f'Error picking best gpu: {e}')
        print(f'Falling back to GPU 0')
        os.environ["CUDA_VISIBLE_DEVICES"] = "0"
        return 0


def plot_torch_hist(parameters, step, checkpoint_dir, name, bins=100, min_val=-1, max_val=1, ymax_f = 0.75, color = 'blue'):
    os.makedirs(checkpoint_dir, exist_ok=True)

    # Flatten and concatenate all parameters into a single tensor
    all_params = torch.cat([p.data.view(-1) for p in parameters])
    norm = torch.norm(all_params)

    # Convert to CPU for plotting
    all_params_cpu = all_params.cpu().float().numpy()

    # Plot histogram
    plt.figure()
    plt.hist(all_params_cpu, bins=bins, density=False, color = color)
    plt.ylim(0, ymax_f * len(all_params_cpu.flatten()))
    plt.xlim(min_val, max_val)
    plt.xlabel('Weight Value')
    plt.ylabel('Count')
    plt.title(f'{name} (std: {np.std(all_params_cpu):.5f}, norm: {norm:.3f}, step {step:03d})')
    plt.savefig(f"{checkpoint_dir}/{name}_hist_{step:04d}.png")
    plt.close()

def plot_curve(value_dict, xlabel, ylabel, title, save_path, log_scale = False, y_lims = None):
    plt.figure()
    for key in value_dict.keys():
        values = value_dict[key]
        plt.plot(range(len(values)), values, label=key)

    if log_scale:
        plt.yscale('log')  # Set y-axis to log scale
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    if y_lims is not None:
        plt.ylim(y_lims[0], y_lims[1])
    plt.title(title)
    plt.legend()
    plt.savefig(save_path)
    plt.close()

# plot the learning rates:
def plot_lrs(lora_lrs, ti_lrs, save_path='learning_rates.png'):
    plt.figure()
    plt.plot(range(len(lora_lrs)), lora_lrs, label='LoRA LR')
    plt.plot(range(len(lora_lrs)), ti_lrs, label='TI LR')
    plt.yscale('log')  # Set y-axis to log scale
    plt.ylim(1e-6, 3e-3)
    plt.xlabel('Step')
    plt.ylabel('Learning Rate')
    plt.title('Learning Rate Curves')
    plt.legend()
    plt.savefig(save_path)
    plt.close()

# plot the learning rates:
def plot_grad_norms(grad_norms, save_path='grad_norms.png'):
    plt.figure()
    plt.plot(range(len(grad_norms['unet'])), grad_norms['unet'], label='unet')

    for i in range(2):
        try:
            plt.plot(range(len(grad_norms[f'text_encoder_{i}'])), grad_norms[f'text_encoder_{i}'], label=f'text_encoder_{i}')
        except:
            pass

    plt.yscale('log')  # Set y-axis to log scale
    plt.ylim(1e-6, 100.0)
    plt.xlabel('Step')
    plt.ylabel('Grad Norm')
    plt.title('Gradient Norms')
    plt.legend()
    plt.savefig(save_path)
    plt.close()

def plot_token_stds(token_std_dict, save_path='token_stds.png'):
    plt.figure()
    anchor_values = []
    for key in token_std_dict.keys():
        tokenizer_i_token_stds = token_std_dict[key]
        for i in range(len(tokenizer_i_token_stds)):
            stds = tokenizer_i_token_stds[i]
            if len(stds) == 0:
                continue
            anchor_values.append(stds[0])
            encoder_index = int(key.split('_')[-1])
            plt.plot(range(len(stds)), stds, label=f'{key}_tok_{i}', linestyle='dashed' if encoder_index > 0 else 'solid')

    plt.xlabel('Step')
    plt.ylabel('Token Embedding Std')
    centre_value = np.mean(anchor_values)
    up_f, down_f = 1.5, 1.25
    try:
        plt.ylim(centre_value/down_f, centre_value*up_f)
    except:
        pass
    plt.title('Token Embedding Std')
    plt.legend()
    plt.savefig(save_path)
    plt.close()

from scipy.signal import savgol_filter
def plot_loss(loss_dict, save_path='losses.png', window_length=31, polyorder=3):

    plt.figure()

    for key in loss_dict.keys():
        losses = loss_dict[key]
        smoothed_losses = [0]
        if len(losses) < window_length:
            continue
        
        smoothed_losses = savgol_filter(losses, window_length, polyorder)
        
        plt.plot(losses, label=key)
        plt.plot(smoothed_losses, label=f'Smoothed {key}', color='red')
        # plt.yscale('log')  # Uncomment if log scale is desired

    plt.xlabel('Step')
    plt.ylabel('Training Loss')
    plt.ylim(0, max(0.01, np.max(smoothed_losses)*1.4))
    plt.legend()
    plt.savefig(save_path)
    plt.close()


def prepare_image(
    pil_image: PIL.Image.Image, w: int = 512, h: int = 512, pipe=None,
) -> torch.Tensor:
    pil_image = pil_image.resize((w, h), resample=Image.BICUBIC, reducing_gap=1)
    image = pipe.image_processor.preprocess(pil_image)
    return image


def prepare_mask(
    pil_image: PIL.Image.Image, w: int = 512, h: int = 512
) -> torch.Tensor:
    pil_image = pil_image.resize((w, h), resample=Image.BICUBIC, reducing_gap=1)
    arr = np.array(pil_image.convert("L"))
    arr = arr.astype(np.float32) / 255.0
    arr = np.expand_dims(arr, 0)
    image = torch.from_numpy(arr).unsqueeze(0)
    return image


class PreprocessedDataset(Dataset):
    def __init__(
        self,
        csv_path: str,
        pipe,
        tokenizer_1,
        tokenizer_2,
        vae_encoder,
        text_encoder_1=None,
        text_encoder_2=None,
        do_cache: bool = False,
        size: int = 512,
        text_dropout: float = 0.0,
        aspect_ratio_bucketing: bool = False,
        train_batch_size: int = None,# required for aspect_ratio_bucketing
        substitute_caption_map: Dict[str, str] = {},
    ):
        super().__init__()
        self.data = pd.read_csv(csv_path)
        self.csv_path = csv_path

        self.caption = self.data["caption"]
        # make it lowercase
        self.caption = self.caption.str.lower()
        for key, value in substitute_caption_map.items():
            self.caption = self.caption.str.replace(key.lower(), value)

        self.image_path = self.data["image_path"]

        if "mask_path" not in self.data.columns:
            self.mask_path = None
        else:
            self.mask_path = self.data["mask_path"]

        self.pipe = pipe

        self.text_encoder_1 = text_encoder_1
        self.text_encoder_2 = text_encoder_2

        self.tokenizer_1 = tokenizer_1
        self.tokenizer_2 = tokenizer_2

        self.vae_encoder = vae_encoder
        self.vae_scaling_factor = self.vae_encoder.config.scaling_factor
        self.text_dropout = text_dropout
        self.size = size

        if do_cache:
            self.vae_latents = []
            self.tokens_tuple = []
            self.masks = []
            self.do_cache = True

            print("Caching latents, masks and token_IDs...\n")

            for idx in range(len(self.data)):
                token, vae_latent, mask = self._process(idx)
                self.vae_latents.append(vae_latent)
                self.tokens_tuple.append(token)
                self.masks.append(mask)

            print(f"\nCached latents masks and token_IDs for {len(self.vae_latents)} images.")
            del self.vae_encoder

        else:
            self.do_cache = False

        if aspect_ratio_bucketing:
            print("Using aspect ratio bucketing.")
            assert train_batch_size is not None, f"Please also provide a `train_batch_size` when you have set `aspect_ratio_bucketing == True`"
            from .utils.aspect_ratio_bucketing import BucketManager
            aspect_ratios = {}
            for idx in range(len(self.data)):
                aspect_ratios[idx] = Image.open(os.path.join(os.path.dirname(self.csv_path), self.image_path[idx])).size

            print(aspect_ratios)
            self.bucket_manager = BucketManager(
                aspect_ratios = aspect_ratios,
                bsz = train_batch_size,
                debug=True
            )
        else:
            print("Not using aspect ratio bucketing.")
            self.bucket_manager = None

    def get_aspect_ratio_bucketed_batch(self):
        assert self.bucket_manager is not None, f"Expected self.bucket_manager to not be None! In order to get an aspect ratio bucketed batch, please set aspect_ratio_bucketing = True and set a value for train_batch_size when doing __init__()"
        indices, resolution = self.bucket_manager.get_batch()

        print(f"Got bucket batch: {indices}, resolution: {resolution}")

        tok1, tok2, vae_latents, masks = [], [], [], []
        
        for idx in indices:

            if  self.tokenizer_2 is None:
                t1, v, m = self.__getitem__(idx = idx, bucketing_resolution=resolution)
            else:
                (t1, t2), v, m = self.__getitem__(idx = idx, bucketing_resolution=resolution)
                tok2.append(t2.unsqueeze(0))

            tok1.append(t1.unsqueeze(0))
            vae_latents.append(v.unsqueeze(0))
            masks.append(m.unsqueeze(0))

        tok1 = torch.cat(tok1, dim = 0)
        if  self.tokenizer_2 is None:
            pass
        else:
            tok2 = torch.cat(tok2, dim = 0)
        vae_latents = torch.cat(vae_latents, dim = 0)
        masks = torch.cat(masks, dim = 0)

        if self.tokenizer_2 is None:
            return (tok1, None), vae_latents, masks
        else:
            return (tok1, tok2), vae_latents, masks

    def __len__(self) -> int:
        return len(self.data)

    @torch.no_grad()
    def _process(
        self, idx: int, bucketing_resolution: tuple = None
    ) -> Tuple[Tuple[torch.Tensor, torch.Tensor], torch.Tensor, torch.Tensor]:
        image_path = self.image_path[idx]
        image_path = os.path.join(os.path.dirname(self.csv_path), image_path)
        image = PIL.Image.open(image_path).convert("RGB")
        if bucketing_resolution is None:
            image = prepare_image(image, w = self.size, h = self.size, pipe = self.pipe).to(
                dtype=self.vae_encoder.dtype, device=self.vae_encoder.device
            )
        else:
            image = prepare_image(image, w = bucketing_resolution[0], h = bucketing_resolution[1], pipe = self.pipe).to(
                dtype=self.vae_encoder.dtype, device=self.vae_encoder.device
            )

        caption = self.caption[idx]
        print(caption)

        # tokenizer_1
        ti1 = self.tokenizer_1(
            caption,
            padding="max_length",
            max_length=self.tokenizer_1.model_max_length,
            truncation=True,
            add_special_tokens=True,
            return_tensors="pt",
        ).input_ids.squeeze()

        if self.tokenizer_2 is None:
            ti2 = None
        else:
            ti2 = self.tokenizer_2(
                caption,
                padding="max_length",
                max_length=self.tokenizer_2.model_max_length,
                truncation=True,
                add_special_tokens=True,
                return_tensors="pt",
            ).input_ids.squeeze()

        vae_latent = self.vae_encoder.encode(image).latent_dist
        dummy_vae_latent = vae_latent.sample()

        if self.mask_path is None:
            mask = torch.ones_like(
                dummy_vae_latent, dtype=self.vae_encoder.dtype, device=self.vae_encoder.device
            )

        else:
            mask_path = self.mask_path[idx]
            mask_path = os.path.join(os.path.dirname(self.csv_path), mask_path)
            mask = PIL.Image.open(mask_path)
            mask = prepare_mask(mask, self.size, self.size).to(
                dtype=self.vae_encoder.dtype, device=self.vae_encoder.device
            )
            
            mask_dtype = mask.dtype
            mask = mask.float()
            mask = torch.nn.functional.interpolate(
                mask, size=(dummy_vae_latent.shape[-2], dummy_vae_latent.shape[-1]), mode="nearest"
            )
            mask = mask.to(dtype=mask_dtype)
            mask = mask.repeat(1, dummy_vae_latent.shape[1], 1, 1)

        assert len(mask.shape) == 4 and len(dummy_vae_latent.shape) == 4

        if ti2 is None: # sd15
            return [ti1], vae_latent, mask.squeeze()
        else: # sdxl
            return [ti1, ti2], vae_latent, mask.squeeze()

    def __getitem__(
        self, idx: int, bucketing_resolution:tuple = None
    ) -> Tuple[Tuple[torch.Tensor, torch.Tensor], torch.Tensor, torch.Tensor]:

        if self.do_cache:
            vae_latent = self.vae_latents[idx].sample() * self.vae_scaling_factor
            return self.tokens_tuple[idx], vae_latent.squeeze(), self.masks[idx]
        else: # This code pathway has not been tested in a long time and might be broken
            tokens, vae_latent, mask = self._process(idx, bucketing_resolution=bucketing_resolution)
            vae_latent = vae_latent.sample() * self.vae_scaling_factor
            return tokens, vae_latent.squeeze(), mask




class TokenEmbeddingsHandler:
    def __init__(self, text_encoders, tokenizers):
        self.text_encoders = text_encoders
        self.tokenizers = tokenizers

        self.train_ids: Optional[torch.Tensor] = None
        self.inserting_toks: Optional[List[str]] = None
        self.embeddings_settings = {}

    def get_trainable_embeddings(self):
        return self.get_embeddings_and_tokens(self.train_ids)

    def get_embeddings_and_tokens(self, indices):
        """
        Get the embeddings and tokens for the given indices
        """
        embeddings, tokens = {}, {}
        for idx, text_encoder in enumerate(self.text_encoders):
            if text_encoder is None:
                continue

            embeddings['txt_encoder_%d' % idx] = []
            tokens['txt_encoder_%d' % idx] = []

            for index in indices:
                token_embedding = text_encoder.text_model.embeddings.token_embedding.weight.data[index]
                embeddings['txt_encoder_%d' % idx].append(token_embedding)

                # also get the corresponding token for this embedding id:
                token = self.tokenizers[idx].convert_ids_to_tokens([index])[0]
                tokens['txt_encoder_%d' % idx].append(token)

        return embeddings, tokens

    def visualize_random_token_embeddings(self, output_dir, n = 6, token_list = None):
        """
        Visualize the embeddings of n random tokens from each text encoder
        """
        if token_list is not None:
            # Convert tokens to indices using the first tokenizer
            indices = self.tokenizers[0].convert_tokens_to_ids(token_list)
        else:
            # Randomly select indices
            n_tokens = len(self.text_encoders[0].text_model.embeddings.token_embedding.weight.data)
            indices = np.random.randint(0, n_tokens, n)

        embeddings, tokens = self.get_embeddings_and_tokens(indices)

        # Visualize the embeddings:
        for idx, text_encoder in enumerate(self.text_encoders):
            if text_encoder is None:
                continue
            for i in range(n):
                token = tokens[f'txt_encoder_{idx}'][i]
                # Strip any backslashes from the token name:
                token = token.replace("/", "_")
                embedding = embeddings[f'txt_encoder_{idx}'][i]
                plot_torch_hist(embedding, 0, os.path.join(output_dir, 'ti_embeddings') , f"frozen_enc_{idx}_tokid_{i}: {token}", min_val=-0.05, max_val=0.05, ymax_f = 0.05, color = 'green')

    def find_nearest_tokens(self, query_embedding, tokenizer, text_encoder, idx, distance_metric, top_k = 5):
        # given a query embedding, compute the distance to all embeddings in the text encoder
        # and return the top_k closest tokens

        assert distance_metric in ["l2", "cosine"], "distance_metric should be either 'l2' or 'cosine'"
        
        # get all non-optimized embeddings:
        index_no_updates = self.embeddings_settings[f"index_no_updates_{idx}"]
        embeddings = text_encoder.text_model.embeddings.token_embedding.weight.data[index_no_updates]

        # compute the distance between the query embedding and all embeddings:
        if distance_metric == "l2":
            diff = (embeddings - query_embedding.unsqueeze(0))**2
            distances = diff.sum(-1)
            distances, indices = torch.topk(distances, top_k, dim=0, largest=False)
        elif distance_metric == "cosine":
            distances = F.cosine_similarity(embeddings, query_embedding.unsqueeze(0), dim=-1)
            distances, indices = torch.topk(distances, top_k, dim=0, largest=True)

        nearest_tokens = tokenizer.convert_ids_to_tokens(indices)
        return nearest_tokens, distances
        

    def print_token_info(self, distance_metric = "cosine"):
        print(f"----------- Closest tokens (distance_metric = {distance_metric}) --------------")
        current_token_embeddings, current_tokens = self.get_trainable_embeddings()
        idx = 0

        for tokenizer, text_encoder in zip(self.tokenizers, self.text_encoders):
            if text_encoder is None:
                idx += 1
                continue

            query_embeddings = current_token_embeddings[f'txt_encoder_{idx}']
            query_tokens = current_tokens[f'txt_encoder_{idx}']

            for token_id, query_embedding in enumerate(query_embeddings):
                nearest_tokens, distances = self.find_nearest_tokens(query_embedding, tokenizer, text_encoder, idx, distance_metric)

                # print the results:
                print(f"txt-encoder {idx}, token {token_id}: {query_tokens[token_id]}:")
                for i, (token, dist) in enumerate(zip(nearest_tokens, distances)):
                    print(f"---> {distance_metric} of {dist:.4f}: {token}")

            idx += 1

    def get_start_embedding(self, text_encoder, tokenizer, example_tokens, unk_token_id = 49407, verbose = False, desired_std_multiplier = 0.0):
        print('-----------------------------------------------')
        # do some cleanup:
        example_tokens = [tok.lower() for tok in example_tokens]
        example_tokens = list(set(example_tokens))

        starting_ids = tokenizer.convert_tokens_to_ids(example_tokens)

        # filter out any tokens that are mapped to unk_token_id:
        example_tokens = [tok for tok, tok_id in zip(example_tokens, starting_ids) if tok_id != unk_token_id]
        starting_ids = [tok_id for tok_id in starting_ids if tok_id != unk_token_id]

        if verbose:
            print("Token mapping:")
            for i, token in enumerate(example_tokens):
                print(f"{token} -> {starting_ids[i]}")

        embeddings, stds = [], []
        for i, token_index in enumerate(starting_ids):
            embedding = text_encoder.text_model.embeddings.token_embedding.weight.data[token_index].clone()
            embeddings.append(embedding)
            stds.append(embedding.std())
            #print(f"token: {example_tokens[i]}, embedding-std: {embedding.std():.4f}, embedding-mean: {embedding.mean():.4f}")

        embeddings = torch.stack(embeddings)
        #print(f"Embeddings: {embeddings.shape}, std: {embeddings.std():.4f}, mean: {embeddings.mean():.4f}")

        if verbose:
            # Compute the squared difference
            squared_diff = (embeddings.unsqueeze(1) - embeddings.unsqueeze(0)) ** 2
            squared_l2_dist = squared_diff.sum(-1)
            l2_distance_matrix = torch.sqrt(squared_l2_dist)

            print("Pairwise L2 Distance Matrix:")
            print(" \t" + "\t".join(example_tokens))
            for i, row in enumerate(l2_distance_matrix):
                print(f"{example_tokens[i]}\t" + "\t".join(f"{dist:.4f}" for dist in row))


        # We're working in cosine-similarity space
        # So first, renormalize the embeddings to have norm 1
        embedding_norms = torch.norm(embeddings, dim=-1, keepdim=True)
        embeddings = embeddings / embedding_norms

        print(f"embedding norms pre normalization:")
        print(embedding_norms)
        print(f"embedding norms post normalization:")
        print(torch.norm(embeddings, dim=-1, keepdim=True))

        print(f"Using {len(embeddings)} embeddings to compute initial embedding...")
        init_embedding = embeddings.mean(dim=0)
        # normalize the init_embedding to have norm 1:
        init_embedding = init_embedding / torch.norm(init_embedding)

        # rescale the init_embedding to have the same std as the average of the embeddings:
        init_embedding = init_embedding * embedding_norms.mean()

        print(f"init_embedding norm: {torch.norm(init_embedding):.4f}, std: {init_embedding.std():.4f}, mean: {init_embedding.mean():.4f}")

        if (desired_std_multiplier is not None) and desired_std_multiplier > 0:
            avg_std        = torch.stack(stds).mean()
            current_std    = init_embedding.std()
            scale_factor   = desired_std_multiplier * avg_std / current_std
            init_embedding = init_embedding * scale_factor
            print(f"Scaled Mean Embedding: std: {init_embedding.std():.4f}, mean: {init_embedding.mean():.4f}")
        
        return init_embedding

    def plot_token_embeddings(self, example_tokens, output_folder = ".", x_range = [-0.05, 0.05]):
        print(f"Plotting embeddings for tokens: {example_tokens}")

        idx = 0
        for tokenizer, text_encoder in zip(self.tokenizers, self.text_encoders):
            if tokenizer is None:
                idx += 1
                continue
                
            token_ids  = tokenizer.convert_tokens_to_ids(example_tokens)
            embeddings = text_encoder.text_model.embeddings.token_embedding.weight.data[token_ids].clone()

            # plot the embeddings histogram:
            for token_name, embedding in zip(example_tokens, embeddings):
                plot_torch_hist(embedding, 0, output_folder, f"tok_{token_name}_{idx}", bins=100, min_val=x_range[0], max_val=x_range[1], ymax_f = 0.05)

            idx += 1

    @property
    def dtype(self):
        return self.text_encoders[0].dtype

    def initialize_new_tokens(self, 
        inserting_toks: List[str],
        starting_toks:  Optional[List[str]] = None,
        seed: int = 0,
    ):
        assert isinstance(
            inserting_toks, list
        ), "inserting_toks should be a list of strings."
        assert all(
            isinstance(tok, str) for tok in inserting_toks
        ), "All elements in inserting_toks should be strings."

        print(f"Initializing new tokens: {inserting_toks}")
        self.inserting_toks = inserting_toks

        seed_everything(seed)
        idx = 0
        for tokenizer, text_encoder in zip(self.tokenizers, self.text_encoders):
            if tokenizer is None:
                idx += 1
                continue

            print(f"Inserting new tokens into tokenizer-{idx}:")
            print(self.inserting_toks)

            special_tokens_dict = {"additional_special_tokens": self.inserting_toks}
            tokenizer.add_special_tokens(special_tokens_dict)
            text_encoder.resize_token_embeddings(len(tokenizer))

            self.train_ids = tokenizer.convert_tokens_to_ids(self.inserting_toks)

            # random initialization of new tokens
            std_token_embedding = (
                text_encoder.text_model.embeddings.token_embedding.weight.data.std(dim=1).mean()
            )
            print(f"Text encoder {idx} token_embedding_std:  {std_token_embedding}")

            if starting_toks is not None:
                assert len(starting_toks) == len(self.inserting_toks), "starting_toks should have the same length as inserting_toks"
                self.starting_ids = tokenizer.convert_tokens_to_ids(starting_toks)
                print(f"Copying embeddings from starting tokens {starting_toks} to new tokens {self.inserting_toks}")
                print(f"Starting ids: {self.starting_ids}")
                # copy the embeddings of the starting tokens to the new tokens
                text_encoder.text_model.embeddings.token_embedding.weight.data[
                    self.train_ids] = text_encoder.text_model.embeddings.token_embedding.weight.data[self.starting_ids].clone()

            else:
                std_multiplier = 1.0
                init_embeddings = (torch.randn(len(self.train_ids), text_encoder.text_model.config.hidden_size).to(device=self.device).to(dtype=self.dtype) * std_token_embedding)
                init_embeddings *= std_multiplier
                
                # clamp the maximum value of the new embeddings to 2*std_token_embedding
                #init_embeddings = torch.clamp(init_embeddings, -2*std_token_embedding, 2*std_token_embedding)
                # renormalize the embeddings to have std = std_token_embedding
                #init_embeddings = init_embeddings / init_embeddings.std(dim=1, keepdim=True) * std_token_embedding

                text_encoder.text_model.embeddings.token_embedding.weight.data[self.train_ids] = init_embeddings.clone()

            self.embeddings_settings[
                f"original_embeddings_{idx}"
            ] = text_encoder.text_model.embeddings.token_embedding.weight.data.clone()
            self.embeddings_settings[f"std_token_embedding_{idx}"] = std_token_embedding

            inu = torch.ones((len(tokenizer),), dtype=torch.bool)
            inu[self.train_ids] = False

            self.embeddings_settings[f"index_no_updates_{idx}"] = inu

            idx += 1

    def pre_optimize_token_embeddings(self, train_dataset, epochs=10):

        ### THIS FUNCTION IS NOT DONE YET
        ### Idea here is to use CLIP-similarity between imgs and prompts to pre-optimize the embeddings

        for idx in range(len(train_dataset)):
            (tok1, tok2), vae_latent, mask = train_dataset[idx]
            image_path = train_dataset.image_path[idx]
            image_path = os.path.join(os.path.dirname(train_dataset.csv_path), image_path)
            image = PIL.Image.open(image_path).convert("RGB")

            print(f"---> Loaded sample {idx}:")
            print("Tokens:")
            print(tok1.shape)
            print(tok2.shape)
            print("Image:")
            print(image.size)

            # tokens to text embeds
            prompt_embeds_list = []
            #for tokenizer, text_encoder in zip(self.tokenizers, self.text_encoders):
            for tok, text_encoder in zip((tok1, tok2), self.text_encoders):
                prompt_embeds_out = text_encoder(
                    tok.to(text_encoder.device),
                    output_hidden_states=True,
                )

                print("prompt_embeds_out:")
                print(prompt_embeds_out.shape)

                pooled_prompt_embeds = prompt_embeds_out[0]
                prompt_embeds = prompt_embeds_out.hidden_states[-2]
                bs_embed, seq_len, _ = prompt_embeds.shape
                prompt_embeds = prompt_embeds.view(bs_embed, seq_len, -1)
                prompt_embeds_list.append(prompt_embeds)

            prompt_embeds = torch.concat(prompt_embeds_list, dim=-1)
            pooled_prompt_embeds = pooled_prompt_embeds.view(bs_embed, -1)

            print("prompt_embeds:")
            print(prompt_embeds.shape)
            print("pooled_prompt_embeds:")
            print(pooled_prompt_embeds.shape)

    def save_embeddings(self, file_path: str, txt_encoder_keys = ["clip_l", "clip_g"]):
        assert (
            self.train_ids is not None
        ), "Initialize new tokens before saving embeddings."
        tensors = {}
        for idx, text_encoder in enumerate(self.text_encoders):
            if text_encoder is None:
                continue
            assert text_encoder.text_model.embeddings.token_embedding.weight.data.shape[
                0
            ] == len(self.tokenizers[0]), "Tokenizers should be the same."
            new_token_embeddings = (
                text_encoder.text_model.embeddings.token_embedding.weight.data[
                    self.train_ids
                ]
            )
            tensors[txt_encoder_keys[idx]] = new_token_embeddings

        save_file(tensors, file_path)

    @property
    def device(self):
        return self.text_encoders[0].device

    def _compute_off_ratio(self, idx):
        # compute the off-std-ratio for the embeddings, to be used for regularization rescaling

        text_encoder = self.text_encoders[idx]
        if text_encoder is None:
            off_ratio = 1.0
            new_embeddings = None
        else:
            index_no_updates    = self.embeddings_settings[f"index_no_updates_{idx}"]
            std_token_embedding = self.embeddings_settings[f"std_token_embedding_{idx}"].float()
            index_updates = ~index_no_updates
            new_embeddings = text_encoder.text_model.embeddings.token_embedding.weight.data[index_updates]#.float()
            new_stds = new_embeddings.std(dim=1)
            off_ratio = std_token_embedding / new_stds.mean().float()

        return off_ratio.float(), new_embeddings

    def fix_embedding_std(self, off_ratio_power = 0.1):
        if off_ratio_power == 0.0:
            return

        std_penalty = 0.0
        idx = 0

        for tokenizer, text_encoder in zip(self.tokenizers, self.text_encoders):
            if text_encoder is None:
                idx += 1
                continue

            index_no_updates = self.embeddings_settings[f"index_no_updates_{idx}"]
            index_updates = ~index_no_updates

            off_ratio, new_embeddings = self._compute_off_ratio(idx)
            std_penalty += (off_ratio - 1.0)**2

            if (off_ratio < 0.95) or (off_ratio > 1.05):
                print(f"WARNING: std-off ratio-{idx} (target-std / embedding-std) = {off_ratio:.4f}, prob not ideal...")                

            # rescale the embeddings to have a more similar std as before:
            new_embeddings = new_embeddings * (off_ratio**off_ratio_power)
            text_encoder.text_model.embeddings.token_embedding.weight.data[
                    index_updates
                ] = new_embeddings.to(device=text_encoder.device).to(dtype=text_encoder.dtype)

            idx += 1


    @torch.no_grad()
    def retract_embeddings(self, print_stds = False):
        idx = 0
        means, stds = [], []

        for tokenizer, text_encoder in zip(self.tokenizers, self.text_encoders):
            if text_encoder is None:
                idx += 1
                continue

            index_no_updates = self.embeddings_settings[f"index_no_updates_{idx}"]
            text_encoder.text_model.embeddings.token_embedding.weight.data[
                index_no_updates
            ] = (
                self.embeddings_settings[f"original_embeddings_{idx}"][index_no_updates]
                .to(device=text_encoder.device)
                .to(dtype=text_encoder.dtype)
            )

            # for the parts that were updated, we can normalize them a bit
            # to have the same std as before
            std_token_embedding = self.embeddings_settings[f"std_token_embedding_{idx}"]

            index_updates = ~index_no_updates
            new_embeddings = (
                text_encoder.text_model.embeddings.token_embedding.weight.data[
                    index_updates
                ]
            )

            idx += 1

            if 0:
                # get the actual embeddings that will get updated:
                inu = torch.ones((len(tokenizer),), dtype=torch.bool)
                inu[self.train_ids] = False
                updateable_embeddings = text_encoder.text_model.embeddings.token_embedding.weight.data[~inu].detach().clone().to(dtype=torch.float32).cpu().numpy()
                
                mean_0, mean_1 = updateable_embeddings[0].mean(), updateable_embeddings[1].mean()
                std_0, std_1 = updateable_embeddings[0].std(), updateable_embeddings[1].std()

                means.append((mean_0, mean_1))
                stds.append((std_0, std_1))

                if print_stds:
                    print(f"Text Encoder {idx} token embeddings:")
                    print(f" --- Means: ({mean_0:.6f}, {mean_1:.6f})")
                    print(f" --- Stds:  ({std_0:.6f}, {std_1:.6f})")

    def _load_embeddings(self, loaded_embeddings, tokenizer, text_encoder):
        # Assuming new tokens are of the format <s_i>
        self.inserting_toks = [f"<s{i}>" for i in range(loaded_embeddings.shape[0])]
        special_tokens_dict = {"additional_special_tokens": self.inserting_toks}
        tokenizer.add_special_tokens(special_tokens_dict)
        text_encoder.resize_token_embeddings(len(tokenizer))

        self.train_ids = tokenizer.convert_tokens_to_ids(self.inserting_toks)
        assert self.train_ids is not None, "New tokens could not be converted to IDs."
        text_encoder.text_model.embeddings.token_embedding.weight.data[
            self.train_ids
        ] = loaded_embeddings.to(device=self.device).to(dtype=self.dtype)

    def load_embeddings(self, file_path: str, txt_encoder_keys = ["clip_l", "clip_g"]):
        if not os.path.exists(file_path):
            file_path = file_path.replace(".pti", ".safetensors")
            if not os.path.exists(file_path):
                raise FileNotFoundError(f"{file_path} does not exist.")

        with safe_open(file_path, framework="pt", device=self.device.type) as f:
            for idx in range(len(self.text_encoders)):
                text_encoder = self.text_encoders[idx]
                tokenizer = self.tokenizers[idx]
                if text_encoder is None:
                    continue
                try:
                    loaded_embeddings = f.get_tensor(txt_encoder_keys[idx])
                except:
                    loaded_embeddings = f.get_tensor(f"text_encoders_{idx}")
                self._load_embeddings(loaded_embeddings, tokenizer, text_encoder)