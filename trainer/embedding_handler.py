
import os
import torch
import torch.nn.functional as F
import numpy as np
import PIL
from tqdm import tqdm
from typing import List, Optional, Dict
from safetensors.torch import save_file, safe_open
import matplotlib.pyplot as plt
from trainer.utils.utils import seed_everything, plot_torch_hist

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
                text_encoder.text_model.embeddings.token_embedding.weight.data[self.train_ids] = init_embeddings.clone()

            self.embeddings_settings[
                f"original_embeddings_{idx}"
            ] = text_encoder.text_model.embeddings.token_embedding.weight.data.clone()
            self.embeddings_settings[f"std_token_embedding_{idx}"] = std_token_embedding

            inu = torch.ones((len(tokenizer),), dtype=torch.bool)
            inu[self.train_ids] = False
            self.embeddings_settings[f"index_no_updates_{idx}"] = inu

            idx += 1

    def plot_tokenid(self, token_id, suffix = '', output_folder = ".", x_range = [-0.05, 0.05]):
        idx = 0
        for tokenizer, text_encoder in zip(self.tokenizers, self.text_encoders):
            if tokenizer is None:
                idx += 1
                continue
            
            embeddings = text_encoder.text_model.embeddings.token_embedding.weight.data[token_id].clone()
            plot_torch_hist(embeddings, 0, output_folder, f"tok_{token_id}_{idx}_{suffix}", bins=100, min_val=x_range[0], max_val=x_range[1], ymax_f = 0.05)
            idx += 1

    def encode_text(self, text, clip_skip = None):
        prompt_embeds_list = []

        for tokenizer, text_encoder in zip(self.tokenizers, self.text_encoders):
            if text_encoder is None:
                continue
            text_input_ids = tokenizer(
                text,
                padding="max_length",
                max_length=tokenizer.model_max_length,
                truncation=True,
                add_special_tokens=True,
                return_tensors="pt",
            ).input_ids

            prompt_embeds = text_encoder(text_input_ids.to(text_encoder.device), output_hidden_states=True)
            
            # We are only ALWAYS interested in the pooled output of the final text encoder
            pooled_prompt_embeds = prompt_embeds[0]
            if clip_skip is None:
                prompt_embeds = prompt_embeds.hidden_states[-2]
            else: # "2" because SDXL always indexes from the penultimate layer.
                prompt_embeds = prompt_embeds.hidden_states[-(clip_skip + 2)]

            prompt_embeds_list.append(prompt_embeds)

        prompt_embeds = torch.concat(prompt_embeds_list, dim=-1)

        return prompt_embeds, pooled_prompt_embeds

    def pre_optimize_token_embeddings(self, config):
        """
        Warmup the token embeddings by optimizing them without using the image denoiser,
        but simply using CLIP-txt and CLIP-img similarity losses

        TODO: add CLIP-img similarity loss into this mix
        --> This requires loading the img-encoder part for each of the txt-encoders and figuring out the correct projection layer
        """
        target_prompt = config.training_attributes["gpt_description"]
        
        if config.token_warmup_steps <= 0 or not target_prompt:
            print("Skipping token embedding warmup.")
            return

        # This is the concept description from chatgpt:
        target_prompt_embeds, target_pooled_prompt_embeds = self.encode_text(target_prompt)
        print(f'Warming up token embeddings with prompt: {target_prompt}...')

        # Setup the token optimizer:
        ti_parameters = []
        for text_encoder in self.text_encoders:
            if text_encoder is not None:
                text_encoder.train()
                for name, param in text_encoder.named_parameters():
                    if "token_embedding" in name:
                        param.requires_grad = True
                        ti_parameters.append(param)
        
        params_to_optimize_ti = [{
                "params": ti_parameters,
                "lr": config.ti_lr,
                "weight_decay": config.ti_weight_decay,
            }]

        optimizer_ti = torch.optim.AdamW(
                params_to_optimize_ti,
                weight_decay=config.ti_weight_decay,
            )

        token_string = config.token_dict["TOK"]
        prompt_template = [
            '{}',
            '{}',
            '{}',
            'a {}',
            '{} image',
            'a picture of {}',
        ]

        #condtioning_regularizer = ConditioningRegularizer(config, self)
        #loss, prompt_embeds_norms = condtioning_regularizer.apply_regularization(loss, prompt_embeds_norms, prompt_embeds, pipe)

        losses = []
        for step in tqdm(range(config.token_warmup_steps)):
            if step % 30 == 0 and config.debug and 0: # disalbe this for now
                for i, token_index in enumerate(self.train_ids):
                    self.plot_tokenid(token_index, suffix = f'token_{i}_{step}', output_folder = f'{config.output_dir}/token_opt')

            # pick a random prompt template and inject the token string:
            prompt_to_optimize = np.random.choice(prompt_template).format(token_string)
            prompt_embeds, pooled_prompt_embeds = self.encode_text(prompt_to_optimize)

            # compute the loss:
            embeds_l2_loss = F.mse_loss(prompt_embeds, target_prompt_embeds)
            pooled_embeds_l2_loss = F.mse_loss(pooled_prompt_embeds, target_pooled_prompt_embeds)
            loss = 4.0 * embeds_l2_loss + pooled_embeds_l2_loss

            # Backward pass:
            retain_graph = step < (config.token_warmup_steps - 1)  # Retain graph for all but the last step
            loss.backward(retain_graph=retain_graph)
            losses.append(loss.item())

            # zero out the gradients of the non-trained text-encoder embeddings
            for embedding_tensor in ti_parameters:
                embedding_tensor.grad.data[:-config.n_tokens, : ] *= 0.

            optimizer_ti.step()
            self.fix_embedding_std(config.off_ratio_power)
            optimizer_ti.zero_grad()

        if config.debug:
            # Plot the losses:
            plt.plot(losses)
            plt.xlabel("Step")
            plt.ylabel("Loss")
            plt.ylim(0, np.max(losses) * 1.1)
            plt.title("Token Embedding Warmup Loss")
            plt.savefig(f'{config.output_dir}/token_warmup_loss.png')
            plt.close()


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