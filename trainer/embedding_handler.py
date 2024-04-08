
import os
import torch
import torch.nn.functional as F
import numpy as np
import PIL
from typing import List, Optional, Dict
from safetensors.torch import save_file, safe_open

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
            image_path = os.path.join(train_dataset.data_dir, image_path)
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