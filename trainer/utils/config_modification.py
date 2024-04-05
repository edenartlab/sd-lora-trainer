from ..config import TrainingConfig

def modify_args_based_on_concept_mode(config: TrainingConfig):
    if config.concept_mode == "face":
        print(f"Face mode is active ----> disabling left-right flips and setting mask_target_prompts to 'face'.")
        config.left_right_flip_augmentation = False  # always disable lr flips for face mode!
        config.mask_target_prompts = "face"
        config.clipseg_temperature = 0.4

    if config.concept_mode == "concept": # gracefully catch any old versions of concept_mode
        config.concept_mode = "object"

    if config.concept_mode == "style": # for styles you usually want the LoRA matrices to absorb a lot (instead of just the token embedding)
        config.l1_penalty = 0.05

    if config.use_dora:
        print(f"Disabling L1 penalty and LoRA weight decay for DORA training.")
        config.l1_penalty = 0.0
        config.lora_weight_decay = 0.0


    # build the inserting_list_tokens and token dict using n_tokens:
    inserting_list_tokens = [f"<s{i}>" for i in range(config.n_tokens)]
    config.inserting_list_tokens = inserting_list_tokens
    config.token_dict = {"TOK": "".join(inserting_list_tokens)}

    return config