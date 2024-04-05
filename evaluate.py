import argparse
from trainer.utils.inference import render_images_eval
from trainer.utils.json_stuff import save_as_json
from trainer.utils.config_modification import modify_args_based_on_concept_mode
from trainer.config import TrainingConfig
from trainer.models import pretrained_models
import clip
from PIL import Image
import torch
import numpy as np
import os
from creator_lora.models.resnet50 import ResNet50MLP

"""
todos:
    - aesthetic scoring (need aesthetic scoring model checkpoint!!)
    - run eval on user-defined captions
"""

device = "cuda" if torch.cuda.is_available() else "cpu"

def filter_prompt(prompt, remove_this = "in the style of <s0><s1>,", replace_with = ""):
    assert  remove_this in prompt, f"Expected '{remove_this}' to be present in the prompt: '{prompt}'"
    return prompt.replace(
        remove_this,
        replace_with
    )

def get_similarity_matrix(a, b, eps=1e-8):
    """
    finds the cosine similarity matrix between each item of a w.r.t each item of b
    a and b are expected to be 2 dimensional
    added eps for numerical stability
    source: https://stackoverflow.com/a/58144658
    """
    a_n, b_n = a.norm(dim=1)[:, None], b.norm(dim=1)[:, None]
    a_norm = a / torch.max(a_n, eps * torch.ones_like(a_n))
    b_norm = b / torch.max(b_n, eps * torch.ones_like(b_n))
    sim_mt = torch.mm(a_norm, b_norm.transpose(0, 1))
    return sim_mt

class Evaluation:
    def __init__(self, image_filenames: list):
        self.image_filenames = image_filenames
        self.image_features = None

    def obtain_image_features(self):

        if self.image_features is None:
            all_image_features  = []
            model, preprocess = clip.load("ViT-B/32", device=device)

            for f in self.image_filenames:
                image = preprocess(Image.open(f)).unsqueeze(0).to(device)
                with torch.no_grad():
                    image_features = model.encode_image(image)
                    all_image_features.append(image_features.float())

            all_image_features = torch.cat(all_image_features, dim = 0)
            self.image_features = all_image_features

        return self.image_features

    def obtain_text_features(self, prompts: list, device):
        model, preprocess = clip.load("ViT-B/32", device=device)
        text = clip.tokenize(prompts).to(device)

        with torch.no_grad():
            text_features = model.encode_text(text)
        return text_features

    def image_text_alignment(self, device, prompts: list):

        image_features = self.obtain_image_features().to(device)
        assert image_features.shape[0] == len(prompts), f'Expected len(prompts) ({len(prompts)}) to have the same number of prompts as the number of images provided: {image_features.shape}'
        text_features = self.obtain_text_features(prompts=prompts, device=device)
        cossim = torch.nn.functional.cosine_similarity(
            text_features, image_features, dim = -1
        ).mean().item()
        return  cossim

    def clip_diversity(self, device: str):

        all_image_features = self.obtain_image_features().to(device)

        distances = 1 - get_similarity_matrix(all_image_features, all_image_features)
        assert distances.shape == (
            all_image_features.shape[0],
            all_image_features.shape[0]
        ), f'Expected the shape of the distance matrix to be (num_images, num_images) i.e {(all_image_features.shape[0], all_image_features.shape[0])} but got: {distances.shape}'
        distances = distances.detach().cpu().numpy()
        # Get the upper triangle:
        upper_triangle = np.triu(distances, k=1).flatten()
        # Drop distances from imgs that are super super similar:
        upper_triangle = upper_triangle[upper_triangle >= 0.2]
        return upper_triangle.mean().item()

    def aesthetic_score(self, device: str, checkpoint_path: str):
        # assert os.path.exists(checkpoint_path), f"invalid checkpoint_path: {checkpoint_path}"
        model = ResNet50MLP(
            model_path=checkpoint_path,
            device = device
        )

        scores = []
        for f in self.image_filenames:
            score = model.predict_score(pil_image=Image.open(f))
            scores.append(score)

        return sum(scores)/len(scores)

def parse_arguments():
    parser = argparse.ArgumentParser(description="Script for generating images based on prompts and computing similarities.")
    

    parser.add_argument("--config_filename", type=str, required=True, default = "sdxl", help="path to config json file")
    parser.add_argument("--lora_path", type=str, required=True,
                        help="Path to LoRa.")
    parser.add_argument("--output_json", type=str, required=True,
                        help="Path to json where we save result values")
    parser.add_argument("--output_folder", type=str, required=True,
                        help="style or face")
    args = parser.parse_args()
    return args

args = parse_arguments()

os.system(f"mkdir -p {args.output_folder}")

config = TrainingConfig.from_json(args.config_filename)
config = modify_args_based_on_concept_mode(config)


image_filenames, prompts = render_images_eval(
    output_folder=args.output_folder,
    concept_mode=config.concept_mode,
    render_size=(1024,1024),
    lora_path=args.lora_path,
    pretrained_model=pretrained_models[config.sd_model_version],
    seed=0,
    is_lora = True,
    trigger_text='TOK' if config.concept_mode != "style" else ", in the style of TOK"
)

if config.concept_mode == "style":
    prompts = [
        filter_prompt(
            x, 
            remove_this="in the style of <s0><s1>,", 
            replace_with=""
        ) 
        for x in prompts
    ]
elif config.concept_mode == "face":
    prompts = [
        filter_prompt(
            x, 
            remove_this="<s0><s1>", 
            replace_with="face"
        ) 
        for x in prompts
    ]
elif config.concept_mode == "object":
    prompts = [
        filter_prompt(
            x, 
            remove_this="<s0><s1>", 
            replace_with="object"
        ) 
        for x in prompts
    ]
else:
    print(prompts)
    raise NotImplementedError("prompt filtering for other concept modes is not implemented")

print(f"Eval prompts:")
for p in prompts:
    print(p)
print(f"\n\n\n")
eval = Evaluation(image_filenames=image_filenames)
clip_diversity = eval.clip_diversity(device=device)

# TODO: replace checkpoint_path=None with the correct checkpoint path
aesthetic_score = eval.aesthetic_score(device=device, checkpoint_path=None)
image_text_alignment = eval.image_text_alignment(device=device, prompts=prompts)

result = {
    "sd_model_version": config.sd_model_version,
    "lora_path": os.path.abspath(args.lora_path),
    "concept_mode": config.concept_mode,
    "output_folder": args.output_folder,
    "scores": {
        "clip_diversity": clip_diversity,
        "aesthetic_score": aesthetic_score,
        "image_text_alignment": image_text_alignment
    }
}

save_as_json(
    dictionary_or_list=result,
    filename=args.output_json
)

"""
Example commands:

python3 evaluate.py  --output_folder eval_images --lora_path lora_models/without_cog_gene--05_00-11-23-sdxl_face_dora/checkpoints/checkpoint-600  --output_json eval_results.json --config_filename training_args_gene.json

python3 evaluate.py  --output_folder eval_images --lora_path lora_models/banny--04_23-53-12-sdxl_object_dora/checkpoints/checkpoint-600  --output_json eval_results_banny.json --config_filename training_args_banny.json

python3 evaluate.py  --output_folder eval_images --lora_path lora_models/clipx_tiny_dora_only---sdxl_style_dora/checkpoints/checkpoint-600  --output_json eval_results_style.json --config_filename training_args.json
"""