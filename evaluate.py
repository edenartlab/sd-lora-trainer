import argparse
from trainer.utils.inference import render_images_eval
from trainer.utils.json_stuff import save_as_json
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
    - replace <s0><s1> with mask_target_prompts in image-text alignment
    - cosine similarity for diversity score
    - run eval on user-defined captions
"""

device = "cuda" if torch.cuda.is_available() else "cpu"

def filter_style_prompt(prompt, remove_this = "in the style of <s0><s1>,"):
    assert  remove_this in prompt, f"Expected '{remove_this}' to be present in the prompt: '{prompt}'"
    return prompt.replace(
        remove_this,
        ""
    )

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
        raise cossim 

    def clip_diversity(self, device: str):

        all_image_features = self.obtain_image_features().to(device)
        distances = torch.cdist(all_image_features, all_image_features, p=2.0)
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
    
    parser.add_argument("--sd_model_version", type=str, required=True, default = "sdxl", help="sdxl or sd15")
    parser.add_argument("--lora_path", type=str, required=True,
                        help="Path to LoRa.")
    parser.add_argument("--output_json", type=str, required=True,
                        help="Path to json where we save result values")
    parser.add_argument("--concept_mode", type=str, required=True,
                        help="style or face")
    parser.add_argument("--output_folder", type=str, required=True,
                        help="style or face")
    args = parser.parse_args()
    return args

args = parse_arguments()

os.system(f"mkdir -p {args.output_folder}")

image_filenames, prompts = render_images_eval(
    output_folder=args.output_folder,
    concept_mode=args.concept_mode,
    render_size=(1024,1024),
    lora_path=args.lora_path,
    pretrained_model=pretrained_models[args.sd_model_version],
    seed=0,
    is_lora = True,
    trigger_text=" in the style of TOK,"
)

if args.concept_mode == "style":
    prompts = [filter_style_prompt(x) for x in prompts]
else:
    raise NotImplementedError("prompt filtering for other concept modes is not implemented yet.")

eval = Evaluation(image_filenames=image_filenames)
clip_diversity = eval.clip_diversity(device=device)

# TODO: replace checkpoint_path=None with the correct checkpoint path
aesthetic_score = eval.aesthetic_score(device=device, checkpoint_path=None)
image_text_alignment = eval.image_text_alignment(device=device, prompts=prompts)

result = {
    "sd_model_version": args.sd_model_version,
    "lora_path": os.path.abspath(args.lora_path),
    "concept_mode": args.concept_mode,
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
Example command
python3 evaluate.py \
--sd_model_version sdxl \
--output_folder eval_images \
--concept_mode style \
--lora_path lora_models/clipx_tiny_dora_only---sdxl_style_dora/checkpoints/checkpoint-600 \
--output_json eval_results.json
"""