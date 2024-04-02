import random, os, cv2, time, json, shutil
from random import shuffle
import numpy as np
from PIL import Image
import pickle
import pandas as pd
import sys

clip_classifier_dir = '/home/xander/Projects/cog/CLIP_active_learning_classifier/CLIP_assisted_data_labeling'
sys.path.append(clip_classifier_dir)
from utils.nn_model import device, SimpleFC
from _1_embed_with_CLIP import CLIP_Feature_Dataset

sys.path.append('..')
from settings import StableDiffusionSettings
from generation import *

def load_prompts(path):
    text_inputs = []
    with open(txt_path) as f:
        for line in f:
            text_inputs.append(line.strip())
    return text_inputs

import clip
def compute_cosine_similarity(image_embeddings, text_embeddings):
    image_embeddings = image_embeddings / torch.norm(image_embeddings, dim=-1, keepdim=True)
    text_embeddings = text_embeddings / torch.norm(text_embeddings, dim=-1, keepdim=True)
    cosine_similarity = torch.matmul(image_embeddings, text_embeddings.T)
    return cosine_similarity

"""


cd /home/xander/Projects/cog/eden-sd-pipelines/eden/xander
python evaluate_checkpoints.py



Given a list of SD checkpoints:
    - load each checkpoint
    - Generate n images (using deterministic seed)
    - Compute the avg perceptual score using the perceptual classifier
    - Compute a diversity metric (e.g. avg pairwise L2 distance in CLIP space)
    - Compute the img-txt alignment (using cosine-sim in clip space)
    - plot the results for all checkpoints

"""


# Render controls:
n_samples     = 197*2
deterministic = 1

txt_path = '/home/xander/Projects/cog/eden-sd-pipelines/eden/random_prompts.txt'
ckpt_dir = '/data/models/eden_ckpts'
outdir   = 'images/evaluate_eden_checkpoints_final'

scoring_model_path = '/home/xander/Projects/cog/CLIP_active_learning_classifier/CLIP_assisted_data_labeling/models/combo_2023-04-25_16:17:14_2.7k_imgs_70_epochs_-1.0000_mse.pkl'
clip_model = ''

clip_model_name = "ViT-L-14-336/openai"  # "ViT-L-14/openai" #SD 1.x  //  "ViT-H-14/laion2b_s32b_b79k" #SD 2.x
clip_model_path = "/home/xander/Projects/cog/cache"


#########################################################

checkpoint_options = sorted([os.path.join(ckpt_dir, f) for f in os.listdir(ckpt_dir)])

if 1:
    checkpoint_options += [
        "stabilityai/stable-diffusion-2-1",
        "runwayml/stable-diffusion-v1-5",
        "dreamlike-art/dreamlike-photoreal-2.0", 
        #"/data/models/EdenI_0.38_to_1.00_from_2.1_no_txt_finetune_20230425-124315/ckpts/EdenI_0.38_to_1.00_from_2.1_no_txt_finetune-ep03-gs11100",
        #"/data/models/EdenI_0.38_to_1.00_from_2.1_no_txt_finetune_20230425-124315/ckpts/EdenI_0.38_to_1.00_from_2.1_no_txt_finetune-ep07-gs22413",
        #"/data/models/EdenI_0.39_to_1.00_from_2.1_no_txt_finetune_20230426-032832/ckpts/EdenI_0.39_to_1.00_from_2.1_no_txt_finetune-ep01-gs01888",
        #"/data/models/EdenI_0.38_to_1.00_from_2.1_txt_finetune_20230425-140646/ckpts/EdenI_0.38_to_1.00_from_2.1_txt_finetune-ep04-gs13095",
    ]

text_inputs        = load_prompts(txt_path)
results_path       = os.path.join(outdir, 'results.csv')

print(f"Sampling from {len(text_inputs)} prompts")

if os.path.exists(results_path):
    # delete results_path from filesystem:
    os.remove(results_path)

results_df = pd.DataFrame(columns = ['ckpt_name', 'aesthetic_score_mean', 'aesthetic_score_std', 'diversity_score'])

#########################################################


def batch_generate_imgs(ckpt_path, n, output_dir):
    print(f"Generating {n} samples to {output_dir}..")
    os.makedirs(output_dir, exist_ok = True)

    for i in range(n):
        print(f"--- Generating img {i} of {n}..")
        seed = i if deterministic else int(time.time())
        seed_everything(seed)

        args = StableDiffusionSettings(
            mode = "generate",
            ckpt=ckpt_path,
            sampler = "euler",
            W = 960,
            H = 768,
            seed = seed,
            text_input = text_inputs[(i+1)%len(text_inputs)],
            init_image_data = None,
            init_image_strength = 0.0,
            steps = 30,
            guidance_scale = random.choice([6,7,8,9]),
            upscale_f = 1.5,
            n_samples = 1,
            #uc_text = '',
        )

        _, generator = generate(args)

        #####################################################################################

        # Save to disk
        ckpt_name = args.ckpt.split('/')[-1]
        name = f'{args.text_input[:40]}_{ckpt_name}_{int(time.time())}'
        name = name.replace("/", "_")

        for i, img in enumerate(generator):
            frame = f'{name}.jpg'
            img.save(os.path.join(output_dir, frame), quality=95)

        # save settings
        settings_filename = f'{output_dir}/{name}.json'
        save_settings(args, settings_filename)

def score_images(image_directory, scoring_model):
    df = pd.DataFrame(columns = ['name', 'aesthetic_score'])

    crop_names = scoring_model.crop_names
    use_img_stat_features = scoring_model.use_img_stat_features

    # get all images:
    img_paths = sorted([f for f in os.listdir(image_directory) if f.endswith('.jpg')])
    for img_path in img_paths:
        feature_path = os.path.join(image_directory, img_path.replace('.jpg', '.pt'))
        feature_dict = torch.load(feature_path)

        clip_features = torch.cat([feature_dict[crop_name] for crop_name in crop_names if crop_name in feature_dict], dim=0).flatten()
        missing_crops = set(crop_names) - set(feature_dict.keys())
        if missing_crops:
            raise Exception(f"Missing crops {missing_crops} for {uuid}, either re-embed the image, or adjust the crop_names variable for training!")

        if use_img_stat_features:
            img_stat_feature_names = [key for key in feature_dict.keys() if key.startswith("img_stat_")]
            img_stat_features = torch.stack([feature_dict[img_stat_feature_name] for img_stat_feature_name in img_stat_feature_names], dim=0).to(device)
            all_features = torch.cat([clip_features, img_stat_features], dim=0)
        else:
            all_features = clip_features 

        output = scoring_model(all_features.unsqueeze(0))
        output = output.detach().cpu().numpy().item()
        new_row = pd.DataFrame({'name': [img_path], 'aesthetic_score': [output]})
        df = pd.concat([df, new_row], ignore_index=True)

    return df

def compute_diversity(ckpt_dir):
    clip_embeddings = []
    images = sorted([f for f in os.listdir(ckpt_dir) if f.endswith('.jpg')])
    for img in images:
        feature_path = os.path.join(ckpt_dir, img.replace('.jpg', '.pt'))
        feature_vector = torch.load(feature_path)['centre_crop'].flatten().to(device).float()
        clip_embeddings.append(feature_vector)

    clip_embeddings = torch.stack(clip_embeddings, dim=0)
    # Compute pairwise distances:
    #distance = nn.PairwiseDistance(p=2)
    #distance = nn.CosineSimilarity(dim=1, eps=1e-6)
    #distances = distance(clip_embeddings, clip_embeddings)
    distances = torch.cdist(clip_embeddings, clip_embeddings, p=2.0)

    distances = distances.detach().cpu().numpy()

    # Get the upper triangle:
    upper_triangle = np.triu(distances, k=1)

    return upper_triangle.flatten()

def compute_img_txt_alignment(image_directory, clip_model_name):
    df = pd.DataFrame(columns = ['name', 'alignment_score'])
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model, preprocess = clip.load(clip_model_name, device=device)
    
    cosine_similarities = []

    img_paths = sorted([f for f in os.listdir(image_directory) if f.endswith('.jpg')])

    for img_path in img_paths:
            image = Image.open(os.path.join(image_directory, img_path)).convert("RGB")
            preprocessed_image = preprocess(image).unsqueeze(0).to(device)

            # Load corresponding json file
            json_file = os.path.splitext(img_path)[0] + ".json"
            json_path = os.path.join(image_directory, json_file)
            with open(json_path, "r") as f:
                json_data = json.load(f)
                text_input = json_data["text_input"]

            # Encode image and text
            with torch.no_grad():
                image_embeddings = model.encode_image(preprocessed_image)
                text_embeddings = model.encode_text(clip.tokenize(text_input).to(device))

            # Compute cosine similarity
            cosine_similarity = compute_cosine_similarity(image_embeddings, text_embeddings)

            new_row = pd.DataFrame({'name': [img_path], 'alignment_score': [cosine_similarity.item()]})
            df = pd.concat([df, new_row], ignore_index=True)

    return df

###############################################################


# Evaluate images with classifier:
with open(scoring_model_path, "rb") as file:
    scoring_model = pickle.load(file)
    scoring_model = scoring_model.to(device)

results = {}
for ckpt_path in checkpoint_options:

    ckpt_name = os.path.basename(ckpt_path)
    ckpt_out_dir = f'{outdir}/{ckpt_name}'

    try:
        images_present = len([f for f in os.listdir(ckpt_out_dir) if f.endswith('.jpg')])
    except:
        images_present = 0

    if images_present == n_samples:
        print(f"Images for {ckpt_name} already generated!")
    else:
        if os.path.exists(ckpt_out_dir): # do a full restart to make sure
            shutil.rmtree(ckpt_out_dir)
        batch_generate_imgs(ckpt_path, n_samples, ckpt_out_dir)

    if 1:
        # embed images with CLIP:
        batch_size = 4
        dataset = CLIP_Feature_Dataset(ckpt_out_dir, clip_model_name, batch_size, 
            clip_model_path = clip_model_path, 
            force_reencode = False,
            num_workers = 0)
        dataset.process()

    df_aesthetic = score_images(ckpt_out_dir, scoring_model)
    img_scores = df_aesthetic['aesthetic_score'].tolist()

    # Compute diversity metric:
    pairwise_distances = compute_diversity(ckpt_out_dir)
    # Drop distances from imgs that are super super similar:
    pairwise_distances = pairwise_distances[pairwise_distances >= 0.2]

    # Compute alignment metric:
    df_alignment = compute_img_txt_alignment(ckpt_out_dir, "ViT-B/32")
    alignment_scores = df_alignment['alignment_score'].tolist()

    # merge df_aesthetic and df_alignment:
    df = pd.merge(df_aesthetic, df_alignment, on='name')
    df.to_csv(f'{outdir}/{ckpt_name}_aesthetic_and_alignment_scores.csv', index = False)

    # Save ckpt-averaged results:
    results[ckpt_name] = {
        'aesthetic_scores': img_scores, 
        'alignment_scores': alignment_scores,
        'pairwise_distances':  pairwise_distances,
        }

    print(f"Evaluation of {ckpt_name} done!")


print("All done! Plotting results...")
import matplotlib.pyplot as plt

ckpt_names = results.keys()
colors = plt.cm.terrain(np.linspace(0, 1, len(ckpt_names)))
colormap = dict(zip(ckpt_names, colors))
figure_size=(20, 14)

def violin_plot(results, n_samples, ckpt_names, metric, outdir, figure_size):
    fig, ax = plt.subplots(figsize=figure_size)
    offsets = [0.0, 0.03]
    for idx, ckpt_name in enumerate(ckpt_names):
        scores = results[ckpt_name][metric]
        vplot = ax.violinplot(scores, positions=[idx], widths=0.9, showmeans=True, showextrema=True, showmedians=False)
        
        for pc in vplot['bodies']:
            pc.set_facecolor(colormap[ckpt_name])
            pc.set_edgecolor('black')
            pc.set_alpha(0.75)

        # Set the color of other plot elements
        for partname in ('cbars', 'cmins', 'cmaxes', 'cmeans'):
            vp = vplot[partname]
            vp.set_edgecolor('black')
            vp.set_linewidth(1)

        # Add an annotation for the checkpoint name
        offset = offsets[idx % len(offsets)]
        ax.annotate(ckpt_name, xy=(idx, 0.93 + offset), xycoords=('data', 'axes fraction'), xytext=(0, 5), textcoords='offset points', fontsize=8, ha='center')


    # Remove the x-axis labels (we'll use annotations instead)
    ax.set_xticks(range(len(ckpt_names)))
    ax.set_xticklabels([''] * len(ckpt_names))
    ax.set_xlabel('Checkpoint')
    ax.set_ylabel(metric)
    ax.set_title(f'{metric} vs Checkpoint ({n_samples} imgs)')

    plt.tight_layout()
    plt.savefig(os.path.join(outdir, f'{metric}_vs_checkpoint_.png'))

violin_plot(results, n_samples, ckpt_names, "aesthetic_scores", outdir, figure_size)
violin_plot(results, n_samples, ckpt_names, "alignment_scores", outdir, figure_size)
violin_plot(results, n_samples, ckpt_names, "pairwise_distances", outdir, figure_size)




