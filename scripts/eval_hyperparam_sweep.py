import os
import json
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from collections import defaultdict
from sklearn.metrics import r2_score

# Step 2: Define a function to count JPG files in a directory
def count_jpg_files(directory):
    return len([f for f in os.listdir(directory) if f.lower().endswith('.jpg')])

# Step 3: Define a function to find and load the training_args.json file
def load_training_args(directory):
    for root, dirs, files in os.walk(directory):
        if 'training_args.json' in files:
            with open(os.path.join(root, 'training_args.json'), 'r') as f:
                return json.load(f)
    return None

# Step 4: Traverse the directory structure and collect data
def collect_data(root_dir, mode):
    data = []
    for root, dirs, files in os.walk(root_dir):
        if "checkpoints" in dirs:
            checkpoints_dir = os.path.join(root,"checkpoints")

            if mode == "final_checkpoint":
                checkpoint_dirs = sorted([f for f in os.listdir(checkpoints_dir) if os.path.isdir(os.path.join(checkpoints_dir, f))])
                checkpoint_dir = os.path.join(checkpoints_dir, checkpoint_dirs[-1])
                score = count_jpg_files(checkpoint_dir)
                training_args = load_training_args(checkpoint_dir)
            if mode == "n_validation_grids":
                score = count_jpg_files(checkpoints_dir)
                training_args = load_training_args(checkpoints_dir)
                    
            data.append((training_args, score))

    print(f"Collected data from {len(data)} runs")
    return data

# Step 5: Process the collected data to identify varying hyperparameters
def identify_varying_hyperparams(data, skip_params=['output_dir', 'start_time', 'name']):
    all_params = set().union(*[set(args.keys()) for args, _ in data])
    varying_params = {}
    
    def make_hashable(val):
        if isinstance(val, dict):
            return tuple(sorted((k, make_hashable(v)) for k, v in val.items()))
        elif isinstance(val, list):
            return tuple(make_hashable(v) for v in val)
        elif isinstance(val, set):
            return frozenset(make_hashable(v) for v in val)
        return val

    for param in all_params:
        if param in skip_params:
            continue
        try:
            values = [make_hashable(args.get(param)) for args, _ in data if param in args]
            unique_values = set(values)
            
            if len(unique_values) > 1:
                # Check if all values are numeric
                try:
                    numeric_values = [float(v) for v in unique_values]
                    varying_params[param] = set(numeric_values)
                except ValueError:
                    # If not all numeric, keep as is
                    varying_params[param] = unique_values
                
                print(f"---> Parameter '{param}' varies across runs")
                
                # Special handling for dictionary-type parameters
                if all(isinstance(v, dict) for v in values):
                    print(f"Dictionary values for '{param}':")
                    for v in unique_values:
                        print(f"  {v}")
                elif len(unique_values) <= 5:  # Print up to 5 unique values
                    print(f"Unique values: {unique_values}")
                else:
                    print(f"Number of unique values: {len(unique_values)}")
        except TypeError as e:
            print(f"Warning: Could not process values for parameter '{param}'. Error: {e}")
            #print(f"Values: {[args.get(param) for args, _ in data if param in args]}")
    
    return varying_params

def create_plots(data, varying_params, outdir, top = 0.15):
    os.makedirs(outdir, exist_ok=True)
    
    for param, values in varying_params.items():
        if all(isinstance(v, dict) for v in values):
            print(f"Skipping plot for dictionary parameter '{param}'")
            continue
        
        plt.figure(figsize=(12, 8))
        param_data = defaultdict(list)
        all_scores = []
        
        for args, score in data:
            if param in args:
                value = args[param]
                value_str = str(value)
                param_data[value_str].append(score)
                all_scores.append(score)
        
        # Calculate global top threshold
        global_top_percent = np.percentile(all_scores, 100*(1-top))
        
        # Sort the values
        try:
            values_list = sorted(param_data.keys(), key=float)
        except ValueError:
            values_list = sorted(param_data.keys())
        
        all_x = []
        all_y = []
        
        for i, value_str in enumerate(values_list):
            scores = param_data[value_str]

            # Apply jitter
            jittered_x = np.random.normal(i, 0.1, size=len(scores))
            jittered_y = np.array(scores) + np.random.normal(0, 0.01 * max(scores), size=len(scores))

            # Use global top 25% threshold
            top_mask = np.array(scores) >= global_top_percent

            # Emphasize top 25% scores using JITTERED coordinates for plotting
            sns.scatterplot(x=jittered_x[top_mask], y=jittered_y[top_mask], alpha=0.6, color='black', marker='X', s=40, linewidth=1)

            # Plot all scores using JITTERED coordinates
            sns.scatterplot(x=jittered_x, y=jittered_y, alpha=0.6, label=value_str)
                
            all_x.extend([i] * len(scores))
            all_y.extend(scores)
        
        # Calculate trendline for all data
        x = np.array(all_x)
        y = np.array(all_y)
        
        z = np.polyfit(x, y, 1)
        p = np.poly1d(z)
        plt.plot(range(len(values_list)), p(range(len(values_list))), "r--", alpha=0.8,
                 label=f'All data: y={z[0]:.2f}x+{z[1]:.2f}\nR²: {r2_score(y, p(x)):.4f}')
        
        # Calculate trendline for global top 25% scoring datapoints
        top_mask = y >= global_top_percent
        x_top = x[top_mask]
        y_top = y[top_mask]
        
        z_top = np.polyfit(x_top, y_top, 1)
        p_top = np.poly1d(z_top)
        plt.plot(range(len(values_list)), p_top(range(len(values_list))), "g--", alpha=0.8,
                 label=f'Top 25%: y={z_top[0]:.2f}x+{z_top[1]:.2f}\nR²: {r2_score(y_top, p_top(x_top)):.4f}')
        
        plt.xlabel(param)
        plt.ylabel('Score')
        plt.title(f'Effect of {param} on Score')
        
        # Set x-ticks and labels
        plt.xticks(range(len(values_list)), values_list, rotation=45, ha='right')
        
        # Adjust legend
        plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
        
        plt.tight_layout()
        
        # Save figure with error handling
        try:
            plt.savefig(f'{outdir}/{param}_vs_score.png', dpi=200, bbox_inches='tight')
        except ValueError:
            print(f"Warning: Failed to save image for {param}. Skipping...")
        
        plt.close()

    print(f"Plots have been saved as PNG files in {outdir}")

if __name__ == "__main__":
    root_dir = "/home/rednax/SSD2TB/Github_repos/diffusion_trainer/lora_models/faces_final"
    outdir = os.path.join('.', os.path.basename(root_dir))
    
    # Collect data
    data = collect_data(root_dir, mode = "final_checkpoint")
    #data = collect_data(root_dir, mode = "n_validation_grids")
    
    # Identify varying hyperparameters
    varying_params = identify_varying_hyperparams(data)

    # Create plots
    create_plots(data, varying_params, outdir)