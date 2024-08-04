import os
import json
import matplotlib.pyplot as plt
from collections import defaultdict

# Step 1: Import necessary libraries (done above)

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
def collect_data(root_dir):
    data = []
    for root, dirs, files in os.walk(root_dir):
        if 'checkpoints' in dirs:
            checkpoints_dir = os.path.join(root, 'checkpoints')
            score = count_jpg_files(checkpoints_dir)
            training_args = load_training_args(root)
            if training_args:
                data.append((training_args, score))
            else:
                print(f"Warning: No training_args.json found in {root}")
    print(f"Collected data from {len(data)} runs")
    return data

# Step 5: Process the collected data to identify varying hyperparameters
def identify_varying_hyperparams(data):
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
        try:
            values = set(make_hashable(args.get(param)) for args, _ in data if param in args)
            if len(values) > 1:
                varying_params[param] = values
                print(f"---> Parameter '{param}' varies across runs")
        except TypeError as e:
            print(f"Warning: Could not hash values for parameter '{param}'. Error: {e}")
            print(f"Values: {[args.get(param) for args, _ in data if param in args]}")
    
    return varying_params


# Step 6: Create visual plots for each varying hyperparameter
import numpy as np
from scipy import stats
import matplotlib.pyplot as plt
from collections import defaultdict

def create_plots(data, varying_params):
    for param, values in varying_params.items():
        plt.figure(figsize=(12, 8))
        param_data = defaultdict(list)
        
        for args, score in data:
            if param in args:
                value = args[param]
                value_str = str(value)
                param_data[value_str].append(score)
        
        values_list = sorted(param_data.keys())
        all_x = []
        all_y = []
        
        for i, value_str in enumerate(values_list):
            scores = param_data[value_str]
            jittered_x = np.random.normal(i, 0.1, size=len(scores))
            plt.scatter(jittered_x, scores, alpha=0.6, label=value_str)
            all_x.extend([i] * len(scores))
            all_y.extend(scores)
        
        # Calculate trendline
        x = np.array(all_x)
        y = np.array(all_y)
        z = np.polyfit(x, y, 1)
        p = np.poly1d(z)
        
        # Calculate R-squared
        r_squared = 1 - (sum((y - p(x))**2) / ((len(y) - 1) * np.var(y, ddof=1)))
        
        # Plot trendline
        plt.plot(x, p(x), "r--", alpha=0.8, 
                 label=f'Trendline: y={z[0]:.2f}x+{z[1]:.2f}\nR²: {r_squared:.4f}')
        
        plt.xlabel(param)
        plt.ylabel('Score')
        plt.title(f'Effect of {param} on Score')
        
        # Adjust x-axis labels
        if len(values_list) > 10:
            plt.xticks(range(0, len(values_list), len(values_list)//10), 
                       [values_list[i] for i in range(0, len(values_list), len(values_list)//10)], 
                       rotation=45, ha='right')
        else:
            plt.xticks(range(len(values_list)), values_list, rotation=45, ha='right')
        
        # Adjust legend
        if len(values_list) > 10:
            plt.legend(title="Legend", bbox_to_anchor=(1.05, 1), loc='upper left')
        else:
            plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
        
        plt.tight_layout()
        
        # Save figure with error handling
        try:
            plt.savefig(f'{param}_vs_score.png', dpi=200, bbox_inches='tight')
        except ValueError:
            print(f"Warning: Failed to save image for {param}. skipping..")
            
        
        plt.close()

    print("Plots have been saved as PNG files in the current directory.")

if __name__ == "__main__":
    root_dir = "/home/rednax/SSD2TB/Github_repos/diffusion_trainer/lora_models/PLANTOID"
    
    # Collect data
    data = collect_data(root_dir)
    
    # Identify varying hyperparameters
    varying_params = identify_varying_hyperparams(data)

    # Create plots
    create_plots(data, varying_params)
    
    print("Plots have been saved as PNG files in the current directory.")