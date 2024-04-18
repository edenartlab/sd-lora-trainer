import os
import json
import matplotlib.pyplot as plt
import seaborn as sns
from collections import defaultdict
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score


# Define paths
exp_dir = "/home/rednax/SSD2TB/Github_repos/diffusion_trainer/lora_models/jana_sdxl_grid_2"
config_dir = "/home/rednax/SSD2TB/Github_repos/diffusion_trainer/gridsearch_configs/gridsearch_sdxl"

# Initialize a dictionary to hold parameter values and associated scores
parameters = defaultdict(lambda: defaultdict(list))

# Step 1: Loop over each experiment subdirectory
for i, exp_subdir in enumerate(os.listdir(exp_dir)):
    exp_path = os.path.join(exp_dir, exp_subdir)
    checkpoints_path = os.path.join(exp_path, "checkpoints")

    # Step 2: Get the score by counting the number of .jpg files in the checkpoints subdir
    if os.path.isdir(checkpoints_path):
        score = sum(1 for _ in os.listdir(checkpoints_path) if _.endswith('.jpg'))
    
        # Match the experiment folder with its corresponding JSON file
        json_file_name = exp_subdir.split('--')[0] + ".json"
        json_file_name = json_file_name.replace('__','_')
        json_path = os.path.join(config_dir, json_file_name)
        
        # Step 3: Load the corresponding .json file
        if os.path.isfile(json_path):
            with open(json_path, 'r') as file:
                config = json.load(file)
                # Step 4: Append all key/value pairs to the total experiment dictionary
                for key, value in config.items():
                    parameters[key]['values'].append(value)
                    parameters[key]['scores'].append(score)
        else:
            print(f"Could not find JSON file for experiment {exp_subdir}")

import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score
from sklearn.preprocessing import LabelEncoder

def plot_parameters(parameters):
    for param, data in parameters.items():
        values = np.array(data['values'])
        scores = np.array(data['scores'])
        
        # Determine if values are numeric
        if values.dtype.kind in 'bifc':  # Numeric types
            # Add noise directly to values
            jittered_values = values + np.random.normal(0, 0.01 * np.max(values), values.shape)
        else:
            # Encode string values to integers for plotting
            encoder = LabelEncoder()
            values = encoder.fit_transform(values)
            jittered_values = values + np.random.normal(0, 0.1, values.shape)

        # Skip plotting if there is only one unique value for the parameter
        if len(np.unique(values)) <= 1:
            continue
        
        # Fit a linear regression model to the encoded values if categorical
        model = LinearRegression()
        values_reshaped = values.reshape(-1, 1)  # Reshape for sklearn
        model.fit(values_reshaped, scores)
        predicted_scores = model.predict(values_reshaped)

        # add some jitter to the scores:
        jittered_scores = scores + np.random.normal(0, 0.02 * np.max(scores), scores.shape)

        # Calculate R² value
        r_squared = r2_score(scores, predicted_scores)
        
        # Plot data points
        sns.scatterplot(x=jittered_values, y=jittered_scores, alpha=0.6)
        
        # Plot trendline
        sns.lineplot(x=np.sort(values), y=predicted_scores[np.argsort(values)], color='red', label=f'Fit: y={model.coef_[0]:.2f}x+{model.intercept_:.2f}, R²={r_squared:.2f}')

        # Set plot title and labels
        plt.title(f'Influence of {param} on the score')
        plt.xlabel(param)
        plt.ylabel('Score')
        plt.legend()
        
        # Save and close the plot
        plt.savefig(f'res_{param}.png')
        plt.close()

# Call the updated function with your parameters dictionary
plot_parameters(parameters)