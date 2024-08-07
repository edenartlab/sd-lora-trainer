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
render_dir = "/home/rednax/SSD2TB/Xander_Tools/sd15_face_sweep/lora_models"
config_dir = "/home/rednax/SSD2TB/Xander_Tools/sd15_face_sweep/xander_adiff_lora"

ignore_threshold_relative = 0.0  # ignore any datapoint with a score below this threshold

filters = {
    "resolution": 512
}

output_dir = f"gridsearch_configs/results/{os.path.basename(config_dir)}"
output_suffix = f"{os.path.basename(render_dir)}"

# Initialize a dictionary to hold parameter values and associated scores
parameters = defaultdict(lambda: defaultdict(list))

# Step 1: Loop over each experiment subdirectory
for i, exp_subdir in enumerate(sorted(os.listdir(render_dir))):
    exp_path = os.path.join(render_dir, exp_subdir)
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

                # Filter out experiments that do not match the filters
                if not all(config[key] == value for key, value in filters.items()):
                    continue

                # Step 4: Append all key/value pairs to the total experiment dictionary
                for key, value in config.items():
                    parameters[key]['values'].append(value)
                    parameters[key]['scores'].append(score)
        else:
            print(f"Could not find JSON file for experiment {exp_subdir}")


# Print the parameters['output_dir'] with the highest scores (there are usually multiple ties):
max_score = max(parameters['output_dir']['scores'])
best_output_dirs = [output_dir for output_dir, score in zip(parameters['output_dir']['values'], parameters['output_dir']['scores']) if score == max_score]
for best_output_dir in best_output_dirs:
    print(f"Best output_dir: {best_output_dir} with score {max_score}")

import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score
from sklearn.preprocessing import LabelEncoder

os.makedirs(output_dir, exist_ok=True)
print(f"Saving results to {output_dir}...")

def plot_parameters(parameters):
    for param, data in parameters.items():
        values = np.array(data['values'])
        scores = np.array(data['scores'])

        # filter based on the ignore_threshold:
        ignore_threshold = ignore_threshold_relative * np.max(scores)
        mask = scores > ignore_threshold
        values = values[mask]
        scores = scores[mask]

        noise_strength_values = 0.02
        noise_strength_scores = 0.02
        
        # Initialize variables for original categorical labels
        original_labels = None
        
        # Determine if values are numeric
        if values.dtype.kind in 'bifc':  # Numeric types
            # Add noise directly to values
            jittered_values = values + np.random.normal(0, noise_strength_values * (np.max(values) - np.min(values)), values.shape)
        else:
            # Encode string values to integers for plotting
            encoder = LabelEncoder()
            original_labels = values.copy()
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
        jittered_scores = scores + np.random.normal(0, noise_strength_scores * np.max(scores), scores.shape)

        # Calculate R² value
        r_squared = r2_score(scores, predicted_scores)
        
        # Plot data points
        sns.scatterplot(x=jittered_values, y=jittered_scores, alpha=0.6)
        
        # Plot trendline
        sns.lineplot(x=np.sort(values), y=predicted_scores[np.argsort(values)], color='red', label=f'R²={r_squared:.2f}')

        # Set plot title and labels
        plt.title(f'Influence of {param} on the score')
        if original_labels is not None:
            # Set x-axis labels to the original categorical labels
            unique_values = np.unique(values)
            plt.xticks(ticks=unique_values, labels=encoder.inverse_transform(unique_values), rotation=45, ha='right')
        else:
            plt.xlabel(param)
        plt.ylabel('Score')
        plt.legend()
        
        # Save and close the plot
        plt.savefig(f'{output_dir}/res_{param}_{output_suffix}.png')
        plt.close()

# Call the updated function with your parameters dictionary
plot_parameters(parameters)