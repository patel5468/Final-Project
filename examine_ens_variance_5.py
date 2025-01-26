import sys
import os
import pickle
import numpy as np
import matplotlib.pyplot as plt
from datetime import datetime, timedelta

def load_pickle_files(start_date, end_date, interval, variable_name, ensemble_type, directory):
    """
    Load pickle files between the specified dates for the given variable and ensemble type.
    """
    data = []
    current_date = datetime.strptime(start_date, "%Y%m%d")
    end_date = datetime.strptime(end_date, "%Y%m%d")
    
    while current_date <= end_date:
        file_name = f"{variable_name}_{ensemble_type}_{current_date.strftime('%Y%m%d%H%M')}_variance.pkl"
        file_path = os.path.join(directory, file_name)
        print(f"Loading file: {file_path}")
        if os.path.exists(file_path):
            with open(file_path, 'rb') as f:
                try:
                    data.append(pickle.load(f))
                except Exception as e:
                    print(f"Error loading {file_name}: {e}")
        else:
            print(f"File not found: {file_path}")
        current_date += timedelta(days=interval)
    
    return data

def convert_to_float(array):
    """
    Ensure the array is of type float for mathematical operations. Handles conversion errors.
    """
    try:
        return np.array(array, dtype=np.float64)
    except Exception as e:
        print(f"Error converting array to float: {e}")
        return np.zeros_like(array, dtype=np.float64)

def compute_normalized_variance(perturbed_data, reference_data):
    """
    Computes normalized variance: Norm Variance = Variance of Perturbed Ensemble / Variance of Reference Ensemble.
    Avoids division by zero and ensures data compatibility.
    """
    normalized_data = []
    for p, r in zip(perturbed_data, reference_data):
        normalized_entry = {}
        for key in p.keys():
            if key == "date":
                normalized_entry[key] = p[key]
            else:
                # Ensure the arrays are of numeric type
                perturbed_array = convert_to_float(p[key])
                reference_array = convert_to_float(r[key])
                
                # Avoid division by zero
                normalized_entry[key] = np.divide(
                    perturbed_array, 
                    reference_array, 
                    out=np.zeros_like(perturbed_array), 
                    where=(reference_array != 0)
                )
        normalized_data.append(normalized_entry)
    return normalized_data

def plot_variance_diagrams(data, variable_name, plot_type):
    """
    Generate level-time and/or latitude-time diagrams to show how variances grow over time.
    """
    times = [datetime.strptime(entry['date'], "%Y%m%d%H%M") for entry in data]
    large_variances = [entry['large scale average variance'] for entry in data]
    medium_variances = [entry['medium scale average variance'] for entry in data]
    small_variances = [entry['small scale average variance'] for entry in data]
    
    fig, axes = plt.subplots(3, 1, figsize=(12, 18))
    
    for ax, variances, scale in zip(axes, [large_variances, medium_variances, small_variances], ['Large', 'Medium', 'Small']):
        var_matrix = np.mean(np.array(variances), axis=(2, 3))  # Averaging over the last two dimensions
        im = ax.imshow(var_matrix.T, aspect='auto', origin='lower', extent=[0, len(times)-1, 0, var_matrix.shape[1]])
        
        # Subsample time ticks for readability
        tick_step = max(len(times) // 10, 1)  # At most 10 labels
        ax.set_xticks(range(0, len(times), tick_step))
        ax.set_xticklabels([times[i].strftime('%Y-%m-%d') for i in range(0, len(times), tick_step)], rotation=45)
        
        # Format Y-axis labels
        ax.set_yticks(range(0, var_matrix.shape[1], max(var_matrix.shape[1] // 10, 1)))
        ax.set_title(f'{scale} Scale {plot_type.capitalize()} Variance over Time for {variable_name}')
        ax.set_xlabel('Time')
        ax.set_ylabel('Vertical Levels')
        fig.colorbar(im, ax=ax, orientation='vertical')
    
    plt.tight_layout()
    output_file = f'{variable_name}_{plot_type}_variance.png'
    plt.savefig(output_file, bbox_inches='tight', dpi=300)
    print(f"Plot saved: {output_file}")

def main():
    start_date = sys.argv[1]
    end_date = sys.argv[2]
    interval_days = int(sys.argv[3])
    variable_name = sys.argv[4]
    plot_type = sys.argv[5]  # perturbed_ens, reference_ens, or normalized
    directory = '/fs/scratch/PAS2856/AS4194_Project/PatelShuvo'
    
    # Load perturbed and reference data
    perturbed_data = load_pickle_files(start_date, end_date, interval_days, variable_name, 'perturbed_ens', directory)
    reference_data = load_pickle_files(start_date, end_date, interval_days, variable_name, 'reference_ens', directory)
    
    if not perturbed_data or not reference_data:
        print("No data loaded. Please check the input parameters and the existence of pickle files.")
        sys.exit(1)
    
    if plot_type == "normalized":
        # Compute normalized variance
        data = compute_normalized_variance(perturbed_data, reference_data)
    elif plot_type == "perturbed_ens":
        data = perturbed_data
    elif plot_type == "reference_ens":
        data = reference_data
    else:
        print(f"Invalid plot type: {plot_type}")
        sys.exit(1)
    
    # Plot variance diagrams
    plot_variance_diagrams(data, variable_name, plot_type)

if __name__ == "__main__":
    main()
