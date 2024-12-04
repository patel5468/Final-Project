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
        print(file_path)
        if os.path.exists(file_path):
            with open(file_path, 'rb') as f:
                data.append(pickle.load(f))
        current_date += timedelta(days=interval)
    
    return data

def calculate_normalized_variances(data):
    """
    Calculate normalized variances for each scale.
    """
    normalized_data = []
    for entry in data:
        #Extract variances for perturbed and reference ensembles
        large_perturbed = np.array(entry['large scale average variance'])
        medium_perturbed = np.array(entry['medium scale average variance'])
        small_perturbed = np.array(entry['small scale average variance'])
        
        large_reference = np.array(entry['large scale reference variance'])
        medium_reference = np.array(entry['medium scale reference variance'])
        small_reference = np.array(entry['small scale reference variance'])
        
        #Calculate normalized variances
        normalized_entry = {
            'date': entry['date'],
            'large scale normalized variance': large_perturbed / large_reference,
            'medium scale normalized variance': medium_perturbed / medium_reference,
            'small scale normalized variance': small_perturbed / small_reference,
        }
        normalized_data.append(normalized_entry)
    return normalized_data

def plot_variance_diagrams(data, variable_name, ensemble_type):
    """
    Generate level-time and/or latitude-time diagrams to show how ensemble variances grow over time.
    """
    #Extract the times, large, medium, and small scale normalized variances from the data:
    times = [datetime.strptime(entry['date'], "%Y%m%d%H%M") for entry in data]
    large_variances = [entry['large scale normalized variance'] for entry in data]
    medium_variances = [entry['medium scale normalized variance'] for entry in data]
    small_variances = [entry['small scale normalized variance'] for entry in data]
    
    fig, axes = plt.subplots(3, 1, figsize=(12, 18)) #Create empty figure with 3 subplots and size of 12x18
    
    for ax, variances, scale in zip(axes, [large_variances, medium_variances, small_variances], ['Large', 'Medium', 'Small']): #For loop over each subplot axis ('ax'), corresponding variance data (large, medium, small), 
                                                                                                                                #and label for the scale (Large, Medium, Small) to generate separate plots for each scale.
        var_matrix = np.mean(np.array(variances), axis=(2, 3))  # Averaging over the last two dimensions

        im = ax.imshow(var_matrix.T, aspect='auto', origin='lower', extent=[0, len(times)-1, 0, var_matrix.shape[1]]) 
                #Show the 2D array 'var_matrix' as an image on the axes, transpose 'var_matrix'and set extent to scale the axes based on times.
        tick_step = max(len(times) // 10, 1)  # At most 10 labels
        ax.set_xticks(range(0, len(times), tick_step)) #Set the X-axis tick positions at intervals of tick_step to avoid overcrowding the labels.
        ax.set_xticklabels([times[i].strftime('%Y-%m-%d') for i in range(0, len(times), tick_step)], rotation=45) #Set x axis labels to corresponding dates and rotate them 45 degrees for readability
        
        #Format Y-axis labels
        ax.set_yticks(range(0, var_matrix.shape[1], max(var_matrix.shape[1] // 10, 1))) #Set the Y-axis tick positions at intervals of tick_step to avoid overcrowding the labels
        ax.set_title(f'{scale} Scale Variance over Time for {variable_name} ({ensemble_type})')
        ax.set_xlabel('Time') #Set x axis label
        ax.set_ylabel('Vertical Levels') #Set y axis label
        fig.colorbar(im, ax=ax, orientation='vertical') #Create colorbar and set orientation to vertical
    
    plt.tight_layout() #Clean up figure to look nice and eliminate excess whitespace
    plt.savefig('sample_figure4_final.png', bbox_inches='tight', dpi=300) #Save fig

def main():
    start_date = sys.argv[1]
    end_date = sys.argv[2]
    interval_days = int(sys.argv[3])
    variable_name = sys.argv[4]
    ensemble_type = sys.argv[5]
    directory = '/fs/scratch/PAS2856/AS4194_Project/PatelShuvo'
    
    data = load_pickle_files(start_date, end_date, interval_days, variable_name, ensemble_type, directory)
    
    if not data:
        print("No data loaded. Please check the input parameters and the existence of pickle files.")
        sys.exit(1)
    
    normalized_data = calculate_normalized_variances(data) #Call function for normalized variance calculations
    plot_variance_diagrams(normalized_data, variable_name, ensemble_type) #Plot the variance by calling the function to plot

if __name__ == "__main__":
    main()
