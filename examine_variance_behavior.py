import sys
import os
import pickle
import numpy as np
import matplotlib.pyplot as plt
from datetime import datetime, timedelta

def load_pickle_files(start_date, end_date, interval, variable_name, ensemble_type, directory): #Function to load pickle file based on command input criteria
    """
    Load pickle files between the specified dates for the given variable and ensemble type.
    Inputs:
        start_date (str): Start date in the format 'YYYYMMDD'.
        end_date (str): End date in the format 'YYYYMMDD'.
        interval (int): Number of days to increment between file loads.
        variable_name (str): Name of the variable (e.g., 'u', 't', 'q') to load data for.
        ensemble_type (str): Type of ensemble (e.g., 'perturbed', 'reference') to load.
        directory (str): Directory where the pickle files are located.

    Returns:
        list: A list of data loaded from the pickle files.
    """
    data = [] #Creates an empty list to store loaded data
    #Use datetime to easily manipute late the start and end dates
    current_date = datetime.strptime(start_date, "%Y%m%d") 
    end_date = datetime.strptime(end_date, "%Y%m%d")
    
    while current_date <= end_date: #Loop through the date range, incrementing by the specified interval in days
        file_name = f"{variable_name}_{ensemble_type}_{current_date.strftime('%Y%m%d%H%M')}_variance.pkl"  #Construct the file name using the variable, ensemble type, and date
        file_path = os.path.join(directory, file_name) #Create file path by joining the directory and file name
        print(f"Loading file: {file_path}") #Print a message indicating which file is being loaded
        if os.path.exists(file_path): #Check if the file exists at the specified path
            with open(file_path, 'rb') as f: #Open and load the pickle file
                try: 
                    data.append(pickle.load(f)) #Append the loaded data to the list
                except Exception as e: #Handle any exceptions that occur during file loading
                    print(f"Error loading {file_name}: {e}")
        else:
            print(f"File not found: {file_path}") #Print a error message if the file does not exist
        current_date += timedelta(days=interval) #Increment the current date by the specified interval to avoid infinite looping
    
    return data #Return the list of loaded data

def convert_to_float(array):
    """
    Ensure the array is of type float for mathematical operations. Handles conversion errors.
    """
    try:  #Try to convert the inputed array to a np array with data type float64
        return np.array(array, dtype=np.float64)
    except Exception as e:
        print(f"Error converting array to float: {e}") #If an error occurs, print the error message
        return np.zeros_like(array, dtype=np.float64) #Return an array of zeros with the same shape as the input

def compute_normalized_variance(perturbed_data, reference_data): #Function to compute the normalized variance based on perturbed and reference data
    """
    Computes normalized variance: Norm Variance = Variance of Perturbed Ensemble / Variance of Reference Ensemble.
    Avoids division by zero and ensures data compatibility.
    """
    normalized_data = [] #Initialize an empty list to store the normalized variance results
    for p, r in zip(perturbed_data, reference_data): #For loop to loop over the paired entries from perturbed_data and reference_data
        normalized_entry = {}  #Create an empty dictionary to store the normalized data for the current entry
        for key in p.keys(): #Loop over the keys in the perturbed data dictionary
            if key == "date": #If the key is "date", copy it directly to the normalized entry
                normalized_entry[key] = p[key]
            else: #Convert the data to numeric arrays to ensure compatibility
                # Ensure the arrays are of numeric type
                perturbed_array = convert_to_float(p[key])
                reference_array = convert_to_float(r[key])
                
                #Compute normalized variance using numpy's divide function
                # Avoid division by zero
                normalized_entry[key] = np.divide( #Division
                    perturbed_array, #Numerator: perturbed variance
                    reference_array, #Denominator: reference variance
                    out=np.zeros_like(perturbed_array), 
                    where=(reference_array != 0) #Perform division only where denominator is non-zero
                )
        normalized_data.append(normalized_entry) #Append the normalized entry to the results list
    return normalized_data #Return the list of dictionaries containing normalized variances

def plot_variance_diagrams(data, variable_name, plot_type):
    """
    Generate level-time and/or latitude-time diagrams to show how variances grow over time.
    """
    times = [datetime.strptime(entry['date'], "%Y%m%d%H%M") for entry in data] #Extract the time values as datetime objects from the data to be plotted
    large_variances = [entry['large scale average variance'] for entry in data]  #Extract large-scale variances from the data to be plotted
    medium_variances = [entry['medium scale average variance'] for entry in data] #Extract medium-scale variances from the data to be plotted
    small_variances = [entry['small scale average variance'] for entry in data] #Extract small-scale variances from the data to be plotted
    
    fig, axes = plt.subplots(3, 1, figsize=(12, 18)) #Create empty figure with 3 subplots and size of 12x18
    
    for ax, variances, scale in zip(axes, [large_variances, medium_variances, small_variances], ['Large', 'Medium', 'Small']): #For loop over each subplot axis ('ax'), corresponding variance data (large, medium, small), 
                                                                                                                                #and label for the scale (Large, Medium, Small) to generate separate plots for each scale.
        var_matrix = np.mean(np.array(variances), axis=(2, 3))  # Averaging over the last two dimensions
        im = ax.imshow(var_matrix.T, aspect='auto', origin='lower', extent=[0, len(times)-1, 0, var_matrix.shape[1]]) #Show the 2D array 'var_matrix' as an image on the axes, transpose 'var_matrix'and set extent to scale the axes based on times.
        
        # Subsample time ticks for readability
        tick_step = max(len(times) // 10, 1)  # At most 10 labels 
        ax.set_xticks(range(0, len(times), tick_step)) #Set the X-axis tick positions at intervals of tick_step to avoid overcrowding the labels.
        ax.set_xticklabels([times[i].strftime('%Y-%m-%d') for i in range(0, len(times), tick_step)], rotation=45) #Set x axis labels to corresponding dates and rotate them 45 degrees for readability
        
        # Format Y-axis labels
        ax.set_yticks(range(0, var_matrix.shape[1], max(var_matrix.shape[1] // 10, 1))) #Set the Y-axis tick positions at intervals of tick_step to avoid overcrowding the labels
        ax.set_title(f'{scale} Scale {plot_type.capitalize()} Variance over Time for {variable_name}') #Set title based on variable being plotted
        ax.set_xlabel('Time')  #Set x axis label
        ax.set_ylabel('Vertical Levels') #Set y axis label
        fig.colorbar(im, ax=ax, orientation='vertical') #Create colorbar and set orientation to vertical
    
    plt.tight_layout() #Clean up figure to look nice and eliminate excess whitespace
    output_file = f'{variable_name}_{plot_type}_variance.png' #Define the output name
    plt.savefig(output_file, bbox_inches='tight', dpi=300) #Save fig
    print(f"Plot saved: {output_file}") #Confirmation of the image being saved

def main():
    start_date = sys.argv[1] #Command input for the start date
    end_date = sys.argv[2] #Command input for the end date
    interval_days = int(sys.argv[3]) #Command input for the interval
    variable_name = sys.argv[4] #Command input for the variable
    plot_type = sys.argv[5]  # perturbed_ens, reference_ens, or normalized
    directory = '/fs/scratch/PAS2856/AS4194_Project/PatelShuvo' #Location of the pickle files
    
    # Load perturbed and reference data
    perturbed_data = load_pickle_files(start_date, end_date, interval_days, variable_name, 'perturbed_ens', directory) #Load the perturbed datafile based on command input
    reference_data = load_pickle_files(start_date, end_date, interval_days, variable_name, 'reference_ens', directory) #Load the reference data based on the command input
    
    if not perturbed_data or not reference_data: #Check if either the perturbed_data or reference_data variables are empty or don't exist
        print("No data loaded. Please check the input parameters and the existence of pickle files.") #Print an error message to inform that no data has been loaded
        sys.exit(1) #Exit the program
    
    if plot_type == "normalized": #Check the type of plot 
        # Compute normalized variance
        data = compute_normalized_variance(perturbed_data, reference_data)  #If the plot type is "normalized", compute the normalized variance using the perturbed data and reference data
    elif plot_type == "perturbed_ens": #If the plot type is 'perturbed_ens', use the perturbed data directly
        data = perturbed_data 
    elif plot_type == "reference_ens":  #If the plot type is 'reference_ens', use the reference data directly
        data = reference_data
    else:
        print(f"Invalid plot type: {plot_type}") #If the plot type is invalid, print error message
        sys.exit(1) #Exit the program if plot type is invalid
    
    #Call the function plot_variance_diagrams above to plot variance diagrams
    plot_variance_diagrams(data, variable_name, plot_type)

if __name__ == "__main__":
    main()
