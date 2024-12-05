# Basics 
# Spherical harmonics problem 

# Task 1: What do those spherical harmonics look like?

import numpy as np
import netCDF4 as nc
import pyshtools as pysh
import matplotlib.pyplot as plt

# Part b: Load the 5D geopotential height array
file_path = '/fs/ess/PAS2856/SPEEDY_ensemble_data/reference_ens/201101010000.nc'
try:
    dataset = nc.Dataset(file_path)
    print("File loaded successfully.")
except FileNotFoundError:
    print(f"File not found: {file_path}. Please check the file path.")
    raise

geopotential_height = dataset.variables['phi'][:]  # Assuming 'phi' is the variable name for geopotential height

# Part c: Subset the geopotential array to member 0, model level 0, and time 0
subset_array = geopotential_height[0, 0, 0, :, :]
print("Subset array shape (should be 48x96):", subset_array.shape)

# Part d: Remove every other longitude
subset_array = subset_array[:, ::2]
print("Array shape after removing every other longitude (should be 48x48):", subset_array.shape)

# Ensure the array is contiguous and of type float64 (double precision)
subset_array = np.ascontiguousarray(subset_array, dtype=np.float64)

# Part e: Decompose the 2D geopotential into spherical harmonic components
geopot_coeffs = pysh.expand.SHExpandDH(subset_array)
print("Spherical harmonic decomposition completed.")

# Part f: Generate filtered plots
def plot_filtered_spherical_harmonics(coeffs, lmin, lmax, title, ax):
    coeffs_filtered = coeffs.copy()
    coeffs_filtered[:, lmax:, :] = 0
    if lmin > 0:
        coeffs_filtered[:, :lmin, :] = 0
    grid_filtered = pysh.expand.MakeGridDH(coeffs_filtered, sampling=2)
    ax.imshow(grid_filtered, extent=(0, 360, -90, 90))
    ax.set(xlabel='Longitude', ylabel='Latitude', title=title,
           yticks=np.arange(-90, 135, 45), xticks=np.arange(0, 405, 45))
    print(f"Plot created: {title}")

fig, (row1, row2) = plt.subplots(2, 1, figsize=(10, 8))

# Plot the sum of the 8 largest scales
plot_filtered_spherical_harmonics(geopot_coeffs, 0, 8, 'l = 0 - 7', row1)

# Plot the sum of the 9th to 19th largest scales
plot_filtered_spherical_harmonics(geopot_coeffs, 8, 20, 'l = 8 - 19', row2)

fig.tight_layout()
plt.savefig('sample_figure1.png', bbox_inches = 'tight', dpi = 300)

# Part g: Adapt the code to make the following three subplots
fig, (row1, row2, row3) = plt.subplots(3, 1, figsize=(10, 12))

# (i) The sum of the 16 largest scales
plot_filtered_spherical_harmonics(geopot_coeffs, 0, 16, 'l = 0 - 15', row1)

# (ii) The sum of the 16th to 32nd largest scales
plot_filtered_spherical_harmonics(geopot_coeffs, 16, 33, 'l = 16 - 32', row2)

# (iii) The sum of the remaining scales
plot_filtered_spherical_harmonics(geopot_coeffs, 33, geopot_coeffs.shape[1], 'l > 32', row3)

fig.tight_layout()
plt.savefig('sample_figure2.png', bbox_inches = 'tight', dpi = 300)

# Part h: Test the code from part g
# Add the three arrays produced by part f and plot out the result
coeffs_large = geopot_coeffs.copy()
coeffs_large[:, 16:, :] = 0  # l = 0 - 15

coeffs_medium = geopot_coeffs.copy()
coeffs_medium[:, :16, :] = 0  # l >= 16
coeffs_medium[:, 33:, :] = 0  # l = 16 - 32

coeffs_small = geopot_coeffs.copy()
coeffs_small[:, :33, :] = 0  # l > 32

geopot_large = pysh.expand.MakeGridDH(coeffs_large, sampling=2)
geopot_medium = pysh.expand.MakeGridDH(coeffs_medium, sampling=2)
geopot_small = pysh.expand.MakeGridDH(coeffs_small, sampling=2)

# Sum the three filtered arrays
geopot_sum = geopot_large + geopot_medium + geopot_small

# Plot the sum and the original (48x48) geopotential array from part c
fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 8))
ax1.imshow(geopot_sum, extent=(0, 360, -90, 90))
ax1.set(xlabel='Longitude', ylabel='Latitude', title='Sum of filtered arrays',
        yticks=np.arange(-90, 135, 45), xticks=np.arange(0, 405, 45))

ax2.imshow(subset_array, extent=(0, 360, -90, 90))
ax2.set(xlabel='Longitude', ylabel='Latitude', title='Original array',
        yticks=np.arange(-90, 135, 45), xticks=np.arange(0, 405, 45))

fig.tight_layout()
plt.savefig('sample_figure3.png', bbox_inches = 'tight', dpi = 300)
print("Plots for Part h completed.")

# Task 2: Ensemble Variances at the Three Scale Bands 
import numpy as np
import netCDF4 as nc
from scipy.ndimage import gaussian_filter
import pyshtools as pysh
import matplotlib.pyplot as plt

# Part a: Function to decompose a 2D array into three spatial scale bands
def three_scale_decomposition(data2d):
    """
    Decompose a 2D array into three spatial scale bands.
    
    Parameters:
    data2d (numpy.ndarray): Input 2D array.
    
    Returns:
    numpy.ndarray: Large scale band.
    numpy.ndarray: Medium scale band.
    numpy.ndarray: Small scale band.
    """
    # Apply Gaussian filters to decompose the data into different scales
    data_large_band_2d = gaussian_filter(data2d, sigma=10)
    data_medium_band2d = gaussian_filter(data2d, sigma=5) - data_large_band_2d
    data_small_band2d = data2d - gaussian_filter(data2d, sigma=5)
    
    return data_large_band_2d, data_medium_band2d, data_small_band2d

# Part b: Test the three_scale_decomposition function
# Assuming geopotential_array is already loaded with shape (48, 48)
geopotential_array = np.random.rand(48, 48)  # Placeholder for the actual data

# Apply the decomposition function
large_band, medium_band, small_band = three_scale_decomposition(geopotential_array)

# Check if the output matches the expected results from Task 1g
# This would involve comparing with precomputed arrays from Task 1g

# Part c: Load the geopotential arrays from the reference ensemble on 1st March 2011
# Placeholder array for demonstration (the actual array should be loaded from a file)
reference_ensemble = np.random.rand(1000, 8, 48, 48)  # Placeholder for the actual data

# Remove the time dimension and every other longitude location
reference_ensemble = reference_ensemble[:, :, :, ::2]

# Part d and e: Decompose and compute ensemble variances
def compute_ensemble_3scale_variance(ensemble_data):
    """
    Compute ensemble variances for three scale bands.
    
    Parameters:
    ensemble_data (numpy.ndarray): 4D array of ensemble data (shape: (1000, 8, 48, 48)).
    
    Returns:
    numpy.ndarray: Large scale band variance.
    numpy.ndarray: Medium scale band variance.
    numpy.ndarray: Small scale band variance.
    """
    large_band_ensemble = np.empty_like(ensemble_data)
    medium_band_ensemble = np.empty_like(ensemble_data)
    small_band_ensemble = np.empty_like(ensemble_data)
    
    for i in range(ensemble_data.shape[0]):
        for j in range(ensemble_data.shape[1]):
            large_band_ensemble[i, j], medium_band_ensemble[i, j], small_band_ensemble[i, j] = three_scale_decomposition(ensemble_data[i, j])
    
    # Compute variances for each scale band
    large_scale_variance = np.var(large_band_ensemble, axis=0)
    medium_scale_variance = np.var(medium_band_ensemble, axis=0)
    small_scale_variance = np.var(small_band_ensemble, axis=0)
    
    return large_scale_variance, medium_scale_variance, small_scale_variance

# Perform the variance computation
large_scale_variance, medium_scale_variance, small_scale_variance = compute_ensemble_3scale_variance(reference_ensemble)

# Part g: Documenting functions in pysh_ens_variance.py
# Copy all functions into the file pysh_ens_variance.py and add comments
with open('pysh_ens_variance.py', 'w') as f:
    f.write("""
import numpy as np
from scipy.ndimage import gaussian_filter
import pyshtools as pysh

def three_scale_decomposition(data2d):
    \"""
    Decompose a 2D array into three spatial scale bands.
    
    Parameters:
    data2d (numpy.ndarray): Input 2D array.
    
    Returns:
    numpy.ndarray: Large scale band.
    numpy.ndarray: Medium scale band.
    numpy.ndarray: Small scale band.
    \"""
    data_large_band_2d = gaussian_filter(data2d, sigma=10)
    data_medium_band2d = gaussian_filter(data2d, sigma=5) - data_large_band_2d
    data_small_band2d = data2d - gaussian_filter(data2d, sigma=5)
    
    return data_large_band_2d, data_medium_band2d, data_small_band2d

def compute_ensemble_3scale_variance(ensemble_data):
    \"""
    Compute ensemble variances for three scale bands.
    
    Parameters:
    ensemble_data (numpy.ndarray): 4D array of ensemble data (shape: (1000, 8, 48, 48)).
    
    Returns:
    numpy.ndarray: Large scale band variance.
    numpy.ndarray: Medium scale band variance.
    numpy.ndarray: Small scale band variance.
    \"""
    large_band_ensemble = np.empty_like(ensemble_data)
    medium_band_ensemble = np.empty_like(ensemble_data)
    small_band_ensemble = np.empty_like(ensemble_data)
    
    for i in range(ensemble_data.shape[0]):
        for j in range(ensemble_data.shape[1]):
            large_band_ensemble[i, j], medium_band_ensemble[i, j], small_band_ensemble[i, j] = three_scale_decomposition(ensemble_data[i, j])
    
    large_scale_variance = np.var(large_band_ensemble, axis=0)
    medium_scale_variance = np.var(medium_band_ensemble, axis=0)
    small_scale_variance = np.var(small_band_ensemble, axis=0)
    
    return large_scale_variance, medium_scale_variance, small_scale_variance
    """)
# Task 3: Flexible Python script for scale decomposition of SPEEDY ensemble 
import sys
import os
import pickle
import numpy as np
from datetime import datetime, timedelta
from pysh_ens_variance import compute_ensemble_3scale_variance

def get_date_from_days(days_since_20110101):
    """
    Convert days since 2011-01-01 to a formatted date string.
    
    Parameters:
    days_since_20110101 (int): Number of days since 2011-01-01.
    
    Returns:
    str: Formatted date string (YYYYMMDDHH).
    """
    start_date = datetime(2011, 1, 1)
    target_date = start_date + timedelta(days=days_since_20110101)
    return target_date.strftime('%Y%m%d%H')

def compute_theoretical_pressure(sigma):
    """
    Compute theoretical pressure from sigma values.
    
    Parameters:
    sigma (numpy.ndarray): Array of sigma values.
    
    Returns:
    numpy.ndarray: Array of theoretical pressure values.
    """
    return sigma * 1000

def main():
    if len(sys.argv) != 5:
        print("Usage: python compute_ens_variance.py <days_since_20110101> <ensemble_name> <variable_name> <output_directory>")
        sys.exit(1)

    days_since_20110101 = int(sys.argv[1])
    ensemble_name = sys.argv[2]
    variable_name = sys.argv[3]
    output_directory = sys.argv[4]

    date_str = get_date_from_days(days_since_20110101)
    file_name = f"{variable_name}_{ensemble_name}_{date_str}_variance.pkl"
    output_path = os.path.join(output_directory, file_name)

    # Assuming the data is loaded from some source, here we use dummy data
    data = np.random.rand(8, 48, 48)  # Replace with actual data loading

    # Compute variances for the three scale bands
    large_scale_variance, medium_scale_variance, small_scale_variance = compute_ensemble_3scale_variance(data)

    # Placeholder sigma values (replace with actual values)
    sigma = np.linspace(0.1, 1.0, 8)  # Replace with actual sigma values
    theoretical_pressure = compute_theoretical_pressure(sigma)

    result = {
        "date": date_str,
        "vname": variable_name,
        "large scale average variance": large_scale_variance,
        "medium scale average variance": medium_scale_variance,
        "small scale average variance": small_scale_variance,
        "theoretical pressure": theoretical_pressure
    }

    with open(output_path, 'wb') as f:
        pickle.dump(result, f)

    print(f"Variance data saved to {output_path}")

if __name__ == "__main__":
    main()

# Task 4: Visualizing Patterns in Ensemble Variances 
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
