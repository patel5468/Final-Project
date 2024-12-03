# Task 3: Flexible Python script for scale decomposition of SPEEDY ensemble 
import sys
import os
import pickle
import numpy as np
from datetime import datetime, timedelta
import netCDF4 as nc
from pysh_ens_variance import compute_ensemble_3scale_variance 
import pyshtools as pysh

def get_date_from_days(days_since_20110101):
    """
    Convert days since 2011-01-01 to a formatted date string.
    
    Parameters:
    days_since_20110101 (int): Number of days since 2011-01-01.
    
    Returns:
    str: Formatted date string (YYYYMMDD%H).
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
    if len(sys.argv) != 6:
        print("Usage: python compute_ens_variance.py <days_since_20110101> <ensemble_name> <variable_name> <netcdf_file> <output_directory>")
        sys.exit(1)

    days_since_20110101 = int(sys.argv[1]) #Days since for date command input
    ensemble_name = sys.argv[2] #accepted inputs: reference_ens or perturbed_ens
    variable_name = sys.argv[3] #Command input for variable names
    #netcdf_file = sys.argv[4]
    output_directory = sys.argv[4] #Command input for the output directory

    date_str = get_date_from_days(days_since_20110101) #calls get_date_from_days function on inputted date
    file_name = f"{variable_name}_{ensemble_name}_{date_str}_variance.pkl" #Names the file for pickle storage
    output_path = os.path.join(output_directory, file_name) #Specifies the output path based on file name and command inputted output directory

    #Construct the filename for the netCDF file based on the provided ensemble and date
    netcdf_file = f"/fs/ess/PAS2856/SPEEDY_ensemble_data/{ensemble_name}/{get_date_from_days(days_since_20110101)}.nc"

    # Read data from the netCDF file
    with nc.Dataset(netcdf_file, 'r') as dataset:
        geopot_original = np.array(dataset.variables[variable_name][:, 0, :, :, ::2], dtype='f8') #read in 4D array for compute_ensemble_3scale_variance and get every other lon value for shape of (1000, 8, 48, 48)
        sigma = dataset.variables['lev'][:] #'lev' = atmosphere_sigma_coordinate from netCDF file
    # Perform the spherical harmonic expansion
    geopot_coeffs = pysh.expand.SHExpandDH(geopot_original)


    # Compute variances for the three scale bands
    large_scale_variance, medium_scale_variance, small_scale_variance = compute_ensemble_3scale_variance(geopot_original)
    
    #theoretical pressure using sigma from netCDF file:  
    theoretical_pressure = compute_theoretical_pressure(sigma)
    #Create a dictionary to store the results
    result = {
        "date": date_str, #Date in string format after get_date_from_days funtion run on the command input
        "vname": variable_name, #The name or symbol of the variable being processed from command input
        "large scale average variance": large_scale_variance, #Variance at the large scale level from compute_ensemble_3scale_variance
        "medium scale average variance": medium_scale_variance, #Variance at the medium scale level from compute_ensemble_3scale_variance
        "small scale average variance": small_scale_variance, #Variance at the small scale level from compute_ensemble_3scale_variance
        "theoretical pressure": theoretical_pressure #Theoretical pressure value from compute_theoretical_pressure based on sigma from the netCDF file
    }

    with open(output_path, 'wb') as f: #Open the file specified by output_path in write mode
        pickle.dump(result, f)  #Use pickle to save the 'result' dictionary and write it to the file

    print(f"Variance data saved to {output_path}") #Confirmation message to indicate that the data was saved successfully

if __name__ == "__main__": #Check if this script is being run directly (not imported as a module)
    main() #Calls the main function to start the program