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

    days_since_20110101 = int(sys.argv[1]) 
    ensemble_name = sys.argv[2] #accepted inputs: reference_ens or perturbed_ens
    variable_name = sys.argv[3] 
    #netcdf_file = sys.argv[4]
    output_directory = sys.argv[5]

    date_str = get_date_from_days(days_since_20110101)
    file_name = f"{variable_name}_{ensemble_name}_{date_str}_variance.pkl"
    output_path = os.path.join(output_directory, file_name)

    #Construct the filename for the netCDF file based on the provided ensemble and date
    netcdf_file = f"/fs/ess/PAS2856/SPEEDY_ensemble_data/{ensemble_name}/{get_date_from_days(days_since_20110101)}.nc"

    # Read data from the netCDF file
    with nc.Dataset(netcdf_file, 'r') as dataset:
        geopot_original = np.array(dataset.variables[variable_name][:, 0, :, :, ::2], dtype='f8') #read in 4D array for compute_ensemble_3scale_variance
        sigma = dataset.variables['lev'][:] #'lev' = atmosphere_sigma_coordinate from netCDF file
    # Remove every other longitude to make the shape (48,48)
    #geopot_subsetted = np.array(geopot_original[:, ::2], dtype='f8')
    
    # Perform the spherical harmonic expansion
    geopot_coeffs = pysh.expand.SHExpandDH(geopot_original)
    '''
    #Extract data for all model levels and compute variances for each level across ensemble members
    small_scale_variance = np.zeros_like(geopot_coeffs[0, 0, :, :])  
    medium_scale_variance = np.zeros_like(geopot_coeffs[0, 0, :, :]) 
    large_scale_variance = np.zeros_like(geopot_coeffs[0, 0, :, :])

    for level in range(geopot_coeffs.shape[2]):  #Loop over all 8 model levels
        level_data = geopot_coeffs[:, 0, level, :, :]  #Shape (ensemble, lat, lon) for a specific level
        
        #Compute the variance for the current level across the ensemble members
        l_scale, m_scale, s_scale = compute_ensemble_3scale_variance(level_data)
        
        #Store the variances for the current level
        large_scale_variance[level, :, :] = l_scale
        medium_scale_variance[level, :, :] = m_scale
        small_scale_variance[level, :, :] = s_scale
    '''

    # Compute variances for the three scale bands
    large_scale_variance, medium_scale_variance, small_scale_variance = compute_ensemble_3scale_variance(geopot_original)
    
    #theoretical pressure using sigma from netCDF file:  
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