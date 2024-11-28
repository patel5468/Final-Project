#Activate the environment
#>>conda activate final_proj_env
#Import necessary packages
import numpy as np
import netCDF4 as nc
import pyshtools as pysh
import matplotlib.pyplot as plt

# -------- TASK 1 --------

#Load the NetCDF file containing geopotential data
f = nc.Dataset('/fs/ess/PAS2856/SPEEDY_ensemble_data/reference_ens/201101010000.nc', 'r')

#create variable geopotential based on ncdump and subset the array to have only lat and lon
geopot_original = f.variables['phi'][0, 0, 0, :, :]
#print(geopot_original.shape)#Verify shape is (48,96)

#remove every other longitude to make the shape (48,48)
geopot_subsetted = np.array(geopot_original[:, ::2], dtype='f8')  #This means...(all lat, every other lon) yields shape (48,48)
#print(geopot_subsetted.shape) #Verify shape is (48,48)

#Perform the spherical harmonic expansion
geopot_coeffs = pysh.expand.SHExpandDH(geopot_subsetted)
'''
#Power Spectrum Calculation
power_per_l = pysh.spectralanalysis.spectrum(geopot_coeffs)

#Plot the power spectrum
degrees = np.arange(geopot_coeffs.shape[1]) #simple way to get the range of degrees for longitude
fig, ax = plt.subplots(1, 1, figsize=(6, 6)) #create empty figure
ax.plot(degrees, power_per_l) #plot the power spectrum
ax.set(yscale='log', xscale='log', xlabel='Spherical harmonic degree', ylabel='Power')
ax.grid() #Add grids
plt.title("Power Spectrum") #add title to plot
plt.savefig('Power_spectrum.png')#Save the file as a png

#Second power spectrum plot
power_per_lm = pysh.spectralanalysis.spectrum(geopot_coeffs, unit='per_lm')

fig, ax = plt.subplots(1, 1)
ax.plot(degrees, power_per_lm)
ax.set(xscale='log', yscale='log', xlabel='Spherical harmonic degree', ylabel='Power')
ax.grid()
plt.title("Power Spectrum per 1m")
plt.savefig('Power_spectrum_per1m')

#Third power spectrum
power_per_dlogl = pysh.spectralanalysis.spectrum(geopot_coeffs, unit='per_dlogl', base=2.)

fig, ax = plt.subplots(1, 1)
ax.plot(degrees, power_per_dlogl)
ax.set_yscale('log', base=2)
ax.set_xscale('log', base=2)
ax.set(ylabel='Power', xlabel='Spherical harmonic degree')
ax.grid()
plt.title("Power Over Infinite Log. Degree Band")
plt.savefig('Power_spectrum_infinite_deg.png')

#Filter Setup 
geopot_coeffs_filtered = geopot_coeffs.copy()
lmax = 8
geopot_coeffs_filtered[:,lmax:,:] = 0 #Colon next to lmax (lmax:) to indicate I want values larger than the lmax specified
topo_filtered = pysh.expand.MakeGridDH(geopot_coeffs_filtered, sampling=2)

geopot_coeffs_filtered2 = geopot_coeffs.copy()
lmin, lmax = 8, 20
geopot_coeffs_filtered2[:,:lmin, :] = 0 #The colon to the left of the lmin (:lmin)represents all values less than the lmin specified. 
geopot_coeffs_filtered2[:,lmax:, :] = 0 #The colon to the right of the lmax (lmax:) represents all values greater than the lmax specified
topo_filtered2 = pysh.expand.MakeGridDH(geopot_coeffs_filtered2, sampling=2)

fig, (row1,row2) = plt.subplots(2,1) #Create empty figure

row1.imshow(topo_filtered, extent=(0,360,-90,90))
row1.set(xlabel='Longitude', ylabel='Latitude', title='1 = 0-7', 
            yticks=np.arange(-90,135,45), xticks=np.arange(0,405,45)
        )

row2.imshow(topo_filtered2, extent=(0, 360, -90, 90))
row2.set(xlabel='Longitude', ylabel='Latitude', title='l = 8 - 19', 
            yticks=np.arange(-90,135,45), xticks=np.arange(0,405,45)
        )

plt.tight_layout()
plt.savefig('test.png')
'''

#Filter for 8th largest scale
geopot_coeffs_filtered1 = geopot_coeffs.copy() #copies the indices and data of geopot_coeffs for manipulation without altering original dataset
lmax1 = 8 #sets the max (start point) to the 8th largest scale
geopot_coeffs_filtered1[:, lmax1:, :] = 0  #Set values for l > 8 to 0
topo_filtered1 = pysh.expand.MakeGridDH(geopot_coeffs_filtered1, sampling=1) #Performs the spherical harmonics expansion on the filtered dataset

#2nd plot: Sum of the 8th to 16nd largest scales (l = 8 to 16)
geopot_coeffs_filtered2 = geopot_coeffs.copy() #copies the indices and data of geopot_coeffs for manipulation without altering original dataset
lmin2, lmax2 = 8, 16 #Sets the range of largest scales as 8 : 16
geopot_coeffs_filtered2[:, :lmin2, :] = 0  #Set values for l < 8 to 0
geopot_coeffs_filtered2[:, lmax2:, :] = 0  #Set values for l > 16 to 0
topo_filtered2 = pysh.expand.MakeGridDH(geopot_coeffs_filtered2, sampling=1) #Performs the spherical harmonics expansion on the filtered dataset

#3rd plot: Sum of the remaining scales (l > 16)
geopot_coeffs_filtered3 = geopot_coeffs.copy() #copies the indices and data of geopot_coeffs for manipulation without altering original dataset
lmin3 = 16 #sets the min value to 16th largest scales
geopot_coeffs_filtered3[:, :lmin3, :] = 0  #Set values for l < 16 to 0
topo_filtered3 = pysh.expand.MakeGridDH(geopot_coeffs_filtered3, sampling=1) #Performs the spherical harmonics expansion on the filtered dataset

#Create empty figure with 3 rows and 1 column for plots
fig, axes = plt.subplots(3, 1, figsize=(8, 12))

#Plot the sum of the 16 largest scales and save it as the first plot
axes[0].imshow(topo_filtered1, extent=(0, 360, -90, 90))  #1st matplotlib imshow plot in figure 
axes[0].set(xlabel='Longitude', ylabel='Latitude', title='1 = 0 - 7', #setting labels and titles as well as ranges for x and y axis
            yticks=np.arange(-90, 135, 45), xticks=np.arange(0, 405, 45)
        )

#Plot the sum of the 16th to 32nd largest scales and save it as the second plot
axes[1].imshow(topo_filtered2, extent=(0, 360, -90, 90))  #2nd matplotlib imshow plot in figure 
axes[1].set(xlabel='Longitude', ylabel='Latitude', title='l = 8 - 16', #setting labels and titles as well as ranges for x and y axis
            yticks=np.arange(-90, 135, 45), xticks=np.arange(0, 405, 45)
        )

#Plot the sum of the remaining scales (l > 32) and save it as the third plot
axes[2].imshow(topo_filtered3, extent=(0, 360, -90, 90))  #3rd matplotlib imshow plot in figure 
axes[2].set(xlabel='Longitude', ylabel='Latitude', title='l > 16',  #setting labels and titles as well as ranges for x and y axis
            yticks=np.arange(-90, 135, 45), xticks=np.arange(0, 405, 45)
        )

#Format the figure to fit each plot within it and eliminate excess white space
plt.tight_layout()
#Save the figure as a PNG
#plt.savefig('geopotential_scales.png')

#Sum the three topographic grids
topo_sum = topo_filtered1 + topo_filtered2 + topo_filtered3
#print(topo_sum.shape)

#Plot the sum
fig, ax = plt.subplots(figsize=(8, 6)) #creates an empty figure to plot the sums
ax.imshow(topo_sum, extent=(0, 360, -90, 90)) #matplotlib imshow to plot the sums as "a pseudocolor image"
ax.set(xlabel='Longitude', ylabel='Latitude', title='Sum of all scales', #setting labels and titles as well as ranges for x and y axis
        yticks=np.arange(-90, 135, 45), xticks=np.arange(0, 405, 45)
        )

#Adjust layout
plt.tight_layout()
#Save the result
#plt.savefig('sum_of_all_scales.png')

# -------- TASK 2 --------

def three_scale_decomposition(data2d): #function named three_scale_decomposition which takes a 2d array as an input
    """
    Decompose a 2D array (data2d) into three spatial scale bands:
    - Large scale (low frequencies)
    - Medium scale (intermediate frequencies)
    - Small scale (high frequencies)
    
    Parameters:
    -----------
    data2d : 2D numpy array
        Input 2D array representing the geopotential or other spatial data.
    
    Returns:
    --------
    data_large_band_2d : 2D numpy array
        The 2D array with large scales (low-frequency components).
    
    data_medium_band_2d : 2D numpy array
        The 2D array with medium scales (intermediate-frequency components).
    
    data_small_band_2d : 2D numpy array
        The 2D array with small scales (high-frequency components).
    """
    
    #Perform spherical harmonic expansion
    geopot_coeffs = pysh.expand.SHExpandDH(data2d)
    
    #Define the scale bands for decomposition:
        #Large scale: l = 0 to 7
        #Medium scale: l = 8 to 16
        #Small scale: l > 16
    #The above scales are based on Task 1 definitions

    #Filter for large scale (0 to 7)
    geopot_coeffs_large = geopot_coeffs.copy() #Creates copy of geopot_coeffs to manipulate for the large scale
    lmax_large = 7 #max for large scale
    geopot_coeffs_large[:, lmax_large+1:, :] = 0  #Set all geopotential coefficients beyond lmax_large to zero.
    data_large_band_2d = pysh.expand.MakeGridDH(geopot_coeffs_large, sampling=1) #Performs spherical harmonic expansion to the large scale dataset
    
    #Filter for medium scale (8 to 16)
    geopot_coeffs_medium = geopot_coeffs.copy() #Creates copy of geopot_coeffs to manipulate for the medium scale
    lmin_medium, lmax_medium = 8, 16 #range of scale
    geopot_coeffs_medium[:, :lmin_medium, :] = 0 #Set all geopotential coefficients below lmin_medium to zero.
    geopot_coeffs_medium[:, lmax_medium+1:, :] = 0  #Set all geopotential coefficients above lmax_medium to zero. 
    data_medium_band_2d = pysh.expand.MakeGridDH(geopot_coeffs_medium, sampling=1) #Performs spherical harmonic expansion to the medium scale dataset
    
    #Filter for small scale (l > 16)
    geopot_coeffs_small = geopot_coeffs.copy() #Creates copy of geopot_coeffs to manipulate for the small scale
    lmin_small = 16 #min for small scale
    geopot_coeffs_small[:, :lmin_small, :] = 0 #Set all geopotential coefficients below lmin_small to zero.
    data_small_band_2d = pysh.expand.MakeGridDH(geopot_coeffs_small, sampling=1) #Performs spherical harmonic expansion to the small scale dataset
    
    return data_large_band_2d, data_medium_band_2d, data_small_band_2d #returns the expanded datasets (2d arrays)

#Testing the function by applying it to the geopotential array of member 0, model number 0, and time index 0


#The function decomposes the data into three components: 
#-->data_large: captures the large-scale components.
#-->data_medium: captures the medium-scale components.
#-->data_small: captures the small-scale components.
data_large, data_medium, data_small = three_scale_decomposition(geopot_subsetted)

fig, axes = plt.subplots(3, 1, figsize=(8, 12)) #Creates empty figure with 3 rows, 1 column. Specifies figsize to 8x12

axes[0].imshow(data_large, extent=(0, 360, -90, 90)) #First plot to show large scale components
axes[0].set_title('Large Scale (l = 0-7)') #Sets title for the plot
axes[1].imshow(data_medium, extent=(0, 360, -90, 90)) #Second plot to show medium scale components
axes[1].set_title('Medium Scale (l = 8-16)') #Sets title for the plot
axes[2].imshow(data_small, extent=(0, 360, -90, 90)) #Third plot to show small scale components
axes[2].set_title('Small Scale (l > 16)') #Sets title for the plot

plt.tight_layout() #Fixes the plot to be neat and tidy
#plt.savefig('three_scale_decomposition_test.png')
#print(data_large.shape, data_medium.shape, data_small.shape) #OUTPUT: (48, 48) (48, 48) (48, 48)
    #Task 2.b, yes the three arrays (three_scale_decomposition_test.png) match the output from  task 1 g (geopotential_scales.png). 

#Load March 1 2011 geopotential arrays: 
f_march = nc.Dataset('/fs/ess/PAS2856/SPEEDY_ensemble_data/reference_ens/201103010000.nc', 'r')
#Load geopotential data keeping everything except time...Format via ncdump = (ensemble, time, lev, lat, lon)
geopot_march = f.variables['phi'][:, 0, :, :, ::2] #This loads all data except time, and loads every other longitude value
#print(geopot_march.shape) #confirm that shape = (1000,8,48,48)

#Initialize arrays to store the decomposition results
data_large_band_2d = np.zeros((1000, 8, 48, 48), dtype='f8')
data_medium_band_2d = np.zeros((1000, 8, 48, 48), dtype='f8')
data_small_band_2d = np.zeros((1000, 8, 48, 48), dtype='f8')

#Loop over all ensemble members and apply three_scale_decomposition
for ens in range(geopot_march.shape[0]):  #Loop over 1000 ensemble members
    for mod in range(geopot_march.shape[1]):  #Loop over 8 model levels
        march_geopot_2d = np.array(geopot_march[ens, mod, :, :], dtype='f8') #Extract 2D geopotential data for each ensemble member and level
        #I added np.array specifically to specify float type to 64 instead of 32 because 32 was not working and it worked after converting the float type...similar issue occured above.
        #print(march_geopot_2d.shape) #--> (48,48)
        
        #Apply three_scale_decomposition to each 2D array
        data_large, data_medium, data_small = three_scale_decomposition(march_geopot_2d)

        #Store the results in the corresponding arrays
        data_large_band_2d[ens, mod, :, :] = data_large
        data_medium_band_2d[ens, mod, :, :] = data_medium
        data_small_band_2d[ens, mod, :, :] = data_small

#Verify the resulting shape for each of the decomposed arrays
#print(data_large_band_2d.shape,data_medium_band_2d.shape,data_small_band_2d.shape) #OUTPUT--> (1000, 8, 48, 48) (1000, 8, 48, 48) (1000, 8, 48, 48)


#Compute the variance for each scale band array (large, medium, and small)
def compute_ensemble_variance(data_band_2d):
    """
    Compute the ensemble variance for each spatial scale band.
    
    Parameters:
    -----------
    data_band_2d : 4D numpy array
        The 4D array of shape (ensemble, model levels, lat, lon)
    
    Returns:
    --------
    variance : 3D numpy array
        The computed variance for each spatial scale, with shape (model_levels, lat, lon)
    """
    #Calculate the variance along the ensemble axis (axis 0)
    variance = np.var(data_band_2d, axis=0)  #Variance along the first axis (ensemble dimension)
    
    return variance

#Compute variances for large, medium, and small scale bands
variance_large = compute_ensemble_variance(data_large_band_2d)
variance_medium = compute_ensemble_variance(data_medium_band_2d)
variance_small = compute_ensemble_variance(data_small_band_2d)

#Verify the resulting shape for each variance array
#print(variance_large.shape, variance_medium.shape, variance_small.shape )  #OUTPUT --> (8, 48, 48) (8, 48, 48) (8, 48, 48)

#Plotting variance for large, medium, and small scale bands because...Why,not?
fig, axes = plt.subplots(3, 1, figsize=(8, 12))

axes[0].imshow(variance_large[0, :, :], extent=(0, 360, -90, 90))
axes[0].set_title('Variance - Large Scale (l = 0-7)')
axes[1].imshow(variance_medium[0, :, :], extent=(0, 360, -90, 90))
axes[1].set_title('Variance - Medium Scale (l = 8-16)')
axes[2].imshow(variance_small[0, :, :], extent=(0, 360, -90, 90))
axes[2].set_title('Variance - Small Scale (l > 16)')

plt.tight_layout()
#plt.savefig('variance_visualization.png')

#Part d and e: Decompose and compute ensemble variances--> Shuvo's part from final_project.py (for my reference)
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
    
    for i in range(ensemble_data.shape[0]): #Iterate over the rows of the ensemble data (ensemble_data.shape[0])
        for j in range(ensemble_data.shape[1]): #Iterate over the columns of the ensemble data (ensemble_data.shape[1])
            #Perform a three-scale decomposition on each individual element (ensemble_data[i, j])
            #Decompose into large, medium, and small scale components
            large_band_ensemble[i, j], medium_band_ensemble[i, j], small_band_ensemble[i, j] = three_scale_decomposition(ensemble_data[i, j])
    
    #Compute variances for each scale band   
    large_scale_variance = np.var(large_band_ensemble, axis=0)
    medium_scale_variance = np.var(medium_band_ensemble, axis=0)
    small_scale_variance = np.var(small_band_ensemble, axis=0)
    
    return large_scale_variance, medium_scale_variance, small_scale_variance

#Sanity Checks for compute_ensemble_3scale_variance funtion
# --------- TASK 3 ---------- #
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
    #data = np.random.rand(8, 48, 48)  # Replace with actual data loading 

    # Compute variances for the three scale bands
    large_scale_variance, medium_scale_variance, small_scale_variance = compute_ensemble_3scale_variance(geopot_subsetted)

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