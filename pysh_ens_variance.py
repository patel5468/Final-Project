#Activate the environment
#>>conda activate final_proj_env
# Import necessary packages
import numpy as np
import netCDF4 as nc
import pyshtools as pysh
import matplotlib.pyplot as plt

# -------- TASK 1 --------

# Load the NetCDF file containing geopotential data
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

# Plot the power spectrum
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
geopot_coeffs_filtered1 = geopot_coeffs.copy()
lmax1 = 8
geopot_coeffs_filtered1[:, lmax1:, :] = 0  #Set values for l > 8 to 0
topo_filtered1 = pysh.expand.MakeGridDH(geopot_coeffs_filtered1, sampling=1)

#2nd plot: Sum of the 8th to 16nd largest scales (l = 8 to 16)
geopot_coeffs_filtered2 = geopot_coeffs.copy()
lmin2, lmax2 = 8, 16
geopot_coeffs_filtered2[:, :lmin2, :] = 0  #Set values for l < 8 to 0
geopot_coeffs_filtered2[:, lmax2:, :] = 0  #Set values for l > 16 to 0
topo_filtered2 = pysh.expand.MakeGridDH(geopot_coeffs_filtered2, sampling=1)

#3rd plot: Sum of the remaining scales (l > 16)
geopot_coeffs_filtered3 = geopot_coeffs.copy()
lmin3 = 16
geopot_coeffs_filtered3[:, :lmin3, :] = 0  #Set values for l < 16 to 0
topo_filtered3 = pysh.expand.MakeGridDH(geopot_coeffs_filtered3, sampling=1)

#Create empty figure with 3 rows and 1 column for plots
fig, axes = plt.subplots(3, 1, figsize=(8, 12))

#Plot the sum of the 16 largest scales and save it as the first plot
axes[0].imshow(topo_filtered1, extent=(0, 360, -90, 90))
axes[0].set(xlabel='Longitude', ylabel='Latitude', title='1 = 0 - 7', 
            yticks=np.arange(-90, 135, 45), xticks=np.arange(0, 405, 45)
        )

#Plot the sum of the 16th to 32nd largest scales and save it as the second plot
axes[1].imshow(topo_filtered2, extent=(0, 360, -90, 90))
axes[1].set(xlabel='Longitude', ylabel='Latitude', title='l = 8 - 16', 
            yticks=np.arange(-90, 135, 45), xticks=np.arange(0, 405, 45)
        )

#Plot the sum of the remaining scales (l > 32) and save it as the third plot
axes[2].imshow(topo_filtered3, extent=(0, 360, -90, 90))
axes[2].set(xlabel='Longitude', ylabel='Latitude', title='l > 16', 
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
fig, ax = plt.subplots(figsize=(8, 6))
ax.imshow(topo_sum, extent=(0, 360, -90, 90))
ax.set(xlabel='Longitude', ylabel='Latitude', title='Sum of all scales', 
        yticks=np.arange(-90, 135, 45), xticks=np.arange(0, 405, 45)
        )

#Adjust layout
plt.tight_layout()
#Save the result
#plt.savefig('sum_of_all_scales.png')

# -------- TASK 2 --------

def three_scale_decomposition(data2d):
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
    
    #Perform spherical harmonic expansion to get the coefficients
    geopot_coeffs = pysh.expand.SHExpandDH(data2d)
    
    #Define the scale bands for decomposition:
        #Large scale: l = 0 to 7
        #Medium scale: l = 8 to 16
        #Small scale: l > 16
    #The above scales are based on Task 1 definitions

    #Filter for large scale (0 to 7)
    geopot_coeffs_large = geopot_coeffs.copy()
    lmax_large = 7
    geopot_coeffs_large[:, lmax_large+1:, :] = 0  
    data_large_band_2d = pysh.expand.MakeGridDH(geopot_coeffs_large, sampling=1)
    
    #Filter for medium scale (8 to 16)
    geopot_coeffs_medium = geopot_coeffs.copy()
    lmin_medium, lmax_medium = 8, 16
    geopot_coeffs_medium[:, :lmin_medium, :] = 0 
    geopot_coeffs_medium[:, lmax_medium+1:, :] = 0  
    data_medium_band_2d = pysh.expand.MakeGridDH(geopot_coeffs_medium, sampling=1)
    
    #Filter for small scale (l > 16)
    geopot_coeffs_small = geopot_coeffs.copy()
    lmin_small = 16
    geopot_coeffs_small[:, :lmin_small, :] = 0 
    data_small_band_2d = pysh.expand.MakeGridDH(geopot_coeffs_small, sampling=1)
    
    return data_large_band_2d, data_medium_band_2d, data_small_band_2d

#Testing the function by applying it to the geopotential array of member 0, model number 0, and time index 0

data_large, data_medium, data_small = three_scale_decomposition(geopot_subsetted)

fig, axes = plt.subplots(3, 1, figsize=(8, 12))

axes[0].imshow(data_large, extent=(0, 360, -90, 90))
axes[0].set_title('Large Scale (l = 0-7)')
axes[1].imshow(data_medium, extent=(0, 360, -90, 90))
axes[1].set_title('Medium Scale (l = 8-16)')
axes[2].imshow(data_small, extent=(0, 360, -90, 90))
axes[2].set_title('Small Scale (l > 16)')

plt.tight_layout()
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

# Loop over all ensemble members and apply three_scale_decomposition
for ens in range(geopot_march.shape[0]):  # Loop over 1000 ensemble members
    for mod in range(geopot_march.shape[1]):  # Loop over 8 model levels
        march_geopot_2d = np.array(geopot_march[ens, mod, :, :], dtype='f8') # Extract 2D geopotential data for each ensemble member and level
        #I added np.array specifically to specify float type to 64 instead of 32 because 32 was not working and it worked after converting the float type...similar issue occured above.
        #print(march_geopot_2d.shape) #--> (48,48)
        
        #Apply three_scale_decomposition to each 2D array
        data_large, data_medium, data_small = three_scale_decomposition(march_geopot_2d)

        # Store the results in the corresponding arrays
        data_large_band_2d[ens, mod, :, :] = data_large
        data_medium_band_2d[ens, mod, :, :] = data_medium
        data_small_band_2d[ens, mod, :, :] = data_small

#Verify the resulting shape for each of the decomposed arrays
#print(data_large_band_2d.shape,data_medium_band_2d.shape,data_small_band_2d.shape) #OUTPUT--> (1000, 8, 48, 48) (1000, 8, 48, 48) (1000, 8, 48, 48)





