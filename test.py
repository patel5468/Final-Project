#Activate the environment
#>>conda activate final_proj_env
# Import necessary packages
import numpy as np
import netCDF4 as nc
import pyshtools as pysh
import matplotlib.pyplot as plt

# Load the NetCDF file containing geopotential data
f = nc.Dataset('/fs/ess/PAS2856/SPEEDY_ensemble_data/reference_ens/201103130000.nc', 'r')

#create variable geopotential based on ncdump and subset the array to have only lat and lon
geopot_original = f.variables['phi'][0, 0, 0, :, :]
#print(geopot_original.shape)#Verify shape is (48,96)
#remove everyother longitude to make the shape (48,48)
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

#Filter for 16th largest scale
geopot_coeffs_filtered1 = geopot_coeffs.copy()
lmax1 = 16
geopot_coeffs_filtered1[:, lmax1:, :] = 0  #Set values for l > 15 to 0
topo_filtered1 = pysh.expand.MakeGridDH(geopot_coeffs_filtered1, sampling=2)

#2nd plot: Sum of the 16th to 32nd largest scales (l = 16 to 31)
geopot_coeffs_filtered2 = geopot_coeffs.copy()
lmin2, lmax2 = 16, 32
geopot_coeffs_filtered2[:, :lmin2, :] = 0  #Set values for l < 16 to 0
geopot_coeffs_filtered2[:, lmax2:, :] = 0  #Set values for l > 31 to 0
topo_filtered2 = pysh.expand.MakeGridDH(geopot_coeffs_filtered2, sampling=2)

#3rd plot: Sum of the remaining scales (l > 31)
geopot_coeffs_filtered3 = geopot_coeffs.copy()
lmin3 = 32
geopot_coeffs_filtered3[:, :lmin3, :] = 0  #Set values for l < 32 to 0
topo_filtered3 = pysh.expand.MakeGridDH(geopot_coeffs_filtered3, sampling=2)

#Create empty figure with 3 rows and 1 column for plots
fig, axes = plt.subplots(3, 1, figsize=(8, 12))

#Plot the sum of the 16 largest scales and save it as the first plot
axes[0].imshow(topo_filtered1, extent=(0, 360, -90, 90))
axes[0].set(xlabel='Longitude', ylabel='Latitude', title='1 = 0 - 15', 
            yticks=np.arange(-90, 135, 45), xticks=np.arange(0, 405, 45)
        )

#Plot the sum of the 16th to 32nd largest scales and save it as the second plot
axes[1].imshow(topo_filtered2, extent=(0, 360, -90, 90))
axes[1].set(xlabel='Longitude', ylabel='Latitude', title='l = 16 - 31', 
            yticks=np.arange(-90, 135, 45), xticks=np.arange(0, 405, 45)
        )

#Plot the sum of the remaining scales (l > 31) and save it as the third plot
axes[2].imshow(topo_filtered3, extent=(0, 360, -90, 90))
axes[2].set(xlabel='Longitude', ylabel='Latitude', title='l > 31', 
            yticks=np.arange(-90, 135, 45), xticks=np.arange(0, 405, 45)
        )

#Format the figure to fit each plot within it and eliminate excess white space
plt.tight_layout()
#Save the figure as a PNG
plt.savefig('geopotential_scales.png')

#Sum the three topographic grids
topo_sum = topo_filtered1 + topo_filtered2 + topo_filtered3
print(topo_sum.shape)

#Plot the sum
fig, ax = plt.subplots(figsize=(8, 6))
ax.imshow(topo_sum, extent=(0, 360, -90, 90))
ax.set(xlabel='Longitude', ylabel='Latitude', title='Sum of all scales', 
        yticks=np.arange(-90, 135, 45), xticks=np.arange(0, 405, 45)
        )

#Adjust layout
plt.tight_layout()
#Save the result
plt.savefig('sum_of_all_scales.png')

