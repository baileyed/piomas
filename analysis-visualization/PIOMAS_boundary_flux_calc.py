# -*- coding: utf-8 -*-
"""
Created on Tue Feb 13 12:31:54 2024

@author: eliza
"""

#%%
import numpy as np
# import os
from netCDF4 import Dataset
import pandas as pd
from scipy.interpolate import interp1d, interp2d, RegularGridInterpolator
# from scipy.interpolate import griddata
from scipy.stats import linregress, theilslopes
import seaborn as sns
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import matplotlib.ticker as mticker
import matplotlib.transforms as mtransforms
import cartopy
import cartopy.crs as ccrs
import cartopy.feature as cfeature
from cartopy.mpl.gridliner import LONGITUDE_FORMATTER, LATITUDE_FORMATTER
from geopy.distance import distance
from pyproj import Proj
from shapely.geometry import shape
#%%
# years = np.arange(2005, 2023)
month_days = np.array([31,28,31,30,31,30,30,31,30,31,30,31])
month_days_leap = np.array([31,29,31,30,31,30,30,31,30,31,30,31])

zero_indexs = np.array([258,253,255,249,260,256,253,247,257,256,251,249,249,253,255,254,246,255,253])
may_index = np.array([120,120,120,121,120,120,120,121,120,120,120,121,120,120,120,121,120,120,120])

nps = ccrs.NorthPolarStereo(central_longitude=-145, true_scale_latitude=None, globe=None)
extent = [-175, -105, 65.5, 82.25]
dims = (120,360)

### Define the box of the region ###
left = [np.linspace(70,78,20), np.linspace(-160,-160,20)]
right = [np.linspace(70,78,20), np.linspace(-120,-120,20)]
top = [np.linspace(78,78,20), np.linspace(-120,-160,20)]
bottom = [np.linspace(70,70,20), np.linspace(-120,-160,20)]

rho_i = 900*1e9 #kg km^-3
rho_i_m = 900 #kg m^-3
L_i = 2.67e5 #J kg^-1 (1e6)
#%%
co = {"type": "Polygon", 
      "coordinates": [[(-120, 78),
                       (-120, 70),
                       (-160, 70),
                       (-160, 78)]]}
lon, lat = zip(*co['coordinates'][0])
pa = Proj("+proj=aea +lat_1=70 +lat_2=78 +lat_0=74 +lon_0=-145")
x, y = pa(lon, lat)
cop = {"type": "Polygon", "coordinates": [zip(x, y)]}
area = shape(cop).area  #in square meters

p1 = (78, 200)
p2 = (78, 240)
dist_NB = distance(p1, p2).km*1000 #m

p1 = (70, 200)
p2 = (78, 200)
dist_WB = distance(p1, p2).km*1000 #~m
#%%
alpha = np.array(pd.read_csv(r'E:\PIOMAS\alpha.dat', header=None, delim_whitespace=True))
flat_alpha = alpha.ravel()
alpha = flat_alpha.reshape(dims)
#%%
def interpolate_lat(data, lat, lon, target_latitude, indexes):
    x = indexes[0]
    y = indexes[1]
    
    # Assuming 'velocities' is a corresponding array of velocities at each index
    lats = lat[x,y]
    # lons = lon[x,y]
    
    # Find the unique x-values (since they represent different groups)
    unique_x = np.unique(x)
    
    # Initialize arrays to store interpolated velocities and corresponding longitudes    
    new_data = np.full((len(data)), np.nan)  
    for i in range(len(data)):
        velocities = data[i]
        # Loop through each group of x-values
        interpolated_data = []
        velocities = velocities[x,y]
        for unique_x_value in unique_x:
            # Find indices corresponding to the current group of x-values
            group_indices = np.where(x == unique_x_value)[0]
            
            if len(group_indices)>1:
            
                # Extract y-values and velocities for the current group
                group_y = y[group_indices]
                group_lat = lats[group_indices]
                group_velocities = velocities[group_indices]
                
                # Sort the group based on y-values
                sorted_indices = np.argsort(group_y)
                # sorted_group_y = group_y[sorted_indices]
                sorted_group_lat = group_lat[sorted_indices]
                sorted_group_velocities = group_velocities[sorted_indices]
                
                # Interpolate velocities at y=78
                interp_func = interp1d(sorted_group_lat, sorted_group_velocities, kind='linear', fill_value="extrapolate")
                interpolated_velocity = interp_func(target_latitude)
            
                # # Determine the corresponding longitude
                # interp_func = interp1d(sorted_group_lat, sorted_group_lon, kind='linear')
                # corresponding_longitude = interp_func(78) # Calculate longitude based on your data
            
                # Append the interpolated velocity and corresponding longitude to the arrays
                interpolated_data.append(interpolated_velocity)
            # interpolated_longitudes.append(corresponding_longitude)
        new_data[i] = np.nanmean(interpolated_data)
        
    return new_data

def interpolate_lon(data, lat, lon, target_longitude, indexes):
    x = indexes[0]
    y = indexes[1]
    
    # Assuming 'velocities' is a corresponding array of velocities at each index
    # lats = lat[x,y]
    lons = lon[x,y]
    
    # Find the unique x-values (since they represent different groups)
    unique_x = np.unique(x)
    
    # Initialize arrays to store interpolated velocities and corresponding longitudes    
    new_data = np.full((len(data)), np.nan)  
    for i in range(len(data)):
        velocities = data[i]
        # Loop through each group of x-values
        interpolated_data = []
        velocities = velocities[x,y]
        for unique_x_value in unique_x:
            # Find indices corresponding to the current group of x-values
            group_indices = np.where(x == unique_x_value)[0]
            
            if len(group_indices)>1:
            
                # Extract y-values and velocities for the current group
                group_y = y[group_indices]
                group_lon = lons[group_indices]
                group_velocities = velocities[group_indices]
                
                # Sort the group based on y-values
                sorted_indices = np.argsort(group_y)
                # sorted_group_y = group_y[sorted_indices]
                sorted_group_lon = group_lon[sorted_indices]
                sorted_group_velocities = group_velocities[sorted_indices]
                
                # Interpolate velocities at y=78
                interp_func = interp1d(sorted_group_lon, sorted_group_velocities, kind='linear', fill_value="extrapolate")
                interpolated_velocity = interp_func(target_longitude)
            
                # # Determine the corresponding longitude
                # interp_func = interp1d(sorted_group_lat, sorted_group_lon, kind='linear')
                # corresponding_longitude = interp_func(78) # Calculate longitude based on your data
            
                # Append the interpolated velocity and corresponding longitude to the arrays
                interpolated_data.append(interpolated_velocity)
            # interpolated_longitudes.append(corresponding_longitude)
        new_data[i] = np.nanmean(interpolated_data)    
    return new_data
#%%
def calculate_NS_boundary_flux(lat, lon, lat_other, lon_other, v, sic, hi, indices, indices_other):
    new_v  = interpolate_lat(v, lat, lon, 78, indices)
    new_sic  = interpolate_lat(sic, lat_other, lon_other, 78, indices_other)
    new_hi = interpolate_lat(hi, lat_other, lon_other, 78, indices_other)
    
    p1 = (78, 200)
    p2 = (78, 240)
    dist = distance(p1, p2).km #~km
    
    area_flux = np.full(len(new_v),np.nan)
    volume_flux = np.full(len(new_v),np.nan)
    thick_flux = np.full(len(new_v),np.nan)
    vflux_err = np.full(len(new_v),np.nan)
    for day in range(len(new_v)):
        day_v=new_v[day]*86.4 #~km/day
        daily_sic = new_sic[day]
        daily_hi = new_hi[day]/1e3 #~km
        a_flux = (day_v*daily_sic)
        v_flux =(day_v*daily_sic*daily_hi)
        area_flux[day] = np.nanmean(a_flux)*dist # ~km^2 s^-1
        volume_flux[day] = np.nanmean(v_flux)*dist # ~km^3 s^-1
        thick_flux[day] = np.nanmean(v_flux)/dist # ~km s^-1
        vflux_err[day] = ((0.17e-3/daily_hi)+(9/day_v))*100
    return -area_flux , -volume_flux, -thick_flux, new_v, vflux_err

def calculate_W_boundary_flux(lat, lon, lat_other, lon_other, u, sic, hi, indices, indices_other):
    
    new_u = interpolate_lon(u, lat, lon, 200, indices)
    new_sic = interpolate_lon(sic, lat_other, lon_other, 200, indices_other)
    new_hi = interpolate_lon(hi, lat_other, lon_other, 200, indices_other)
    
    p1 = (70, 200)
    p2 = (78, 200)
    dist = distance(p1, p2).km #~km
    
    area_flux = np.full(len(u),np.nan)
    volume_flux = np.full(len(u),np.nan)
    thick_flux = np.full(len(u),np.nan)
    vflux_err = np.full(len(u),np.nan)
    for day in range(len(u)):
        day_u=new_u[day]*86.4 #~km/day
        daily_sic = new_sic[day]
        daily_hi = new_hi[day]/1e3 #~km
        a_flux = (day_u*daily_sic)
        v_flux =(day_u*daily_sic*daily_hi)
        area_flux[day] = np.nanmean(a_flux)*dist # ~km^2 s^-1
        volume_flux[day] = np.nanmean(v_flux)*dist # ~km^3 s^-1
        thick_flux[day] = np.nanmean(v_flux)/dist # ~km s^-1
        vflux_err[day] = ((0.17e-3/daily_hi)+(9/day_u))*100
    return area_flux , volume_flux, thick_flux, new_u, vflux_err

def calculate_E_boundary_flux(lat, lon, lat_other, lon_other, u, sic, hi, indices, indices_other):
    
    new_u = interpolate_lon(u, lat, lon, 240, indices)
    new_sic = interpolate_lon(sic, lat_other, lon_other, 240, indices_other)
    new_hi = interpolate_lon(hi, lat_other, lon_other, 240, indices_other)
    
    p1 = (70, 240)
    p2 = (78, 240)
    dist = distance(p1, p2).km #~km
    
    area_flux = np.full(len(u),np.nan)
    volume_flux = np.full(len(u),np.nan)
    thick_flux = np.full(len(u),np.nan)
    vflux_err = np.full(len(u),np.nan)
    for day in range(len(u)):
        day_u=new_u[day]*86.4 #~km/day
        daily_sic = new_sic[day]
        daily_hi = new_hi[day]/1e3 #~km
        a_flux = (day_u*daily_sic)
        v_flux =(day_u*daily_sic*daily_hi)
        area_flux[day] = np.nanmean(a_flux)*dist # ~km^2 s^-1
        volume_flux[day] = np.nanmean(v_flux)*dist # ~km^3 s^-1
        thick_flux[day] = np.nanmean(v_flux)/dist # ~km s^-1
        vflux_err[day] = ((0.17e-3/daily_hi)+(9/day_u))*100
    return area_flux , volume_flux, thick_flux, new_u, vflux_err
#%%
year1 = 1979
# year2 = 1979
year2 = 2023
years = np.arange(year1,year2+1)
# Open files outside the loop
thick_nc_files = [Dataset("E:\PIOMAS\processed\daily_thickness\{year}.nc".format(year=year)) for year in years]
sic_nc_files = [Dataset("E:\PIOMAS\processed\daily_concentration\{year}.nc".format(year=year)) for year in years]
u_nc_files = [Dataset(r"E:\PIOMAS\processed\daily_velocity\u\{year}.nc".format(year=year)) for year in years]
v_nc_files = [Dataset(r"E:\PIOMAS\processed\daily_velocity\v\{year}.nc".format(year=year)) for year in years]

north_AF_05_22 = []
north_VF_05_22 = []
north_TF_05_22 = []
north_vels = []
north_err = []

west_AF_05_22 = [] 
west_VF_05_22 = []
west_TF_05_22 = []
west_vels = []
west_err = []


east_AF_05_22 = [] 
east_VF_05_22 = []
east_TF_05_22 = []
east_vels = []
east_err = []


daily_heff = []
daily_heff_std = []

tot_V = []
tot_V_std = []

tot_V_sum = []
tot_V_sum_std = []

tot_V_fall = []
tot_V_fall_std = []

tot_V_win = []
tot_V_win_std = []

for year in years:
    print('starting', year)
    hi_nc = thick_nc_files[year-year1]
    lat_other = hi_nc.variables['latitude'][:]
    lon_other = hi_nc.variables['longitude'][:]

    hi = hi_nc.variables['thickness'][:]

    sic_nc = sic_nc_files[year-year1]
    sic = sic_nc.variables['concentration'][:]

    u_nc = u_nc_files[year-year1]
    u = u_nc.variables['u'][:]
    lat = u_nc.variables['latitude'][:]
    lon = u_nc.variables['longitude'][:]
    

    v_nc = v_nc_files[year-year1]
    v = v_nc.variables['v'][:]

    u_rot = np.zeros_like(u)
    v_rot = np.zeros_like(v)
    ### Apply Rotation Matrix ###
    for i in range(len(u)):
        u_rot[i] = u[i] * np.cos(np.deg2rad(-alpha)) + v[i] * np.sin(np.deg2rad(-alpha))
        v_rot[i] = -u[i] * np.sin(np.deg2rad(-alpha)) + v[i] * np.cos(np.deg2rad(-alpha))
    
    condition_1 = np.abs(lat - 78) < 1
    condition_2 = (lon >= 200) & (lon <= 240)
    indices_N = np.where(condition_1 & condition_2)  
    
    condition_1 = np.abs(lat_other - 78) < 1
    condition_2 = (lon_other >= 200) & (lon_other <= 240)
    indices_Nother = np.where(condition_1 & condition_2)
    
    north_AF, north_VF, north_TF, north_vel, nerr = calculate_NS_boundary_flux(lat, lon, lat_other, lon_other, v_rot, sic, hi, indices_N, indices_Nother)
        
    condition_1 = np.abs(lon - 200) < 4
    condition_2 = (lat >= 69) & (lat <= 78)
    indices_W = np.where(condition_1 & condition_2)
    
    condition_1 = np.abs(lon_other - 200) < 4
    condition_2 = (lat_other >= 69) & (lat_other <= 78)
    indices_Wother = np.where(condition_1 & condition_2)
        
    west_AF, west_VF, west_TF, west_vel, werr = calculate_W_boundary_flux(lat, lon, lat_other, lon_other, u_rot, sic, hi, indices_W, indices_Wother)
    
    condition_1 = np.abs(lon - 240) < 4
    condition_2 = (lat >= 69) & (lat <= 78)
    indices_E = np.where(condition_1 & condition_2)
    
    condition_1 = np.abs(lon_other - 240) < 4
    condition_2 = (lat_other >= 69) & (lat_other <= 78)
    indices_Eother = np.where(condition_1 & condition_2)
        
    east_AF, east_VF, east_TF, east_vel, eerr = calculate_E_boundary_flux(lat, lon, lat_other, lon_other, u_rot, sic, hi, indices_E, indices_Eother)
    
    north_AF_05_22.append(north_AF)
    north_VF_05_22.append(north_VF)
    north_TF_05_22.append(north_TF)
    north_vels.append(north_vel)
    north_err.append(nerr)
    
    west_AF_05_22.append(west_AF)
    west_VF_05_22.append(west_VF)
    west_TF_05_22.append(west_TF)
    west_vels.append(west_vel)
    west_err.append(werr)
    
    east_AF_05_22.append(east_AF)
    east_VF_05_22.append(east_VF)
    east_TF_05_22.append(east_TF)
    east_vels.append(east_vel)
    east_err.append(eerr)
    
    # Define the condition for the Beaufort Sea
    condition_lon = (lon_other >= 200) & (lon_other <= 240)
    condition_lat = (lat_other >= 69) & (lat_other <= 78)
    # Find the indices of the Beaufort Sea
    indices = np.where(condition_lon & condition_lat)
    
    ### Calculate Daily Mean Thickness for each year for Beaufort Sea ###
    daily_heff.append(np.nanmean(hi[:, indices[0], indices[1]], axis=1))
    
    ### Calculate Total Volume Changes for Beaufort Sea ###
    
    # Calculate mean and standard deviation for each hiday entry
    hi_BS_mean = np.nansum(np.diff(hi[:, indices[0], indices[1]], axis=0),axis=0)
    tot_V.append(np.nanmean(hi[:, indices[0], indices[1]])*rho_i_m*L_i*area) #~J
    tot_V_std.append(np.nanstd(hi_BS_mean)*rho_i_m*L_i*area) #~J
    
    hi_BS_mean = np.nansum(np.diff(hi[:120, indices[0], indices[1]], axis=0),axis=0)
    tot_V_win.append(np.nanmean(hi_BS_mean)*rho_i_m*L_i*area) #~J
    tot_V_win_std.append(np.nanstd(hi_BS_mean)*rho_i_m*L_i*area) #~J
    
    hi_BS_mean = np.nansum(np.diff(hi[120:258, indices[0], indices[1]], axis=0),axis=0)
    tot_V_sum.append(np.nanmean(hi_BS_mean)*rho_i_m*L_i*area) #~J
    tot_V_sum_std.append(np.nanstd(hi_BS_mean)*rho_i_m*L_i*area) #~J
    
    hi_BS_mean = np.nansum(np.diff(hi[258:, indices[0], indices[1]], axis=0),axis=0)
    tot_V_fall.append(np.nanmean(hi_BS_mean)*rho_i_m*L_i*area) #~J
    tot_V_fall_std.append(np.nanstd(hi_BS_mean)*rho_i_m*L_i*area) #~J

# Close files
for file in thick_nc_files + sic_nc_files + u_nc_files + v_nc_files:
    file.close()
#%%
# north_VF_05_22 = np.array(north_VF_05_22)
# east_VF_05_22 = np.array(east_VF_05_22)
# west_VF_05_22 = np.array(west_VF_05_22)
#%%
# Find the maximum size among all arrays
max_size = max(len(arr) for arr in daily_heff)

# Pad arrays to match the maximum size
padded_arrays = []
for arr in daily_heff:
    padded_array = np.pad(arr, (0, max_size - len(arr)), 
                          mode='constant', constant_values=np.nan)
    padded_arrays.append(padded_array)

# Stack the padded arrays into a 2D array
stacked_arrays = np.vstack(padded_arrays)

# Calculate element-wise mean while ignoring NaN values
daily_heff_mean = np.nanmean(stacked_arrays, axis=0)
daily_heff_std = np.nanstd(stacked_arrays, axis=0)
#%%
# Find the maximum size among all arrays
max_size = max(len(arr) for arr in north_vels)

# Pad arrays to match the maximum size
padded_arrays = []
for arr in north_vels:
    padded_array = np.pad(arr, (0, max_size - len(arr)), 
                          mode='constant', constant_values=np.nan)
    padded_arrays.append(padded_array)

# Stack the padded arrays into a 2D array
stacked_arrays = np.vstack(padded_arrays)

window_size=10
# Calculate element-wise mean while ignoring NaN values
daily_nvels_mean = pd.Series(np.nanmean(stacked_arrays, axis=0)).rolling(window=window_size, min_periods=1).mean()
daily_nvels_std = pd.Series(np.nanstd(stacked_arrays, axis=0)).rolling(window=window_size, min_periods=1).mean()

# Find the maximum size among all arrays
max_size = max(len(arr) for arr in west_vels)

# Pad arrays to match the maximum size
padded_arrays = []
for arr in west_vels:
    padded_array = np.pad(arr, (0, max_size - len(arr)), 
                          mode='constant', constant_values=np.nan)
    padded_arrays.append(padded_array)

# Stack the padded arrays into a 2D array
stacked_arrays = np.vstack(padded_arrays)

# Calculate element-wise mean while ignoring NaN values
daily_wvels_mean = pd.Series(np.nanmean(stacked_arrays, axis=0)).rolling(window=window_size, min_periods=1).mean()
daily_wvels_std = pd.Series(np.nanstd(stacked_arrays, axis=0)).rolling(window=window_size, min_periods=1).mean()
#%%
fig, axs = plt.subplot_mosaic([['(a)'],['(b)'],['(c)']],
                              layout='constrained', figsize=(30, 30))

for label, ax in axs.items():
    # label physical distance to the left and up:
    trans = mtransforms.ScaledTranslation(-20/72, 7/72, fig.dpi_scale_trans)
    ax.text(-0.05, 1.05, label, transform=ax.transAxes + trans,
            fontsize=40, va='bottom', fontfamily='serif')

# fig, ax = plt.subplots(nrows=2, ncols=2, figsize=(25, 20), width_ratios=[25, 10])
ax1 = axs['(a)']
ax2 = axs['(b)']
ax3 = axs['(c)']

# # Define the window size for the running mean
# window_size = 10
# runmean_ohc = pd.Series(ohc_mean_05_22).rolling(window=window_size, min_periods=1).mean()
# runstd_ohc = pd.Series(ohc_std_05_22).rolling(window=window_size, min_periods=1).mean()

# print(np.nanmax(runmean_ohc))

days = np.arange(1,len(daily_heff_mean)+1)
ax1.plot(days, daily_heff_mean, color='k', linestyle='-', label='OHC', lw=3)
ax1.fill_between(days, daily_heff_mean-daily_heff_std, 
                 daily_heff_mean+daily_heff_std,  alpha=0.25, color='k')
# mean_zero_idx = np.where(fneto_daily_mean[242:] <= 0)[0]+242
ax1.axvline(120, color='k', lw=4)
ax1.axvline(258, color='k', lw=4)
ax1.set_xlim(1, 365)
ax1.set_ylim(0, 3)
ax1.set_ylabel(r'Ice Thickness (m)', fontsize=30, color='black')
ax1.set_xticks([1,31,59,90,120,151,181,212,243,273,304,334,365])
ax1.set_xticklabels(['Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun', 
                     'Jul', 'Aug', 'Sep', 'Oct', 'Nov', 'Dec', 'Jan'])
ax1.tick_params(labelsize=30, direction='in', length=16, width=4, color='black',
                labelcolor='black', labelbottom=True, pad=15)
ax1.grid(True, linestyle='--', zorder=0)

days = np.arange(1,len(daily_nvels_mean)+1)
ax2.plot(days, -daily_nvels_mean, color='k', linestyle='-', label='OHC', lw=3)
ax2.fill_between(days, -daily_nvels_mean-daily_nvels_std, 
                 -daily_nvels_mean+daily_nvels_std,  alpha=0.25, color='k')
# mean_zero_idx = np.where(fneto_daily_mean[242:] <= 0)[0]+242
ax2.axhline(0, color='k', lw=4)
ax2.axvline(120, color='k', lw=4)
ax2.axvline(258, color='k', lw=4)
ax2.set_xlim(1, 365)
# ax1.set_ylim(0, 3)
ax2.set_ylabel(r'North Boundary Ice Velocity (m/s)', fontsize=30, color='black')
ax2.set_xticks([1,31,59,90,120,151,181,212,243,273,304,334,365])
ax2.set_xticklabels(['Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun', 
                     'Jul', 'Aug', 'Sep', 'Oct', 'Nov', 'Dec', 'Jan'])
ax2.tick_params(labelsize=30, direction='in', length=16, width=4, color='black',
                labelcolor='black', labelbottom=True, pad=15)
ax2.grid(True, linestyle='--', zorder=0)

days = np.arange(1,len(daily_wvels_mean)+1)
ax3.plot(days, daily_wvels_mean, color='k', linestyle='-', label='OHC', lw=3)
ax3.fill_between(days, daily_wvels_mean-daily_wvels_std, 
                 daily_wvels_mean+daily_wvels_std,  alpha=0.25, color='k')
# mean_zero_idx = np.where(fneto_daily_mean[242:] <= 0)[0]+242
ax3.axhline(0, color='k', lw=4)
ax3.axvline(120, color='k', lw=4)
ax3.axvline(258, color='k', lw=4)
ax3.set_xlim(1, 365)
# ax1.set_ylim(0, 3)
ax3.set_ylabel(r'West Boundary Ice Velocity (m/s)', fontsize=30, color='black')
ax3.set_xticks([1,31,59,90,120,151,181,212,243,273,304,334,365])
ax3.set_xticklabels(['Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun', 
                     'Jul', 'Aug', 'Sep', 'Oct', 'Nov', 'Dec', 'Jan'])
ax3.tick_params(labelsize=30, direction='in', length=16, width=4, color='black',
                labelcolor='black', labelbottom=True, pad=15)
ax3.grid(True, linestyle='--', zorder=0)
#%%  
# mean_u = np.nanmean(u[3:6], axis=0)*86.4
# mean_v = np.nanmean(v[3:6], axis=0)*86.4

# # Calculate the block size for aggregation
# block_size_x = mean_u.shape[0] // 30
# block_size_y = mean_u.shape[1] // 90

# # Reshape the data into blocks and calculate the mean value for each block
# reduced_u = np.zeros((30, 90))
# reduced_v = np.zeros((30, 90))
# reduced_lat = np.zeros((30, 90))
# reduced_lon = np.zeros((30, 90))

# for i in range(30):
#     for j in range(90):
#         reduced_u[i, j] = np.mean(mean_u[i*block_size_x : (i+1)*block_size_x, 
#                                           j*block_size_y : (j+1)*block_size_y])
#         reduced_v[i, j] = np.mean(mean_v[i*block_size_x : (i+1)*block_size_x, 
#                                           j*block_size_y : (j+1)*block_size_y])
#         reduced_lat[i, j] = np.mean(ulat[i*block_size_x : (i+1)*block_size_x, 
#                                           j*block_size_y : (j+1)*block_size_y])
#         reduced_lon[i, j] = np.mean(ulon[i*block_size_x : (i+1)*block_size_x, 
#                                            j*block_size_y : (j+1)*block_size_y])
                                          
# fig, axs = plt.subplot_mosaic([['(a)']],
#                                     per_subplot_kw={'(a)':{'projection': nps}},
#                                   layout='constrained', figsize=(30, 30))
    
# for label, ax in axs.items():
#     # label physical distance to the left and up:
#     trans = mtransforms.ScaledTranslation(-20/72, 7/72, fig.dpi_scale_trans)
#     ax.text(-0.05, 1.05, label, transform=ax.transAxes + trans,
#             fontsize=40, va='bottom', fontfamily='serif')
                
# ax = axs['(a)']

# ax.set_extent(extent, ccrs.PlateCarree())  # Use PlateCarree for defining the extent
# # ax.add_feature(cartopy.feature.LAND, zorder=3, edgecolor='black')
# ax.pcolormesh(lon, lat, np.nanmean(hi[:90], axis=0),cmap='viridis', 
#               transform=ccrs.PlateCarree(), 
#               vmin=0, vmax=3)
# # Q = ax.quiver(reduced_lon, reduced_lat, 
# #               reduced_u,
# #               # np.zeros(np.shape(reduced_u)),
# #               reduced_v, 
# #               transform=ccrs.PlateCarree(), 
# #               scale=100, zorder=1, color='k',
# #               width=0.0015, headwidth=4, 
# #               headlength=3, headaxislength=3)
# # qk = ax.quiverkey(Q, 0.9, 1.05, 8, r'$8 \frac{km}{day}$', labelpos='E',
# #                   coordinates='axes', fontproperties={'size':30})
# ax.scatter(lon[indices_N], lat[indices_N], c='red',s=300, transform=ccrs.PlateCarree())
# ax.scatter(lon[indices_W], lat[indices_W], c='red',s=300, transform=ccrs.PlateCarree())
# ax.plot(left[1], left[0], transform=ccrs.PlateCarree(), c = 'k', linewidth=5, zorder=1)
# ax.plot(right[1], right[0], transform=ccrs.PlateCarree(), c = 'k', linewidth=5, zorder=1)
# ax.plot(top[1], top[0], transform=ccrs.PlateCarree(), c = 'k', linewidth=5, zorder=1)
#%%
fig, axs = plt.subplot_mosaic([['(a)','(b)'],['(c)','(d)'],['(e)','(f)']], 
                              layout='constrained', figsize=(40, 30))

for label, ax in axs.items():
    # label physical distance to the left and up:
    trans = mtransforms.ScaledTranslation(-20/72, 7/72, fig.dpi_scale_trans)
    ax.text(-0.05, 1.05, label, transform=ax.transAxes + trans,
            fontsize=40, va='bottom', fontfamily='serif')

ax1 = axs['(a)']
ax2 = axs['(b)']
ax3 = axs['(c)']
ax4 = axs['(d)']
ax5 = axs['(e)']
ax6 = axs['(f)']

N_19 = north_AF_05_22[-5]
N_20 = north_AF_05_22[-4]
N_21 = north_AF_05_22[-3]

W_19 = west_AF_05_22[-5]
W_20 = west_AF_05_22[-4]
W_21 = west_AF_05_22[-3]

N_19_20 = np.hstack((N_19[273:],N_20[:151]))/1e5
W_19_20 = np.hstack((W_19[273:],W_20[:151]))/1e5

N_19_20_cumsum = np.nancumsum(N_19_20)
W_19_20_cumsum = np.nancumsum(W_19_20)

N_20_21 = np.hstack((N_20[273:],N_21[:151]))/1e5
W_20_21 = np.hstack((W_20[273:],W_21[:151]))/1e5

N_20_21_cumsum = np.nancumsum(N_20_21)
W_20_21_cumsum = np.nancumsum(W_20_21)

# Pad arrays to match the maximum size
N_arrays = []
W_arrays = []
yi=2007
for year in range(2007,2019):
    ny1 = north_AF_05_22[year-yi]
    ny2 = north_AF_05_22[(year+1)-yi]
    N_y1y2 = np.hstack((ny1[273:], ny2[:151]))
    N_arrays.append(np.nancumsum(N_y1y2))
    wy1 = west_AF_05_22[year-yi]
    wy2 = west_AF_05_22[(year+1)-yi]
    W_y1y2 = np.hstack((wy1[273:], wy2[:151]))
    W_arrays.append(np.nancumsum(W_y1y2))
# Stack the padded arrays into a 2D array
stacked_arrays = np.vstack(N_arrays)
# Calculate element-wise mean while ignoring NaN values
N_AF_mean = np.nanmean(stacked_arrays, axis=0)
N_AF_std = np.nanstd(stacked_arrays, axis=0)

stacked_arrays = np.vstack(W_arrays)
# Calculate element-wise mean while ignoring NaN values
W_AF_mean = np.nanmean(stacked_arrays, axis=0)
W_AF_std = np.nanstd(stacked_arrays, axis=0)

ax1.plot(N_19_20_cumsum, c='red', lw=3)
ax1.plot(N_20_21_cumsum, c='blue', lw=3)
ax1.plot(N_AF_mean/1e5, c='k',lw=3)
ax1.fill_between(range(len(N_AF_mean)), (N_AF_mean-N_AF_std)/1e5, 
                  (N_AF_mean+N_AF_std)/1e5,  alpha=0.25, color='black')
ax1.axhline(0, c='k',ls='--')
ax1.set_xlim(0,243)
ax1.set_ylim(-7.5,7.5)
ax1.set_xticks([0,31,61,92,123,151,182,212,243])
ax1.set_xticklabels(['Oct', 'Nov', 'Dec', 'Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun'])
ax1.set_yticks([-7.5,-5,-2.5,0,2.5,5,7.5])
ax1.set_xlabel('Month', fontsize=30)
ax1.set_ylabel('Northern Boundary \n\nCumulative Ice Area Flux \n(10$^5$ km$^2$)', fontsize=30)
ax1.tick_params(labelsize=30, direction='in', length=16, width=4, color='black',
                labelcolor='black', labelbottom=True, pad=15, 
                bottom=True, top=True, left=True, right=True)
ax1.set_title('Cumulative Ice Area Flux (10$^5$ km$^2$)', fontsize=30)

ax3.plot(W_19_20_cumsum, c='red', lw=3)
ax3.plot(W_20_21_cumsum, c='blue', lw=3)
ax3.plot((W_AF_mean)/1e5, c='k',lw=3)
ax3.fill_between(range(len(W_AF_mean)), (W_AF_mean-W_AF_std)/1e5, 
                  (W_AF_mean+W_AF_std)/1e5,  alpha=0.25, color='black')
ax3.axhline(0, c='k',ls='--')
ax3.set_xlim(0,243)
ax3.set_ylim(-10,5)
ax3.set_xticks([0,31,61,92,123,151,182,212,243])
ax3.set_xticklabels(['Oct', 'Nov', 'Dec', 'Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun'])
ax3.set_yticks([-10,-7.5,-5,-2.5,0,2.5,5])
ax3.set_xlabel('Month', fontsize=30)
ax3.set_ylabel('Western Boundary \n\nCumulative Ice Area Flux \n(10$^5$ km$^2$)', fontsize=30)
ax3.tick_params(labelsize=30, direction='in', length=16, width=4, color='black',
                labelcolor='black', labelbottom=True, pad=15, 
                bottom=True, top=True, left=True, right=True)

ax5.plot((N_19_20_cumsum+W_19_20_cumsum), c='red', lw=3)
ax5.plot((N_20_21_cumsum+W_20_21_cumsum), c='blue', lw=3)
ax5.plot((N_AF_mean+W_AF_mean)/1e5, c='k',lw=3)
ax5.fill_between(range(len(N_AF_mean)),
                  ((N_AF_mean-N_AF_std)+(W_AF_mean-W_AF_std))/1e5, 
                  ((N_AF_mean+N_AF_std)+(W_AF_mean+W_AF_std))/1e5,  
                  alpha=0.25, color='black')
ax5.axhline(0, c='k',ls='--')
ax5.set_xlim(0,243)
ax5.set_ylim(-10,5)
ax5.set_xticks([0,31,61,92,123,151,182,212,243])
ax5.set_xticklabels(['Oct', 'Nov', 'Dec', 'Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun'])
ax5.set_yticks([-10,-7.5,-5,-2.5,0,2.5,5])
ax5.set_xlabel('Month', fontsize=30)
ax5.set_ylabel('Northern and Western Boundaries \n\nCumulative Ice Area Flux \n(10$^5$ km$^2$)', 
                fontsize=30)
ax5.tick_params(labelsize=30, direction='in', length=16, width=4, color='black',
                labelcolor='black', labelbottom=True, pad=15, 
                bottom=True, top=True, left=True, right=True)

N_19 = north_VF_05_22[-5]
N_20 = north_VF_05_22[-4]
N_21 = north_VF_05_22[-3]

W_19 = west_VF_05_22[-5]
W_20 = west_VF_05_22[-4]
W_21 = west_VF_05_22[-3]

N_19_20 = np.hstack((N_19[273:],N_20[:151]))
W_19_20 = np.hstack((W_19[273:],W_20[:151]))

N_19_20_cumsum = np.nancumsum(N_19_20)
W_19_20_cumsum = np.nancumsum(W_19_20)

N_20_21 = np.hstack((N_20[273:],N_21[:151]))
W_20_21 = np.hstack((W_20[273:],W_21[:151]))

N_20_21_cumsum = np.nancumsum(N_20_21)
W_20_21_cumsum = np.nancumsum(W_20_21)

# Pad arrays to match the maximum size
N_arrays = []
W_arrays = []
yi=2007
for year in range(2007,2019):
    ny1 = north_VF_05_22[year-yi]
    ny2 = north_VF_05_22[(year+1)-yi]
    N_y1y2 = np.hstack((ny1[273:], ny2[:151]))
    N_arrays.append(np.nancumsum(N_y1y2))
    wy1 = west_VF_05_22[year-yi]
    wy2 = west_VF_05_22[(year+1)-yi]
    W_y1y2 = np.hstack((wy1[273:], wy2[:151]))
    W_arrays.append(np.nancumsum(W_y1y2))
# Stack the padded arrays into a 2D array
stacked_arrays = np.vstack(N_arrays)
# Calculate element-wise mean while ignoring NaN values
N_VF_mean = np.nanmean(stacked_arrays, axis=0)
N_VF_std = np.nanstd(stacked_arrays, axis=0)

stacked_arrays = np.vstack(W_arrays)
# Calculate element-wise mean while ignoring NaN values
W_VF_mean = np.nanmean(stacked_arrays, axis=0)
W_VF_std = np.nanstd(stacked_arrays, axis=0)

ax2.plot(N_19_20_cumsum/1e2, c='red', lw=3)
ax2.plot(N_20_21_cumsum/1e2, c='blue', lw=3)
ax2.plot((N_VF_mean)/1e2, c='k',lw=3)
ax2.fill_between(range(len(N_VF_mean)), (N_VF_mean-N_VF_std)/1e2, 
                  (N_VF_mean+N_VF_std)/1e2,  alpha=0.25, color='black')
ax2.axhline(0, c='k',ls='--')
ax2.set_xlim(0,243)
ax2.set_ylim(-5,10)
ax2.set_xticks([0,31,61,92,123,151,182,212,243])
ax2.set_xticklabels(['Oct', 'Nov', 'Dec', 'Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun'])
ax2.set_yticks([-5,-2.5,0,2.5,5,7.5,10])
ax2.set_xlabel('Month', fontsize=30)
ax2.set_ylabel('Cumulative Ice Volume Flux \n(10$^2$ km$^3$)', fontsize=30)
ax2.tick_params(labelsize=30, direction='in', length=16, width=4, color='black',
                labelcolor='black', labelbottom=True, pad=15, 
                bottom=True, top=True, left=True, right=True)
ax2.set_title('Cumulative Ice Volume Flux (10$^2$ km$^3$)', fontsize=30)

ax4.plot(W_19_20_cumsum/1e2, c='red', lw=3)
ax4.plot(W_20_21_cumsum/1e2, c='blue', lw=3)
ax4.plot((W_VF_mean)/1e2, c='k',lw=3)
ax4.fill_between(range(len(W_VF_mean)), (W_VF_mean-W_VF_std)/1e2, 
                  (W_VF_mean+W_VF_std)/1e2,  alpha=0.25, color='black')
ax4.axhline(0, c='k',ls='--')
ax4.set_xlim(0,243)
ax4.set_ylim(-10,5)
ax4.set_xticks([0,31,61,92,123,151,182,212,243])
ax4.set_xticklabels(['Oct', 'Nov', 'Dec', 'Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun'])
ax4.set_yticks([-10,-7.5,-5,-2.5,0,2.5,5])
ax4.set_xlabel('Month', fontsize=30)
ax4.set_ylabel('Cumulative Ice Volume Flux \n(10$^2$ km$^3$)', fontsize=30)
ax4.tick_params(labelsize=30, direction='in', length=16, width=4, color='black',
                labelcolor='black', labelbottom=True, pad=15, 
                bottom=True, top=True, left=True, right=True)

ax6.plot((N_19_20_cumsum+W_19_20_cumsum)/1e2, c='red', lw=3)
ax6.plot((N_20_21_cumsum+W_20_21_cumsum)/1e2, c='blue', lw=3)
ax6.plot((N_VF_mean+W_VF_mean)/1e2, c='k',lw=3)
ax6.fill_between(range(len(N_VF_mean)),
                  ((N_VF_mean-N_VF_std)+(W_VF_mean-W_VF_std))/1e2, 
                  ((N_VF_mean+N_VF_std)+(W_VF_mean+W_VF_std))/1e2,  
                  alpha=0.25, color='black')
ax6.axhline(0, c='k',ls='--')
ax6.set_xlim(0,243)
ax6.set_ylim(-10,5)
ax6.set_xticks([0,31,61,92,123,151,182,212,243])
ax6.set_xticklabels(['Oct', 'Nov', 'Dec', 'Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun'])
ax6.set_yticks([-10,-7.5,-5,-2.5,0,2.5,5])
ax6.set_xlabel('Month', fontsize=30)
ax6.set_ylabel('Cumulative Ice Volume Flux \n(10$^2$ km$^3$)', fontsize=30)
ax6.tick_params(labelsize=30, direction='in', length=16, width=4, color='black',
                labelcolor='black', labelbottom=True, pad=15, 
                bottom=True, top=True, left=True, right=True)
#%%
# fig, axs = plt.subplot_mosaic([['(a)','(b)'],['(c)','(d)'],['(e)','(f)']], 
#                               layout='constrained', figsize=(55, 25))

# for label, ax in axs.items():
#     # label physical distance to the left and up:
#     trans = mtransforms.ScaledTranslation(-20/72, 7/72, fig.dpi_scale_trans)
#     ax.text(-0.05, 1.05, label, transform=ax.transAxes + trans,
#             fontsize=40, va='bottom', fontfamily='serif')

# ax1 = axs['(a)']
# ax2 = axs['(b)']
# ax3 = axs['(c)']
# ax4 = axs['(d)']
# ax5 = axs['(e)']
# ax6 = axs['(f)']

# N_summer_sum=[]
# W_summer_sum=[]
# Net_summer_sum=[]

# N_fall_sum=[]
# W_fall_sum=[]
# Net_fall_sum=[]
# leap = 1980
# for year in years:
#     N = north_AF_05_22[year-year1]
#     W = west_AF_05_22[year-year1]
#     Net = N+W
#     N_summer_sum.append(np.nansum(N[120:243])/1e5)
#     W_summer_sum.append(np.nansum(W[120:243])/1e5)
#     Net_summer_sum.append(np.nansum(Net[120:243])/1e5)

#     N_fall_sum.append(np.nansum(N[243:])/1e5)
#     W_fall_sum.append(np.nansum(W[243:])/1e5)
#     Net_fall_sum.append(np.nansum(Net[243:])/1e5)

# ax1.plot(years, N_summer_sum, c='red', lw=3, label='Summer')
# ax1.plot(years, N_fall_sum, c='blue', lw=3, label='Fall')
# ax1.axhline(0, c='k',ls='--')
# # ax1.set_ylim(-7.5,7.5)
# # ax1.set_xticks(years)
# # ax1.set_yticks([-7.5,-5,-2.5,0,2.5,5,7.5])
# ax1.set_xlabel('Month', fontsize=30)
# ax1.set_ylabel('Northern Boundary \n\nCumulative Ice Area Flux \n(10$^5$ km$^2$)', fontsize=30)
# ax1.tick_params(labelsize=30, direction='in', length=16, width=4, color='black',
#                 labelcolor='black', labelbottom=True, pad=15, 
#                 bottom=True, top=True, left=True, right=True)
# ax1.set_title('Cumulative Ice Area Flux (10$^5$ km$^2$)', fontsize=30)

# ax3.plot(years, W_summer_sum, c='red', lw=3, label='Summer')
# ax3.plot(years, W_fall_sum, c='blue', lw=3, label='Fall')
# ax3.axhline(0, c='k',ls='--')
# # ax3.set_xlim(0,243)
# # ax3.set_ylim(-10,5)
# # ax3.set_xticks(years)
# ax3.set_xlabel('Month', fontsize=30)
# ax3.set_ylabel('Western Boundary \n\nCumulative Ice Area Flux \n(10$^5$ km$^2$)', fontsize=30)
# ax3.tick_params(labelsize=30, direction='in', length=16, width=4, color='black',
#                 labelcolor='black', labelbottom=True, pad=15, 
#                 bottom=True, top=True, left=True, right=True)
# ax3.legend(fontsize=30)

# ax5.plot(years, Net_summer_sum, c='red', lw=3, label='Summer')
# ax5.plot(years, Net_fall_sum, c='blue', lw=3, label='Fall')
# ax5.axhline(0, c='k',ls='--')
# # ax5.set_xlim(0,243)
# # ax5.set_ylim(-10,5)
# # ax5.set_xticks(years)
# # ax5.set_yticks([-10,-7.5,-5,-2.5,0,2.5,5])
# ax5.set_xlabel('Month', fontsize=30)
# ax5.set_ylabel('Northern and Western Boundaries \n\nCumulative Ice Area Flux \n(10$^5$ km$^2$)', 
#                fontsize=30)
# ax5.tick_params(labelsize=30, direction='in', length=16, width=4, color='black',
#                 labelcolor='black', labelbottom=True, pad=15, 
#                 bottom=True, top=True, left=True, right=True)
# N_summer_sum=[]
# W_summer_sum=[]
# Net_summer_sum=[]

# N_fall_sum=[]
# W_fall_sum=[]
# Net_fall_sum=[]
# for year in years:
#     N = north_VF_05_22[year-year1]
#     W = west_VF_05_22[year-year1]
#     Net = N+W
    
#     N_summer_sum.append(np.nansum(N[120:243])/1e2)
#     W_summer_sum.append(np.nansum(W[120:243])/1e2)
#     Net_summer_sum.append(np.nansum(Net[120:243])/1e2)
    
#     N_fall_sum.append(np.nansum(N[243:])/1e2)
#     W_fall_sum.append(np.nansum(W[243:])/1e2)
#     Net_fall_sum.append(np.nansum(Net[243:])/1e2)
    
# ax2.plot(years,N_summer_sum, c='red', lw=3, label='Summer')
# ax2.plot(years,N_fall_sum, c='blue', lw=3, label='Fall')
# ax2.axhline(0, c='k',ls='--')
# # ax2.set_xlim(0,243)
# # ax2.set_ylim(-5,10)
# # ax2.set_xticks(years)
# # ax2.set_xticklabels(['Oct', 'Nov', 'Dec', 'Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun'])
# # ax2.set_yticks([-5,-2.5,0,2.5,5,7.5,10])
# ax2.set_xlabel('Month', fontsize=30)
# ax2.set_ylabel('Cumulative Ice Volume Flux \n(10$^2$ km$^3$)', fontsize=30)
# ax2.tick_params(labelsize=30, direction='in', length=16, width=4, color='black',
#                 labelcolor='black', labelbottom=True, pad=15, 
#                 bottom=True, top=True, left=True, right=True)
# ax2.set_title('Cumulative Ice Volume Flux (10$^2$ km$^3$)', fontsize=30)

# ax4.plot(years, W_summer_sum, c='red', lw=3, label='Summer')
# ax4.plot(years, W_fall_sum, c='blue', lw=3, label='Fall')
# ax4.axhline(0, c='k',ls='--')
# # ax4.set_xlim(0,243)
# # ax4.set_ylim(-10,5)
# # ax4.set_xticks(years)
# # ax4.set_xticklabels(['Oct', 'Nov', 'Dec', 'Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun'])
# # ax4.set_yticks([-10,-7.5,-5,-2.5,0,2.5,5])
# ax4.set_xlabel('Month', fontsize=30)
# ax4.set_ylabel('Cumulative Ice Volume Flux \n(10$^2$ km$^3$)', fontsize=30)
# ax4.tick_params(labelsize=30, direction='in', length=16, width=4, color='black',
#                 labelcolor='black', labelbottom=True, pad=15, 
#                 bottom=True, top=True, left=True, right=True)

# ax6.plot(years, Net_summer_sum, c='red', lw=3, label='Summer')
# ax6.plot(years, Net_fall_sum, c='blue', lw=3, label='Fall')
# ax6.axhline(0, c='k',ls='--')
# # ax6.set_xlim(0,243)
# # ax6.set_ylim(-10,5)
# # ax6.set_xticks(years)
# # ax6.set_xticklabels(['Oct', 'Nov', 'Dec', 'Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun'])
# # ax6.set_yticks([-10,-7.5,-5,-2.5,0,2.5,5])
# ax6.set_xlabel('Month', fontsize=30)
# ax6.set_ylabel('Cumulative Ice Volume Flux \n(10$^2$ km$^3$)', fontsize=30)
# ax6.tick_params(labelsize=30, direction='in', length=16, width=4, color='black',
#                 labelcolor='black', labelbottom=True, pad=15, 
#                 bottom=True, top=True, left=True, right=True)
#%%
# ERA_summer_path = r'ERA5_summer_BS_atmo.pkl'
# ERA_summer = pd.read_pickle(ERA_summer_path)

# ERA_fall_path = r'ERA5_freeze_up_BS_atmo.pkl'
# ERA_fall = pd.read_pickle(ERA_fall_path)
# #%%
# ERA_lat = np.linspace(78,70,33)
# ERA_lon = np.linspace(-160,-120,161)
# #%%
# net_ao = np.zeros(len(ERA_summer))
# net_ai = np.zeros(len(ERA_summer))

# for i in range(len(ERA_summer)):
#     net_ai[i] = np.sum(ERA_summer['net_atmo_ice'][i])/((zero_indexs[i]-may_index[i])*86400)
#     net_ao[i] = np.sum(ERA_summer['net_atmo_ocean'][i])/((zero_indexs[i]-may_index[i])*86400)
# #%%
# fig, axs = plt.subplot_mosaic([['(a)'],['(b)']],
#                               layout='constrained', figsize=(30, 20))

# for label, ax in axs.items():
#     # label physical distance to the left and up:
#     trans = mtransforms.ScaledTranslation(-20/72, 7/72, fig.dpi_scale_trans)
#     ax.text(-0.05, 1.05, label, transform=ax.transAxes + trans,
#             fontsize=40, va='bottom', fontfamily='serif')

# ax1 = axs['(a)']
# ax2 = axs['(b)']

# ax1.plot(years, net_ao/1e3, label='Atmo-Ocean')
# ax1.set_ylabel('Atmosphere-Ocean Heat Flux \n(kW m$^{-2}$)', fontsize=30)
# ax1.set_xticks(years)
# ax1.tick_params(labelsize=30, direction='in', length=16, width=4, color='black',
#                 labelcolor='black', labelbottom=True, pad=15, 
#                 bottom=True, top=True, left=True, right=True)

# ax2.plot(years, net_ai/1e3, label='Atmo-Ice')
# ax2.set_ylabel('Atmosphere-Ice Heat Flux \n(kW m$^{-2}$)', fontsize=30)
# ax2.set_xticks(years)
# ax2.tick_params(labelsize=30, direction='in', length=16, width=4, color='black',
#                 labelcolor='black', labelbottom=True, pad=15,
#                 bottom=True, top=True, left=True, right=True)
# #%%
# fig, axs = plt.subplot_mosaic([['(a)']],
#                                     per_subplot_kw={'(a)':{'projection': nps}},
#                                   layout='constrained', figsize=(30, 30))
    
# for label, ax in axs.items():
#     # label physical distance to the left and up:
#     trans = mtransforms.ScaledTranslation(-20/72, 7/72, fig.dpi_scale_trans)
#     ax.text(-0.05, 1.05, label, transform=ax.transAxes + trans,
#             fontsize=40, va='bottom', fontfamily='serif')
                
# ax = axs['(a)']

# extent = [-165, -115, 69, 79]
# ax.set_extent(extent, ccrs.PlateCarree())  # Use PlateCarree for defining the extent
# ax.add_feature(cartopy.feature.LAND, zorder=3, edgecolor='black')
# ax.pcolormesh(ERA_lon, ERA_lat, ERA_summer['net_atmo_ice'][0]/11923200, cmap='viridis', 
#               transform=ccrs.PlateCarree())
# ax.plot(left[1], left[0], transform=ccrs.PlateCarree(), c = 'k', linewidth=5, zorder=1)
# ax.plot(right[1], right[0], transform=ccrs.PlateCarree(), c = 'k', linewidth=5, zorder=1)
# ax.plot(top[1], top[0], transform=ccrs.PlateCarree(), c = 'k', linewidth=5, zorder=1)
#%%
fig, axs = plt.subplot_mosaic([['(a)','(b)'],['(c)','(d)'],['(e)','(f)']],
                              layout='constrained', figsize=(60, 30), sharex=True)

for label, ax in axs.items():
    # label physical distance to the left and up:
    trans = mtransforms.ScaledTranslation(-20/72, 7/72, fig.dpi_scale_trans)
    ax.text(-0.05, 1.05, label, transform=ax.transAxes + trans,
            fontsize=40, va='bottom', fontfamily='serif')

ax1 = axs['(a)']
ax2 = axs['(c)']
ax3 = axs['(e)']

ax4 = axs['(b)']
ax5 = axs['(d)']
ax6 = axs['(f)']

days = np.arange(1,366)
window_size = 30
for year in years:
    N = north_VF_05_22[year-year1]*rho_i*L_i
    W = west_VF_05_22[year-year1]*rho_i*L_i
    Net = N+W
    
    N = pd.Series(N).rolling(window=window_size, min_periods=1).mean()
    W = pd.Series(W).rolling(window=window_size, min_periods=1).mean()
    Net = pd.Series(Net).rolling(window=window_size, min_periods=1).mean()
    
    ax1.plot(days, N/1e2, label=year, lw=2)
    ax2.plot(days, W/1e2, label=year, lw=2)
    ax3.plot(days, Net/1e2, label=year, lw=2)
    
    N = north_vels[year-year1]
    W = west_vels[year-year1]
    
    N = pd.Series(N).rolling(window=window_size, min_periods=1).mean()
    W = pd.Series(W).rolling(window=window_size, min_periods=1).mean()
    
    ax5.plot(days, N/1e2, label=year, lw=2)
    ax6.plot(days, W/1e2, label=year, lw=2)
    
    ax4.plot(days, daily_heff[year-year1], lw=2)

ax1.set_ylabel('Northern Boundary Flux \n(10$^2$ km$^3$)', fontsize=30)
ax1.set_xlim(1,365)
# ax1.set_ylim(-140,-50)
ax1.axhline(0, color='k', ls=':')
ax1.set_xticks([1,31,59,90,120,151,181,212,243,273,304,334,365])
ax1.set_xticklabels(['Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun', 
                     'Jul', 'Aug', 'Sep', 'Oct', 'Nov', 'Dec', 'Jan'])
ax1.tick_params(labelsize=30, direction='in', length=16, width=4, color='black',
                labelcolor='black', labelbottom=True, pad=15, 
                bottom=True, top=True, left=True, right=True)
ax1.grid(True, zorder=0)

ax2.set_ylabel('Western Boundary Flux \n(10$^2$ km$^3$)', fontsize=30)
ax2.set_xlim(1,365)
# ax2.set_ylim(-200,100)
ax2.axhline(0, color='k', ls=':')
ax2.set_xticks([1,31,59,90,120,151,181,212,243,273,304,334,365])
ax2.set_xticklabels(['Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun', 
                     'Jul', 'Aug', 'Sep', 'Oct', 'Nov', 'Dec', 'Jan'])
ax2.tick_params(labelsize=30, direction='in', length=16, width=4, color='black',
                labelcolor='black', labelbottom=True, pad=15, 
                bottom=True, top=True, left=True, right=True)
ax2.grid(True, zorder=0)
# ax2.legend(fontsize=30)

ax3.set_ylabel('Net Boundary Flux \n(10$^2$ km$^3$)', fontsize=30)
ax3.set_xlim(1,365)
# ax3.set_ylim(-1,1)
ax3.axhline(0, color='k', ls=':')
ax3.set_xticks([1,31,59,90,120,151,181,212,243,273,304,334,365])
ax3.set_xticklabels(['Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun', 
                     'Jul', 'Aug', 'Sep', 'Oct', 'Nov', 'Dec', 'Jan'])
ax3.tick_params(labelsize=30, direction='in', length=16, width=4, color='black',
                labelcolor='black', labelbottom=True, pad=15, 
                bottom=True, top=True, left=True, right=True)
ax3.grid(True, zorder=0)

ax4.set_ylabel('Ice Thickness (m)', fontsize=30)
ax4.set_xlim(1,365)
# ax3.set_ylim(-1,1)
ax4.axhline(0, color='k', ls=':')
ax4.set_xticks([1,31,59,90,120,151,181,212,243,273,304,334,365])
ax4.set_xticklabels(['Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun', 
                     'Jul', 'Aug', 'Sep', 'Oct', 'Nov', 'Dec', 'Jan'])
ax4.tick_params(labelsize=30, direction='in', length=16, width=4, color='black',
                labelcolor='black', labelbottom=True, pad=15, 
                bottom=True, top=True, left=True, right=True)
ax4.grid(True, zorder=0)
#%%
fig, axs = plt.subplot_mosaic([['(a)'],['(b)']],#,['(c)']],
                              layout='constrained', figsize=(30, 20), sharex=True)

for label, ax in axs.items():
    # label physical distance to the left and up:
    trans = mtransforms.ScaledTranslation(-20/72, 7/72, fig.dpi_scale_trans)
    ax.text(-0.05, 1.05, label, transform=ax.transAxes + trans,
            fontsize=40, va='bottom', fontfamily='serif')

ax1 = axs['(a)']
ax2 = axs['(b)']
# ax3 = axs['(c)']

V_tot = np.array(tot_V)/1e20
# V_tot = np.full(len(years),np.nan)
# V_tot[:-1] = (np.array(tot_V_win[1:])+np.array(tot_V_fall[:-1])+np.array(tot_V_sum[:-1]))/1e20

ax1.plot(years, V_tot, label='Net Volume loss', color='k', lw=4)
# ax1.errorbar(years, V_tot, yerr=np.array(tot_V_std)/1e20, fmt='-o', linewidth=2, color='k', capsize=5)
ax1.fill_between(years, V_tot-0.416, V_tot+0.416,  alpha=0.25, color='black')
ax1.set_ylabel(r'$\Delta$V$_{tot}$ ($\times$ 10$^{20}$ J)', fontsize=30)
ax1.set_ylabel(r'Mean V$_{tot}$ ($\times$ 10$^{20}$ J)', fontsize=30)
# ax1.set_ylim(-2.5,2.5)
# ax1.set_ylim(-0.5,0.5)
# ax1.set_xticks(years)
ax1.tick_params(labelsize=30, direction='in', length=16, width=4, color='black',
                labelcolor='black', labelbottom=True, pad=15, 
                bottom=True, top=True, left=True, right=True)
ax1.grid(True, zorder=0)

res = linregress(years, V_tot)
print(res.intercept,res.slope,res.stderr)
ax1.plot(years, res.intercept + res.slope*years, color = 'dimgrey', ls = '--', label='fitted line', lw=3)
# ax1.text(2006, 1, 
#           'Slope = {slope} $\pm$ {std}'.format(slope=round(res.slope,3), 
#                                               std=round(res.stderr*1.96,3)), 
#           color='k', fontsize=30)
ax1.text(2006, 5.5, 
          'Slope = {slope} $\pm$ {std}'.format(slope=round(res.slope,3), 
                                              std=round(res.stderr*1.96,3)), 
          color='k', fontsize=30)

# Net_sum=[]
# for year in years:
#     N = north_VF_05_22[year-year1]*rho_i*L_i
#     E = east_VF_05_22[year-year1]*rho_i*L_i
#     W = west_VF_05_22[year-year1]*rho_i*L_i
#     Net = N+E+W
#     Net_sum.append(np.nansum(Net))
Net_sum = np.full(len(years),np.nan)
for year in range(len(years)-1):
    N1 = north_VF_05_22[year]*rho_i*L_i
    E1 = east_VF_05_22[year]*rho_i*L_i
    W1 = west_VF_05_22[year]*rho_i*L_i
    N2 = north_VF_05_22[year+1]*rho_i*L_i
    E2 = east_VF_05_22[year+1]*rho_i*L_i
    W2 = west_VF_05_22[year+1]*rho_i*L_i
    N = np.hstack((N1[120:],N2[:120]))
    E = np.hstack((E1[120:],E2[:120]))
    W = np.hstack((W1[120:],W2[:120]))
    Net = N+E+W
    Net_sum[year] = np.nansum(Net)
    
D_tot = Net_sum/1e20
T_tot = (V_tot-Net_sum)/1e20
# Qtd = (T**2-D**2)/(T**2+D**2)

ax2.bar(years-0.2, D_tot, width=0.4, label='$\Delta$V$_{advect}$', color='blue')
ax2.bar(years+0.2, T_tot, width=0.4, label='$\Delta$V$_{thermo}$', color='red')
ax2.set_ylabel(r'$\Delta$V ($\times$ 10$^{20}$ J)', fontsize=30)
ax2.set_ylim(-2.5,2.5)
# ax2.set_xticks(years)
ax2.tick_params(labelsize=30, direction='in', length=16, width=4, color='black',
                labelcolor='black', labelbottom=True, pad=15, 
                bottom=True, top=True, left=True, right=True)
ax2.grid(True, zorder=0)
ax2.legend(fontsize=30)

res = linregress(years[:-1], T_tot[:-1])
ax2.plot(years[:-1], res.intercept + res.slope*years[:-1], color = 'crimson', ls = '--', label='fitted line', lw=3)
ax2.text(1982, 1.5, 
         'Slope = {slope} $\pm$ {std}'.format(slope=round(res.slope,3), 
                                              std=round(res.stderr*1.96,3)), 
         color='crimson', fontsize=30)

res = linregress(years[:-1], D_tot[:-1])
ax2.plot(years[:-1], res.intercept + res.slope*years[:-1], color = 'navy', ls = '--', label='fitted line', lw=3)
ax2.text(1982, -1.9, 
         'Slope = {slope} $\pm$ {std}'.format(slope=round(res.slope,3), 
                                              std=round(res.stderr*1.96,3)), 
         color='navy', fontsize=30)

plt.suptitle('Total Year', fontsize=35)
#%%
fig, axs = plt.subplot_mosaic([['(a)','(b)'],['(c)','(d)']],
                              layout='constrained', figsize=(40, 20), sharex=True)

for label, ax in axs.items():
    # label physical distance to the left and up:
    trans = mtransforms.ScaledTranslation(-20/72, 7/72, fig.dpi_scale_trans)
    ax.text(-0.05, 1.05, label, transform=ax.transAxes + trans,
            fontsize=40, va='bottom', fontfamily='serif')

ax1 = axs['(a)']
ax2 = axs['(c)']
ax3 = axs['(b)']
ax4 = axs['(d)']

V_tot = np.full(len(years),np.nan)
V_tot[:-1] = (np.array(tot_V_win[1:])+np.array(tot_V_fall[:-1]))/1e20

ax1.set_title('Growth Season (Sep 15 to May 1)', fontsize=35)

ax1.plot(years[:-1], V_tot[:-1], label='Net Volume loss', color='k', lw=4)
ax1.fill_between(years[:-1], V_tot[:-1]-0.416, V_tot[:-1]+0.416,  alpha=0.25, color='black')
# ax1.errorbar(years[:-1], V_tot, yerr=np.array(tot_V_win_std[:-1])/1e20, fmt='-o', linewidth=2, color='k', capsize=5)
ax1.set_ylabel(r'$\Delta$V$_{tot}$ ($\times$ 10$^{20}$ J)', fontsize=30)
ax1.set_ylim(0.5,5)
# ax1.set_xticks(years)
ax1.tick_params(labelsize=30, direction='in', length=16, width=4, color='black',
                labelcolor='black', labelbottom=True, pad=15, 
                bottom=True, top=True, left=True, right=True)
ax1.grid(True, zorder=0)

res = linregress(years[:-1], V_tot[:-1])
ax1.plot(years[:-1], res.intercept + res.slope*years[:-1], color = 'dimgrey', ls = '--', label='fitted line', lw=3)
ax1.text(2006, 1.6, 
         'Slope = {slope} $\pm$ {std}'.format(slope=round(res.slope,3), 
                                              std=round(res.stderr*1.96,3)), 
         color='k', fontsize=30)

Net_sum = np.full(len(years),np.nan)
for year in range(len(years)-1):
    N1 = north_VF_05_22[year]*rho_i*L_i
    E1 = east_VF_05_22[year]*rho_i*L_i
    W1 = west_VF_05_22[year]*rho_i*L_i
    N2 = north_VF_05_22[year+1]*rho_i*L_i
    E2 = east_VF_05_22[year+1]*rho_i*L_i
    W2 = west_VF_05_22[year+1]*rho_i*L_i
    N = np.hstack((N1[258:],N2[:120]))
    E = np.hstack((E1[258:],E2[:120]))
    W = np.hstack((W1[258:],W2[:120]))
    Net = N+E+W
    Net_sum[year] = np.nansum(Net)/1e20
    
D_grow = Net_sum
T_grow = V_tot-D_grow
# Qtd = (T**2-D**2)/(T**2+D**2)

ax2.bar(years-0.2, D_grow, width=0.4, label='$\Delta$V$_{advect}$', color='blue',
        yerr=np.abs(0.416*D_grow), capsize=4)
ax2.bar(years+0.2, T_grow, width=0.4, label='$\Delta$V$_{thermo}$', color='red',
        yerr=np.abs(0.416*D_grow), capsize=4)
ax2.set_ylabel(r'$\Delta$V ($\times$ 10$^{20}$ J)', fontsize=30)
ax2.set_ylim(-3,6)
# ax2.set_xticks(years)
ax2.tick_params(labelsize=30, direction='in', length=16, width=4, color='black',
                labelcolor='black', labelbottom=True, pad=15, 
                bottom=True, top=True, left=True, right=True)
ax2.grid(True, zorder=0)
ax2.legend(fontsize=30)

res = linregress(years[:-1], T_grow[:-1])
ax2.plot(years[:-1], res.intercept + res.slope*years[:-1], color = 'crimson', ls = '--', label='fitted line', lw=3)
ax2.text(1989, 4.3, 
         'Slope = {slope} $\pm$ {std}'.format(slope=round(res.slope,3), 
                                              std=round(res.stderr*1.96,3)), 
         color='crimson', fontsize=30)

res = linregress(years[:-1], D_grow[:-1])
ax2.plot(years[:-1], res.intercept + res.slope*years[:-1], color = 'navy', ls = '--', label='fitted line', lw=3)
ax2.text(2011, -1.9, 
         'Slope = {slope} $\pm$ {std}'.format(slope=round(res.slope,3), 
                                              std=round(res.stderr*1.96,3)), 
         color='navy', fontsize=30)

### Melt Season Plots###
ax3.set_title('Melt Season (May 1 to Sep 15)', fontsize=35)
V_tot = np.array(tot_V_sum)/1e20
ax3.plot(years, V_tot, label='Net Volume loss', color='k', lw=4)
ax3.fill_between(years, V_tot-0.416, V_tot+0.416,  alpha=0.25, color='black')
# ax1.errorbar(years, V_tot, yerr=np.array(tot_V_sum_std)/1e20, fmt='-o', linewidth=2, color='k', capsize=5)
ax3.set_ylabel(r'$\Delta$V$_{tot}$ ($\times$ 10$^{20}$ J)', fontsize=30)
ax3.set_ylim(-5,-0.5)
# ax1.set_xticks(years)
ax3.tick_params(labelsize=30, direction='in', length=16, width=4, color='black',
                labelcolor='black', labelbottom=True, pad=15, 
                bottom=True, top=True, left=True, right=True)
ax3.grid(True, zorder=0)

res = linregress(years, V_tot)
ax3.plot(years, res.intercept + res.slope*years, color = 'dimgrey', ls = '--', label='fitted line', lw=3)
ax3.text(2001, -1.9, 
         'Slope = {slope} $\pm$ {std}'.format(slope=round(res.slope,3), 
                                              std=round(res.stderr*1.96,3)), 
         color='k', fontsize=30)

Net_summer_sum=[]
Net_err=[]
for year in years:
    N = north_VF_05_22[year-year1]*rho_i*L_i
    E = east_VF_05_22[year-year1]*rho_i*L_i
    W = west_VF_05_22[year-year1]*rho_i*L_i
    Nerr = north_err[year-year1]*rho_i*L_i
    Eerr = east_err[year-year1]*rho_i*L_i
    Werr = west_err[year-year1]*rho_i*L_i
    Net = N+E+W
    net_err = Nerr+Eerr+Werr
    Net_summer_sum.append(np.nansum(Net[120:258]))
    Net_err.append(np.nansum(net_err[120:258]))
    
D_melt = np.array(Net_summer_sum)/1e20
T_melt = (np.array(tot_V_sum)-np.array(Net_summer_sum))/1e20
# Qtd = (T**2-D**2)/(T**2+D**2)

ax4.bar(np.array(years)-0.2, D_melt, width=0.4, label='$\Delta$V$_{advect}$', color='blue',
        yerr=np.abs(0.416*D_melt), capsize=4)
ax4.bar(np.array(years)+0.2, T_melt, width=0.4, label='$\Delta$V$_{thermo}$', color='red',
        yerr=np.abs(0.416*D_melt), capsize=4)
# ax4.errorbar(years, D_melt, yerr=np.abs(0.416*D_melt), 
#               fmt='-o', linewidth=3, color='blue')

ax4.set_ylabel(r'$\Delta$V ($\times$ 10$^{20}$ J)', fontsize=30)
ax4.set_ylim(-6,3)
# ax2.set_xticks(years)
ax4.tick_params(labelsize=30, direction='in', length=16, width=4, color='black',
                labelcolor='black', labelbottom=True, pad=15, 
                bottom=True, top=True, left=True, right=True)
ax4.grid(True, zorder=0)
ax4.legend(fontsize=30, loc='lower left')

res = linregress(years, T_melt)
ax4.plot(years, res.intercept + res.slope*years, color = 'crimson', ls = '--', label='fitted line', lw=3)
ax4.text(1991, -4.5, 
         'Slope = {slope} $\pm$ {std}'.format(slope=round(res.slope,3), 
                                              std=round(res.stderr*1.96,3)), 
         color='crimson', fontsize=30)

res = linregress(years, D_melt)
ax4.plot(years, res.intercept + res.slope*years, color = 'navy', ls = '--', label='fitted line', lw=3)
ax4.text(1992, 1.4, 
         'Slope = {slope} $\pm$ {std}'.format(slope=round(res.slope,3), 
                                              std=round(res.stderr*1.96,3)), 
         color='navy', fontsize=30)
#%%
timeseries = pd.DataFrame({'year' : years,
                            'dyna_tot': D_tot,
                            'therm_tot': T_tot, 
                            'dyna_melt' : D_melt,
                            'therm_melt' : T_melt,
                            'dyna_grow' : D_grow,
                            'therm_grow' : T_grow})
timeseries.to_pickle(r'PIOMAS_volume_change_fluxes.pkl')
#%%
fig, axs = plt.subplot_mosaic([['(a)'],['(b)']],#['(c)']],#['(d)']],
                              layout='constrained', figsize=(30, 20), sharex=True)

for label, ax in axs.items():
    # label physical distance to the left and up:
    trans = mtransforms.ScaledTranslation(-20/72, 7/72, fig.dpi_scale_trans)
    ax.text(-0.05, 1.05, label, transform=ax.transAxes + trans,
            fontsize=40, va='bottom', fontfamily='serif')

# ax1 = axs['(a)']
ax2 = axs['(a)']
ax3 = axs['(b)']
# ax4 = axs['(d)']

# N_sum=[]
# E_sum=[]
# W_sum=[]
# for year in years:
#     N = north_VF_05_22[year-year1]*rho_i*L_i
#     E = east_VF_05_22[year-year1]*rho_i*L_i
#     W = west_VF_05_22[year-year1]*rho_i*L_i
#     N_sum.append(np.nansum(N)/1e20)
#     E_sum.append(np.nansum(E)/1e20)
#     W_sum.append(np.nansum(W)/1e20)

# ax1.bar(years-0.2, N_sum, width=0.25, label='$\Delta$V$_{advect, N}$', color='darkorchid')
# ax1.bar(years, E_sum, width=0.25, label='$\Delta$V$_{advect, E}$', color='green')
# ax1.bar(years+0.2, W_sum, width=0.25, label='$\Delta$V$_{advect, W}$', color='orange')
# ax1.set_ylabel(r'$\Delta$V$_{advect}$ ($\times$ 10$^{20}$ J)', fontsize=30)
# ax1.set_ylim(-3.25,4.25)
# ax1.tick_params(labelsize=30, direction='in', length=16, width=4, color='black',
#                 labelcolor='black', labelbottom=True, pad=15, 
#                 bottom=True, top=True, left=True, right=True)
# ax1.grid(True, zorder=0)
# ax1.legend(fontsize=30, loc='lower left', bbox_to_anchor=(0.865, 1.01), 
#            fancybox=False, framealpha=1, edgecolor="inherit")

# res = linregress(years, N_sum)
# ax1.plot(years, res.intercept + res.slope*years, color = 'indigo', ls = '--', label='fitted line', lw=3)
# ax1.text(1990.5, 3.8, 
#          'Slope = {slope} $\pm$ {std}'.format(slope=round(res.slope,3), 
#                                               std=round(res.stderr*1.96,3)), 
#          color='indigo', fontsize=30)

# res = linregress(years, E_sum)
# ax1.plot(years, res.intercept + res.slope*years, color = 'green', ls = '--', label='fitted line', lw=3)
# ax1.text(1990.5, -2.3, 
#          'Slope = {slope} $\pm$ {std}'.format(slope=round(res.slope,3), 
#                                               std=round(res.stderr*1.96,3)), 
#          color='green', fontsize=30)

# res = linregress(years, W_sum)
# ax1.plot(years, res.intercept + res.slope*years, color = 'orangered', ls = '--', label='fitted line', lw=3)
# ax1.text(1990.5, -2.9, 
#          'Slope = {slope} $\pm$ {std}'.format(slope=round(res.slope,3), 
#                                               std=round(res.stderr*1.96,3)), 
#          color='orangered', fontsize=30)

# ax1.set_title('Total Year', fontsize=35)

N_sum=[]
E_sum=[]
W_sum=[]
for i in range(len(years)-1):
    N1 = north_VF_05_22[i]*rho_i*L_i/1e20
    E1 = east_VF_05_22[i]*rho_i*L_i/1e20
    W1 = west_VF_05_22[i]*rho_i*L_i/1e20
    N2 = north_VF_05_22[i+1]*rho_i*L_i/1e20
    E2 = east_VF_05_22[i+1]*rho_i*L_i/1e20
    W2 = west_VF_05_22[i+1]*rho_i*L_i/1e20
    N_sum.append(np.nansum(np.hstack((N1[258:],N2[:120]))))
    E_sum.append(np.nansum(np.hstack((E1[258:],E2[:120]))))
    W_sum.append(np.nansum(np.hstack((W1[258:],W2[:120]))))
    
N_sum=np.array(N_sum)
E_sum=np.array(E_sum)
W_sum=np.array(W_sum)

ax2.bar(years[:-1]-0.2, N_sum, width=0.25, label='$\Delta$V$_{advect, N}$', color='darkorchid',
        yerr=np.abs(0.3*N_sum), capsize=4)
ax2.bar(years[:-1], E_sum, width=0.25, label='$\Delta$V$_{advect, E}$', color='green',
        yerr=np.abs(0.3*E_sum), capsize=4)
ax2.bar(years[:-1]+0.2, W_sum, width=0.25, label='$\Delta$V$_{advect, W}$', color='orangered',
        yerr=np.abs(0.3*W_sum), capsize=4)
ax2.set_ylabel(r'$\Delta$V$_{advect}$ ($\times$ 10$^{20}$ J)', fontsize=30)
ax2.set_ylim(-4,4)
ax2.tick_params(labelsize=30, direction='in', length=16, width=4, color='black',
                labelcolor='black', labelbottom=True, pad=15, 
                bottom=True, top=True, left=True, right=True)
ax2.grid(True, zorder=0)
ax2.legend(fontsize=30, loc='lower left', bbox_to_anchor=(0.865, 1.01), 
            fancybox=False, framealpha=1, edgecolor="inherit")

res = linregress(years[:-1], N_sum)
ax2.plot(years[:-1], res.intercept + res.slope*years[:-1], color = 'indigo', ls = '--', label='fitted line', lw=3)
ax2.text(1990.5, 3.1, 
         'Slope = {slope} $\pm$ {std}'.format(slope=round(res.slope,3), 
                                              std=round(res.stderr*1.96,3)), 
         color='indigo', fontsize=30)

res = linregress(years[:-1], E_sum)
ax2.plot(years[:-1], res.intercept + res.slope*years[:-1], color = 'green', ls = '--', label='fitted line', lw=3)
ax2.text(1990.5, -2.5, 
         'Slope = {slope} $\pm$ {std}'.format(slope=round(res.slope,3), 
                                              std=round(res.stderr*1.96,3)), 
         color='green', fontsize=30)

res = linregress(years[:-1], W_sum)
ax2.plot(years[:-1], res.intercept + res.slope*years[:-1], color = 'orangered', ls = '--', label='fitted line', lw=3)
ax2.text(1990.5, -3, 
         'Slope = {slope} $\pm$ {std}'.format(slope=round(res.slope,3), 
                                              std=round(res.stderr*1.96,3)), 
         color='orangered', fontsize=30)

ax2.set_title('Growth  Season', fontsize=35)

N_sum=[]
E_sum=[]
W_sum=[]
for year in years:
    N = north_VF_05_22[year-year1]*rho_i*L_i
    E = east_VF_05_22[year-year1]*rho_i*L_i
    W = west_VF_05_22[year-year1]*rho_i*L_i
    N_sum.append(np.nansum(N[120:258])/1e20)
    E_sum.append(np.nansum(E[120:258])/1e20)
    W_sum.append(np.nansum(W[120:258])/1e20)

N_sum=np.array(N_sum)
E_sum=np.array(E_sum)
W_sum=np.array(W_sum)

ax3.bar(np.array(years)-0.2, N_sum, width=0.25, label='$\Delta$V$_{advect, N}$', color='darkorchid',
        yerr=np.abs(0.3*N_sum), capsize=4)
ax3.bar(np.array(years), E_sum, width=0.25, label='$\Delta$V$_{advect, E}$', color='green',
        yerr=np.abs(0.3*E_sum), capsize=4)
ax3.bar(np.array(years)+0.2, W_sum, width=0.25, label='$\Delta$V$_{advect, W}$', color='orangered',
        yerr=np.abs(0.3*W_sum), capsize=4)
ax3.set_ylabel(r'$\Delta$V$_{advect}$ ($\times$ 10$^{20}$ J)', fontsize=30)
ax3.set_ylim(-4,4)
ax3.tick_params(labelsize=30, direction='in', length=16, width=4, color='black',
                labelcolor='black', labelbottom=True, pad=15, 
                bottom=True, top=True, left=True, right=True)
ax3.grid(True, zorder=0)

res = linregress(years, N_sum)
ax3.plot(years, res.intercept + res.slope*years, color = 'indigo', ls = '--', label='fitted line', lw=3)
ax3.text(1990.5, 1.6, 
         'Slope = {slope} $\pm$ {std}'.format(slope=round(res.slope,3), 
                                              std=round(res.stderr*1.96,3)), 
         color='indigo', fontsize=30)

res = linregress(years, E_sum)
ax3.plot(years, res.intercept + res.slope*years, color = 'green', ls = '--', label='fitted line', lw=3)
ax3.text(1990.5, -2.5, 
         'Slope = {slope} $\pm$ {std}'.format(slope=round(res.slope,3), 
                                              std=round(res.stderr*1.96,3)), 
         color='green', fontsize=30)

res = linregress(years, W_sum)
ax3.plot(years, res.intercept + res.slope*years, color = 'orangered', ls = '--', label='fitted line', lw=3)
ax3.text(1990.5, -2.9, 
         'Slope = {slope} $\pm$ {std}'.format(slope=round(res.slope,3), 
                                              std=round(res.stderr*1.96,3)), 
         color='orangered', fontsize=30)

ax3.set_title('Melt Season', fontsize=35)
#%%
fig, axs = plt.subplot_mosaic([['(a)'],['(b)']],#['(d)']],
                              layout='constrained', figsize=(30, 20), sharex=True)

for label, ax in axs.items():
    # label physical distance to the left and up:
    trans = mtransforms.ScaledTranslation(-20/72, 7/72, fig.dpi_scale_trans)
    ax.text(-0.05, 1.05, label, transform=ax.transAxes + trans,
            fontsize=40, va='bottom', fontfamily='serif')

ax2 = axs['(a)']
ax3 = axs['(b)']
# ax3 = axs['(c)']
# ax4 = axs['(d)']

# N_sum=[]
# E_sum=[]
# W_sum=[]
# for year in years:
#     N = north_VF_05_22[year-year1]/north_AF_05_22[year-year1]
#     E = east_VF_05_22[year-year1]/east_AF_05_22[year-year1]
#     W = west_VF_05_22[year-year1]/west_AF_05_22[year-year1]
#     N_sum.append(np.nanmean(N)*1e3)
#     E_sum.append(np.nanmean(E)*1e3)
#     W_sum.append(np.nanmean(W)*1e3)
    
# N_sum=np.array(N_sum)
# E_sum=np.array(E_sum)
# W_sum=np.array(W_sum)
# con1 = W_sum>0
# con2 = W_sum<5
# mask = con1 & con2

# ax1.plot(years, N_sum, label='Northern Boundary', color='darkorchid')
# ax1.plot(years, E_sum, label='Eastern Boundary', color='green')
# ax1.plot(years[mask], W_sum[mask], label='Western Boundary', color='orangered')
# ax1.fill_between(years, N_sum-0.17, N_sum+0.17,  alpha=0.25, color='darkorchid')
# ax1.fill_between(years, E_sum-0.17, E_sum+0.17,  alpha=0.25, color='green')
# ax1.fill_between(years[mask], W_sum[mask]-0.17, W_sum[mask]+0.17,  alpha=0.25, color='orangered')
# ax1.set_ylabel(r'Ice Thickness (m)', fontsize=30)
# ax1.set_ylim(0,5)
# ax1.tick_params(labelsize=30, direction='in', length=16, width=4, color='black',
#                 labelcolor='black', labelbottom=True, pad=15, 
#                 bottom=True, top=True, left=True, right=True)
# ax1.grid(True, zorder=0)
# ax1.legend(fontsize=30, loc='lower left', bbox_to_anchor=(0.865, 1.01), 
#            fancybox=False, framealpha=1, edgecolor="inherit")

# res = linregress(years, N_sum)
# ax1.plot(years, res.intercept + res.slope*years, color = 'indigo', ls = '--', label='fitted line', lw=3)
# ax1.text(1981, 4.3, 
#          'Slope = {slope} $\pm$ {std}'.format(slope=round(res.slope,3), 
#                                               std=round(res.stderr*1.96,3)), 
#          color='indigo', fontsize=30)

# res = linregress(years, E_sum)
# ax1.plot(years, res.intercept + res.slope*years, color = 'green', ls = '--', label='fitted line', lw=3)
# ax1.text(1981, 0.75, 
#          'Slope = {slope} $\pm$ {std}'.format(slope=round(res.slope,3), 
#                                               std=round(res.stderr*1.96,3)), 
#          color='green', fontsize=30)

# res = linregress(years[mask], W_sum[mask])
# ax1.plot(years, res.intercept + res.slope*years, color = 'orangered', ls = '--', label='fitted line', lw=3)
# ax1.text(1981, 0.25, 
#          'Slope = {slope} $\pm$ {std}'.format(slope=round(res.slope,3), 
#                                               std=round(res.stderr*1.96,3)), 
#          color='orangered', fontsize=30)

# ax1.set_title('Total Year', fontsize=35)

N_sum=[]
E_sum=[]
W_sum=[]
for i in range(len(years)-1):
    N1 = north_VF_05_22[i]/north_AF_05_22[i]*1e3
    E1 = east_VF_05_22[i]/east_AF_05_22[i]*1e3
    W1 = west_VF_05_22[i]/west_AF_05_22[i]*1e3
    N2 = north_VF_05_22[i+1]/north_AF_05_22[i+1]*1e3
    E2 = east_VF_05_22[i+1]/east_AF_05_22[i+1]*1e3
    W2 = west_VF_05_22[i+1]/west_AF_05_22[i+1]*1e3
    N_sum.append(np.nanmean(np.hstack((N1[258:],N2[:120]))))
    E_sum.append(np.nanmean(np.hstack((E1[258:],E2[:120]))))
    W_sum.append(np.nanmean(np.hstack((W1[258:],W2[:120]))))
    
N_sum=np.array(N_sum)
E_sum=np.array(E_sum)
W_sum=np.array(W_sum)
con1 = W_sum>0
con2 = W_sum<10
mask = con1 & con2

ax2.plot(years[:-1], N_sum, label='$\Delta$V$_{advect, N}$', color='darkorchid')
ax2.plot(years[:-1], E_sum, label='$\Delta$V$_{advect, E}$', color='green')
ax2.plot(years[:-1], W_sum[mask], label='$\Delta$V$_{advect, W}$', color='orangered')
ax2.fill_between(years[:-1], N_sum-0.17, N_sum+0.17,  alpha=0.25, color='darkorchid')
ax2.fill_between(years[:-1], E_sum-0.17, E_sum+0.17,  alpha=0.25, color='green')
ax2.fill_between(years[:-1], W_sum[mask]-0.17, W_sum[mask]+0.17,  alpha=0.25, color='orangered')
ax2.set_ylabel(r'Ice Thickness (m)', fontsize=30)
ax2.set_ylim(0,5)
ax2.tick_params(labelsize=30, direction='in', length=16, width=4, color='black',
                labelcolor='black', labelbottom=True, pad=15, 
                bottom=True, top=True, left=True, right=True)
ax2.grid(True, zorder=0)

res = linregress(years[:-1], N_sum)
ax2.plot(years[:-1], res.intercept + res.slope*years[:-1], color = 'indigo', ls = '--', label='fitted line', lw=3)
ax2.text(1981, 4.3, 
         'Slope = {slope} $\pm$ {std}'.format(slope=round(res.slope,3), 
                                              std=round(res.stderr*1.96,3)), 
         color='indigo', fontsize=30)

res = linregress(years[:-1], E_sum)
ax2.plot(years[:-1], res.intercept + res.slope*years[:-1], color = 'green', ls = '--', label='fitted line', lw=3)
ax2.text(1981, 0.75, 
         'Slope = {slope} $\pm$ {std}'.format(slope=round(res.slope,3), 
                                              std=round(res.stderr*1.96,3)), 
         color='green', fontsize=30)

res = linregress(years[:-1], W_sum[mask])
ax2.plot(years[:-1], res.intercept + res.slope*years[:-1], color = 'orangered', ls = '--', label='fitted line', lw=3)
ax2.text(1981, 0.25, 
         'Slope = {slope} $\pm$ {std}'.format(slope=round(res.slope,3), 
                                              std=round(res.stderr*1.96,3)), 
         color='orangered', fontsize=30)

ax2.set_title('Growth Season', fontsize=35)


N_sum=[]
E_sum=[]
W_sum=[]
for year in years:
    N = north_VF_05_22[year-year1]/north_AF_05_22[year-year1]
    E = east_VF_05_22[year-year1]/east_AF_05_22[year-year1]
    W = west_VF_05_22[year-year1]/west_AF_05_22[year-year1]
    N_sum.append(np.nanmean(N[120:258])*1e3)
    E_sum.append(np.nanmean(E[120:258])*1e3)
    W_sum.append(np.nanmean(W[120:258])*1e3)
    
N_sum=np.array(N_sum)
E_sum=np.array(E_sum)
W_sum=np.array(W_sum)

ax3.plot(years, N_sum, label='$\Delta$V$_{advect, N}$', color='darkorchid')
ax3.plot(years, E_sum, label='$\Delta$V$_{advect, E}$', color='green')
ax3.plot(years, W_sum, label='$\Delta$V$_{advect, W}$', color='orangered')
ax3.fill_between(years, N_sum-0.17, N_sum+0.17,  alpha=0.25, color='darkorchid')
ax3.fill_between(years, E_sum-0.17, E_sum+0.17,  alpha=0.25, color='green')
ax3.fill_between(years, W_sum-0.17, W_sum+0.17,  alpha=0.25, color='orangered')
ax3.set_ylabel(r'Ice Thickness (m)', fontsize=30)
ax3.set_ylim(0,5)
ax3.tick_params(labelsize=30, direction='in', length=16, width=4, color='black',
                labelcolor='black', labelbottom=True, pad=15, 
                bottom=True, top=True, left=True, right=True)
ax3.grid(True, zorder=0)

res = linregress(years, N_sum)
ax3.plot(years, res.intercept + res.slope*years, color = 'indigo', ls = '--', label='fitted line', lw=3)
ax3.text(1981, 4.3, 
         'Slope = {slope} $\pm$ {std}'.format(slope=round(res.slope,3), 
                                              std=round(res.stderr*1.96,3)), 
         color='indigo', fontsize=30)

res = linregress(years, E_sum)
ax3.plot(years, res.intercept + res.slope*years, color = 'green', ls = '--', label='fitted line', lw=3)
ax3.text(1981, 0.75, 
         'Slope = {slope} $\pm$ {std}'.format(slope=round(res.slope,3), 
                                              std=round(res.stderr*1.96,3)), 
         color='green', fontsize=30)

res = linregress(years, W_sum)
ax3.plot(years, res.intercept + res.slope*years, color = 'orangered', ls = '--', label='fitted line', lw=3)
ax3.text(1981, 0.25, 
         'Slope = {slope} $\pm$ {std}'.format(slope=round(res.slope,3), 
                                              std=round(res.stderr*1.96,3)), 
         color='orangered', fontsize=30)

ax3.set_title('Melt Season', fontsize=35)
#%%
fig, axs = plt.subplot_mosaic([['(a)'],['(b)']],#['(d)']],
                              layout='constrained', figsize=(30, 20), sharex=True)

for label, ax in axs.items():
    # label physical distance to the left and up:
    trans = mtransforms.ScaledTranslation(-20/72, 7/72, fig.dpi_scale_trans)
    ax.text(-0.05, 1.05, label, transform=ax.transAxes + trans,
            fontsize=40, va='bottom', fontfamily='serif')

ax2 = axs['(a)']
ax3 = axs['(b)']
# ax3 = axs['(c)']
# ax4 = axs['(d)']

# N_sum=[]
# E_sum=[]
# W_sum=[]
# for year in years:
#     N = -north_vels[year-year1]
#     E = east_vels[year-year1]
#     W = west_vels[year-year1]
#     N_sum.append(np.nanmean(N[:])*100)
#     E_sum.append(np.nanmean(E[:])*100)
#     W_sum.append(np.nanmean(W[:])*100)
    
# N_sum=np.array(N_sum)
# E_sum=np.array(E_sum)
# W_sum=np.array(W_sum)

# ax1.bar(years-0.2, N_sum, width=0.25, label='u$_{N}$', color='darkorchid',
#         yerr=np.abs(0.3*N_sum), capsize=4)
# ax1.bar(years, E_sum, width=0.25, label='u$_{E}$', color='green',
#         yerr=np.abs(0.3*E_sum), capsize=4)
# ax1.bar(years+0.2, W_sum, width=0.25, label='u$_{W}$', color='orangered',
#         yerr=np.abs(0.3*W_sum), capsize=4)
# ax1.set_ylabel(r'Mean Velocity (cm s$^{-1}$)', fontsize=30)
# ax1.set_ylim(-8,5)
# ax1.tick_params(labelsize=30, direction='in', length=16, width=4, color='black',
#                 labelcolor='black', labelbottom=True, pad=15, 
#                 bottom=True, top=True, left=True, right=True)
# ax1.grid(True, zorder=0)
# ax1.legend(fontsize=30, loc='lower left', bbox_to_anchor=(0.865, 1.01), 
#            fancybox=False, framealpha=1, edgecolor="inherit")

# res = linregress(years, N_sum)
# ax1.plot(years, res.intercept + res.slope*years, color = 'indigo', ls = '--', label='fitted line', lw=3)
# ax1.text(1981, 3, 
#           'Slope = {slope} $\pm$ {std}'.format(slope=round(res.slope,3), 
#                                               std=round(res.stderr*1.96,3)), 
#           color='indigo', fontsize=30)

# res = linregress(years, E_sum)
# ax1.plot(years, res.intercept + res.slope*years, color = 'green', ls = '--', label='fitted line', lw=3)
# ax1.text(1981, -4.9, 
#           'Slope = {slope} $\pm$ {std}'.format(slope=round(res.slope,3), 
#                                               std=round(res.stderr*1.96,3)), 
#           color='green', fontsize=30)

# res = linregress(years, W_sum)
# ax1.plot(years, res.intercept + res.slope*years, color = 'orangered', ls = '--', label='fitted line', lw=3)
# ax1.text(1981, -5.9, 
#           'Slope = {slope} $\pm$ {std}'.format(slope=round(res.slope,3), 
#                                               std=round(res.stderr*1.96,3)), 
#           color='orangered', fontsize=30)

# ax1.set_title('Total Year', fontsize=35)

N_sum=[]
E_sum=[]
W_sum=[]
for i in range(len(years)-1):
    N1 = -north_vels[i]
    N2 = -north_vels[i+1]
    E1 = east_vels[i]
    E2 = east_vels[i+1]
    W1 = west_vels[i]
    W2 = west_vels[i+1]
    N_sum.append(np.nanmean(np.hstack((N1[258:],N2[:120])))*100)
    E_sum.append(np.nanmean(np.hstack((E1[258:],E2[:120])))*100)
    W_sum.append(np.nanmean(np.hstack((W1[258:],W2[:120])))*100)

N_sum=np.array(N_sum)
E_sum=np.array(E_sum)
W_sum=np.array(W_sum)

ax2.bar(years[:-1]-0.2, N_sum, width=0.25, label='$\Delta$V$_{advect, N}$', color='darkorchid',
        yerr=np.abs(0.3*N_sum), capsize=4)
ax2.bar(years[:-1], E_sum, width=0.25, label='$\Delta$V$_{advect, W}$', color='green',
        yerr=np.abs(0.3*E_sum), capsize=4)
ax2.bar(years[:-1]+0.2, W_sum, width=0.25, label='$\Delta$V$_{advect, W}$', color='orangered',
        yerr=np.abs(0.3*W_sum), capsize=4)
ax2.set_ylabel(r'Mean Velocity (cm s$^{-1}$)', fontsize=30)
ax2.set_ylim(-8,5)
ax2.tick_params(labelsize=30, direction='in', length=16, width=4, color='black',
                labelcolor='black', labelbottom=True, pad=15, 
                bottom=True, top=True, left=True, right=True)
ax2.grid(True, zorder=0)

res = linregress(years[:-1], N_sum)
ax2.plot(years[:-1], res.intercept + res.slope*years[:-1], color = 'indigo', ls = '--', label='fitted line', lw=3)
ax2.text(1981, 3, 
          'Slope = {slope} $\pm$ {std}'.format(slope=round(res.slope,3), 
                                              std=round(res.stderr*1.96,3)), 
          color='indigo', fontsize=30)

res = linregress(years[:-1], E_sum)
ax2.plot(years[:-1], res.intercept + res.slope*years[:-1], color = 'green', ls = '--', label='fitted line', lw=3)
ax2.text(1981, -4.9, 
          'Slope = {slope} $\pm$ {std}'.format(slope=round(res.slope,3), 
                                              std=round(res.stderr*1.96,3)), 
          color='green', fontsize=30)

res = linregress(years[:-1], W_sum)
ax2.plot(years[:-1], res.intercept + res.slope*years[:-1], color = 'orangered', ls = '--', label='fitted line', lw=3)
ax2.text(1981, -5.9, 
          'Slope = {slope} $\pm$ {std}'.format(slope=round(res.slope,3), 
                                              std=round(res.stderr*1.96,3)), 
          color='orangered', fontsize=30)

ax2.set_title('Growth  Season', fontsize=35)

N_sum=[]
E_sum=[]
W_sum=[]
for year in years:
    N = -north_vels[year-year1]
    E = east_vels[year-year1]
    W = west_vels[year-year1]
    N_sum.append(np.nanmean(N[120:258])*100)
    E_sum.append(np.nanmean(E[120:258])*100)
    W_sum.append(np.nanmean(W[120:258])*100)
    
N_sum=np.array(N_sum)
E_sum=np.array(E_sum)
W_sum=np.array(W_sum)

ax3.bar(years-0.2, N_sum, width=0.25, label='$\Delta$V$_{advect, N}$', color='darkorchid',
        yerr=np.abs(0.3*N_sum), capsize=4)
ax3.bar(years, E_sum, width=0.25, label='$\Delta$V$_{advect, W}$', color='green',
        yerr=np.abs(0.3*E_sum), capsize=4)
ax3.bar(years+0.2, W_sum, width=0.25, label='$\Delta$V$_{advect, W}$', color='orangered',
        yerr=np.abs(0.3*W_sum), capsize=4)
ax3.set_ylabel(r'Mean Velocity (cm s$^{-1}$)', fontsize=30)
ax3.set_ylim(-8,5)
ax3.tick_params(labelsize=30, direction='in', length=16, width=4, color='black',
                labelcolor='black', labelbottom=True, pad=15, 
                bottom=True, top=True, left=True, right=True)
ax3.grid(True, zorder=0)

res = linregress(years, N_sum)
ax3.plot(years, res.intercept + res.slope*years, color = 'indigo', ls = '--', label='fitted line', lw=3)
ax3.text(1981, 3, 
          'Slope = {slope} $\pm$ {std}'.format(slope=round(res.slope,3), 
                                              std=round(res.stderr*1.96,3)), 
          color='indigo', fontsize=30)

res = linregress(years, E_sum)
ax3.plot(years, res.intercept + res.slope*years, color = 'green', ls = '--', label='fitted line', lw=3)
ax3.text(1981, -4.9, 
          'Slope = {slope} $\pm$ {std}'.format(slope=round(res.slope,3), 
                                              std=round(res.stderr*1.96,3)), 
          color='green', fontsize=30)

res = linregress(years, W_sum)
ax3.plot(years, res.intercept + res.slope*years, color = 'orangered', ls = '--', label='fitted line', lw=3)
ax3.text(1981, -5.9, 
          'Slope = {slope} $\pm$ {std}'.format(slope=round(res.slope,3), 
                                              std=round(res.stderr*1.96,3)), 
          color='orangered', fontsize=30)

ax3.set_title('Melt Season', fontsize=35)
#%%
# advect_07_19 = []
# for year in range(2007,2020):
#     thick_nc = Dataset(r"E:\PIOMAS\processed\advection\{year}.nc".format(year=year))
#     lat = thick_nc.variables['latitude'][:]
#     lon = thick_nc.variables['longitude'][:]
#     a = thick_nc.variables['advection'][:]
#     thick_nc.close()
    
#     a[0] = a[0]*86400*31
#     a[1] = a[1]*86400*28
#     a[2] = a[2]*86400*31
    
#     advect_07_19.append(np.nanmean(a[:3], axis=0))
    
# advect_clim_07_19 = np.nanmean(advect_07_19, axis=0)
# print(np.shape(advect_clim_07_19))
# #%%
# thick_nc = Dataset(r"E:\PIOMAS\processed\advection\{year}.nc".format(year=2021))
# lat = thick_nc.variables['latitude'][:]
# lon = thick_nc.variables['longitude'][:]
# a = thick_nc.variables['advection'][:]
# thick_nc.close()

# a[0] = a[0]*86400*31
# a[1] = a[1]*86400*28
# a[2] = a[2]*86400*31

# a = np.nanmean(a[:3], axis=0) - advect_clim_07_19

# fig, axs = plt.subplot_mosaic([['(a)']],
#                               per_subplot_kw={'(a)':{'projection': nps}},
#                               layout='constrained', figsize=(30, 30))

# for label, ax in axs.items():
#     # label physical distance to the left and up:
#     trans = mtransforms.ScaledTranslation(-20/72, 7/72, fig.dpi_scale_trans)
#     ax.text(-0.05, 1.05, label, transform=ax.transAxes + trans,
#             fontsize=40, va='bottom', fontfamily='serif')

# ax = axs['(a)']

# extent = [-165, -115, 69, 79]
# ax.set_extent(extent, ccrs.PlateCarree())  # Use PlateCarree for defining the extent
# ax.add_feature(cartopy.feature.LAND, zorder=3, edgecolor='black')
# ax.pcolormesh(lon, lat, -a, cmap='Spectral', 
#               transform=ccrs.PlateCarree(), vmin=-0.3, vmax=0.3)
# ax.plot(left[1], left[0], transform=ccrs.PlateCarree(), c = 'k', linewidth=5, zorder=1)
# ax.plot(right[1], right[0], transform=ccrs.PlateCarree(), c = 'k', linewidth=5, zorder=1)
# ax.plot(top[1], top[0], transform=ccrs.PlateCarree(), c = 'k', linewidth=5, zorder=1)