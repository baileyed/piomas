import numpy as np
import pandas as pd
import struct
import xarray as xr
# import matplotlib.pyplot as plt
# from cartoplot import cartoplot

grids = {}

dims = (120,360)
# dims = (360,120)

leap=1980


### For scalar quantites ###
# for i in ['lon','lat']:
    
#     grid = np.array(pd.read_csv(f'E:/PIOMAS/{i}grid.dat',
#                                 header=None,
#                                 delim_whitespace=True))

#     flat_grid = grid.ravel()
    
# #     if i == 'lon':

#     shaped_grid = flat_grid.reshape(dims)
        
# #     else:
        
# #         shaped_grid = flat_grid.reshape((360,120))
    
#     grids[i] = shaped_grid
    
### For vector quantites ###
for i in ['lon','lat']:    
    grid = np.array(pd.read_csv("E:\PIOMAS\grid_vector.dat",
                                header=None,
                                delim_whitespace=True))

    flat_grid = grid.ravel()
    
    if i == 'lat':
        shaped_grid = flat_grid[:43200].reshape(dims)
    else:
        shaped_grid = flat_grid[43200:86400].reshape(dims)
        
#         shaped_grid = flat_grid.reshape((360,120))
    
    grids[i] = shaped_grid


def process_piomas(year, leap_year=False):
    
    binary_dir = f'E:/PIOMAS/raw_binary/daily_velocity/uiday.H{year}'

    ############################################################
    
    ### For daily datasets ###
    if leap==True:
        days = 366
    else:
        days = 365
    
    ### For monthly datasets ###
    # days=12
    
    # Read File
        
    with open(binary_dir, mode='rb') as file:
    
        fileContent = file.read()
        
        
        if len(fileContent) % 4 != 0:
            print("Error: Invalid file size.")
            exit()
        
        data = struct.unpack("f" * (len(fileContent)// 4), fileContent)
        
        print(len(data)/days)
        
        if len(data) % days != 0:
            print("Error: Invalid data length.")
            exit()
    ############################################################
    
    # Put it in a 3D array
        
        ### For scalar quantites ###
        # native_data = np.full((days,dims[0],dims[1]),np.nan)
        # print(native_data)
        
        ### For vector quantities ###
        native_data = np.full((2, days, dims[0], dims[1]), np.nan)

        for month in range(1, days+1):
            start = (month - 1) * (dims[0] * dims[1])*2 #+ 518400
            end = month * (dims[0] * dims[1])*2 #+ 518400
            
            # Ensure indices are within bounds
            if end <= len(data):
                thickness_list = np.array(data[start:end])
                # print(thickness_list)
        
                # # Reshape the flattened data to a 2D array
                # gridded = thickness_list.reshape(dims[0], dims[1])
        
                # # Store the gridded data in the 3D array
                # native_data[month - 1, :, :] = gridded
                
                # Reshape the flattened data to a 2D array
                gridded = thickness_list.reshape(2, dims[0], dims[1])
        
                # Store the gridded data in the 3D array
                native_data[0, month - 1, :, :] = gridded[0]
                native_data[1, month - 1, :, :] = gridded[1]
                
            else:
                print(f"Error: Data for month {month} is out of bounds.")
                
        
    ############################################################
    
    # # Put it in a 3D array
        
    #     ### For monthly datasets
    #     native_data = np.full((12,dims[0],dims[1]),np.nan)
        
    #     for month in range(1,13):
            
    #         start = (month-1)*(dims[0]*dims[1]) + 43200
    #         end = month*(dims[0]*dims[1]) + 43200
    #         thickness_list = np.array(data[start:end])
            
    #         gridded = thickness_list.reshape(dims[0],dims[1])
    #         native_data[month-1,:,:] = gridded
            
    #         break
            
          
    ############################################################
        
    # Output to NetCDF4
        # print(native_data)
        ds = xr.Dataset( data_vars={'v':(['t','x','y'],native_data[1])},

                         coords =  {'longitude':(['x','y'],grids['lon']),
                                    'latitude':(['x','y'],grids['lat']),
                                    'day':(['t'],np.array(range(1,days+1)))})
        
        ds.attrs['data_name'] = 'Monthly mean Piomas sea ice velocity data'
        
        ds.attrs['description'] = """Sea ice velocity in m/s on the lat/lon grid, 
                                    data produced by University of Washington Polar Science Center"""
        
        ds.attrs['year'] = f"""These data are for the year {year}"""
        
        ds.attrs['citation'] = """When using this data please use the citation: 
                                Zhang, Jinlun and D.A. Rothrock: Modeling global sea 
                                ice with a thickness and enthalpy distribution model 
                                in generalized curvilinear coordinates,
                                Mon. Wea. Rev. 131(5), 681-697, 2003."""
        
        ds.attrs['code to read'] = """  # Example code to read a month of this data 
    
                                        def read_month_of_piomas(year,month):
    
                                            data_dir = 'output/' 

                                            with xr.open_dataset(f'{data_dir}{year}.nc') as data: 

                                                ds_month = data.where(int(month) == data.month, drop =True) 

                                                return(ds_month)"""
        
        ds.attrs['python author'] = """Robbie Mallett wrote this python code. If there's a problem with it, 
                                        email him at robbie.mallett.17@ucl.ac.uk"""
                                
        
        

        # output_dir = f'output/'
        # output_dir = 'E:/PIOMAS/processed/'
        output_dir = 'E:/PIOMAS/processed/daily_velocity/v/'

        ds.to_netcdf(f'{output_dir}{year}.nc','w')

    return native_data

for year in range(1979,2024):
    if year==leap:
        x = process_piomas(year, leap_year=True)
        leap+=4
    else:
        x = process_piomas(year, leap_year=False)

