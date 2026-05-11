#!/usr/bin/env python
# coding: utf-8

# # Convert Bedmap gpkg files to gpq's (speeds up data loading downstream)

# In[1]:


import os
import geopandas as gpd

# path to directory containing the .gpkg files
folder = os.path.expanduser("/Users/abbyhartley/Desktop/glaciology/amery/bedmap_data")
gpkg_files = [os.path.join(folder, f) for f in os.listdir(folder) if f.endswith('.gpkg')]

# loop through each file and convert it to .gpq (GeoParquet)
for gpkg_path in gpkg_files:
    try:
        # load the GeoPackage
        gdf = gpd.read_file(gpkg_path)
        
        # construct new .gpq file path
        gpq_path = gpkg_path.replace('.gpkg', '.gpq')

        # save as GeoParquet
        gdf.to_parquet(gpq_path, index=False)
        print(f"Converted: {gpkg_path} → {gpq_path}")
    
    except Exception as e:
        print(f"Failed to convert {gpkg_path}: {e}")
