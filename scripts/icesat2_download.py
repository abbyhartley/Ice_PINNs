#!/usr/bin/env python
# coding: utf-8

# # Downloading IceSAT-2 data into a geoparquet

# In[2]:


import earthaccess
import geopandas as gpd
import h5py
import numpy as np
import pandas as pd
import rioxarray 
import shapely.geometry
import tqdm


# In[5]:


auth = earthaccess.login() # you'll need an earthaccess account to access the data
box_path = './amery_ice_shelf_4326.gpkg' # amery region cropped and saved from OpenAltimetry
bbox = gpd.read_file(box_path)
minx = bbox.geometry.bounds['minx'][0]
miny = bbox.geometry.bounds['miny'][0]
maxx = bbox.geometry.bounds['maxx'][0]
maxy = bbox.geometry.bounds['maxy'][0]

bbox.geometry.bounds


# In[6]:


# Set up spatiotemporal query for ATL06 ice product
granules = earthaccess.search_data(short_name="ATL06",
    cloud_hosted=True,
    bounding_box=(minx, miny, maxx, maxy),  # xmin, ymin, xmax, ymax
    temporal=("2019-07-01", "2019-12-31"),) # 6mo of data (only need 3mo for completeness though)

granules[-1]  # visualize last data granule


# In[7]:


get_ipython().run_cell_magic('time', '', 'file_obj = earthaccess.open(granules=[granules[0]])[0]\natl_file = h5py.File(name=file_obj, mode="r")\natl_file.keys()\n')


# In[8]:


# orientation - 0: backward, 1: forward, 2: transition
orient = atl_file["orbit_info"]["sc_orient"][:]
if orient == 0:
    strong_beams = ["gt1l", "gt2l", "gt3l"]
elif orient == 1:
    strong_beams = ["gt3r", "gt2r", "gt1r"]
strong_beams


# In[18]:


atl_file['gt1r']['land_ice_segments']['atl06_quality_summary']


# In[10]:


# function to get gdfs with height, quality, beam and date info
meta_data = dict(granules[0].items())
date = meta_data['meta']['native-id'].split('_')[1]
pd_date = gpd.pd.to_datetime(date)

gdf_list = []

for beam in strong_beams:
    try:
        h_li = atl_file[f"{beam}/land_ice_segments/h_li"][:]
        lon = atl_file[f"{beam}/land_ice_segments/longitude"][:]
        lat = atl_file[f"{beam}/land_ice_segments/latitude"][:]
        quality = atl_file[f"{beam}/land_ice_segments/atl06_quality_summary"][:]
        
        if len(h_li) == 0:
            print(f"⚠️ Beam {beam} has no data — skipping.")
            continue

        gdf = gpd.GeoDataFrame(
            data={"h_li": h_li,
                "h_li_sigma": atl_file[f"{beam}/land_ice_segments/h_li_sigma"][:],
                "atl06_quality_summary": quality,
                "beam": f'{beam}',
                "date": pd_date},
            geometry=gpd.points_from_xy(x=lon, y=lat),
            crs="OGC:CRS84",
        )
        gdf_list.append(gdf)
    
    except KeyError as e:
        print(f"Missing dataset for beam {beam}: {e}")
    except Exception as e:
        print(f"Unexpected error with beam {beam}: {e}")


# In[11]:


gdf_concat = gpd.pd.concat(gdf_list)


# In[12]:


gdf_concat[gdf_concat['atl06_quality_summary']==0]


# In[13]:


gdf_concat.to_parquet(path="ATL06_point_cloud_date_6mo.gpq", compression="zstd", schema_version="1.1.0")


# In[14]:


# atl_file['METADATA'].keys()
meta_data = dict(granules[0].items())
date = meta_data['meta']['native-id'].split('_')[1]
gpd.pd.to_datetime(date)
meta_data['meta']['native-id'][:-3]


# In[15]:


def granule2gdf(granule, bounds):
    """
    Converts an ATL06 granule into a GeoDataFrame and filters by bounding box.

    Parameters:
        granule: an earthaccess granule object
        bounds: tuple (minx, miny, maxx, maxy)

    Returns:
        gdf_concat: filtered GeoDataFrame
        file_name: filename stem for saving
    """
    # read granule with h5py
    file_obj = earthaccess.open(granules=[granule])[0]
    atl_file = h5py.File(name=file_obj, mode="r")

    # determine strong beams from spacecraft orientation
    orient = atl_file["orbit_info"]["sc_orient"][:]
    if orient == 0:
        strong_beams = ["gt1l", "gt2l", "gt3l"]
    elif orient == 1:
        strong_beams = ["gt3r", "gt2r", "gt1r"]

    # get a date from the granule item
    meta_data = dict(granule.items())
    file_name = meta_data['meta']['native-id'][:-3]
    date = meta_data['meta']['native-id'].split('_')[1]
    pd_date = gpd.pd.to_datetime(date)

    gdf_list = []
    for beam in strong_beams:
        gdf = gpd.GeoDataFrame(
            data={
                "h_li": atl_file[f"{beam}/land_ice_segments/h_li"][:],
                "h_li_sigma": atl_file[f"{beam}/land_ice_segments/h_li_sigma"][:],
                "atl06_quality_summary": atl_file[f"{beam}/land_ice_segments/atl06_quality_summary"][:],
                "beam": f'{beam}',
                "date": pd_date},
            geometry=gpd.points_from_xy(
                x=atl_file[f"{beam}/land_ice_segments/longitude"][:],
                y=atl_file[f"{beam}/land_ice_segments/latitude"][:],
            ),
            crs="OGC:CRS84",
        )
        gdf_list.append(gdf)

    # concat all strong beams
    gdf_concat = gpd.pd.concat(gdf_list)
    
    # keep only high-quality segments
    gdf_concat = gdf_concat[gdf_concat['atl06_quality_summary'] == 0]

    # apply bounding box filter
    minx, miny, maxx, maxy = bounds
    gdf_concat = gdf_concat.cx[minx:maxx, miny:maxy]

    return gdf_concat, file_name


# In[17]:


# iterate through granules and save as gpq
bounds = (minx, miny, maxx, maxy)

for granule in granules: 
    gdf, file_name = granule2gdf(granule, bounds)
    if len(gdf) > 0:
        gdf.to_parquet(path=f"{file_name}.gpq", compression="zstd", schema_version="1.1.0")
    else:
        print(f"Skipping {file_name} — empty or invalid GeoDataFrame.")


# In[19]:


from glob import glob
gdfs = [gpd.read_parquet(f) for f in glob("*.gpq")]
combined = gpd.pd.concat(gdfs, ignore_index=True)


# In[20]:


combined.to_parquet("ATL06_combined_6mo.gpq", compression="zstd")


# In[21]:


print("Unique dates:", combined['date'].nunique())
print("Unique beams:", combined['beam'].unique())
print(combined['date'].value_counts())


# In[22]:


print(combined.dtypes)
print(combined.columns)
print(combined.crs)

