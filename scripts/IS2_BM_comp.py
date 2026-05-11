#!/usr/bin/env python
# coding: utf-8

# In[13]:


import geopandas as gpd
import matplotlib.pyplot as plt
import numpy as np
import os
import pandas as pd
import h5py
from shapely.geometry import Point
from sklearn.metrics import mean_squared_error
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score
from matplotlib.colors import LogNorm


# In[14]:


# load ICESat-2 data
icesat2 = gpd.read_parquet("ATL06_combined_6mo.gpq")
icesat2.head()
icesat2.crs  # should still be "OGC:CRS84"


# In[15]:


# clean ICESat-2 heights
icesat2 = icesat2[(icesat2['h_li'].notna()) &
    (icesat2['h_li'] > 0) &
    (icesat2['h_li'] < 1000) &  
    (icesat2["atl06_quality_summary"] == 0)]

icesat2['h_li'][:5] # land-ice segment height in m


# In[16]:


# mask out grounded ice

# Load latitude and longitude from Point_F
# maybe also use point_H for a refined mask??
with h5py.File("IS2_grounding.h5", "r") as f:
    lat_f = f["Point_F/latitude"][:]
    lon_f = f["Point_F/longitude"][:]

# Create GeoDataFrame
geometry_f = [Point(lon, lat) for lon, lat in zip(lon_f, lat_f)]
grounding_f = gpd.GeoDataFrame(geometry=geometry_f, crs="EPSG:4326")  # WGS 84

# Clip grounding line to Amery Ice Shelf
amery_poly = gpd.read_file("amery_ice_shelf_4326.gpkg").to_crs("EPSG:4326")
grounding_f_amery = gpd.clip(grounding_f, amery_poly)


# In[17]:


# Reproject both to EPSG:3031 for buffering
grounding_f_amery_proj = grounding_f_amery.to_crs("EPSG:3031")
icesat2_proj = icesat2.to_crs("EPSG:3031")

# Buffer the grounding line by 10 km
buffered_grounding = grounding_f_amery_proj.buffer(10_000)

# Combine buffer polygons into one
buffer_union = buffered_grounding.unary_union


# In[18]:


# Drop points within 10 km of the grounding line
icesat2_floating = icesat2_proj[~icesat2_proj.geometry.intersects(buffer_union)]
icesat2_floating_amery = gpd.clip(icesat2_floating, amery_poly.to_crs("EPSG:3031"))

icesat2_floating_amery = icesat2_floating_amery[
    (icesat2_floating_amery['h_li'] > 0) &
    (icesat2_floating_amery['h_li'] < 120)  # 120m seems reasonable
].copy()


# In[19]:


amery_proj = amery_poly.to_crs("EPSG:3031") # ensure all geometries are in the same CRS


# In[20]:


# visualize grounding zone/buffer on amery

fig, ax = plt.subplots(figsize=(8, 8))
# plot amery outline
amery_proj.boundary.plot(ax=ax, color='black', linewidth=1, label="Amery boundary")
# plot grounding line buffer
gpd.GeoSeries(buffer_union).plot(ax=ax, color='lightgray', alpha=0.5, label="10 km buffer (grounding zone)")
# plot filtered ICESat-2 points (floating only)
icesat2_floating_amery.plot(ax=ax, markersize=1, color='blue', label="ICESat-2 (floating)")

# add grounding line points for context
grounding_f_amery_proj = grounding_f_amery.to_crs("EPSG:3031")
grounding_f_amery_proj.plot(ax=ax, markersize=5, color='red', label="Grounding line (Point F)")

ax.set_title("ICESat-2 Floating Ice near Amery Ice Shelf", fontsize=14)
ax.set_xlabel("Easting (m)")
ax.set_ylabel("Northing (m)")
ax.legend(loc='lower left')
ax.set_aspect("equal")
plt.grid(True)
plt.tight_layout()
plt.show()


# In[11]:


icesat2_floating_amery['h_li'].describe()


# In[21]:


# convert ICESat-2 height data to thickness

rho_s = 1027  # seawater density (kg/m^3)
rho_i = 918   # ice density (kg/m^3)
rho_a = 1     # firn-air density (kg/m^3)
Ha = 18.5       # firn-air column thickness (m) 

# hydrostatic ice thickness estimate from Chartrand+23
icesat2_floating_amery["H_E"] = (icesat2_floating_amery["h_li"] * (rho_s / (rho_s - rho_i)) -
    Ha * ((rho_i - rho_a) / (rho_s - rho_i)))

icesat2_floating_amery['H_E'][:5] 


# In[13]:


print(icesat2_floating_amery[['h_li', 'H_E']].describe())


# In[22]:


# Bedmap shapePoints
folder = os.path.expanduser("/Users/abbyhartley/Desktop/glaciology/amery/bedmap_data")  # directory with .gpkg files
gpkg_files = [os.path.join(folder, f) for f in os.listdir(folder) if f.endswith('.gpkg')]
# load and combine all GPKG files
bedmap_points = gpd.GeoDataFrame(pd.concat([gpd.read_file(f) for f in gpkg_files], ignore_index=True))

# ensure same CRS before spatial comparison
bedmap_points = bedmap_points.to_crs(icesat2.crs)
print(f"Loaded {len(bedmap_points)} Bedmap3 points")
# now we can start comparing bedmap3 and ICESat-2 thickness data (for Amery specifically)


# In[70]:


print(bedmap_points.columns)


# In[71]:


bedmap_points['Mean_thick'].describe()


# In[23]:


# clean bedmap3 thickness data
bedmap_points = bedmap_points[(bedmap_points["Mean_thick"].notna()) &
    (bedmap_points["Mean_thick"] > 0) &
    (bedmap_points["Mean_thick"] < 2000)] 

bedmap_points['Mean_thick'][:5]


# In[25]:


# make hist of bedmap thickness vs ours after masking out grounding line data

# extract thickness arrays
icesat_thickness = icesat2_floating_amery["H_E"]
bedmap_thickness = bedmap_points["Mean_thick"]

# plot histogram
plt.figure(figsize=(8, 4))
plt.hist(icesat_thickness, bins=100, alpha=0.6, label="ICESat-2 H_E (Hydrostatic)", color="indianred")
plt.hist(bedmap_thickness, bins=100, alpha=0.6, label="Bedmap Mean_thick (Radar)", color="steelblue")

plt.xlabel("Ice Thickness (m)", fontsize=12)
plt.ylabel("Number of Points", fontsize=12)
plt.title("ICESat-2 vs. Bedmap Ice Thickness Distributions", fontsize=14)
plt.xlim(0, 1000)
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.show()


# In[75]:


print(np.mean(icesat_thickness), np.mean(bedmap_thickness))


# ## optimize rho_a and Ha to calibrate icesat2 to bedmap

# In[ ]:


# from geopandas.tools import sjoin_nearest

# # Reproject both datasets to meters
# icesat_proj = icesat2_floating_amery.to_crs("EPSG:3031")
# bedmap_proj = bedmap_points.to_crs("EPSG:3031")

# # Drop -9999 (invalid) Bedmap points just to be safe
# bedmap_valid = bedmap_proj[bedmap_proj['Mean_thick'] > 0].copy()

# # Nearest neighbor match (max 1 km)
# comparison = sjoin_nearest(
#     icesat_proj[['geometry', 'h_li']],
#     bedmap_valid[['geometry', 'Mean_thick']],
#     how='inner',
#     distance_col='distance_m',
#     max_distance=1000
# ).dropna(subset=['Mean_thick'])

# # Now we have side-by-side ICESat-2 heights and Bedmap thicknesses
# h_li_matched = comparison['h_li'].to_numpy()
# bedmap_matched = comparison['Mean_thick'].to_numpy()

# print(f"Matched {len(h_li_matched)} pairs")


# In[ ]:


# from scipy.optimize import differential_evolution

# # optimization objective (based on matched data)
# def matched_objective(params):
#     rho_a, Ha = params
#     H_E = (h_li_matched * (rho_s / (rho_s - rho_i)) -
#            Ha * ((rho_i - rho_a) / (rho_s - rho_i)))
#     return np.mean((H_E - bedmap_matched)**2)  # MSE between matched values

# bounds = [(0, 10), (0, 30)]

# # 0ptimize
# result = differential_evolution(matched_objective, bounds)
# opt_rho_a, opt_Ha = result.x

# print(f"Optimized rho_a: {opt_rho_a:.3f} kg/m³")
# print(f"Optimized Ha: {opt_Ha:.3f} m")


# In[26]:


# reproject ICESat-2 and Bedmap3 points to match Amery polygon CRS
icesat2_floating_amery = icesat2_floating_amery.to_crs(amery_poly.crs)
bedmap_points = bedmap_points.to_crs(amery_poly.crs)

# now we can safely clip!
icesat2_amery = gpd.clip(icesat2_floating_amery, amery_poly)
bedmap_amery = gpd.clip(bedmap_points, amery_poly)

print(bedmap_amery.columns)
print(icesat2_amery.columns)


# In[27]:


# test out a nearest neighbor match
from geopandas.tools import sjoin_nearest

# project to Antarctic Polar Stereographic (m) for a distance comparison
icesat2_amery_proj = icesat2_amery.to_crs("EPSG:3031")
bedmap_amery_proj = bedmap_amery.to_crs("EPSG:3031")

# nearest neighbor match
# remove invalid Bedmap3 points (no data)
bedmap_amery_valid = bedmap_amery_proj[bedmap_amery_proj['Mean_thick'] != -9999]
# rerun the spatial join
comparison = sjoin_nearest(icesat2_amery_proj[['geometry', 'H_E']],
    bedmap_amery_valid[['geometry', 'Mean_thick']],
    how='inner',
    distance_col='distance_m',
    max_distance=1000)


# In[34]:


# visualize how icesat2 and bedmap3 data overlap
fig, ax = plt.subplots(figsize=(10, 10))
bedmap_amery_proj.plot(ax=ax, color='blue', markersize=0.5, alpha = 0.2, label='Bedmap')
icesat2_amery_proj.plot(ax=ax, color='red', markersize=0.5, alpha = 0.2, label='ICESat-2')
plt.legend()
plt.title("ICESat-2 vs Bedmap Locations (EPSG:3031)")
plt.savefig('BM_IS2_overlap_6mo.png')
plt.show()


# In[20]:


print(len(icesat2_amery_proj))
print(len(bedmap_amery_proj))

print(icesat2_amery_proj.crs)
print(bedmap_amery_proj.crs)


# In[84]:


print(comparison.shape)
print(comparison[['H_E', 'Mean_thick', 'distance_m']].dropna().head())


# In[28]:


# filter out invalid or zero vals to avoid division errors
valid = comparison[(comparison['Mean_thick'] > 0) & (comparison['H_E'] > 0)]
valid['thick_ratio'] = valid['H_E'] / valid['Mean_thick']


# In[33]:


from sklearn.linear_model import LinearRegression

# extract valid matched data
bedmap_thick = valid['Mean_thick'].values.reshape(-1, 1)  # x-axis
icesat_thick = valid['H_E'].values.reshape(-1, 1)         # y-axis

# fit linear regression model 
reg = LinearRegression().fit(bedmap_thick, icesat_thick)
slope = reg.coef_[0][0]
intercept = reg.intercept_[0]
r2 = reg.score(bedmap_thick, icesat_thick)

plt.figure(figsize=(8, 8))
plt.scatter(bedmap_thick, icesat_thick, s=1, alpha=0.1, c='royalblue', label='Matched points')
plt.plot([0, bedmap_thick.max()], [0, bedmap_thick.max()], 'k--', label='1:1 line')
plt.plot(
    [0, bedmap_thick.max()],
    [intercept, slope * bedmap_thick.max() + intercept],
    'r-', label=f'Best fit: y={slope:.2f}x + {intercept:.2f}, $R^2$={r2:.2f}')
plt.xlabel('Bedmap Thickness (m)')
plt.ylabel('ICESat-2 Thickness (m)')
plt.title('Comparison of Ice Thickness (ICESat-2 vs. Bedmap)')
plt.xlim(0, 1010)
plt.ylim(0, 1050)
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.show()


# In[24]:


valid['residual'] = valid['H_E'] - valid['Mean_thick']
fig, ax = plt.subplots(figsize=(10, 8))
valid.plot(column='residual', cmap='coolwarm', legend=True, ax=ax, markersize=1)
ax.set_title('ICESat-2 - Bedmap Thickness Residual (m)')
plt.show()


# In[41]:


# try varying Ha locally

# rerun the spatial join, including h_li this time
comparison = sjoin_nearest(
    icesat2_amery_proj[['geometry', 'H_E', 'h_li']], 
    bedmap_amery_valid[['geometry', 'Mean_thick']],
    how='inner',
    distance_col='distance_m',
    max_distance=1000)

# only for valid thickness comparisons
valid = comparison[(comparison['Mean_thick'] > 0) & (comparison['H_E'] > 0)]

# calculate spatial Ha needed for each point to match Bedmap
valid['Ha_local'] = ((valid['h_li'] * (rho_s / (rho_s - rho_i)) - valid['Mean_thick']) *
                     ((rho_s - rho_i) / (rho_i - rho_a)))
corrected_icesat = (valid['H_E'] - 340) / 0.6
icesat_thick = corrected_icesat.values.reshape(-1, 1)


# In[26]:


fig, ax = plt.subplots(figsize=(10, 8))
valid.plot(column='Ha_local', cmap='viridis', ax=ax, legend=True, markersize=1)
ax.set_title('Inferred Spatial $H_a(x, y)$ to Match Bedmap Thickness')
plt.show()


# In[42]:


# recompute thickness 
valid['H_E_adjusted'] = (valid['h_li'] * (rho_s / (rho_s - rho_i)) -
                         valid['Ha_local'] * ((rho_i - rho_a) / (rho_s - rho_i)))


# In[43]:


# check for improvement 

x = valid['Mean_thick'].values.reshape(-1, 1)
y = valid['H_E_adjusted'].values

model = LinearRegression().fit(x, y)
slope = model.coef_[0]
intercept = model.intercept_
r2 = r2_score(y, model.predict(x))

print(f"Slope: {slope:.2f}, Intercept: {intercept:.2f}, R²: {r2:.3f}")


# In[52]:


reg = LinearRegression().fit(bedmap_thick, icesat_thick)
slope = reg.coef_[0][0]
intercept = reg.intercept_[0]
r2 = reg.score(bedmap_thick, icesat_thick)

plt.figure(figsize=(8, 8))
plt.scatter(bedmap_thick, icesat_thick, s=1, alpha=0.1, c='royalblue', label='Matched points')
plt.plot([0, bedmap_thick.max()], [0, bedmap_thick.max()], 'k--', label='1:1 line')
plt.plot(
    [0, bedmap_thick.max()],
    [intercept, slope * bedmap_thick.max() + intercept],
    'r-', label=f'Best fit: y={slope:.2f}x + {intercept:.2f}, $R^2$={r2:.2f}')
plt.xlabel('Bedmap Thickness (m)')
plt.ylabel('ICESat-2 Thickness (m)')
plt.title('Comparison of Ice Thickness (ICESat-2 vs. Bedmap)')
plt.ylim(0, 1050)
plt.xlim(0, 1010)
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.savefig('icesat_bedmap_thickness_scatter.png')
plt.show()


# In[57]:


# Calculate RMSD from the 1:1 line
rmsd = np.sqrt(mean_squared_error(bedmap_thick, icesat_thick))
print(f"RMSD from 1:1 line: {rmsd:.2f} m")


# In[56]:


# linear regression model
reg = LinearRegression().fit(bedmap_thick, icesat_thick)
slope = reg.coef_[0][0]
intercept = reg.intercept_[0]
r2 = reg.score(bedmap_thick, icesat_thick)

# plot with density
plt.hist2d(bedmap_thick.flatten(), icesat_thick.flatten(), bins=150, cmap='viridis', norm=LogNorm())
plt.colorbar(label='log10(N points)')

# 1:1 and best fit lines
xmax = bedmap_thick.max()
plt.plot([0, xmax], [0, xmax], 'k--', label='1:1 line')
plt.plot(
    [0, xmax],
    [intercept, slope * xmax + intercept],
    'r-', label=f'Best fit: y={slope:.2f}x + {intercept:.2f}, $R^2$={r2:.2f}')

plt.xlabel('Bedmap Thickness (m)')
plt.ylabel('ICESat-2 Thickness (m)')
plt.title('Comparison of Ice Thickness (ICESat-2 vs. Bedmap)')
plt.ylim(0, 1050)
plt.xlim(0, 1010)
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.savefig('icesat_bedmap_thickness_density.png')
plt.show()


# In[29]:


# save IceSAT-2 thickness data to new gpq 

# start with original ICESat-2 GeoDataFrame in EPSG:3031
icesat2_all = icesat2_floating_amery.copy()  # already projected to EPSG:3031 and cleaned

# valid df includes a subset of points with successful Bedmap matches
# filter to positive corrected thickness values
valid = valid[corrected_icesat > 0].copy()
valid["H_E_corrected"] = corrected_icesat[corrected_icesat > 0].values

# merge corrected thickness back into the full ICESat-2 GeoDataFrame by geometry
icesat2_all["H_E_corrected"] = pd.NA  # add empty column
icesat2_all.set_geometry("geometry", inplace=True)
valid.set_geometry("geometry", inplace=True)
icesat2_all = icesat2_all.merge(
    valid[["geometry", "H_E_corrected"]],
    on="geometry",
    how="left")

# set CRS to original (OGC:CRS84) and save
icesat2_all = icesat2_all.to_crs("OGC:CRS84")
icesat2_all.to_parquet("ICESat2_corrected_thickness.gpq")

print("Saved ICESat-2 data with corrected thickness to 'ICESat2_corrected_thickness.gpq'!")


# In[30]:


icesat2_clean = gpd.read_parquet("ICESat2_corrected_thickness.gpq")
icesat2_clean.head()
icesat2_clean.crs  # should still be "OGC:CRS84"


# In[32]:


# drop unnecessary or duplicate columns
icesat2_clean = icesat2_clean.drop(columns=["H_E_corrected_x", "H_E_corrected_y"], errors="ignore")
icesat2_clean = icesat2_clean.rename(columns={"H_E": "H_E_corrected"})
# re-save the cleaned gpq
icesat2_clean.to_parquet("ATL06_thick.gpq")


# In[34]:


icesat2_clean = icesat2_clean[icesat2_clean["H_E_corrected"] > 0].copy()
icesat2_clean.to_parquet("ATL06_thick.gpq")


# In[35]:


icesat2_clean["H_E_corrected"].describe()


# In[36]:


icesat2_clean.describe()

