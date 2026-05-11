# Ice_PINNs
Ice Viscosity Inversion Using PINNs:

Bedmap data download: This site lets you visualize Bedmap 1, 2, and 3 tracks on Antarctica to decide which tracks/data to download for your ice shelf. You can then download the data with these links: Bedmap1, Bedmap 2, Bedmap 3. For a viscosity inversion problem, you’ll want the thickness profiles (not raw radar data), so it’s best to download the relevant shapePoints gpkg files. For example, we needed data from INGV 2003 TALOS-DOME, PRIC 2016 CHA2 & 2017 CHA3, and RNRF 1971 LAMBERT-AMERY for the Amery ice shelf. Optional: Convert these gpkgs to gpqs using gpkg_to_gpq.ipynb to speed up data loading later.

IceSAT-2 data download: See icesat2_download.ipynb to extract relevant IceSAT-2 data using Earthaccess and save it as one combined gpq.

Bedmap/IceSAT-2 comparison & calibration: IS2_BM_comp.ipynb loads in IceSAT-2 and Bedmap data, converts IceSAT-2 heights to thicknesses using Eqn (1) from Chartrand+23, then calibrates IceSAT-2 thicknesses to the Bedmap dataset by optimizing the firn-air column thickness and density parameters and saves the new IceSAT-2 data as a gpq.

Extra: gpkg_to_gpq.py converts gpkg files to gpqs :-)
