# Ice_PINNs
Ice Viscosity Inversion Using PINNs:

Bedmap data download: [This site](https://bedmap.scar.org/) lets you visualize Bedmap 1, 2, and 3 tracks on Antarctica to decide which tracks/data to download for your ice shelf. You can then download the data with these links: [Bedmap 1](https://ramadda.data.bas.ac.uk/repository/entry/show?entryid=synth:925ac4ec-2a9d-461a-bfaa-6314eb0888c8:L0JFRE1BUDFfMTk2Ni0yMDAwX0FJUl9CTTE=), [Bedmap 2](https://ramadda.data.bas.ac.uk/repository/entry/show?entryid=0f90d926-99ce-43c9-b536-0c7791d1728b), and [Bedmap 3](https://ramadda.data.bas.ac.uk/repository/entry/show?entryid=a72a50c6-a829-4e12-9f9a-5a683a1acc4a). For a viscosity inversion problem, you’ll want the thickness profiles (not raw radar data), so it’s best to download the relevant shapePoints gpkg files. For example, we needed data from INGV 2003 TALOS-DOME, PRIC 2016 CHA2 & 2017 CHA3, and RNRF 1971 LAMBERT-AMERY for the Amery ice shelf. Optional: Convert these gpkgs to gpqs using gpkg_to_gpq.ipynb to speed up data loading later.

IceSAT-2 data download: See icesat2_download.py to extract relevant IceSAT-2 data using Earthaccess and save it as one combined gpq.

Bedmap/IceSAT-2 comparison & calibration: IS2_BM_comp.ipynb loads in IceSAT-2 and Bedmap data, converts IceSAT-2 heights to thicknesses using Eqn (1) from Chartrand+23, then calibrates IceSAT-2 thicknesses to the Bedmap dataset by optimizing the firn-air column thickness and density parameters and saves the new IceSAT-2 data as a gpq.

Extra: gpkg_to_gpq.py converts gpkg files to gpqs :-)
