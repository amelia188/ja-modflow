'''
refactor code from Modflow 2005 to Modflow 6
'''
# IMPORT LIBRARIES
import os
import shutil
import numpy as np
import matplotlib.pyplot as plt
import geopandas as gpd
import rasterio
import flopy
from shapely.geometry import Point
from flopy.export.vtk import Vtk 

# SETUP AND CONFIGURATION

# Files from local directory
shp_path = r"C:/Users/ameli/Documents/Cermes/JA-project/Upper_RioCobre_SWMU.shp"
dem_path = r"C:/Users/ameli/Documents/Cermes/JA-project/riocobre2.tif"

model_name = "RioCobre_Modflow6"
workspace = r"C:/Users/amelia/Documents/Cermes/JA-project/Modflow6_model"
if not os.path.exists(workspace):
    os.makedirs(workspace)

# Define the Modflow 6 executable name
exe_name = "mf2005" 
    
# Create the FloPy model object
mf = flopy.modflow.Modflow(model_name, exe_name=exe_name, model_ws=workspace)


# Define discretization 
#
# Grid parameters

nlay = 3
nrow 
ncol
delr
delc
ztop
zbotm

# Create the discretization package
dis = flopy.modflow.ModflowDis(mf, nlay=nlay, nrow=nrow, ncol=ncol, delr=delr, delc=delc, top=ztop, botm=zbotm) 


# Define Hydraulic Properties (FloPy LPF package)

# Define Boundary conditions

# Define Initial Conditions (FloPy BAS package)




# Run the model
success, buff = mf.run_model(silent=True)