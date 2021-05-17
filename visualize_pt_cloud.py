import numpy as np
import pyvista as pv
from pyvista import examples

# REQUIRES PYTHON 3.7
# 'F:/Documents/SFM_Project/points_3d_baseline.npz'
points = np.load('points_3d_rp40.npz')['point_cloud']

point_cloud = pv.PolyData(points)
print(point_cloud)
point_cloud.plot(eye_dome_lighting=True)
