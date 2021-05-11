import numpy as np
import pyvista as pv
from pyvista import examples

# REQUIRES PYTHON 3.7/8
points = np.load('points_3d.npz')['point_cloud']

point_cloud = pv.PolyData(points)
print(point_cloud)
point_cloud.plot(eye_dome_lighting=True)
