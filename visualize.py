import numpy as np
import pyvista as pv
from pyvista import examples

# REQUIRES PYTHON 3.7
# 'F:/Documents/SFM_Project/points_3d_baseline.npz'
# points = np.load('points3d_8_2.npz')['point_cloud']


class PointCloudVisualizer:
    def __init__(self, points):
        self.points = points

    def visualize(self):
        points1 = np.empty((0, 3))
        pt_bound = 5

        for point in self.points:
            point_abs = np.linalg.norm(point)
            # if abs(point[0]) < pt_bound and abs(point[1]) < pt_bound and abs(point[2]) < pt_bound:
            if point_abs < pt_bound:
                points1 = np.append(points1, [point], axis=0)

        print(f"{len(self.points)} loaded. {len(points1)} acquired after filtering with bound of {pt_bound}.")
        point_cloud = pv.PolyData(points1)
        print(point_cloud)
        point_cloud.plot(eye_dome_lighting=True)
