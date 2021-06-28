import numpy as np
import pyvista as pv


class PointCloudVisualizer:
    """
    A visualizer object that contains 3D points to be visualized.
    """
    def __init__(self, points):
        self.points = points

    def visualize(self):
        """
        Visualizes 3D points by generating a point cloud using pyvista.
        """
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
