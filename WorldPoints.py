import numpy as np
import pandas as pd


class WorldPointSet:
    """
    Set for storing and retrieving 3D points for SFM.
    world_points property contains the points that will be visualized.
    """

    def __init__(self, world_points=np.array([])):
        self.world_points = world_points
        self.view_ids = None
        self.count = 0
        self.correspondences = pd.DataFrame(columns=['PointIndex', 'ViewId', 'FeatureIndex'])

    def add_world_points(self, points):
        assert(points.shape[1] == 3)  # ensures only 3 columns are present
        np.append(self.world_points, points)

    def remove_world_points(self, point_indices):
        pass

    def add_correspondences(self, viewId, pointIndices, featureIndices):
        pass

    def remove_correspondences(self, correspondences):
        pass

    def save_to_file(self, filename):
        """
        Saves to file the world coordinates as an (N x 3) array.
        :param filename: Name of file to store world coordinates.
        :return: None
        """
        pass

    def load_from_file(self, filename):
        pass
