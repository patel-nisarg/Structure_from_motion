import numpy as np
import pandas as pd
import logging


class WorldPointSet:
    """
    Dataset for storing and retrieving 3D points for SFM.
    world_points property contains the points that will be visualized.
    """

    def __init__(self, world_points=np.empty((0, 3), dtype=np.float64), add_redundant_views=False):
        self.world_points = world_points
        self.correspondences = pd.DataFrame(columns=['ViewId', 'FeatureIndex'])
        self.add_redundant_views = add_redundant_views

    def add_world_points(self, points):
        """
        Add world points to self.world_points numpy array (N x 3).

        :params points: (N x 3) 3D points to add to world_points
        """
        self.world_points = np.append(self.world_points, points, axis=0)

    def add_correspondences(self, world_coords, view1, view2):
        """
        Add view keypoints and 3D points (world coordinates) to world point dataset object.
        Also checks if 3D points already exist in set and ignores those points.

        :params world_coords: 3D points to add
        :params view1: first view from which to add keypoints
        :params view2: second view from which to add keypoints
        """
        ViewIds = [view1.id, view2.id]
        feature_indices = view1.tracked_pts[view2.id]

        if self.correspondences.empty:
            logging.info("Found empty world points set. Appending 3D coordinates from baseline...")
            for i, world_coord in enumerate(world_coords):
                self.correspondences = self.correspondences.append({'ViewId': ViewIds,
                                                                    'FeatureIndex': [feature_indices[0][i],
                                                                                     feature_indices[1][i]]},
                                                                   ignore_index=True)
                self.add_world_points(points=world_coord.reshape((1, 3)))
            logging.info(f"Appended {len(world_coords)} 3D points to world coordinates set.")

        else:
            logging.info("Appending 3D coordinates to existing world points set...")
            for i, world_coord in enumerate(world_coords):
                pt_in_set = np.any(np.isclose(world_coord, self.world_points))
                if pt_in_set and self.add_redundant_views:
                    index = np.argwhere(np.isclose(world_coord, self.world_points))[0][0]
                    # append new views and 2D points (feature_indices) to
                    self.correspondences.at[index, 'FeatureIndex'] += feature_indices[0][i]
                    self.correspondences.at[index, 'FeatureIndex'] += feature_indices[1][i]
                    self.correspondences.at[index, 'ViewId'] += ViewIds[0]
                    self.correspondences.at[index, 'ViewId'] += ViewIds[1]
                    self.add_world_points(points=world_coord.reshape((1, 3)))
                elif pt_in_set:
                    continue
                else:
                    self.correspondences = self.correspondences.append({'ViewId': ViewIds,
                                                                        'FeatureIndex': [feature_indices[0][i],
                                                                                         feature_indices[1][i]]},
                                                                       ignore_index=True)
                    self.add_world_points(points=world_coord.reshape((1, 3)))

            logging.info(f"Appended {len(world_coords)} 3D points to world coordinates set.")

    def save_to_file(self, filename):
        """
        Saves to file the world coordinates as an (N x 3) array.
        :param filename: Name of file to store world coordinates.
        :return: None
        """
        np.savez(filename, point_cloud=self.world_points)

    def load_from_file(self, filename):
        """
        Loads world points (3D points) from file.

        :params filename: Name of file to load world coordinates.
        """
        self.world_points = np.load(filename)
