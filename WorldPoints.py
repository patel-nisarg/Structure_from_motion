import numpy as np
import pandas as pd
import logging


class WorldPointSet:
    """
    Set for storing and retrieving 3D points for SFM.
    world_points property contains the points that will be visualized.
    """

    def __init__(self, world_points=np.empty((0, 3), dtype=np.float64)):
        self.world_points = world_points
        self.view_ids = None
        self.count = 0
        self.correspondences = pd.DataFrame(columns=['PointIndex', 'ViewId', 'FeatureIndex'])

    def add_world_points(self, points):
        self.world_points = np.append(self.world_points, points, axis=0)

    def remove_world_points(self, point_indices):
        pass

    def add_correspondences(self, world_coords, view1, view2):
        ViewIds = [view1.id, view2.id]
        feature_indices = view1.tracked_pts[view2.id]

        if self.correspondences.empty:
            logging.info("Found empty world points set. Appending 3D coordinates from baseline...")
            for i, world_coord in enumerate(world_coords):
                self.correspondences = self.correspondences.append({'PointIndex': i,
                                                                    'ViewId': ViewIds,
                                                                    'FeatureIndex': [feature_indices[0][i],
                                                                                     feature_indices[1][i]]},
                                                                   ignore_index=True)
                self.add_world_points(points=world_coord.reshape((1, 3)))
            logging.info(f"Appended {len(world_coords)} 3D points to world coordinates set.")
        else:
            logging.info("Appending 3D coordinates to existing world points set...")
            set_length = len(self.correspondences)
            for i, world_coord in enumerate(world_coords):
                if world_coord in self.world_points:
                    # print("WP already in WP Set.")
                    continue
                self.correspondences = self.correspondences.append({'PointIndex': set_length + i,
                                                                    'ViewId': ViewIds,
                                                                    'FeatureIndex': [feature_indices[0][i],
                                                                                     feature_indices[1][i]]},
                                                                   ignore_index=True)
                self.add_world_points(points=world_coord.reshape((1, 3)))
            logging.info(f"Appended {len(world_coords)} 3D points to world coordinates set.")

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
