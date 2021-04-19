import logging
import os
import hashlib

import cv2 as cv
import numpy as np


def create_view_ID(img_name):
    hash_object = hashlib.sha256(str(img_name).encode('utf-8'))
    return "V" + hash_object.hexdigest()[:12]


class ImageView:
    """
    An image view containing its pose and features. Each view also comes with a dictionary for
    2D:3D point correspondence.
    """

    def __init__(self, image_path, features_path=None):
        self.image_path = image_path
        self.name = os.path.basename(image_path)[:-4]
        self.id = create_view_ID(self.name)
        self.features_path = features_path
        self.rotation = np.zeros((3, 3))
        self.position = np.zeros((3, 1))
        self.keypoints = None
        self.descriptors = None
        self.world_points = {}  # (keypoint):3D point pair {():[]}
        self.tracked_pts = {}  #

        self.image = cv.imread(self.image_path)
        if self.image is None:
            raise FileNotFoundError("Could not load image from file!")
        else:
            logging.info(msg=f"{self.name} from filepath {self.image_path} opened successfully.")

    def extract_features(self, write_to_file=False):
        """
        Extracts features from View object. Optionally, saves features to file in .npz format.
        """
        sift = cv.SIFT_create()
        self.keypoints, self.descriptors = sift.detectAndCompute(self.image, None)

        if write_to_file:
            self.write_features()
        return None

    def read_features(self):
        """
        Load features from file. File format should be "image_001.npz"
        """
        data = np.load(self.features_path + self.name + ".npz")
        self.descriptors = data['descriptors']
        self.keypoints = data['keypoints']

    def write_features(self):
        """
        Write image features to file. Write format should be "image_001.npz"
        """
        filename = self.features_path + self.name
        logging.info(f'Writing features of image {self.name} to file...')
        np.savez(filename, keypoints=self.keypoints, descriptors=self.descriptors)
        logging.info('Writing features complete.')

    def get_3D_correspondences(self):
        return self.world_points.items()

    def get_pose(self):
        """
        Returns pose for the view in the form of a tuple (R, C).
        :return: (R, C). Where R is a (3x3) rotation matrix and t is a (3x1) position vector.
        """
        return [self.rotation, self.position]

    def __setitem__(self, keys, values):
        assert(len(keys) == len(values))
        for i, key in enumerate(keys):
            self.world_points[key] = values[i]
