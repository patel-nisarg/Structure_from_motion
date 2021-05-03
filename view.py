import logging
import os
import hashlib

import cv2 as cv
import numpy as np


def create_view_ID(img):
    """
    Generates 12-digit ID of image to make sure each image has a unique identifier. When saving and reading
    image features, two images with the same filename will have unique identifiers.
    :param img: Image to create an ID for.
    :return:
    """
    img = cv.resize(img, dsize=(10, 10), interpolation=cv.INTER_NEAREST)  # rescale with nearest neighbour
    hash_object = hashlib.sha256(str(img).encode('utf-8'))
    return "view_" + hash_object.hexdigest()[:12]


class ImageView:
    """
    An image view containing its pose and features. Each view also comes with a dictionary for
    2D:3D point correspondence.
    """

    def __init__(self, image_path, features_path=""):
        self.image_path = image_path
        self.name = os.path.basename(image_path)[:-4]
        self.features_path = features_path
        self.rotation = np.zeros((3, 3))
        self.position = np.zeros((3, 1))
        self.keypoints = None
        self.descriptors = None
        self.world_points = {}  # (keypoint):3D point pair {():[]}
        self.tracked_pts = {}  #
        self.translation = np.zeros((3, 1))

        self.image = cv.imread(self.image_path)
        if self.image is None:
            raise FileNotFoundError("Could not load image from file!")
        else:
            logging.info(msg=f"{self.name} from filepath {self.image_path} opened successfully.")
        self.id = create_view_ID(self.image)

    def extract_features(self, write_to_file=False):
        """
        Extracts features from View object. Optionally, saves features to file in .npz format.
        """
        logging.info(f"Extracting features from {self.name}...")
        sift = cv.SIFT_create()
        self.keypoints, self.descriptors = sift.detectAndCompute(self.image, None)
        logging.info(f"Feature extraction complete.")
        if write_to_file:
            self.write_features()
        return None

    def read_features(self):
        """
        Load features from file. File format should be "image_001.npz"
        """
        def unpack_keypoint(data):
            try:
                kpts = data['keypoints']
                desc = data['descriptors']
                keypoints = [cv.KeyPoint(x, y, _size, _angle, _response, int(_octave), int(_class_id))
                             for x, y, _size, _angle, _response, _octave, _class_id in list(kpts)]
                return keypoints, np.array(desc)
            except(IndexError):
                return np.array([]), np.array([])
        try:
            data = np.load(self.features_path + self.id + ".npz")
            self.keypoints, self.descriptors = unpack_keypoint(data)
            logging.info(f"Existing features for {self.name} found in features directory.")
        except FileNotFoundError:
            logging.info(f"Features for {self.name} not found in {self.features_path}.")

    def write_features(self):
        """
        Write image features to file. Write format should be "image_001.npz"
        """
        def pack_keypoint(keypoints, descriptors):
            kpts = np.array([[kp.pt[0], kp.pt[1], kp.size,
                              kp.angle, kp.response, kp.octave,
                              kp.class_id]
                             for kp in keypoints])
            desc = np.array(descriptors)
            return kpts, desc

        filename = self.features_path + self.id
        kpts, desc = pack_keypoint(self.keypoints, self.descriptors)
        logging.info(f'Writing features of image {self.name} to file...')
        np.savez(filename, keypoints=kpts, descriptors=desc)
        logging.info('Features saved.')

    def get_3D_correspondences(self):
        return self.world_points.items()

