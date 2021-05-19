import cv2 as cv
import numpy as np

from WorldPoints import WorldPointSet
from utils import *


class Baseline:
    """
    Creates the baseline from two views.
    """

    def __init__(self, view1, view2, K, keypoints):
        """
        K: Camera intrinsic calibration matrix.
        view1: View of first image from which to get baseline. By default R and t for this view will be I_3x3 and 0.
        view2: View of second image from which to extract baseline information.
        """
        self.X2 = keypoints[0]
        self.X1 = keypoints[1]
        self.K = K
        self.view1 = view1
        self.view2 = view2
        self.fundamental_mat = np.zeros((3, 3))
        self.essential_mat = np.zeros((3, 3))

    def __call__(self, *args, **kwargs):
        # Feature matching two views
        # self.feature_match_baseline()
        # Calculate fundamental matrix
        self.calc_fundamental_matrix()
        # Calculate essential matrix
        self.calc_essential_matrix()
        # Get four camera poses for view2
        self.view1.rotation = np.eye(3)
        C2, R2 = camera_pose_extraction(self.essential_mat)
        # Disambiguate poses
        X_4 = []
        x1, x2 = self.view1.tracked_pts[self.view2.id]
        for i in range(len(R2)):
            print(f"Pose {i + 1}")
            X_n = triangulate_points(self.K, self.view1.translation, self.view1.rotation,
                                     C2[i], R2[i], x1, x2)
            X_4.append(X_n)

        X, self.view2.rotation, self.view2.translation = pose_disambiguation(x2, self.K, C2, R2, X_4)
        wpSet = WorldPointSet(add_redundant_views=False)
        print(self.view1.tracked_pts[self.view2.id][0].shape)
        X = store_3Dpoints_to_views(X, self.view1, self.view2, self.K, error_threshold=1.0)
        wpSet.add_correspondences(X, self.view1, self.view2)
        np.savez('points_3d_baseline', point_cloud=wpSet.world_points)
        return wpSet

    def calc_fundamental_matrix(self, save=False):
        FM_METHOD = cv.FM_RANSAC
        self.fundamental_mat, mask = cv.findFundamentalMat(self.X1, self.X2, method=FM_METHOD)
        self.view1.tracked_pts[self.view2.id] = (self.X1[mask.ravel() == 1], self.X2[mask.ravel() == 1])
        self.view2.tracked_pts[self.view1.id] = (self.X2[mask.ravel() == 1], self.X1[mask.ravel() == 1])
        # self.view1.tracked_pts[self.view2.id] = (self.X1, self.X2)
        # self.view2.tracked_pts[self.view1.id] = (self.X2, self.X1)
        if save:
            np.savez('fundamental_matrix', F=self.fundamental_mat)

    def calc_essential_matrix(self, save=False):
        self.essential_mat = self.K.T @ self.fundamental_mat @ self.K
        U, _, V = np.linalg.svd(self.essential_mat)
        self.essential_mat = U @ np.array([[1, 0, 0], [0, 1, 0], [0, 0, 0]]) @ V
        self.essential_mat = self.essential_mat / np.linalg.norm(self.essential_mat)
        if save:
            np.savez('essential_matrix', E=self.essential_mat)

    def feature_match_baseline(self, matcher_type="bf", NUM_NEAREST_NEIGHBOURS=2):
        """
        Performs feature matching of the two baseline views.
        :param view1:
        :param view2:
        :param matcher_type:
        :param NUM_NEAREST_NEIGHBOURS:
        :return:
        """
        NN_RATIO_THRESHOLD = 0.8

        if matcher_type == "flann":
            FLANN_INDEX_KDTREE = 1
            index_params = dict(algorithm=FLANN_INDEX_KDTREE, trees=5)
            search_params = dict(checks=50)
            matcher = cv.FlannBasedMatcher(index_params, search_params)
        elif matcher_type == 'bf':
            matcher = cv.BFMatcher()
        else:
            raise Exception("Please enter a valid matcher type! Either 'bf' or 'flann' are accepted.")

        # Read features from file. If features file does not exist, extract from images and save to file.
        for view in [self.view1, self.view2]:
            view.read_features()
            if view.descriptors is None:
                view.extract_features()
                view.write_features()

        logging.info(f"Generating feature matches between {self.view1.name} and {self.view2.name}.")
        matches = matcher.knnMatch(self.view1.descriptors, self.view2.descriptors, k=NUM_NEAREST_NEIGHBOURS)
        logging.info(f"{len(matches)} matches found.")
        good_matches = []
        X1 = []
        X2 = []
        for m, n in matches:
            if m.distance < NN_RATIO_THRESHOLD * n.distance:
                good_matches.append(m)
                X1.append(self.view1.keypoints[m.queryIdx].pt)
                X2.append(self.view2.keypoints[m.trainIdx].pt)

        logging.info(f"{len(good_matches)} good matches found from {len(matches)} matches.")

        self.X1 = np.array(X1, dtype=np.float64)
        self.X2 = np.array(X2, dtype=np.float64)

        return None

    def recover_pose(self):
        pass