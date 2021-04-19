import cv2 as cv
import numpy as np
from utils import *


class Baseline:
    """
    Creates the baseline from two views.
    """
    def __init__(self, view1, view2, K):
        """
        K: Camera intrinsic calibration matrix.
        view1: View of first image from which to get baseline. By default R and t for this view will be I_3x3 and 0.
        view2: View of second image from which to extract baseline information.
        """
        self.K = K
        self.view1 = view1
        self.view2 = view2
        self.fundamental_mat = np.zeros((3, 3))
        self.essential_mat = np.zeros((3, 3))

    def calc_fundamental_matrix(self, save=False):
        pass

    def calc_essential_matrix(self, save=False):
        self.view1.keypoints

    def return_views(self):
        return self.view1, self.view2

    def feature_match_baseline(self, matcher_type="bf", NUM_NEAREST_NEIGHBOURS=2):
        """
        Performs feature matching of the two baseline views.
        :param view1:
        :param view2:
        :param matcher_type:
        :param NUM_NEAREST_NEIGHBOURS:
        :return:
        """
        if matcher_type == "flann":
            FLANN_INDEX_KDTREE = 0
            index_params = dict(algorithm=FLANN_INDEX_KDTREE, trees=5)
            search_params = dict(checks=50)
            matcher = cv.FlannBasedMatcher(index_params, search_params)
        elif matcher_type == 'bf':
            matcher = cv.BFMatcher()
        else:
            raise Exception("Please enter a valid matcher type! Either 'bf' or 'flann' are accepted.")


        self.view1.extract_features(write_to_file=True)
        self.view2.extract_features(write_to_file=True)
        return None