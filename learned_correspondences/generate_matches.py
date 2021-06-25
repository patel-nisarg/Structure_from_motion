import pickle
import numpy as np
import cv2
import os
import sys
from network import MyNetwork
from evaluate import test_pair
from itertools import combinations

BASE_PATH = os.path.join(os.getcwd(), "demo")
sys.path.append(BASE_PATH)

def match(im_1, im_2, n_kp=2000):
    sift = cv2.SIFT_create(
        nfeatures=n_kp, contrastThreshold=1e-5)

    cv_kp_1, cv_desc_1 = sift.detectAndCompute(im_1, None)
    cv_kp_2, cv_desc_2 = sift.detectAndCompute(im_2, None)

    matcher = cv2.BFMatcher(cv2.NORM_L2)
    matches_all = matcher.knnMatch(cv_desc_1, cv_desc_2, k=2)

    # scores = np.array([x.distance for x in matches_all])
    # idx1 = np.array([x.queryIdx for x in matches_all])
    # idx2 = np.array([x.trainIdx for x in matches_all])
    #
    # kp1_x = np.array([k.pt[0] for k in cv_kp_1])[idx1]
    # kp1_y = np.array([k.pt[1] for k in cv_kp_1])[idx1]
    # kp1 = np.stack((kp1_x, kp1_y), axis=1)
    #
    # kp2_x = np.array([k.pt[0] for k in cv_kp_2])[idx2]
    # kp2_y = np.array([k.pt[1] for k in cv_kp_2])[idx2]
    # kp2 = np.stack((kp2_x, kp2_y), axis=1)

    good_matches = []
    X1 = []  # points in image 1
    X2 = []  # corresponding points in image 2

    for m, n in matches_all:
        if m.distance < 0.8 * n.distance:
            good_matches.append(m)
            X1.append(cv_kp_1[m.queryIdx].pt)
            X2.append(cv_kp_2[m.trainIdx].pt)
    if len(good_matches) < 20:
        print(f"Found insufficient matches.")
    kp1 = np.array(X1)
    kp2 = np.array(X2)

    return kp1, kp2


def run_model(mynet, kp1, kp2, shape_im_1, shape_im_2, focal_len_1=None, focal_len_2=None):
    # normalize coordinates: if focal lengths are not given, use simpler normalization
    norm_factor_1 = focal_len_1 if focal_len_1 is not None else np.max(shape_im_1[:2])
    norm_factor_2 = focal_len_2 if focal_len_2 is not None else np.max(shape_im_2[:2])
    kp1 = (kp1 - 0.5 * np.array(shape_im_1[:2])[::-1]) / norm_factor_1
    kp2 = (kp2 - 0.5 * np.array(shape_im_2[:2])[::-1]) / norm_factor_2
    # concatenate
    xs = np.expand_dims(np.concatenate((kp1, kp2), axis=1), 0)
    # run
    ys = test_pair(mynet, xs)
    return ys


def match_img_list(img_txt, save=False):
    """
    Finds feature matches between .
    :param save: Flag to save unfiltered matches to file.
    :param img_txt: *.txt file path containing string list of image paths to feature match. Separator: \n
    :return: features_list: Numpy array of (N x [kp1, kp2, im1.shape, im2.shape, im1_name, im2_name]).
                            Where N is all combination of matches.
    """
    with open(img_txt, 'r') as f:
        img_paths = f.readlines()
    img_paths = [x.strip() for x in img_paths]
    tot_combinations = int((np.math.factorial(len(img_paths))) / (2 * np.math.factorial(len(img_paths) - 2)))
    print(f"{len(img_paths)} images found. Creating {tot_combinations} total feature matches.")
    # need a list of all combinations of list img_paths for [I_k, I_k'] matches
    img_pairs = list(combinations(img_paths, r=2))  # r = number of elements to choose from in nCr(n, r)
    features_list = []
    for pair in img_pairs:
        im1, im2 = cv2.imread(pair[0]), cv2.imread(pair[1])
        im1_name, im2_name = os.path.basename(pair[0])[:-4], os.path.basename(pair[1])[:-4]
        kp1, kp2 = match(im1, im2)
        print(f"Found {len(kp1)} matches between image{im1_name} and image{im2_name}.")
        features_list.append([kp1, kp2, im1.shape, im2.shape, im1_name, im2_name])

    if save:
        np.savez('feature_matches_unfiltered', feature_matches=features_list)

    return np.array(features_list)


def filter_matches(unfiltered_matches, filter_model, save=False):
    """
    :param save: Flag to save filtered matches to file.
    :param filter_model: Neural network model used to filter matches
    :param unfiltered_matches: Numpy array of (N x [kp1, kp2, im1.shape, im2.shape, im1_name, im2_name]).
                               Where N is all combination of matches.
    :return: filtered_matches: Numpy array of pair of matches now filtered with filter_model.
                                (N x [kp1, kp2, im1_name, im2_name])
    """
    filtered_matches = []

    for match_pair in unfiltered_matches:
        kp1, kp2, im_1_shape, im_2_shape = match_pair[0], match_pair[1], match_pair[2], match_pair[3]
        im1_name, im2_name = match_pair[4], match_pair[5]

        # run model
        y = run_model(filter_model, kp1, kp2, im_1_shape, im_2_shape)

        # resort matches according to model output
        sorted_idx = np.argsort(-y).squeeze()
        kp1 = kp1[sorted_idx, :]
        kp2 = kp2[sorted_idx, :]
        scores = y.squeeze()[sorted_idx]

        # filter out matches with negative scores
        kp1 = kp1[scores > 0]
        kp2 = kp2[scores > 0]
        print(f"Filtered matches to {len(kp1)} between image{match_pair[4]} and image{match_pair[5]}.")
        filtered_matches.append([kp1, kp2, im1_name, im2_name])

    if save:
        np.savez('feature_matches_filtered', filtered_matches=filtered_matches)

    return np.array(filtered_matches)


class FeatureMatchGenerator:
    def __init__(self, IMG_PATHS_FILE):
        self.img_paths = IMG_PATHS_FILE

    def create_feature_matches(self):
        # load unfiltered_matches:
        # feature_matches = np.load('feature_matches_unfiltered.npz')['feature_matches']
        # uncomment the following line if unfiltered matches do not exist
        feature_matches = match_img_list(self.img_paths, save=True)

        with open('config.pickle', 'rb') as handle:
            config = pickle.load(handle)

        mynet = MyNetwork(config)
        mynet.load_model()

        # generate filtered matched. Note matches are not cv.KeyPoint objects but 2D points
        # example shape is filtered_matches[img_combinations[kps1, kps2[2D point]], img1_name, img_2 name]
        filter_matches(feature_matches, mynet, save=True)
