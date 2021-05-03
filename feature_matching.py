import cv2 as cv
import numpy as np
import matplotlib.pyplot as plt

data = np.load('calibration_data.npz')

distortion_params = data['distortion_params']
K = data['calibratoin_matrix']

# Matcher parameters
NN_RATIO_THRESHOLD = 0.75
NUM_NEAREST_NEIGHBOURS = 2

# Read images
img1 = cv.imread('image_001.jpg', cv.IMREAD_GRAYSCALE)
img2 = cv.imread('image_002.jpg', cv.IMREAD_GRAYSCALE)

if img1 is None or img2 is None:
    raise FileNotFoundError('Could not find images!')

# Initialize SIFT object
sift = cv.SIFT_create()

# Detect keypoints on images and find descriptors
kp1, descriptors1 = sift.detectAndCompute(img1, None)
kp2, descriptors2 = sift.detectAndCompute(img2, None)

keyp1 = np.array([kp1[idx].pt for idx in range(0, len(kp1))], dtype=np.float32).reshape(-1, 2)


# BFMatcher with k=2
bf = cv.BFMatcher()
matches = bf.knnMatch(descriptors1, descriptors2, k=NUM_NEAREST_NEIGHBOURS)

# FLANN_INDEX_KDTREE = 0
# index_params = dict(algorithm=FLANN_INDEX_KDTREE, trees=5)
# search_params = dict(checks=50)
#
# flann = cv.FlannBasedMatcher(index_params, search_params)
# matches = flann.knnMatch(descriptors1, descriptors2, k=2)

# Ratio test for preliminary filtering
good_matches = []
X1 = []  # points in image 1
X2 = []  # corresponding points in image 2

for m, n in matches:
    if m.distance < NN_RATIO_THRESHOLD * n.distance:
        good_matches.append(m)
        X1.append(kp1[m.queryIdx].pt)
        X2.append(kp2[m.trainIdx].pt)

# Save keypoints that have good (above threshold) correspondence
X1 = np.array(X1)
X2 = np.array(X2)
info = """
This data file contains point correspondences from two images.
Keypoints from image 1 are stored in array 'X1'.
Corresponding keypoints from image 2 are stored in array 'X2'.
"""
np.savez('img_correspondences', info=info, X1=X1, X2=X2)

img3 = cv.drawMatches(img1, kp1, img2, kp2, good_matches, None, flags=2)
cv.imwrite('feature_matched_img.png', img3)
