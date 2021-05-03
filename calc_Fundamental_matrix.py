import numpy as np
import cv2 as cv

correspondences_file = 'img_correspondences.npz'
FM_METHOD = cv.FM_RANSAC

# Note on F computation method from OpenCV docs:
# FM_7POINT, for a 7 - point algorithm. N=7
# FM_8POINT, for an 8 - point algorithm. N≥8
# FM_RANSAC, for the RANSAC algorithm. N≥8
# FM_LMEDS, for the LMedS algorithm. N≥8

data = np.load(correspondences_file)

# Load points from two images
X1_original = np.float32(data['X1'])
X2_original = np.float32(data['X2'])

# Calculate fundamental matrix and get mask
F, mask = cv.findFundamentalMat(X1_original, X2_original,
                                method=FM_METHOD)

# We select only inlier points
pts1 = X1_original[mask.ravel() == 1]
pts2 = X2_original[mask.ravel() == 1]
info = """
Contains corresponding points (x1, x2) between two images after 
calculating Fundamental matrix via RANSAC and obtaining mask.
"""

np.savez('ransac_correspondences', info=info, x1=pts1, x2=pts2)
