import numpy as np
from utils import *
import cv2 as cv

data = np.load('ransac_correspondences.npz')
x1 = data['x1']
x2 = data['x2']

F = np.load('fundamental_mat.npz')['F']
K = np.load('calibration_data.npz')['calibratoin_matrix']
E = K.T @ F @ K

# Define first camera pose as I and [0, 0, 0]
R1 = np.eye(3)
C1 = np.zeros((3, 1))
# Get the four possible C2 and R2 for second image
C2, R2 = camera_pose_extraction(E)

# Get 3D points
X_4 = []
for i in range(len(R2)):
    C_2 = np.expand_dims(C2[i], axis=1)
    X_4.append(linear_triangulation(K, C1, R1, C_2, R2[i], x1, x2))

# The correct camera pose
X = pose_disambiguation(C2, R2, X_4)

