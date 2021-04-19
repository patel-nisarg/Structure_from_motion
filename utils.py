import logging

import numpy as np
import cv2 as cv


def feature_match(view1, view2, matcher_type="bf", NUM_NEAREST_NEIGHBOURS=2, homography_threshold=0.50):
    """
    Feature matches two images(views) using BF matcher by default.
    Returns points that overlap between two views as well as the degree of overlap from 0 to 1.

    :param view1: View of first image to match.
    :param view2: View of second image to match.
    :param matcher_type: Feature matcher type. Either "bf" for Brute Force or "flann" for FLANN based matcher.
    :param NUM_NEAREST_NEIGHBOURS: Number of nearest neighbors for knn Match.
    :param homography_threshold: Threshold for homography overlap between two views.
    :return: view1kps: view1 matching keypoints coordinates. A Numpy Masked Array object (N x 3) is returned.
    :return: view2kps: view2 matching keypoints coordinates. A Numpy Masked Array object (N x 3) is returned.
    :return: amount_overlap: amount of keypoint overlap. Ranges from 0 to 1.
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

    matches = matcher.knnMatch(view1.descriptors, view2.descriptors, k=NUM_NEAREST_NEIGHBOURS)

    good_matches = []
    X1 = np.array([])  # points in image 1
    X2 = np.array([])  # corresponding points in image 2

    for m, n in matches:
        if m.distance < 0.8 * n.distance:
            good_matches.append(m)
            np.append(X1, view1.keypoints[m.queryIdx].pt)
            np.append(X2, view2.keypoints[m.trainIdx].pt)

    if len(good_matches) > 10:
        H, s = cv.findHomography(X1, X2, cv.RANSAC, 4)
        s = np.asarray(s, dtype=bool)
        overlap_amount = sum(s) / len(s)
        if overlap_amount >= homography_threshold:
            view1_kps = X1[s]
            view2_kps = X2[s]
            return view1_kps, view2_kps, overlap_amount
        else:
            logging.info(f"Not enough point overlap found between {view1.name} and {view2.name}.")
            return None, None, None
    else:
        logging.info(f"Insufficient feature matches between {view1.name} and {view2.name}.")

    return None


def check_determinant(R, threshold=1e-7):
    """
    :param R: Rotation matrix (3x3)
    :param threshold: Radius within -1 to check for
    :return: Bool: True if determinant of R is not -1. False if determinant is approximately -1.
    """
    det = np.linalg.det(R)

    if det + 1 < threshold:
        return False
    else:
        return True


def camera_pose_extraction(E):
    """
    (input) E: Essential matrix
    (output) C_set and R_set: Four sets of camera centers and rotations.
    """
    W = np.array([[0, -1, 0],
                  [1, 0, 0],
                  [0, 0, 1]])
    U, D, V = np.linalg.svd(E)
    C1 = U[:, -1]
    R1 = U @ W @ V.T

    C2 = -U[:, -1]
    R2 = U @ W @ V.T

    C3 = U[:, -1]
    R3 = U @ W.T @ V.T

    C4 = -U[:, -1]
    R4 = U @ W.T @ V.T

    C = np.asarray([C1, C2, C3, C4], dtype=np.float32)
    R = np.asarray([R1, R2, R3, R4], dtype=np.float32)

    for i in range(len(R)):
        if check_determinant(R[i]):
            R[i] = -R[i]
            C[i] = -C[i]
    return C, R


def vec2skew(v):
    return [[0, -v[2], v[1]],
            [v[2], 0, -v[0]],
            [-v[2], v[0], 0]]


def linear_triangulation(K, C1, R1, C2, R2, x1, x2):
    """LinearTriangulation
  Find 3D positions of the point correspondences using the relative
  position of one camera from another
  Inputs:
      C1 - size (3 x 1) translation of the first camera pose
      R1 - size (3 x 3) rotation of the first camera pose
      C2 - size (3 x 1) translation of the second camera
      R2 - size (3 x 3) rotation of the second camera pose
      x1 - size (N x 2) matrix of points in image 1
      x2 - size (N x 2) matrix of points in image 2, each row corresponding
        to x1
  Outputs: 
      X - size (N x 3) matrix who's rows represent the 3D triangulated
        points"""

    P1 = K @ np.hstack((R1, -R1 @ C1))
    P2 = K @ np.hstack((R2, -R2 @ C2))
    X = np.zeros((len(x1), 4))
    for i in range(len(x1)):
        pt1 = vec2skew(np.append(x1[i], 1))
        pt2 = vec2skew(np.append(x2[i], 1))
        A = np.concatenate((pt1 @ np.asarray(P1), pt2 @ np.asarray(P2)), axis=0)
        _, _, V = np.linalg.svd(A)
        X[i] = V[:, -1] / V[-1, -1]
    return X


def pose_disambiguation(C_set, R_set, X_set):
    """
    (INPUT) Cset and Rset: four configurations of camera centers and rotations
    (INPUT) Xset: four sets of triangulated points from four camera pose configurations
    (OUTPUT) C and R: the correct camera pose
    (OUTPUT) X0: the 3D triangulated points from the correct camera pose

    The sign of the Z element in the camera coordinate system indicates the location of
    the 3D point with respect to the camera, i.e., a 3D point X is in front of a
    camera (C, R) if r3(X-C) > 0 where r3 is the third row of R. Not all triangulated points
    satisfy this condition due to the presence of correspondence noise. The best camera
    configuration, (C, R, X) is the one that produces the maximum number of points satisfying
    the chirality condition. """

    def check_chirality(C, R, X):
        """For each point in X, checks the condition that r_3(X-C) > 0.
        Returns the total number of points in X that meets this condition."""
        num_points = 0
        for x in X:
            if R[:, -1] @ (x[:-1] - C) < 0:
                num_points += 1
        return num_points

    error = []
    for i in range(len(R_set)):
        error.append(check_chirality(C_set[i], R_set[i], X_set[i]))

    return X_set[np.argmin(error)]


# Does not seem to work. Throws memory error.
def triangulate_points(K, C1, R1, C2, R2, x1, x2):
    P1 = K @ np.hstack((R1, -R1 @ C1))
    P2 = K @ np.hstack((R2, -R2 @ C2))
    x1 = x1.T
    x2 = x2.T
    X = cv.triangulatePoints(P1, P2, x1, x2)
    return X


def write_pose_to_file(view):
    """
    Writes the pose of view object to a file.
    :param pose:
    """
    pass


def visualize_pt_cloud(world_coords):
    pass


def compute_pose(view, completed_views):
    """
    Compute pose of current view from completed_views using Linear PnP.
    Also updates the tracked points (view.tracked_pts['completed_view']) between current view and completed views.
    :param view: Current view for which to compute pose.
    :param completed_views: List containing all View Ids that have been 3D reconstructed.
    :return:
    """

    def linear_pnp():
        pass

    position = []
    rotation = []

    return position, rotation


def get_paths_from_txt(filename):
    logging.info("Reading image paths text file...")
    with open(filename, 'r') as f:
        image_paths = f.readlines()
    image_paths = [x.strip() for x in image_paths]
    logging.info(f"{len(image_paths)} images found.")
    return image_paths
