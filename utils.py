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
        FLANN_INDEX_KDTREE = 1
        index_params = dict(algorithm=FLANN_INDEX_KDTREE, trees=5)
        search_params = dict(checks=50)
        matcher = cv.FlannBasedMatcher(index_params, search_params)
    elif matcher_type == 'bf':
        matcher = cv.BFMatcher()
    else:
        raise Exception("Please enter a valid matcher type! Either 'bf' or 'flann' are accepted.")

    matches = matcher.knnMatch(view1.descriptors, view2.descriptors, k=NUM_NEAREST_NEIGHBOURS)

    good_matches = []
    X1 = []  # points in image 1
    X2 = []  # corresponding points in image 2

    for m, n in matches:
        if m.distance < 0.8 * n.distance:
            good_matches.append(m)
            X1.append(view1.keypoints[m.queryIdx].pt)
            X2.append(view2.keypoints[m.trainIdx].pt)

    X1 = np.array(X1)
    X2 = np.array(X2)

    if len(good_matches) > 20:
        logging.info(f"{len(good_matches)} good matches found from {len(matches)} matches "
                     f"between {view1.name} and {view2.name}.")
        # return matches between X1 and X2

        # H, mask = cv.findHomography(X1, X2, cv.RANSAC, 4)
        # overlap_amount = sum(mask) / len(mask)
        # if overlap_amount >= homography_threshold:
        #     view1_kps = X1[mask.ravel() == 1]
        #     view2_kps = X2[mask.ravel() == 1]
        #     return view1_kps, view2_kps, mask, overlap_amount
        # else:
        #     logging.info(f"Not enough point overlap found between {view1.name} and {view2.name}. "
        #                  f"Overlap, Threshold:{overlap_amount}{homography_threshold}")
        #     return None
        view1.tracked_pts[view2.id] = (X1, X2)
        view2.tracked_pts[view1.id] = (X2, X1)
        return X1, X2
    else:
        logging.info(f"Insufficient feature matches between {view1.name} and {view2.name}.")

    return None


def check_determinant(R, threshold=1e-9):
    """
    :param R: Rotation matrix (3x3)
    :param threshold: Radius within -1 to check for
    :return: Bool: True if determinant of R is not -1. False if determinant is approximately -1.
    """
    det = np.linalg.det(R)

    if det + 1.0 < threshold:
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
    C1 = U[:, -1].reshape((3, 1))
    R1 = U @ W @ V

    C2 = -U[:, -1].reshape((3, 1))
    R2 = U @ W @ V

    C3 = U[:, -1].reshape((3, 1))
    R3 = U @ W.T @ V

    C4 = -U[:, -1].reshape((3, 1))
    R4 = U @ W.T @ V

    C = np.asarray([C1, C2, C3, C4], dtype=np.float64)
    R = np.asarray([R1, R2, R3, R4], dtype=np.float64)

    for i in range(len(R)):
        if not check_determinant(R[i]):
            R[i] = -R[i]
            C[i] = -C[i]

    return C, R


def vec2skew(v):
    return [[0, -v[2], v[1]],
            [v[2], 0, -v[0]],
            [-v[2], v[0], 0]]


def baseline_triangulation(K, C1, R1, C2, R2, x1, x2, inhomogeneous=True):
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
      inhomogeneous: If true, returns X as N x 3 instead of N x 4
    Outputs:
      X - size (N x 3) OR (N x 4) matrix who's rows represent the 3D triangulated
        points"""
    logging.info("Triangulating baseline views...")
    P1 = K @ np.hstack((R1, C1))
    P2 = K @ np.hstack((R2, C2.reshape((3, 1))))
    X = np.zeros((len(x1), 4))
    for i in range(len(x1)):
        pt1 = vec2skew(np.append(x1[i], 1))
        pt2 = vec2skew(np.append(x2[i], 1))
        A = np.concatenate((pt1 @ np.asarray(P1), pt2 @ np.asarray(P2)), axis=0)
        _, _, V = np.linalg.svd(A)
        X[i] = V.T[:, -1] / V.T[-1, -1]
    if inhomogeneous:
        X = np.delete(X, -1, 1)
    return X


def pose_disambiguation(x2, K, C2, R2, X_n):
    """
    Returns the pose that minimizes reprojection error.

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
    reprojection_error = []
    for j, X_set in enumerate(X_n):
        X_set_error = []
        for i, point_3d in enumerate(X_set):
            error, reprojected_pt = calculate_reprojection_error(point_3d, x2[i], K, R2[j],
                                                                 C2[j])
            # if error > 50.0:
            # print(x2[i], reprojected_pt)
            X_set_error.append(error)
        reprojection_error.append(np.mean(X_set_error))
    print(reprojection_error)
    X = X_n[np.argmin(reprojection_error)]
    R = R2[np.argmin(reprojection_error)]
    C = C2[np.argmin(reprojection_error)]

    # def check_chirality(C, R, X):
    #     """For each point in X, checks the condition that r_3(X-C) > 0.
    #     Returns the total number of points in X that meets this condition."""
    #     num_points = 0
    #     for x in X:
    #         if R[:, -1] @ (x[:-1] - C) < 0:
    #             num_points += 1
    #     return num_points
    #
    # condition_met = []
    # for i in range(len(R_set)):
    #     condition_met.append(check_chirality(C_set[i], R_set[i], X_set[i]))
    #
    # X = np.delete(X_set[np.argmax(condition_met)], -1, 1)
    # R = R_set[np.argmax(condition_met)]
    # print(X.shape)
    return X, R, C


def triangulate_points(K, t1, R1, t2, R2, x1, x2, print_error=False):
    """
    - Normalizes 2D points in homogeneous coordinates by multiplying them by inverse of intrinsic calibration matrix
    - Triangulates 3D points based on normalized 2D points and projection matrices for two views
    - returns 3D points as (N x 3)
    :param K:
    :param t1:
    :param R1:
    :param t2:
    :param R2:
    :param x1:
    :param x2:
    :param print_error:
    :return:
    """
    logging.info("Triangulating 3D points...")
    # K_inv = np.linalg.inv(K)
    # P1 = np.hstack((R1, t1))
    # P2 = np.hstack((R2, t2.reshape((3, 1))))
    # u1 = cv.convertPointsToHomogeneous(x1)
    # u2 = cv.convertPointsToHomogeneous(x2)
    if len(x1) < 8:
        return None
    P1 = K @ np.hstack((R1, t1))
    P2 = K @ np.hstack((R2, t2))
    x1 = x1.T
    x2 = x2.T
    # u1_n = np.empty((0, 3))
    # u2_n = np.empty((0, 3))
    # for i in range(len(u1)):
    #     u1_n = np.append(u1_n, (K_inv @ u1[i].T).T, axis=0)
    #     u2_n = np.append(u2_n, (K_inv @ u2[i].T).T, axis=0)
    # u1_n = cv.convertPointsFromHomogeneous(u1_n)
    # u2_n = cv.convertPointsFromHomogeneous(u2_n)
    X = cv.triangulatePoints(projMatr1=P1, projMatr2=P2, projPoints1=x1, projPoints2=x2)
    X = cv.convertPointsFromHomogeneous(X.T)
    error1 = []
    error2 = []
    if print_error:
        for i, point_3d in enumerate(X):
            e1, _ = calculate_reprojection_error(point_3d, x1.T[i], K, R1, t1)
            error1.append(e1)
            e2, _ = calculate_reprojection_error(point_3d, x2.T[i], K, R2, t2)
            error2.append(e2)
        print(np.mean(error1), np.mean(error2))
    if np.mean(error1) < 55 and np.mean(error2) < 55:
        return X
    else:
        return None


def compute_pose(view, completed_views, K, dist, img_matches):
    """
    Compute pose of current view from completed_views using Linear PnP.
    Also updates the tracked points (view.tracked_pts['completed_view']) between current view and completed views.
    :param img_matches: Dictionary of image match pairs containing keypoints
    :param view: Current view for which to compute pose.
    :param completed_views: List containing all View Ids that have been 3D reconstructed.
    :param K: camera intrinsic matrix
    :param dist: camera distortion parameters
    :return:
    """
    points_2d = np.empty((0, 2))
    points_3d = np.empty((0, 3))
    pts_found = 0
    for view_n in completed_views:
        match = feature_match(view, view_n)
        # print(view.name, view_n.name)
        # match = img_matches[(view.name, view_n.name)]
        # view.tracked_pts[view_n.id] = match
        # view_n.tracked_pts[view.id] = img_matches[(view_n.name, view.name)]
        # print(view_n.world_points[:, :2].shape)
        if match is not None:
            logging.info(f"{len(match[1])} matches found between {view.name} and {view_n.name}.")
            #     for i, point in enumerate(match[1]):
            #         point = tuple(point.tolist())
            #         if point in view_n.world_points:
            #             point_3d = view_n.world_points[point]
            #             points_2d.append(match[0][i])
            #             points_3d = np.append(points_3d, point_3d)
            for i, point in enumerate(match[1]):
                # find existing 2D/3D point correspondence in view_n that is in
                # the feature match between view and view_n
                index = np.argwhere(np.isclose(view_n.world_points[:, :2], point))
                #print(index)
                if index.size != 0:
                    pts_found += 1
                    point_3d = view_n.world_points[index[0][0], 2:]
                    points_3d = np.append(points_3d, [point_3d], axis=0)

                    point_2d = match[0][i]  # point in View (NOT View_n)
                    points_2d = np.append(points_2d, [point_2d], axis=0)
        print(f"Found {pts_found} 3D points in {view_n.name} matching {view.name}.")
        logging.info(f"Found {pts_found} 3D points in {view_n.name} matching {view.name}.")

    print(points_3d.shape, points_2d.shape)

    reprojection_Error = 8.0
    if len(points_3d) > 12:
        _, rotation, translation, _ = cv.solvePnPRansac(points_3d, points_2d, K, None, confidence=0.99,
                                                        reprojectionError=reprojection_Error, flags=cv.SOLVEPNP_EPNP)
        # PnP spits out a Rotation vector. Convert to Rotation matrix and check validity.
        rotation, _ = cv.Rodrigues(rotation)
        #
        # if check_determinant(rotation):
        #     rotation = -rotation
        logging.info(f"Pose for {view.name} calculated.")
        return rotation, translation
    else:
        return None, None


def get_paths_from_txt(filename):
    logging.info("Reading image paths text file...")
    with open(filename, 'r') as f:
        image_paths = f.readlines()
    image_paths = [x.strip() for x in image_paths]
    logging.info(f"{len(image_paths)} images found.")
    return image_paths


def store_3Dpoints_to_views(X_initial, view1, view2, K, error_threshold=40.0):
    """
    Stores 3D points to the two views that were used to triangulate them. Also filters 3D points with large
    reprojection error and returns that are
    :param K: Intrinsic matrix for camera
    :param view1: First view used to triangulate 3D points.
    :param view2: Second view used to triangulate 3D points.
    :param X_initial: 3D points to filter and store in views.
    """
    view1_points = np.array(view1.tracked_pts[view2.id][0])  # points in view1 that have a triangulated 3D point X
    view2_points = np.array(view2.tracked_pts[view1.id][0])
    rm_indices = []
    for i, point_3d in enumerate(X_initial):
        error1, reproj_pt1 = calculate_reprojection_error(point_3d, view1_points[i][:, np.newaxis], K, view1.rotation,
                                                          view1.translation)
        error2, reproj_pt2 = calculate_reprojection_error(point_3d, view2_points[i][:, np.newaxis], K, view2.rotation,
                                                          view2.translation)
        if error1 < error_threshold and error2 < error_threshold:
            # append to world points of views of reprojection error is lower than error_threshold
            wp1 = np.concatenate((view1_points[i], point_3d.reshape(1, 3)), axis=None)
            view1.world_points = np.append(view1.world_points, [wp1], axis=0)
            wp2 = np.concatenate((view2_points[i], point_3d.reshape(1, 3)), axis=None)
            view2.world_points = np.append(view2.world_points, [wp2], axis=0)
        else:
            rm_indices.append(i)

    # remove 2D points from views that have higher reprojection error than threshold
    view1.tracked_pts[view2.id] = (np.delete(view1.tracked_pts[view2.id][0], rm_indices, 0),
                                   np.delete(view1.tracked_pts[view2.id][1], rm_indices, 0))
    view2.tracked_pts[view1.id] = (np.delete(view2.tracked_pts[view1.id][0], rm_indices, 0),
                                   np.delete(view2.tracked_pts[view1.id][1], rm_indices, 0))
    # remove 3D points from points set that have a higher reprojection error than threshold
    X_fin = np.delete(X_initial, rm_indices, axis=0)

    return X_fin


def remove_outliers(view1, view2):
    """
    Removes outliers between two views by using Fundamental matrix. Used before point triangulation.
    :param view1:
    :param view2:
    :return:
    """

    X1, X2 = view1.tracked_pts[view2.id]
    FM_METHOD = cv.FM_RANSAC

    fundamental_mat, mask = cv.findFundamentalMat(X1, X2, method=FM_METHOD, confidence=0.999, ransacReprojThreshold=8)
    if mask is not None:
        view1.tracked_pts[view2.id] = (X1[mask.ravel() == 1], X2[mask.ravel() == 1])
        view2.tracked_pts[view1.id] = (X2[mask.ravel() == 1], X1[mask.ravel() == 1])

    return view1.tracked_pts[view2.id][0], view2.tracked_pts[view1.id][0]


def calculate_reprojection_error(point_3D, point_2D, K, R, t):
    """Calculates the reprojection error for a 3D point by projecting it back into the image plane"""
    point_2D = point_2D.reshape((2, 1))
    point_3D = point_3D.reshape((3, 1))
    # print(K.shape, R.shape, point_3D.shape, t.shape)
    reprojected_point = K.dot(R.dot(point_3D) + t.reshape((3, 1)))
    reprojected_point = cv.convertPointsFromHomogeneous(reprojected_point.T)[:, 0, :].T
    # print((point_2D - reprojected_point)[1])
    error = np.linalg.norm(point_2D - reprojected_point)
    return error, reprojected_point


def keypoints_to_dict(keypoints, filtered=True):
    img_matches = {}
    if filtered:
        for match in keypoints:
            key = (match[2], match[3])
            val = [match[0], match[1]]
            img_matches[key] = val
            key = (match[3], match[2])
            val = [match[1], match[0]]
            img_matches[key] = val
    else:
        for match in keypoints:
            key = (match[4], match[5])
            val = [match[0], match[1]]
            img_matches[key] = val
            key = (match[5], match[4])
            val = [match[1], match[0]]
            img_matches[key] = val
    return img_matches
