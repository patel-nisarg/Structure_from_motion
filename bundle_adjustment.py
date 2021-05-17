import logging
import time

import cv2 as cv
import numpy as np
from scipy.sparse import lil_matrix
from scipy.optimize import least_squares


# Reference: https://scipy-cookbook.readthedocs.io/items/bundle_adjustment.html

def rotationMatrixToEulerAngles(R):
    sy = np.sqrt(R[0, 0] * R[0, 0] + R[1, 0] * R[1, 0])

    singular = sy < 1e-6

    if not singular:
        x = np.arctan2(R[2, 1], R[2, 2])
        y = np.arctan2(-R[2, 0], sy)
        z = np.arctan2(R[1, 0], R[0, 0])
    else:
        x = np.arctan2(-R[1, 2], R[1, 1])
        y = np.arctan2(-R[2, 0], sy)
        z = 0

    return np.array([x, y, z])


def rotate_points(points, rot_vecs):
    theta = np.linalg.norm(rot_vecs, axis=1)[:, np.newaxis]
    with np.errstate(invalid='ignore'):
        v = rot_vecs / theta
        v = np.nan_to_num(v)
    dot = np.sum(points * v, axis=1)[:, np.newaxis]
    cos_theta = np.cos(theta)
    sin_theta = np.sin(theta)
    return cos_theta * points + sin_theta * np.cross(v, points) + dot * (1 - cos_theta) * v


def project(points, camera_params):
    points_proj = rotate_points(points, camera_params[:, :3])
    points_proj += camera_params[:, 3:6]
    # print(points_proj)
    points_proj = -points_proj[:, :2] / points_proj[:, 2, np.newaxis]
    f = camera_params[:, 6]
    k1 = camera_params[:, 7]
    k2 = camera_params[:, 8]
    n = np.sum(points_proj ** 2, axis=1)
    r = 1 + k1 * n + k2 * n ** 2
    points_proj *= (r * f)[:, np.newaxis]
    return points_proj


def fun(params, n_cameras, n_points, camera_indices, point_indices, points_2d):
    # returns difference between projected 2D points and actual 2D points
    # different procedure here for baseline views
    camera_params = params[:n_cameras * 9].reshape((n_cameras, 9))
    points_3d = params[n_cameras * 9:].reshape((n_points, 3))
    points_proj = project(points_3d[point_indices], camera_params[camera_indices])
    return (points_proj - points_2d).ravel()


def bundle_adjustment_sparsity(n_cameras, n_points, camera_indices, point_indices):
    m = camera_indices.size * 2
    n = n_cameras * 9 + n_points * 3
    A = lil_matrix((m, n), dtype=int)

    i = np.arange(camera_indices.size)
    for s in range(9):
        A[2 * i, camera_indices * 9 + s] = 1
        A[2 * i + 1, camera_indices * 9 + s] = 1

    for s in range(3):
        A[2 * i, n_cameras * 9 + point_indices * 3 + s] = 1
        A[2 * i + 1, n_cameras * 9 + point_indices * 3 + s] = 1
    return A


class BundleAdjustment:
    def __init__(self, wpSet, K, dist, completed_views):
        self.completed_views = completed_views
        self.points_3d = wpSet.world_points
        self.points_2d = []
        self.point_indices = []
        self.camera_indices = []
        self.view_idx = {}
        self.camera_params = []
        self.focal_len = (K[0, 0] + K[1, 1]) / 2
        self.dist = dist[0][:2]
        self.correspondences = wpSet.correspondences
        self.n_cameras = None
        self.n_points = None

    def view2idx(self):
        """
        Takes in a list of views and converts them to indices. For each 2D
        point, a view index is assigned."""
        for view in self.completed_views:
            if view.id not in self.view_idx:
                self.view_idx[view.id] = len(self.view_idx)

            rot_vec, _ = cv.Rodrigues(view.rotation)
            params = np.concatenate((rot_vec, view.translation.reshape((1, 3)), self.focal_len, self.dist), axis=None).tolist()
            # print(view.name, params)
            self.camera_params.append(params)

        self.camera_params = np.array(self.camera_params)
        for i, row in self.correspondences.iterrows():
            self.points_2d.append(row['FeatureIndex'][0])
            self.camera_indices.append(self.view_idx[row['ViewId'][0]])

            self.points_2d.append(row['FeatureIndex'][1])
            self.camera_indices.append(self.view_idx[row['ViewId'][1]])

            self.point_indices.append(i)
            self.point_indices.append(i)

        self.camera_indices = np.array(self.camera_indices)
        self.point_indices = np.array(self.point_indices)
        self.points_2d = np.array(self.points_2d)
        self.points_3d = np.array(self.points_3d)
        self.n_points = self.points_3d.shape[0]
        self.n_cameras = self.camera_params.shape[0]
        logging.info(f"Number of views processed: {self.n_cameras}.")
        logging.info(f"Number of 3D points processed: {self.n_points}.")
        np.savez('optimize_data', camera_params=self.camera_params, points_3d=self.points_3d,
                 camera_indices=self.camera_indices, point_indices=self.point_indices, points_2d=self.points_2d)

    def optimize(self):
        self.view2idx()
        x0 = np.hstack((self.camera_params.ravel(), self.points_3d.ravel()))
        print(len(self.camera_params.ravel()), len(self.points_3d.ravel()))
        f0 = fun(x0, self.n_cameras, self.n_points, self.camera_indices, self.point_indices, self.points_2d)
        A = bundle_adjustment_sparsity(self.n_cameras, self.n_points, self.camera_indices, self.point_indices)
        t0 = time.time()
        res = least_squares(fun, x0, jac_sparsity=A, verbose=2, x_scale='jac', ftol=1e-5, method='trf',
                            args=(self.n_cameras, self.n_points, self.camera_indices, self.point_indices,
                                  self.points_2d))
        t1 = time.time()
        logging.info(f"Optimized {self.n_points} in {t1-t0} seconds.")
        print(res)
        print(len(res['x']))
        return res['x'][len(self.camera_params.ravel()):]
