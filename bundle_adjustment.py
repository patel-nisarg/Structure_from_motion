import logging
import time

import cv2 as cv
import numpy as np
from scipy.sparse import lil_matrix
from scipy.optimize import least_squares


# Reference: https://scipy-cookbook.readthedocs.io/items/bundle_adjustment.html

def project(points, camera_params, K, dist=np.array([])):
    points_proj = []
    for idx in range(len(camera_params)):  # idx applies to both points and cam_params, they are = length vectors
        R = camera_params[idx][:9].reshape(3, 3)
        rvec, _ = cv.Rodrigues(R)
        t = camera_params[idx][9:]
        pt = points[idx]
        pt = np.expand_dims(pt, axis=0)
        pt, _ = cv.projectPoints(pt, rvec, t, K, distCoeffs=dist)
        pt = np.squeeze(np.array(pt))
        points_proj.append(pt)
    return np.array(points_proj)


def fun(params, n_cameras, n_points, camera_indices, point_indices, points_2d, K):
    # returns difference between projected 2D points and actual 2D points
    # different procedure here for baseline views
    camera_params = params[:n_cameras * 12].reshape((n_cameras, 12))
    points_3d = params[n_cameras * 12:].reshape((n_points, 3))
    points_proj = project(points_3d[point_indices], camera_params[camera_indices], K)
    return (points_proj - points_2d).ravel()


def bundle_adjustment_sparsity(n_cameras, n_points, camera_indices, point_indices):
    m = camera_indices.size * 2
    n = n_cameras * 12 + n_points * 3
    A = lil_matrix((m, n), dtype=int)

    i = np.arange(camera_indices.size)
    for s in range(12):
        A[2 * i, camera_indices * 12 + s] = 1
        A[2 * i + 1, camera_indices * 12 + s] = 1

    for s in range(3):
        A[2 * i, n_cameras * 12 + point_indices * 3 + s] = 1
        A[2 * i + 1, n_cameras * 12 + point_indices * 3 + s] = 1
    return A


class BundleAdjustment:
    def __init__(self, wpSet, K, dist, completed_views):
        self.completed_views = completed_views
        self.wpSet = wpSet
        self.points_3d = wpSet.world_points
        self.points_2d = []
        self.point_indices = []
        self.camera_indices = []
        self.view_idx = {}
        self.camera_params = []
        self.focal_len = (K[0, 0] + K[1, 1]) / 2
        self.dist = dist[0][:2]
        self.K = K
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

            rot_vec = np.squeeze(view.rotation)  # 1 x 9
            params = np.concatenate((rot_vec, view.translation.reshape((1, 3))), axis=None).tolist()
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
        fun(x0, self.n_cameras, self.n_points, self.camera_indices, self.point_indices, self.points_2d, self.K)
        A = bundle_adjustment_sparsity(self.n_cameras, self.n_points, self.camera_indices, self.point_indices)
        t0 = time.time()
        res = least_squares(fun, x0, jac_sparsity=A, verbose=2, x_scale='jac', ftol=1e-4, method='trf', xtol=1e-12,
                            args=(self.n_cameras, self.n_points, self.camera_indices, self.point_indices,
                                  self.points_2d, self.K))
        t1 = time.time()
        logging.info(f"Optimized {self.n_points} in {t1-t0} seconds.")

        points_3d = res.x[self.n_cameras * 12:].reshape(self.n_points, 3)
        poses = res.x[:self.n_cameras * 12].reshape(self.n_cameras, 12)

        return poses, points_3d
