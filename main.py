from WorldPoints import WorldPointSet
from baseline import Baseline
from utils import *
from view import ImageView
from bundle_adjustment import BundleAdjustment
import numpy as np
import os
from datetime import datetime


def run():
    image_paths = get_paths_from_txt("temple_ring_img_pths.txt")  # list of filenames for sfm.
    # File names should be in 'image_001.jpg' format
    features_dir = os.path.join(os.getcwd(), "features/")
    # Obtain intrinsic matrix(K) and distortion coefficients for camera
    camera_data = np.load('calibration_data.npz')
    # K, dist = camera_data['calibratoin_matrix'], camera_data['distortion_params']
    # K = np.loadtxt('K.txt')
    K = np.matrix('1520.40 0.00 302.32; 0.00 1525.90 246.87; 0.00 0.00 1.00')
    dist = np.zeros((1, 5))
    # Load first two views. Note, ensure the two desired baseline images are at top of image paths text file.
    view1 = ImageView(image_paths[0], features_path=features_dir)
    view2 = ImageView(image_paths[1], features_path=features_dir)
    # Initialize baseline object with its point correspondences.
    global img_matches
    keypoints = np.load('features\\feature_matches_filtered_tr2.npz', allow_pickle=True)['filtered_matches']
    img_matches = keypoints_to_dict(keypoints)
    global unfiltered_img_matches
    unfiltered_kps = np.load('features\\feature_matches_unfiltered_tr2.npz', allow_pickle=True)['feature_matches']
    unfiltered_img_matches = keypoints_to_dict(unfiltered_kps, filtered=False)
    init_pair = img_matches[('20', '23')]
    baseline = Baseline(view1, view2, K, init_pair)
    # establish baseline -> get F matrix, E matrix, primary poses and 3D points from first two views
    wpSet = baseline()
    points_3d = sfm_loop(image_paths, features_dir, baseline, wpSet, K, dist)
    print(points_3d.shape)
    np.savez('C:\\Users\\Nisarg\\Desktop\\points_3d', point_cloud=points_3d)
    # Visualize point cloud
    # visualize_pt_cloud(world_coords)


def sfm_loop(sfm_images, features_dir, baseline, wpSet, K, dist):
    """
    Main Structure From Motion loop.
    
    :param dist: Camera distortion parameters.
    :param K: Camera intrinsic matrix.
    :param wpSet:
    :param baseline: baseline of first two views.
    :param sfm_images: Text file of image paths.
    :param features_dir: Path of features directory.
    :return: world_coords: 3D world inhomogeneous coordinates.
    :return: camera_poses: Camera poses in P = [R|t] format as a (3x4) matrix.
    """
    # Get new view. Perform linear PnP on new view to get pose information.
    view1 = baseline.view1
    view2 = baseline.view2
    completed_views = [view1, view2]

    def update_3d_points(view, completed_views, K, dist):
        """
        Updates 3D points with the current view.
        :param dist: Camera distortion parameters
        :param K: Camera intrinsic matrix
        :param completed_views: Views from which 3D points have been triangulated.
        :param view: View with which to update world 3D coordinates.
        """
        view.rotation, view.translation = compute_pose(view, completed_views, K, dist, unfiltered_img_matches)

        # if view.rotation is None:
        #     view.rotation, view.translation = compute_pose(view, completed_views, K, dist, unfiltered_img_matches)
        for view_n in completed_views:
            if view_n.id in view.tracked_pts:
                # Before triangulating, remove outliers using F matrix between x1 and x2.
                x1, x2 = remove_outliers(view, view_n)
                print("Number of points to triangulate:", x1.shape[0])
                X = triangulate_points(K, t1=view.translation, R1=view.rotation,
                                       t2=view_n.translation, R2=view_n.rotation, x1=x1, x2=x2,
                                       print_error=True)
                # print('3d points', X)
                # add correspondences to world coordinates
                if X is not None:
                    X = store_3Dpoints_to_views(X, view_n, view, K, error_threshold=2.0)
                    print("Points triangulated: ", X.shape[0])
                    wpSet.add_correspondences(X, view, view_n)  # change WorldPoints.py to skip existing 3D pts
                    return True

    for i, image in enumerate(sfm_images[2:]):
        # extract features of a view
        view = ImageView(image, features_dir)
        view.read_features()
        if view.descriptors is None:
            view.extract_features(write_to_file=True)
        if view not in completed_views:
            update_3d_points(view, completed_views, K, dist)
            completed_views.append(view)
        # Perform bundle adjustment on new view and existing views -> Update 3D points dictionary
        ba = BundleAdjustment(wpSet, K, dist, completed_views)
        poses, wpSet.world_points = ba.optimize()
        for j, view in enumerate(completed_views):
            rot_mat = (poses[j, :9]).reshape(3, 3)
            t_vec = poses[j, 9:].reshape(3, 1)
            view.rotation = rot_mat
            view.translation = t_vec
            view.update_world_points(wpSet)
            view.reproject_view(K, print_error=True)
        np.savez(f'points\\points3d_{i}', point_cloud=wpSet.world_points)

    wpSet.correspondences.to_csv('\point_correspondences.csv')
    np.savetxt("points_3d.csv", wpSet.world_points, delimiter=",")
    return wpSet.world_points


if __name__ == "__main__":
    log_filename = "logs/" + datetime.now().strftime("%Y-%m-%dT%H_%M_%S") + '_sfm_runtime_log.log'
    logging.basicConfig(level=logging.INFO, filename=log_filename)
    run()
