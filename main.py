from WorldPoints import WorldPointSet
from baseline import Baseline
from utils import *
from view import ImageView
from bundle_adjustment import BundleAdjustment
from visualize import PointCloudVisualizer
import numpy as np
import os
from datetime import datetime
import argparse
import sys

BASE_PATH = os.path.join(os.getcwd(), "learned_correspondences")
MODEL_PATH = os.path.join(BASE_PATH, "models")
sys.path.append(BASE_PATH)
sys.path.append(MODEL_PATH)

from learned_correspondences import generate_matches


def run(args):
    """
    Loads in images, finds keypoint features in images. Match keypoint features between image pairs.
    Runs SFM loop to generate poses and 3d points. Visualizes point cloud.
    """
    image_txt_path = args.image_text_path
    image_paths = get_paths_from_txt(image_txt_path)  # list of filenames for sfm.
    
    # File names should be in 'image_001.jpg' format
    features_dir = os.path.join(os.getcwd(), "features")
    
    # Obtain intrinsic matrix(K) and distortion coefficients for camera
    K = load_cal_mat(args.K)
    dist = np.zeros((1, 5))
    
    # Load first two views. Note, ensure the two desired baseline images are at top of image paths text file.
    feature_match_gen = generate_matches.FeatureMatchGenerator(image_txt_path)
    feature_match_gen.create_feature_matches()
    
    global img_matches
    keypoints = np.load('features\\feature_matches_filtered_tr2.npz', allow_pickle=True)['filtered_matches']
    img_matches = keypoints_to_dict(keypoints)

    init_pair = img_matches[('20', '23')]
    view1 = ImageView(image_paths[0], features_path=features_dir)
    view2 = ImageView(image_paths[1], features_path=features_dir)
    
    # Initialize baseline object with its point correspondences.
    baseline = Baseline(view1, view2, K, init_pair)
    
    # establish baseline <- get F matrix, E matrix, primary poses and 3D points from first two views
    wpSet = baseline()
    points_3d = sfm_loop(image_paths, features_dir, baseline, wpSet, K, dist)
    np.savez(os.path.join(os.getcwd(), "\\points_3d"), point_cloud=points_3d)
    
    # Visualize point cloud
    visualizer = PointCloudVisualizer(points_3d)
    visualizer.visualize()


def sfm_loop(sfm_images, features_dir, baseline, wpSet, K, dist):
    """
    Main Structure From Motion loop.
    
    :param sfm_images: Text file of image paths.
    :param features_dir: Path of features directory.
    :param baseline: baseline of first two views.
    :param wpSet: World points data from the baseline.
    :param K: Camera intrinsic matrix.
    :param dist: Camera distortion parameters.
    """

    # Get new view. Perform linear PnP on new view to get pose information.
    view1 = baseline.view1
    view2 = baseline.view2
    completed_views = [view1, view2]

    def update_3d_points(view, completed_views, K):
        """
        Updates 3D points with the current view.
        :param view: View with which to update world 3D coordinates.
        :param completed_views: Views from which 3D points have been triangulated.
        :param K: Camera intrinsic matrix
        """
        view.rotation, view.translation = compute_pose(view, completed_views, K, img_matches)

        for view_n in completed_views:
            if view_n.id in view.tracked_pts:
                # Before triangulating, outliers should be removed using F matrix/RANSAC between x1 and x2.
                x1, x2 = remove_outliers(view, view_n)
                print("Number of points to triangulate:", x1.shape[0])
                X = triangulate_points(K, t1=view.translation, R1=view.rotation,
                                       t2=view_n.translation, R2=view_n.rotation, x1=x1, x2=x2,
                                       print_error=True)
                # add correspondences to world coordinates only if reproj errors are low for img pair
                if X is not None:
                    X = store_3Dpoints_to_views(X, view_n, view, K, error_threshold=2.0)
                    print(f"Found {len(X)} 3D points between image{view_n.name} and image{view.name}.")
                    view.reproject_view(K, print_error=True)
                    wpSet.add_correspondences(X, view, view_n)
        print(f"Found {len(view.world_points)} 3D points for new image{view.name}.")

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
        wpSet.correspondences.to_csv(f'points\\point_correspondences_{i + 1}.csv')
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

    wpSet.correspondences.to_csv('\\point_correspondences.csv')
    np.savetxt("points_3d.csv", wpSet.world_points, delimiter=",")
    return wpSet.world_points


def set_args(parser):
    parser.add_argument('-i', '--image_paths', action='store', type=str, dest='image_text_path',
                        help='File path to *.txt file containing all path of all image files to be reconstructed.')
    parser.add_argument('-K', '--calibration_mat', action='store', type=str, dest='K',
                        help='File path to *.npz or *.txt file containing camera intrinsic matrix.')


if __name__ == "__main__":
    # Generate log file
    log_filename = "logs/" + datetime.now().strftime("%Y-%m-%dT%H_%M_%S") + '_sfm_runtime_log.log'
    logging.basicConfig(level=logging.INFO, filename=log_filename)
    desc = "This module performs 3D reconstruction on a set of images. "
    parser = argparse.ArgumentParser(description=desc)
    set_args(parser)
    args = parser.parse_args()
    run(args)
