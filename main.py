from WorldPoints import WorldPointSet
from baseline import Baseline
from utils import *
from view import ImageView
import numpy as np
import os
from datetime import datetime


def run():
    image_paths = get_paths_from_txt("sfm_image_paths.txt")  # list of filenames for sfm.
    # File names should be in 'image_001.jpg' format
    features_dir = os.path.join(os.getcwd(), "features/")
    # Obtain K matrix.
    K = np.load('calibration_data.npz')['calibratoin_matrix']
    # Load first two views. Note, ensure the two desired baseline images are at top of image paths text file.
    view1 = ImageView(image_paths[0], features_path=features_dir)
    view2 = ImageView(image_paths[1], features_path=features_dir)
    # Initialize baseline object with its point correspondences.
    baseline = Baseline(view1, view2, K)
    # Feature matching two views
    baseline.feature_match_baseline()
    # Calculate fundamental matrix
    baseline.calc_fundamental_matrix(save=True)
    # Calculate essential matrixZ
    baseline.calc_essential_matrix(save=True)
    # Get four camera poses for view2
    view1.rotation = np.eye(3)
    view1.position = np.zeros((3, 1))

    C2, R2 = camera_pose_extraction(baseline.essential_mat)
    # Disambiguate poses
    X_4 = []
    for i in range(len(R2)):
        C_2 = np.expand_dims(C2[i], axis=1)
        X_4.append(linear_triangulation(K, view1.position, view1.rotation,
                                        C_2, R2[i], view1.keypoints, view2.keypoints))

    X = pose_disambiguation(C2, R2, X_4)
    wpSet = WorldPointSet()
    wpSet.add_correspondences()

    #world_coords, camera_poses = sfm_loop(image_paths, features_dir, baseline)

    # Visualize point cloud
    #visualize_pt_cloud(world_coords)


def sfm_loop(sfm_images, features_dir, baseline):
    """
    Main Structure From Motion loop.
    
    :param baseline: baseline of first two views.
    :param sfm_images: Text file of image paths.
    :param features_dir: Path of features directory.
    :return: world_coords: 3D world inhomogeneous coordinates.
    :return: camera_poses: Camera poses in P = [R|t] format as a (3x4) matrix.
    """
    # Get new view. Perform linear PnP on new view to get pose information.

    world_coords = baseline.view1.get_3D_correspondences()
    camera_poses = []
    view1 = baseline.view1
    view2 = baseline.view2

    completed_views = [view1, view2]

    def update_3d_points(view, completed_views):
        """
        Updates 3D points with the current view.
        :param view: View with which to update world 3D coordinates.
        """
        view.position, view.rotation = compute_pose(view, completed_views)
        for view_n in completed_views:
            x1, x2 = view.tracked_pts[view_n.id]
            X = linear_triangulation(baseline.K, C1=view.position, R1=view.rotation,
                                     C2=view_n.position, R2=view_n.rotation, x1=x1, x2=x2)
            for i, kp in enumerate(x1):
                view.world_points[(kp)] = X[i]
                # add correspondences to world coordinates

    for image in sfm_images[2:]:
        # extract features of a view
        view = ImageView(image)
        view.read_features()

        if view.descriptors is None:
            view.extract_features(write_to_file=True)

        if view.id not in completed_views:
            update_3d_points(view, completed_views)

        camera_poses.append(view.get_pose())
        completed_views.append(view)
        # Perform bundle adjustment on new view and existing views -> Update 3D points dictionary

    return world_coords, camera_poses


if __name__ == "__main__":
    log_filename = "logs/" + datetime.now().strftime("%Y-%m-%dT%H_%M_%S") + '_sfm_runtime_log.log'
    logging.basicConfig(level=logging.INFO, filename=log_filename)
    run()
