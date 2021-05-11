# SFM from CAD Model
 3D reconstruction of a 3D printed CAD model.

main.py

- Loads image paths from a text file of images paths. 
- Initializes camera calibration data
     - For OpenMVG images I posted, calibration data is only K matrix in K.txt
     - No distortion parameters
- Calculate baseline views and triangulate 3D points from baseline
   - return these in a WorldPointsSet object
- Run main SFM loop
- SFM Loop
  - Load image/image features if they have already been generated. If not, generate features using SIFT
  - If view has not had it's 3D points extracted ->update_3D_points(view, ...)
  - Update_3d_points:
    - Compute pose using OpenCV's PnP (code for this in utils.py)
    - If view is in a completed view's tracked points (meaning their 2D point correspondences have been generated), then remove outliers between point correspondences
    - Triangulate points between view and completed views
    - Store 3D points in View object as well as World Points Set object
  - completed_views is a list containing pointers to View objects that have been reconstructed
  - Once 3D points are appended to their respective source views and the world point set, bundle adjustment is applied.

baseline.py

- Feature match the two baseline views 
- Calculate fundamental matrix
- Calculate essential matrix
- Set pose for view1 to be R = I_3 and t = (0, 0, 0)
- Calculate 4 possible poses for view2 using camera_pose_extraction(essential_matrix) found in utils.py
- Triangulate 3D points between view1 and four possible poses of view2
- Disambiguate poses by finding the pose that minimizes reprojection error
- Add 3D points to World Points Set object


view.py
- Contains class for View object
- Features for the image are extracted here (unless features are for baseline views which are extracted in baseline.py)
- Rotation matrix stored in view as (3 x 3). During BA, this is converted to Euler angles vector (3 x 1) using Rodrigues.
- WorldPoints are stored in each View like so: {(view1_kp, view2_kp):3d_point,..., (view1_kp, view11_kp):3d_point}
- TrackedPoints are matched points between two images. They are stored for example as view1.tracked_pts = {'view2_id':(view1 kps, view2kps), 'view3_id':(view1kps, view3kps), ..., 'viewn_id':(view1kps, viewnkps)}
   - Each View has an ID associated to it in the form of a hashed string. TrackedPoints takes another View's ID as a key and assigns a tuple containing keypoint matches between the two views.
   - These keypoint matches are used for computing pose via PnP and later filtered using a fundamental matrix between the two views prior to triangulation.

WorldPoints.py

- Stores 3D points, the Views/images that created those 3D points, and the 2D keypoints from the two Views that created it

bundle_adjustment.py

- 

