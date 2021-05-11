# SFM_from_CAD_Model
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
