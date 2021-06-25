import cv2 as cv
import os
import glob
import numpy as np

# Checkerboard contains 25mm squares - 8x6 vertices, 9x7 squares
# 28mm (equivalent) f/1.8 lens with OIS
# https://www.camerafv5.com/devices/manufacturers/google/pixel_3a_xl_bonito_0/
# 12.2Mp 1/2.55-inch sensor with 1.4µm pixel width

pixel_size = 1.4e-6  # number of pixels in 1 square mm

# termination criteria
criteria = (cv.TermCriteria_EPS + cv.TERM_CRITERIA_MAX_ITER, 30, 0.001)

# object points
checker_square_size = np.mean([23.36, 23.35, 23.32, 23.22, 23.43, 23.20, 23.28])  # size in mm

objp = np.zeros((8 * 6, 3), np.float32)
objp[:, :2] = np.mgrid[0:8, 0:6].T.reshape(-1, 2)
objp = checker_square_size * objp

objpoints = []  # 3D point in world coords
imgpoints = []  # 2D point in image coords

path = "F:/Documents/Python Scripts/Phone Calibration Images/"
os.chdir(path)
images = glob.glob("*.jpg")

patternSize = (8, 6)
flags = None

for i, fname in enumerate(images):
    img = cv.imread(fname)
    gray = cv.cvtColor(img, cv.COLOR_BGR2GRAY)

    ret, corners = cv.findChessboardCorners(gray, patternSize, flags)

    print(f'Completed {i + 1} images.')

    if ret is True:
        objpoints.append(objp)

        corners2 = cv.cornerSubPix(gray, corners, winSize=(11, 11), zeroZone=(-1, -1), criteria=criteria)
        imgpoints.append(corners)

        cv.drawChessboardCorners(img, patternSize, corners2, ret)
        # cv.imshow('img', img)
        cv.waitKey(100)

cv.destroyAllWindows()
os.chdir("F:/Documents/Python Scripts/")

ret, mtx, dist, rvecs, tvecs = cv.calibrateCamera(objpoints, imgpoints, gray.shape[::-1], None, None)
# mtx[:2, :] = mtx[:2, :] * (pixel_size * 1e3)  # Converts focal length to mm

mean_error = 0
for i in range(len(objpoints)):
    imgpoints2, _ = cv.projectPoints(objpoints[i], rvecs[i], tvecs[i], mtx, dist)
    error = cv.norm(imgpoints[i], imgpoints2, cv.NORM_L2)/len(imgpoints2)
    mean_error += error

total_reproj_err = mean_error/len(objpoints)
print("total error: {}".format(total_reproj_err))

np.savez('calibration_data', ret=ret, calibratoin_matrix=mtx, distortion_params=dist,
         rotation_vecs=rvecs, translation_vecs=tvecs, reprojection_error=total_reproj_err)

cv.namedWindow("original", cv.WINDOW_NORMAL)        # Create window with freedom of dimensions
im = cv.imread("PXL_20210407_172636685.jpg")        # Read image
imS = cv.resize(im, (1240, 1080))                   # Resize image
cv.imshow("original", imS)                          # Show image
cv.waitKey(100)                                     # Display the image infinitely until any keypress

img = cv.imread('PXL_20210407_172636685.jpg')
h, w = img.shape[:2]
newcameramtx, roi = cv.getOptimalNewCameraMatrix(mtx, dist, (w, h), 1, (w, h))

dst = cv.undistort(img, mtx, dist, None, newcameramtx)
x, y, w, h = roi
dst = dst[y:y+h, x:x+w]
cv.imwrite('calibresult.png', dst)
