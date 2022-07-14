#!/usr/bin/env python

# Camera calibration tool used to find camera distortion matrix
# Saves distortion matrix values in binary numpy file located in /vars/cam_dist_matrix.npz
# Saved values can be used to remove camera lense distortion
# Trains on checkerboard images obtained from get_image.py

import cv2
import numpy as np
import glob

# Define the dimensions of checkerboard
CHECKERBOARD = (8, 6)  # 8 Rows, 10 Columns

# Termination Criteria
criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)

# Prepare object points, like (0,0,0), (1,0,0), (2,0,0)....,(6,5,0)
objp = np.zeros((CHECKERBOARD[1]*CHECKERBOARD[0], 3), np.float32)
objp[:, :2] = np.mgrid[0:CHECKERBOARD[0], 0:CHECKERBOARD[1]].T.reshape(-1, 2)

# Creating vector to store vectors of 3D points for each checkerboard image
objpoints = []

# Creating vector to store vectors of 2D points for each checkerboard image
imgpoints = []

# Extracting path of individual image stored in a given directory
images = glob.glob('./images/camera_calibration/*.png')

cornercnt = 0

for fname in images:
    img = cv2.imread(fname)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    # Find the chess board corners
    # If desired number of corners are found in the image then ret = true
    ret, corners = cv2.findChessboardCorners(gray, CHECKERBOARD, None)

    """
    If desired number of corner are detected,
    we refine the pixel coordinates and display 
    them on the images of checker board
    """
    if ret == True:
        cornercnt += 1
        print("Corner Detected: ", cornercnt)
        objpoints.append(objp)
        # refining pixel coordinates for given 2d points.
        corners2 = cv2.cornerSubPix(gray, corners, (11, 11), (-1, -1), criteria)

        imgpoints.append(corners2)

        # Draw and display the corners
        img = cv2.drawChessboardCorners(img, CHECKERBOARD, corners2, ret)

    cv2.imshow('img', img)
    cv2.waitKey(50)

cv2.destroyAllWindows()

"""
Performing camera calibration by
passing the value of known 3D points (objpoints)
and corresponding pixel coordinates of the
detected corners (imgpoints)
"""
ret, mtx, dist, rvecs, tvecs = cv2.calibrateCamera(objpoints, imgpoints, gray.shape[::-1], None, None)
img = cv2.imread('./images/camera_calibration/c1.png')  # any image can be used, c1.png happened to be an easy test case
h, w = img.shape[:2]
newcameramtx, roi = cv2.getOptimalNewCameraMatrix(mtx, dist, (w, h), 0.97, (w, h))

# Save the distortion matrix to an uncompressed binary numpy file
np.savez('vars/cam_dist_matrix.npz', mtx=mtx, dist=dist, newcameramtx=newcameramtx)

# Undistort the image
dst = cv2.undistort(img, mtx, dist, None, newcameramtx)
cv2.imwrite('./images/camera_calibration/result/undistorted.png', dst)
cv2.imshow('undistort_img', dst)
if cv2.waitKey(5000) & 0xFF == ord('q'):    # Display image for 5 seconds, close anytime using 'q'
    cv2.destroyAllWindows()

# Crop the image
x, y, w, h = roi
dst = dst[y:y+h, x:x+w]
cv2.imwrite('./images/camera_calibration/result/calibresult.png', dst)

# Re-projection Error
mean_error = 0
for i in range(len(objpoints)):
    imgpoints2, _ = cv2.projectPoints(objpoints[i], rvecs[i], tvecs[i], mtx, dist)
    error = cv2.norm(imgpoints[i], imgpoints2, cv2.NORM_L2)/len(imgpoints2)
    mean_error += error

print("total error: {}".format(mean_error/len(objpoints)))
