import numpy as np
import cv2
import os
import glob
import pickle

CHESSBOARD_X_SQUARES = 9
CHESSBOARD_Y_SQUARES = 6
# prepare object points, like (0,0,0), (1,0,0), (2,0,0) ....,(6,5,0)
objp = np.zeros((CHESSBOARD_Y_SQUARES*CHESSBOARD_X_SQUARES,3), np.float32)
objp[:,:2] = np.mgrid[0:CHESSBOARD_X_SQUARES, 0:CHESSBOARD_Y_SQUARES].T.reshape(-1,2)

# Arrays to store object points and image points from all the images.
objpoints = [] # 3d points in real world space
imgpoints = [] # 2d points in image plane.

# Make a list of calibration images
images = glob.glob('camera_cal/calibration*.jpg')

# Step through the list and search for chessboard corners
for idx, fname in enumerate(images):
    img = cv2.imread(fname)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    # Find the chessboard corners
    ret, corners = cv2.findChessboardCorners(gray, (CHESSBOARD_X_SQUARES,CHESSBOARD_Y_SQUARES), None)

    # If found, add object points, image points
    if ret == True:
        objpoints.append(objp)
        imgpoints.append(corners)

        # Draw and display the corners
        cv2.drawChessboardCorners(img, (CHESSBOARD_X_SQUARES,CHESSBOARD_Y_SQUARES), corners, ret)
        write_name = 'output_images/corners_' + os.path.basename(fname)
        cv2.imwrite(write_name, img)
        cv2.imshow(write_name, img)
        cv2.waitKey()


# Test undistortion on an image
img = cv2.imread('camera_cal/calibration1.jpg')
gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
img_size = (img.shape[1], img.shape[0])

cv2.imshow('Original Image', img)
cv2.waitKey()
cv2.imwrite('output_images/test_dist.jpg',img)

# Do camera calibration given object points and image points
ret, mtx, dist, rvecs, tvecs = cv2.calibrateCamera(objpoints, imgpoints, img_size,None,None)

dst = cv2.undistort(img, mtx, dist, None, mtx)
cv2.imwrite('output_images/test_undist.jpg',dst)

# Save the camera calibration result for later use (we won't worry about rvecs / tvecs)
dist_pickle = {}
dist_pickle["mtx"] = mtx
dist_pickle["dist"] = dist
pickle.dump( dist_pickle, open( "output_images/wide_dist_pickle.p", "wb" ) )

# Visualize undistortion
cv2.imshow('Undistorted Image', dst)
cv2.waitKey()

cv2.destroyAllWindows()
