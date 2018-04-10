import pickle
import cv2
import numpy as np

# Function that takes an image, object points, and image points
# performs the image distortion correction and 
# returns the undistorted image
def cal_undistort(img, mtx, dist):
    return cv2.undistort(img, mtx, dist, None, mtx)

# Read in the saved objpoints and imgpoints
dist_pickle = pickle.load( open( "output_images/wide_dist_pickle.p", "rb" ) )
mtx = dist_pickle["mtx"]
dst = dist_pickle["dist"]

# Read in an image
img = cv2.imread('test_images/test1.jpg')

undistorted = cal_undistort(img, mtx, dst)
cv2.imwrite('output_images/undistorted_test1.jpg', undistorted)

cv2.imshow('Original Image', img)
cv2.imshow('Undistorted Image', undistorted)
cv2.waitKey()

cv2.destroyAllWindows()
