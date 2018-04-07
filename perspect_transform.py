import pickle
import cv2
import numpy as np

# Read in the saved camera matrix and distortion coefficients
# These are the arrays you calculated using cv2.calibrateCamera()
dist_pickle = pickle.load( open( "output_images/wide_dist_pickle.p", "rb" ) )
mtx = dist_pickle["mtx"]
dist = dist_pickle["dist"]

# Read in an image
img = cv2.imread('test_images/straight_lines1.jpg')
img_size = (img.shape[1], img.shape[0])

# MODIFY THIS FUNCTION TO GENERATE OUTPUT 
# THAT LOOKS LIKE THE IMAGE ABOVE
def corners_unwarp(img, nx, ny, mtx, dist):
    # Pass in your image into this function
    # Write code to do the following steps
    # 1) Undistort using mtx and dist
    undist = cv2.undistort(img, mtx, dist, None, mtx)

src = np.float32(
    [[255,  677],
     [600,  445],
     [680,  445],
     [1054, 677]])
print(src)

dst = np.float32(
    [[(img_size[0] / 4), 0],
    [(img_size[0] / 4), img_size[1]],
    [(img_size[0] * 3 / 4), img_size[1]],
    [(img_size[0] * 3 / 4), 0]])

print(dst)
cv2.polylines(img,np.int32([src]),True,(0,255,255))

M = cv2.getPerspectiveTransform(src, dst)
warped = cv2.warpPerspective(img, M, img_size, flags=cv2.INTER_LINEAR)

cv2.imshow('With Trapezoid',img)
cv2.waitKey()
cv2.imshow('Warped',warped)
cv2.waitKey()

cv2.imwrite('output_images/straight_lines1_with_trapezoid.jpg',img)
cv2.imwrite('output_images/straight_lines1_warped_trapezoid.jpg',warped)

cv2.destroyAllWindows()