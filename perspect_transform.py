import pickle
import cv2
import numpy as np

# MODIFY THIS FUNCTION TO GENERATE OUTPUT 
# THAT LOOKS LIKE THE IMAGE ABOVE
def cal_undistort(img, mtx, dist):
    # Use cv2.calibrateCamera() and cv2.undistort()  
    undist = cv2.undistort(img, mtx, dist, None, mtx)
    
    # crop the image
    #x,y,w,h = roi
    #undist = undist[y:y+h, x:x+w]
    #undist = np.copy(img)  # Delete this line
    return undist

# Read in the saved camera matrix and distortion coefficients
# These are the arrays you calculated using cv2.calibrateCamera()
dist_pickle = pickle.load( open( "output_images/wide_dist_pickle.p", "rb" ) )
mtx = dist_pickle["mtx"]
dist = dist_pickle["dist"]

# Read in an image
img = cv2.imread('test_images/straight_lines1.jpg')
img_size = (img.shape[1], img.shape[0])
undistorted = cal_undistort(img, mtx, dist)

src = np.float32(
    [[(img_size[0] / 2) - 65, img_size[1] / 2 + 100],
    [((img_size[0] / 6) - 30), img_size[1]],
    [(img_size[0] * 5 / 6) + 70, img_size[1]],
    [(img_size[0] / 2 + 65), img_size[1] / 2 + 100]])

print(src)

dst = np.float32(
    [[(img_size[0] / 4), 0],
    [(img_size[0] / 4), img_size[1]],
    [(img_size[0] * 3 / 4), img_size[1]],
    [(img_size[0] * 3 / 4), 0]])

print(dst)
cv2.polylines(undistorted,np.int32([src]),True,(0,255,255))
cv2.polylines(img,np.int32([src]),True,(0,255,255))

M = cv2.getPerspectiveTransform(src, dst)
warped = cv2.warpPerspective(undistorted, M, img_size, flags=cv2.INTER_LINEAR)

cv2.imshow('With Trapezoid Original',img)
cv2.waitKey()
cv2.imshow('With Trapezoid Undistored',undistorted)
cv2.waitKey()
cv2.imshow('Warped',warped)
cv2.waitKey()

cv2.imwrite('output_images/straight_lines1_with_trapezoid.jpg',img)
cv2.imwrite('output_images/straight_lines1_warped_trapezoid.jpg',warped)

