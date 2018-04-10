import pickle
import cv2
import numpy as np

def cal_undistort(img, mtx, dist):
    undist = cv2.undistort(img, mtx, dist, None, mtx)
    
    return undist

# Read in the saved camera matrix and distortion coefficients
# These are the arrays you calculated using cv2.calibrateCamera()
dist_pickle = pickle.load( open( "output_images/wide_dist_pickle.p", "rb" ) )
mtx = dist_pickle["mtx"]
dist = dist_pickle["dist"]

# Read in an image
img = cv2.imread('test_images/straight_lines1.jpg')
undistorted = cal_undistort(img, mtx, dist)

img_size = (img.shape[1], img.shape[0])

def warper(img, src, dst):
    M = cv2.getPerspectiveTransform(src, dst)
    result = cv2.warpPerspective(img, M, (img.shape[1], img.shape[0]), flags=cv2.INTER_LINEAR)
    return result


src = np.float32(
    [[(img_size[0] / 2) - 65, img_size[1] / 2 + 100],
    [((img_size[0] / 6) - 30), img_size[1]],
    [(img_size[0] * 5 / 6) + 70, img_size[1]],
    [(img_size[0] / 2 + 65), img_size[1] / 2 + 100]])

dst = np.float32(
    [[(img_size[0] / 4), 0],
    [(img_size[0] / 4), img_size[1]],
    [(img_size[0] * 3 / 4), img_size[1]],
    [(img_size[0] * 3 / 4), 0]])

print(src)
print(dst)

warped = warper(undistorted, src, dst)

cv2.polylines(undistorted,np.int32([src]),True,(0,0,255),2)
cv2.polylines(warped,np.int32([dst]),True,(0,0,255),2)
cv2.imwrite('output_images/straight_lines1_with_trapezoid.jpg',undistorted)
cv2.imwrite('output_images/straight_lines1_warped_trapezoid.jpg',warped)

