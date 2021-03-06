import numpy as np
import cv2
import pickle
from moviepy.editor import VideoFileClip


def cal_undistort(img, mtx, dist):
    undist = cv2.undistort(img, mtx, dist, None, mtx)
    return undist

# Read in the saved camera matrix and distortion coefficients
# These are the arrays you calculated using cv2.calibrateCamera()
dist_pickle = pickle.load( open( "output_images/wide_dist_pickle.p", "rb" ) )
mtx = dist_pickle["mtx"]
dist = dist_pickle["dist"]

img_size = (1280, 720)

src = np.float32(
    [[(img_size[0] / 2) - 70, img_size[1] / 2 + 100],
    [((img_size[0] / 6) - 30), img_size[1]],
    [(img_size[0] * 5 / 6) + 70, img_size[1]],
    [(img_size[0] / 2 + 70), img_size[1] / 2 + 100]])

dst = np.float32(
    [[(img_size[0] / 4), 0],
    [(img_size[0] / 4), img_size[1]],
    [(img_size[0] * 3 / 4), img_size[1]],
    [(img_size[0] * 3 / 4), 0]])

M = cv2.getPerspectiveTransform(src, dst)
Minv = cv2.getPerspectiveTransform(dst, src)

# Threshold x gradient
thresh_min = 35
thresh_max = 120

# Threshold color channel
s_thresh_min = 180
s_thresh_max = 255

# Set the width of the windows +/- margin
margin = 45
# Set minimum number of pixels found to recenter window
minpix = 50

# Write some Text

font                    = cv2.FONT_HERSHEY_SIMPLEX
bottomLeftCornerOfLine1 = (10,50)
bottomLeftCornerOfLine2 = (10,80)
fontScale               = 1
fontColor               = (255,255,255)
lineType                = 2

#i = 0
RESULT_CACHE_SIZE = 5
left_cache = []
right_cache = []
left_cr_cache = []
right_cr_cache = []
def process_image(img):
    #global i
    #cv2.imwrite('input_images/test%d.jpg' % (i),img)

    # Convert to HLS color space and separate the S channel
    # Note: img is the undistorted image
    hls = cv2.cvtColor(img, cv2.COLOR_RGB2HLS)
    l_channel = hls[:,:,1]
    s_channel = hls[:,:,2]

    # Sobel x
    sobelx = cv2.Sobel(l_channel, cv2.CV_64F, 1, 0) # Take the derivative in x
    abs_sobelx = np.absolute(sobelx) # Absolute x derivative to accentuate lines away from horizontal
    scaled_sobel = np.uint8(255*abs_sobelx/np.max(abs_sobelx))

    sxbinary = np.zeros_like(scaled_sobel)
    sxbinary[(scaled_sobel >= thresh_min) & (scaled_sobel <= thresh_max)] = 1

    s_binary = np.zeros_like(s_channel)
    s_binary[(s_channel >= s_thresh_min) & (s_channel <= s_thresh_max)] = 1

    # Combine the two binary thresholds
    combined = np.zeros_like(sxbinary)
    combined[(s_binary == 1) | (sxbinary == 1)] = 1

    undistorted = cal_undistort(combined, mtx, dist)
    warped = cv2.warpPerspective(undistorted, M, img_size, flags=cv2.INTER_LINEAR)

    histogram = np.sum(warped[warped.shape[0]//2:,:], axis=0)
    # Create an output image to draw on and  visualize the result
    out_img = np.dstack((warped, warped, warped))*255
    # Find the peak of the left and right halves of the histogram
    # These will be the starting point for the left and right lines
    midpoint = np.int(histogram.shape[0]//2)
    quarterpoint = np.int(histogram.shape[0]//4)
    octopoint = np.int(histogram.shape[0]//8)
    leftx_base = np.argmax(histogram[quarterpoint:midpoint]) + quarterpoint
    rightx_base = np.argmax(histogram[midpoint + octopoint:2*midpoint - octopoint]) + midpoint + octopoint

    # Choose the number of sliding windows
    nwindows = 9
    # Set height of windows
    window_height = np.int(warped.shape[0]//nwindows)
    # Identify the x and y positions of all nonzero pixels in the image
    nonzero = warped.nonzero()
    nonzeroy = np.array(nonzero[0])
    nonzerox = np.array(nonzero[1])
    # Current positions to be updated for each window
    leftx_current = leftx_base
    rightx_current = rightx_base
    # Create empty lists to receive left and right lane pixel indices
    left_lane_inds = []
    right_lane_inds = []

    # Step through the windows one by one
    for window in range(nwindows):
        # Identify window boundaries in x and y (and right and left)
        win_y_low = warped.shape[0] - (window+1)*window_height
        win_y_high = warped.shape[0] - window*window_height
        win_xleft_low = leftx_current - margin
        win_xleft_high = leftx_current + margin
        win_xright_low = rightx_current - margin
        win_xright_high = rightx_current + margin

        # Identify the nonzero pixels in x and y within the window
        good_left_inds = ((nonzeroy >= win_y_low) & (nonzeroy < win_y_high) & 
        (nonzerox >= win_xleft_low) &  (nonzerox < win_xleft_high)).nonzero()[0]
        good_right_inds = ((nonzeroy >= win_y_low) & (nonzeroy < win_y_high) & 
        (nonzerox >= win_xright_low) &  (nonzerox < win_xright_high)).nonzero()[0]
        # Append these indices to the lists
        left_lane_inds.append(good_left_inds)
        right_lane_inds.append(good_right_inds)
        # If you found > minpix pixels, recenter next window on their mean position
        if len(good_left_inds) > minpix:
            leftx_current = np.int(np.mean(nonzerox[good_left_inds]))
        if len(good_right_inds) > minpix:        
            rightx_current = np.int(np.mean(nonzerox[good_right_inds]))

    # Concatenate the arrays of indices
    left_lane_inds = np.concatenate(left_lane_inds)
    right_lane_inds = np.concatenate(right_lane_inds)

    # Extract left and right line pixel positions
    leftx = nonzerox[left_lane_inds]
    lefty = nonzeroy[left_lane_inds] 
    rightx = nonzerox[right_lane_inds]
    righty = nonzeroy[right_lane_inds] 

    # Fit a second order polynomial to each
    left_fit = np.polyfit(lefty, leftx, 2)
    right_fit = np.polyfit(righty, rightx, 2)

    left_cache.append(left_fit)
    right_cache.append(right_fit)
    subset = left_cache[-RESULT_CACHE_SIZE:]
    left_fit[0] = np.sum( x[0] for x in subset ) / len(subset)
    left_fit[1] = np.sum( x[1] for x in subset ) / len(subset)
    left_fit[2] = np.sum( x[2] for x in subset ) / len(subset)

    subset = right_cache[-RESULT_CACHE_SIZE:]
    right_fit[0] = np.sum( x[0] for x in subset ) / len(subset)
    right_fit[1] = np.sum( x[1] for x in subset ) / len(subset)
    right_fit[2] = np.sum( x[2] for x in subset ) / len(subset)

    # Generate x and y values for plotting
    ploty = np.linspace(0, warped.shape[0]-1, warped.shape[0] )
    left_fitx = left_fit[0]*ploty**2 + left_fit[1]*ploty + left_fit[2]
    right_fitx = right_fit[0]*ploty**2 + right_fit[1]*ploty + right_fit[2]

    out_img[nonzeroy[left_lane_inds], nonzerox[left_lane_inds]] = [255, 0, 0]
    out_img[nonzeroy[right_lane_inds], nonzerox[right_lane_inds]] = [0, 0, 255]

    unwarped = cv2.warpPerspective(out_img, Minv, img_size, flags=cv2.INTER_LINEAR)

    # Create an image to draw the lines on
    warp_zero = np.zeros_like(warped).astype(np.uint8)
    color_warp = np.dstack((warp_zero, warp_zero, warp_zero))

    # Recast the x and y points into usable format for cv2.fillPoly()
    pts_left = np.array([np.transpose(np.vstack([left_fitx, ploty]))])
    pts_right = np.array([np.flipud(np.transpose(np.vstack([right_fitx, ploty])))])
    pts = np.hstack((pts_left, pts_right))

    # Draw the lane onto the warped blank image
    cv2.fillPoly(color_warp, np.int_([pts]), (0,255, 0))

    # Warp the blank back to original image space using inverse perspective matrix (Minv)
    newwarp = cv2.warpPerspective(color_warp, Minv, img_size ) 
    # Combine the result with the original image
    result = cv2.addWeighted(img, 1, newwarp, 0.3, 0)

    # Define y-value where we want radius of curvature
    # I'll choose the maximum y-value, corresponding to the bottom of the image
    y_eval = np.max(ploty)
    left_curverad = ((1 + (2*left_fit[0]*y_eval + left_fit[1])**2)**1.5) / np.absolute(2*left_fit[0])
    right_curverad = ((1 + (2*right_fit[0]*y_eval + right_fit[1])**2)**1.5) / np.absolute(2*right_fit[0])
    #print(left_curverad, right_curverad)

    # Define conversions in x and y from pixels space to meters
    ym_per_pix = 30/720 # meters per pixel in y dimension
    xm_per_pix = 3.7/700 # meters per pixel in x dimension

    # Fit new polynomials to x,y in world space
    left_fit_cr = np.polyfit(lefty*ym_per_pix, leftx*xm_per_pix, 2)
    right_fit_cr = np.polyfit(righty*ym_per_pix, rightx*xm_per_pix, 2)

    left_cr_cache.append(left_fit_cr)
    right_cr_cache.append(right_fit_cr)
    subset = left_cr_cache[-RESULT_CACHE_SIZE:]
    left_fit_cr[0] = np.sum( x[0] for x in subset ) / len(subset)
    left_fit_cr[1] = np.sum( x[1] for x in subset ) / len(subset)
    left_fit_cr[2] = np.sum( x[2] for x in subset ) / len(subset)

    subset = right_cr_cache[-RESULT_CACHE_SIZE:]
    right_fit_cr[0] = np.sum( x[0] for x in subset ) / len(subset)
    right_fit_cr[1] = np.sum( x[1] for x in subset ) / len(subset)
    right_fit_cr[2] = np.sum( x[2] for x in subset ) / len(subset)

    # Calculate the new radii of curvature
    left_curverad = ((1 + (2*left_fit_cr[0]*y_eval*ym_per_pix + left_fit_cr[1])**2)**1.5) / np.absolute(2*left_fit_cr[0])
    right_curverad = ((1 + (2*right_fit_cr[0]*y_eval*ym_per_pix + right_fit_cr[1])**2)**1.5) / np.absolute(2*right_fit_cr[0])
    curve_rad = (left_curverad + right_curverad) / 2
    # Now our radius of curvature is in meters
    #print(left_curverad, 'm', right_curverad, 'm')

    img_center_position = img_size[0]/2
    l_fit_x_int = left_fit[0]*y_eval**2 + left_fit[1]*y_eval + left_fit[2]
    r_fit_x_int = right_fit[0]*y_eval**2 + right_fit[1]*y_eval + right_fit[2]
    lane_center_position = (l_fit_x_int + r_fit_x_int) /2
    center_dist = (lane_center_position - img_center_position) * xm_per_pix

    cv2.putText(result,'Radius of Curvature = %6.1f(m)' % curve_rad,
    bottomLeftCornerOfLine1,
    font, 
    fontScale,
    fontColor,
    lineType)

    cv2.putText(result,'Vehicle is %6.2fm left of Center' % center_dist,
    bottomLeftCornerOfLine2,
    font, 
    fontScale,
    fontColor,
    lineType)

    #cv2.imwrite('pipe_images/test%d.jpg' % (i),result)
    #i = i + 1
    return result

img = cv2.imread('test_images/test3.jpg')
curved = process_image(img)
cv2.imwrite("output_images/with_radius_test3.jpg", curved)
    
white_output = 'project_video_output.mp4'
clip1 = VideoFileClip("project_video.mp4")
white_clip = clip1.fl_image(process_image) #NOTE: this function expects color images!!
white_clip.write_videofile(white_output, audio=False)
