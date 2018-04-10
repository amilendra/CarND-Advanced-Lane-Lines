## Writeup Template

---

**Advanced Lane Finding Project**

The goals / steps of this project are the following:

* Compute the camera calibration matrix and distortion coefficients given a set of chessboard images.
* Apply a distortion correction to raw images.
* Use color transforms, gradients, etc., to create a thresholded binary image.
* Apply a perspective transform to rectify binary image ("birds-eye view").
* Detect lane pixels and fit to find the lane boundary.
* Determine the curvature of the lane and vehicle position with respect to center.
* Warp the detected lane boundaries back onto the original image.
* Output visual display of the lane boundaries and numerical estimation of lane curvature and vehicle position.

[//]: # (Image References)

[image0]: ./camera_cal/calibration1.jpg "Original"
[image1]: ./output_images/undistorted_calibration1.jpg "Undistorted"
[image1]: ./examples/undistort_output.png "Undistorted"
[image2]: ./test_images/test1.jpg "Road Transformed"
[image2_undistort]: ./output_images/undistorted_test1.jpg "Road Undistorted"
[image3_input]: ./test_images/signs_vehicles_xygrad.png "Test Image for Binary Thresholding"
[image3]: ./output_images/bin_thresh_signs_vehicles_xygrad.jpg  "Binary Thresholding Example"
[image4_undistorted]: ./output_images/straight_lines1_with_trapezoid.jpg "Undistorted image with src points drawn"
[image4_warped]: ./output_images/straight_lines1_warped_trapezoid.jpg "Warped results with dst points drawn"
[image5]: ./output_images/color_fit_lines.jpg "Fit Visual"
[image6]: ./examples/example_output.jpg "Output"
[video1]: ./project_video.mp4 "Video"

## [Rubric](https://review.udacity.com/#!/rubrics/571/view) Points

### Here I will consider the rubric points individually and describe how I addressed each point in my implementation.  

---

### Writeup / README

#### 1. Provide a Writeup / README that includes all the rubric points and how you addressed each one.  You can submit your writeup as markdown or pdf.  [Here](https://github.com/udacity/CarND-Advanced-Lane-Lines/blob/master/writeup_template.md) is a template writeup for this project you can use as a guide and a starting point.  

You're reading it!

### Camera Calibration

#### 1. Briefly state how you computed the camera matrix and distortion coefficients. Provide an example of a distortion corrected calibration image.

The code for this step is contained in the file called `calibrate.py`.

I start by preparing "object points", which will be the (x, y, z) coordinates of the chessboard corners in the world. Here I am assuming the chessboard is fixed on the (x, y) plane at z=0, such that the object points are the same for each calibration image.  Thus, `objp` is just a replicated array of coordinates, and `objpoints` will be appended with a copy of it every time I successfully detect all chessboard corners in a test image.  `imgpoints` will be appended with the (x, y) pixel position of each of the corners in the image plane with each successful chessboard detection.  

I then used the output `objpoints` and `imgpoints` to compute the camera calibration and distortion coefficients using the `cv2.calibrateCamera()` function.  I applied this distortion correction to the test image using the `cv2.undistort()` function and obtained this result: 

##### Original Image

![alt text][image0]

##### Undistorted Image

![alt text][image1]

### Pipeline (single images)

#### 1. Provide an example of a distortion-corrected image.

To demonstrate this step, I will describe how I apply the distortion correction to one of the test images like this one:
![alt text][image2]

Code for this is in `undistort.py`.
It uses the camera calibration and distortion coefficients found before, and the test image is undistorted using the cv2.undistort function. Here is the undistorted image:

![alt text][image2_undistort]

#### 2. Describe how (and identify where in your code) you used color transforms, gradients or other methods to create a thresholded binary image.  Provide an example of a binary image result.

I mostly used the example code in the lessons and did a combination of color and gradient thresholds to generate a binary image (thresholding steps at lines 63 through 68 in `binary_threshold.py`).  Here's an example of my output for this step.  


##### Input Image

![alt text][image3_input]

##### Output Image

![alt text][image3]

#### 3. Describe how (and identify where in your code) you performed a perspective transform and provide an example of a transformed image.

The code for my perspective transform includes a function called `warper()`, which appears in lines 22 through 25 in the file `perspect_transform.py`.  The `warper()` function takes as inputs an image (`img`), as well as source (`src`) and destination (`dst`) points.  I chose the hardcode the source and destination points in the following manner:

```python
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
```

This resulted in the following source and destination points:

| Source        | Destination   | 
|:-------------:|:-------------:| 
| 575, 460      | 320, 0        | 
| 183, 720      | 320, 720      |
| 1137, 720     | 960, 720      |
| 705, 460      | 960, 0        |

I verified that my perspective transform was working as expected by drawing the `src` and `dst` points onto a test image and its warped counterpart to verify that the lines appear parallel in the warped image.

##### Undistorted image with src points drawn

![alt text][image4_undistorted]

##### Warped results with dst points drawn

![alt text][image4_warped]


#### 4. Describe how (and identify where in your code) you identified lane-line pixels and fit their positions with a polynomial?
See the function `process_image()` in `color_fit_lines.py` for my implementation.
Most of the code is based on the lecture videos and quizes, but made the following small changes.

Because grayscale images can be prone to loss of information due to illumination changes,
I converted the image to hls and used the l and s channels for processing.(lines 48 to 50 in `color_fit_lines.py`)
I tuned the search for lanes using the following heuristic(lines 75 to 79 in `color_fit_lines.py`):
* left lane is searched between the first quarter to the midpoint of the image
* the right lane is searched between 5/8 and 7/8 of the image.

I could fit the lanes into a 2nd order polynomial like this.

![alt text][image5]

#### 5. Describe how (and identify where in your code) you calculated the radius of curvature of the lane and the position of the vehicle with respect to center.

Again I used the example code from the lessons. After fitting the lanes to 2nd order polynomials, I get two equations in the form of `f(y) = A*x^2 + B*x + C`

The radius of the curve at a particular y position was calculated by the equation,
```text
radius = ((1 + (2*A*y + B)^2)^1.5) / abs(2*A)
```

I did this in lines 196 through 197 in my code in `pipeline.py`

#### 6. Provide an example image of your result plotted back down onto the road such that the lane area is identified clearly.

I implemented this step in lines # through # in my code in `yet_another_file.py` in the function `map_lane()`.  Here is an example of my result on a test image:

![alt text][image6]

---

### Pipeline (video)

#### 1. Provide a link to your final video output.  Your pipeline should perform reasonably well on the entire project video (wobbly lines are ok but no catastrophic failures that would cause the car to drive off the road!).

Here's a [link to my video result](./project_video.mp4)

---

### Discussion

#### 1. Briefly discuss any problems / issues you faced in your implementation of this project.  Where will your pipeline likely fail?  What could you do to make it more robust?

Here I'll talk about the approach I took, what techniques I used, what worked and why, where the pipeline might fail and how I might improve it if I were going to pursue this project further.  
