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

[image1]: ./examples/undistort_output.png "Undistorted"
[image2]: ./test_images/test1.jpg "Road Transformed"
[image3]: ./examples/binary_combo_example.jpg "Binary Example"
[image4]: ./examples/warped_straight_lines.jpg "Warp Example"
[image5]: ./examples/color_fit_lines.jpg "Fit Visual"
[image6]: ./examples/example_output.jpg "Output"
[video1]: ./project_video.mp4 "Video"

## [Rubric](https://review.udacity.com/#!/rubrics/571/view) Points

### Here I will consider the rubric points individually and describe how I addressed each point in my implementation.  

---

### Camera Calibration

#### 1. Briefly state how you computed the camera matrix and distortion coefficients. Provide an example of a distortion corrected calibration image.

The code for this step is contained in the second to fourth code cell of the IPython notebook located in "Advanced_Lane_Line_Finding.ipynb" .  

I start by preparing "object points", which will be the (x, y, z) coordinates of the chessboard corners in the world. Here I am assuming the chessboard is fixed on the (x, y) plane at z=0, such that the object points are the same for each calibration image.  Thus, `objp` is just a replicated array of coordinates, and `objpoints` will be appended with a copy of it every time I successfully detect all chessboard corners in a test image.  `imgpoints` will be appended with the (x, y) pixel position of each of the corners in the image plane with each successful chessboard detection.  

I then used the output `objpoints` and `imgpoints` to compute the camera calibration and distortion coefficients using the `cv2.calibrateCamera()` function.  I applied this distortion correction to the test image using the `cv2.undistort()` function and obtained this result: 

![alt text](https://github.com/Vencentlp/Advanced_Lane_Line_Finding/raw/master/output_images/camera_cali/undistort_test_imgage.png)

### Pipeline (single images)

#### 1. Provide an example of a distortion-corrected image.

To demonstrate this step, I will describe how I apply the distortion correction to one of the test images like this one:
![alt text](https://github.com/Vencentlp/Advanced_Lane_Line_Finding/raw/master/test_images/test1.jpg)
For this image, I used `cv2.undistort()` function to undistort. The result of the undistortiong:
![alt text](https://github.com/Vencentlp/Advanced_Lane_Line_Finding/raw/master/output_images/camera_cali/undistort_test_imgage.png)

#### 2. Describe how (and identify where in your code) you used color transforms, gradients or other methods to create a thresholded binary image.  Provide an example of a binary image result.

I used a combination of color and gradient thresholds to generate a binary image (in the fifth to enghth code cells).  Here's an example of my output for this step.  
Firstly, I calculated sobelx and sobel direction and applied threshold to get the soble binary image. (In the 5th code cell )
Secondly, I select the white and yellow color through RGB thresholds.(In the 6th code cell)
Thirdly, I transformed RGB image into HLS image. I applied S and l channel color threshold to get the S and L channel binary image.(In the 7th code cell)
Fourthly, I transformed RGB image into LAB image. I applied B channel color threshold to get B channel binary image.(In the 8th code cell)
Finally, I combined all the four binary images by ((white and yellow select binary image & l channel binary)) & ((s channel binary & sobel binary)) | (b channel binary)

![alt text](https://github.com/Vencentlp/Advanced_Lane_Line_Finding/raw/master/output_images/binary/binarytest1.jpg)

#### 3. Describe how (and identify where in your code) you performed a perspective transform and provide an example of a transformed image.

The code for my perspective transform includes a function called `warp()`, which appears in 9th code cell in the file `Advanced_Lane_Line_Finding.ipynb`.  The `warp()` function takes as inputs an image (`img`), as well as source (`src`) and destination (`dst`) points.  I chose the hardcode the source and destination points manually:



The following source and destination points are as follows:

| Source        | Destination   | 
|:-------------:|:-------------:| 
| 575, 464      | 320, 1        | 
| 707, 464      | 920, 1      |
| 258, 682     | 320, 720      |
| 1049, 682      | 920, 720    |

I verified that my perspective transform was working as expected by drawing the `src` and `dst` points onto a test image and its warped counterpart to verify that the lines appear parallel in the warped image.

![alt text](https://github.com/Vencentlp/Advanced_Lane_Line_Finding/raw/master/output_images/warptest1.png)

#### 4. Describe how (and identify where in your code) you identified lane-line pixels and fit their positions with a polynomial?

Then I did some other stuff and fit my lane lines with a 2nd order polynomial kinda like this(origial image is the test1.jpg like above):

![alt text](https://github.com/Vencentlp/Advanced_Lane_Line_Finding/blob/master/output_images/binary_Lane_lines/Lane_line_withband_test1.jpg)
To get the lane lines image like the above one, I need to take some steps to dect the pixles in the line region.
For the first frame:
1) I need to fistly find where the left and right lane lines are. So I take a histogram of the bottom half of the image.
2) I take the peak positions of the left and right halves of the histogram as the starting position of the left and right lines.
3) Define a window to slide over the line positions to detect line pixles.
4) Slide the window step by step
5) Recenter the window by averaging all the pixles in the window.
6) Concatenate all the found pixles
7) Aplly 2nd polyfit to the pixles found

For the next frame:
I do not need to search the lane lines by sliding windows for that the line position has been determined. All I need to do is to search near the lines found before. After line pixles were found, I applied 2nd polyfit to them.
All the code can be found in code cell 37th and 38th.


#### 5. Describe how (and identify where in your code) you calculated the radius of curvature of the lane and the position of the vehicle with respect to center.

I did this in 39th code cell in 'Advanced_Lane_Line_Finding.ipynb'
To calculate the curvature,t he steps are as following:
1)I first transform the coordinate unit of line pixls found from unit pixle into meter.
2)Then I take a 2nd polyfit for all the points in meter.
3) I calculated the curvature using the formula in the class

To calculate the position of the vehicle with respect to the center, I take the following steps:
1) Assume that the car in the middle of the image 
2) Calculate the middle postion between the left and right lines
3) Calculate the difference of the above 2.
If the difference is negative, the car is on the left side of the lane center,and vice versa.

#### 6. Provide an example image of your result plotted back down onto the road such that the lane area is identified clearly.

I implemented this step in lines # through # in my code in `Advanced_Lane_Line_Finding.ipynb` in the function `draw_lane()` and draw_data().  Here is an example of my result on a test image:

![alt text][image6]

---

### Pipeline (video)

#### 1. Provide a link to your final video output.  Your pipeline should perform reasonably well on the entire project video (wobbly lines are ok but no catastrophic failures that would cause the car to drive off the road!).

Here's a [link to my video result](./project_video.mp4)

---

### Discussion

#### 1. Briefly discuss any problems / issues you faced in your implementation of this project.  Where will your pipeline likely fail?  What could you do to make it more robust?

Here I'll talk about the approach I took, what techniques I used, what worked and why, where the pipeline might fail and how I might improve it if I were going to pursue this project further.  
