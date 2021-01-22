# Stereo
## Functions 
Given 2 iamges that are from left and right view respectively, compute the depth from the original images.

## Dataset
https://vision.middlebury.edu/stereo/data/

mainly use 2005datasets and 2006. datasets

## Algorithm
### Horizontal search
Given a pixel in the left-view image, search the best candidate pixel on the same horizon of the right-view image.

The best candidate pixel are determined by the maximum correlation of 2 slide windows; one is fixed on the left-view, and the other move from left to right in the right-view image.
### Disparity 
depth = f\*T/disparity, where f and T are constants here, and disparity is the x value of 2 pixels in the last step.

## Required packages
opencv-python

numpy
