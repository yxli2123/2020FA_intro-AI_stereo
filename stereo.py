import numpy as np
import cv2

# load images from left and right view
imgL = cv2.imread('./view0.png', cv2.IMREAD_COLOR)
imgR = cv2.imread('./view1.png', cv2.IMREAD_COLOR)

# preprocess: RGB to YCrCb, normalization, standardization
imgL = cv2.cvtColor(imgL, cv2.COLOR_RGB2YCrCb)[:, :, 0]
imgR = cv2.cvtColor(imgR, cv2.COLOR_RGB2YCrCb)[:, :, 0]
imgL = imgL - np.floor(imgL.mean())
imgR = imgR - np.floor(imgR.mean())
imgL = cv2.normalize(src=imgL, dst=imgL, beta=0, alpha=255, norm_type=cv2.NORM_MINMAX)
imgR = cv2.normalize(src=imgR, dst=imgR, beta=0, alpha=255, norm_type=cv2.NORM_MINMAX)
imgL = np.array(imgL, dtype='uint8')
imgR = np.array(imgR, dtype='uint8')

# set parameters for stereo
windowSize = 35
stereo = cv2.StereoSGBM_create(minDisparity=-1,
                               numDisparities=16 * 5,
                               blockSize=25,
                               P1=8 * 3 * windowSize,
                               P2=16 * 3 * windowSize,
                               disp12MaxDiff=12,
                               preFilterCap=10,
                               uniquenessRatio=15,
                               speckleWindowSize=12,
                               speckleRange=63,
                               mode=cv2.STEREO_SGBM_MODE_SGBM_3WAY)

# compute the disparity of 2 images and normalization
disparity = stereo.compute(imgL, imgR)
disparity = cv2.normalize(src=disparity, dst=disparity,
                          beta=0, alpha=255, norm_type=cv2.NORM_MINMAX)
disparity = np.uint8(disparity)

# write depth map
cv2.imwrite('./depth.png', disparity)
