import numpy as np
import cv2
import glob
dims=(640,480)

#define suitable filepath with images
stereoCalib=StereoCalibration(filepath)
calibParams=stereoCalib.stereo_calibrate(dims)
M1=calibParams['M1']
M2=calibParams['M2']
d1=calibParams['d1']
d2=calibParams['d2']
R=calibParams['R']
T=calibParams['T']
E=calibParams['E']
F=calibParams['F']


(leftRectification, rightRectification, leftProjection, rightProjection,dispartityToDepthMap, leftROI, rightROI) = cv2.stereoRectify(M1, d1, M2, d2,dims, R, T,None, None, None, None, None,cv2.CALIB_ZERO_DISPARITY,alpha =-1)
leftMapX, leftMapY = cv2.initUndistortRectifyMap(M1, d1, leftRectification,leftProjection, dims, cv2.CV_32FC1)
rightMapX, rightMapY = cv2.initUndistortRectifyMap(M2, d2, rightRectification,rightProjection, dims, cv2.CV_32FC1)



stereoMatcher = cv2.StereoBM_create()

leftFrame = cv2.imread('left199.jpg')
cv2.imshow('imgl',leftFrame)
cv2.waitKey(0)
rightFrame = cv2.imread('right199.png')
cv2.imshow('imgl',rightFrame	)
cv2.waitKey(0)

fixedLeft = cv2.remap(leftFrame, leftMapX, leftMapY, cv2.INTER_LINEAR)
fixedRight = cv2.remap(rightFrame, rightMapX, rightMapY,cv2.INTER_LINEAR)
cv2.imshow('leftRect',fixedLeft)
cv2.waitKey(0)
cv2.imshow('rightRect',fixedRight)
cv2.waitKey(0)
cv2.destroyAllWindows()

#grayLeft = cv2.cvtColor(fixedLeft, cv2.COLOR_BGR2GRAY)
#grayRight = cv2.cvtColor(fixedRight, cv2.COLOR_BGR2GRAY)
#depth = stereoMatcher.compute(grayLeft, grayRight)



#DEPTH_VISUALIZATION_SCALE = 2048
#cv2.imshow('depth', depth / DEPTH_VISUALIZATION_SCALE)
