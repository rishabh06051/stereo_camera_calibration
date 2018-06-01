import numpy as np
import cv2
import glob
dims=(640,480)
M1=np.array([[563.8907633 ,   0.        , 306.18886608],
       [  0.        , 563.72546218, 244.00094137],
       [  0.        ,   0.        ,   1.        ]])


M2= np.array([[615.53553866,   0.        , 327.05762946],
       [  0.        , 607.62598079, 244.84291992],
       [  0.        ,   0.        ,   1.        ]])

d1= np.array([[-0.2017611 ,  0.58761183,  0.00706433,  0.00638899, -0.32385782]])
      

d2=np.array([[ 4.50667734e-01, -2.58312233e+00, -2.40962112e-03,
         3.26869414e-02,  6.03737354e+00]])

R= np.array([[ 0.99748413,  0.01277911, -0.06972873],
       [-0.01012243,  0.99921419,  0.03832148],
       [ 0.07016365, -0.03751925,  0.99682966]])
T= np.array([[-5.17682622],
       [-0.38246197],
       [ 3.80196113]])

E = np.array([[ 0.01165015, -3.78462383, -0.52694623],
       [ 4.15562091, -0.14564494,  4.89530801],
       [ 0.4339018 , -5.1678707 , -0.22505225]])

F= np.array([[-1.49013111e-04,  4.87415299e-02, -7.89165805e+00],
       [-5.35195386e-02,  1.88866719e-03, -2.06950646e+01],
       [ 9.85568742e+00,  2.33546933e+01,  1.00000000e+00]])


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