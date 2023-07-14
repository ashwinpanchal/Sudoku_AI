import cv2 as cv
import numpy as np

def preprocess(path):
    img = cv.imread(path)
    img = cv.resize(img,(450,450))
    gray = cv.cvtColor(img,cv.COLOR_BGR2GRAY)
    img_blur = cv.GaussianBlur(gray,(5,5),1)
    img_threshold = cv.adaptiveThreshold(img_blur,255,1,1,11,2)
    return img_threshold,img

def big_contours(contours):
    bigg = np.array([])
    max_area = 0
    for i in contours:
        area = cv.contourArea(i)
        if area>50:
            peri = cv.arcLength(i,True)
            approx = cv.approxPolyDP(i,0.02*peri,True)
            if area > max_area and len(approx)==4:
                bigg = approx
                max_area = area
    return bigg,max_area

def arrange(biggest):
    biggest = biggest.reshape((4,2))
    add = biggest.sum(axis = 1)
    points = np.zeros((4,2),dtype = np.int32)
    points[0] = biggest[np.argmin(add)]
    points[3] = biggest[np.argmax(add)]
    diff = np.diff(biggest,axis = 1)
    points[2] = biggest[np.argmax(diff)]
    points[1] = biggest[np.argmin(diff)]
    return points

def wrap_img(path):
    img_thresh,img = preprocess(path)
    img_contour = img.copy()
    img_biggest_contour = img.copy()
    contours,hierarchy = cv.findContours(img_thresh,cv.RETR_EXTERNAL,cv.CHAIN_APPROX_SIMPLE)
    # cv.drawContours(img_contour,contours,-1,(0,255,0),3)
    biggest,maxarea = big_contours(contours)
    points = arrange(biggest)
    points = np.float32(points)
    final_points = np.float32([[0,0],[450,0],[0,450],[450,450]])
    matrix = cv.getPerspectiveTransform(points,final_points)
    wrapped_img = cv.warpPerspective(img,matrix,(450,450))
    return wrapped_img