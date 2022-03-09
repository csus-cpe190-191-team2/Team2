# This is a sample Python script.

# Press Shift+F10 to execute it or replace it with your code.
# Press Double Shift to search everywhere for classes, files, tool windows, actions, and settings.

# In terminal pip install cap
# In terminal pip install numpy
# In terminal pip install open-cv
import cv2
import numpy as np
import utlis


def getLaneCurve(img):
    #### STEP 1
    imgThres = utlis.thresholding(img)

    #### STEP 2
    h,w, c = img.shape
    points = utlis.valTrackbars()
    imgWarp = utlis.warpImg(imgThres,points,w,h)
    imgWarpPoints = utlis.drawPoints(img,points)

    #### STEP 3

    utlis.getHistogram(imgWarp)
    basePoint,imgHist = utlis.getHistogram(imgWarp,display = True)


    cv2.imshow('Thres',imgThres)
    cv2.imshow('Warp', imgWarp)
    cv2.imshow('points', imgWarpPoints)
    cv2.imshow('Histogram', imgHist)

    return None


# Press the green button in the gutter to run the script.
if __name__ == '__main__':
    cap = cv2.VideoCapture(0)
    intialTrackBarVals = [100, 100, 100, 100]
    utlis.initializeTrackbars(intialTrackBarVals)
    frameCounter = 0
    while True:

        success, img = cap.read()
        img = cv2.resize(img,(480,240))
        getLaneCurve(img)

        cv2.imshow('Video',img)
        cv2.waitKey(1)



