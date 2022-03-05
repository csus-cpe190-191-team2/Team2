# This is a sample Python script
# Help determine HSV color values using sliders
# to easily determine best values for lane detection

import cv2
import numpy as np

frameWidth = 340
frameHeight = 240


def empty(x): return x


if __name__ == '__main__':
    cap = cv2.VideoCapture(0)
    cap.set(3, frameWidth)
    cap.set(4, frameHeight)

    if not cap.isOpened():
        raise IOError("Cannot open webcam")

    cv2.namedWindow("HSV")
    cv2.resizeWindow("HSV", 340, 240)
    cv2.createTrackbar("HUE Min", "HSV", 0, 179, empty)
    cv2.createTrackbar("HUE Max", "HSV", 179, 179, empty)
    cv2.createTrackbar("SAT Min", "HSV", 0,  255, empty)
    cv2.createTrackbar("SAT Max", "HSV", 255,  255, empty)
    cv2.createTrackbar("VALUE Min", "HSV", 0,  255, empty)
    cv2.createTrackbar("VALUE Max", "HSV", 255,  255, empty)

    while True:
        ret, frame = cap.read() 
        frameHsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)

        h_min = cv2.getTrackbarPos("HUE Min", "HSV")
        h_max = cv2.getTrackbarPos("HUE Max", "HSV")
        s_min = cv2.getTrackbarPos("SAT Min", "HSV")
        s_max = cv2.getTrackbarPos("SAT Max", "HSV")
        v_min = cv2.getTrackbarPos("VALUE Min", "HSV")
        v_max = cv2.getTrackbarPos("VALUE Max", "HSV")

        lower = np.array([h_min, s_min, v_min])
        upper = np.array([h_max, s_max, v_max])
        mask = cv2.inRange(frameHsv, lower, upper)
        result = cv2.bitwise_and(frame, frame, mask=mask)

        hStack = np.hstack([frame, result])

        # cv2.imshow('Input',frame)
        # cv2.imshow('Result', result)
        cv2.imshow('HSV Color Space', frameHsv)
        cv2.imshow('Mask', mask)
        cv2.imshow("Frame vs. Result", hStack)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()
