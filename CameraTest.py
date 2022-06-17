# This is a sample Python script.

import cv2

if __name__ == '__main__':
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
     raise IOError("Cannot open webcam properly")

    while True:
        ret, frame = cap.read()
        #print('looping')
        frame = cv2.resize( frame, None, fx = 0.5, fy=0.5, interpolation = cv2.INTER_AREA)
        cv2.imshow('Input',frame)

        c = cv2.waitKey(1)
        if c == 27:
            break

    cap.release()
    cv2.destoryALLWindows()
