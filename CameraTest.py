# This is a sample Python script.

# Press Shift+F10 to execute it or replace it with your code.
# Press Double Shift to search everywhere for classes, files, tool windows, actions, and settings.
import cap as cap
import cv2
import cap


# Press the green button in the gutter to run the script.
if __name__ == '__main__':
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
     raise IOError("Cannot open webcam properly")

    while True:
        ret, frame = cap.read()
        frame = cv2.resize( frame, None, fx = 0.5, fy=0.5, interpolation = cv2.INTER_AREA)
        cv2.imshow('Input',frame)

        c = cv2.waitKey(1)
        if c == 27:
            break

    cap.release()
    cv2.destoryALLWindows()


# See PyCharm help at https://www.jetbrains.com/help/pycharm/
