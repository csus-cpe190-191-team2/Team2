# This is a sample Python script
# Help determine HSV color values using sliders
# to easily determine best values for lane detection

import cv2
import numpy as np
# import lane_detection as lane

frameWidth = 480
frameHeight = 240


def empty(x): return x


def get_histogram(img, min_per=0.1, display=False, region=1):

    if region == 1:
        hist_vals = np.sum(img, axis=0)
    else:
        hist_vals = np.sum(img[img.shape[0]//region:,:], axis=0)
    print(hist_vals)  # DEBUG output
    max_value = np.max(hist_vals)
    min_value = min_per*max_value

    index_array = np.where(hist_vals >= min_value)
    base_point = int(np.average(index_array))
    # print(base_point) # DEBUG output

    if display:
        img_hist = np.zeros((img.shape[0], img.shape[1], 3), np.uint8)
        for x, intensity in enumerate(hist_vals):
            cv2.line(img_hist, (x, img.shape[0]), (x, img.shape[0]-int(intensity)//255//region), (255, 0, 255), 1)
            #cv2.circle(img_hist, (base_point, img.shape[0]), 20, (0,255,255), cv2.FILLED)
        return base_point, img_hist

    return base_point


if __name__ == '__main__':
    cap = cv2.VideoCapture(0)   # input device id: 0-3
    #cap = cv2.imread("lane_with_polygon.png")
    cap.set(3, frameWidth)
    cap.set(4, frameHeight)


    if not cap.isOpened():
        raise IOError("Cannot open webcam")

    cv2.namedWindow("HSV")
    cv2.resizeWindow("HSV", frameWidth, frameHeight)
    cv2.createTrackbar("HUE Min", "HSV", 0, 179, empty)
    cv2.createTrackbar("HUE Max", "HSV", 179, 179, empty)
    cv2.createTrackbar("SAT Min", "HSV", 0,  255, empty)
    cv2.createTrackbar("SAT Max", "HSV", 255,  255, empty)
    cv2.createTrackbar("VALUE Min", "HSV", 0,  255, empty)
    cv2.createTrackbar("VALUE Max", "HSV", 255,  255, empty)

    print("Press 'q' to quit")
    while True:
        ret, frame = cap.read()
        #frame = cap # TEST
        frame = cv2.resize(frame, (frameWidth, frameHeight))
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

        mid_point, img_hist = get_histogram(mask, display=True)
        cv2.imshow("Histogram", img_hist)

        # cv2.imshow('Input',frame)
        # cv2.imshow('Result', result)
        cv2.imshow('HSV Color Space', frameHsv)
        cv2.imshow('Mask', mask)
        cv2.imshow("Frame vs. Result", hStack)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()
