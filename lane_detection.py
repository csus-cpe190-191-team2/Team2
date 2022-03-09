import cv2
import numpy as np
from scipy.signal import savgol_filter

cap = cv2.VideoCapture(0)   # input device id: 0-3
curve_list = []
avg_val = 10    # Number of values to average
timer = 0


def empty(x): return x


def get_img(display=False, size=[480, 240]):
    _, img = cap.read()
    img = cv2.resize(img,(size[0], size[1]))
    if display:
        cv2.imshow('IMG', img)

    return img


def thresholding(img):
    imgHsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    lowerWhite = np.array([101, 43, 145])    # Green: 27, 68, 178
    upperWhite = np.array([135, 255, 206])   # Green: 35, 112, 223
    maskWhite = cv2.inRange(imgHsv, lowerWhite, upperWhite)
    return maskWhite


def warp_image(img, points, w, h, inv=False):
    pts1 = np.float32(points)
    pts2 = np.float32([[0, 0], [w, 0], [0, h], [w, h]]) # np.float32([[0, 0], [w, 0], [0, h], [w, h]])
    if inv:
        matrix = cv2.getPerspectiveTransform(pts2, pts1)
    else:
        matrix = cv2.getPerspectiveTransform(pts1, pts2)
    img_warp = cv2.warpPerspective(img, matrix, (w,h))
    return img_warp


def initialize_trackbars(init_trackbar_vals, w_t=480, h_t=240):
    cv2.namedWindow("Trackbars")
    cv2.resizeWindow("Trackbars", 360, 240)
    cv2.createTrackbar("Width Top", "Trackbars", init_trackbar_vals[0], w_t//2, empty)
    cv2.createTrackbar("Height Top", "Trackbars", init_trackbar_vals[1], h_t, empty)
    cv2.createTrackbar("Width Bottom", "Trackbars", init_trackbar_vals[2], w_t // 2, empty)
    cv2.createTrackbar("Height Bottom", "Trackbars", init_trackbar_vals[3], h_t, empty)


def val_trackbars(w_t=480, h_t=240):
    width_top = cv2.getTrackbarPos("Width Top", "Trackbars")
    height_top = cv2.getTrackbarPos("Height Top", "Trackbars")
    width_bottom = cv2.getTrackbarPos("Width Bottom", "Trackbars")
    height_bottom = cv2.getTrackbarPos("Height Bottom", "Trackbars")
    points = np.float32([(width_top, height_top), (w_t - width_top, height_top),
                         (width_bottom, height_bottom), (w_t - width_bottom, height_bottom)])
    return points


def draw_points(img, points):
    for x in range(4):
        cv2.circle(img,(int(points[x][0]), int(points[x][1])), 15, (0,0,255), cv2.FILLED)
    return img


def get_histogram(img, min_per=0.1, display=False, region=1):

    if region == 1:
        hist_vals = np.sum(img, axis=0)
    else:
        hist_vals = np.sum(img[img.shape[0]//region:,:], axis=0)

    hist_vals = savgol_filter(hist_vals, 51, 3) # Smooth out histogram values; reduce noise
    # print(hist_vals)  # DEBUG output
    max_value = np.max(hist_vals)
    min_value = min_per*max_value

    index_array = np.int64(np.where(hist_vals >= min_value))
    index_length = int(index_array[0].size//2)

    # print(index_array[0].size, ' ', index_length, ' ', index_array.shape[1], ' ',
    #       index_array[0][0:index_length], ' ', index_array[0][0])                 # DEBUG

    if index_array.shape[1] > 1:
        base_left = int(np.average(index_array[0][:index_length]))
        base_right = int(np.average(index_array[0][index_length:]))
    else:
        base_left = 0
        base_right = 0
    # base_point = int(np.average(index_array)) # For detecting only one line
    # base_point = int(np.average(np.append(base_left, base_right)))  # For detecting two lines
    base_point = int(np.add(base_left, base_right)//2)  # find midpoint For detecting two lines

    print("left: ", base_left, "\tRight: ", base_right, "\tMidpoint: ", base_point) # DEBUG output

    if display:
        img_hist = np.zeros((img.shape[0], img.shape[1], 3), np.uint8)
        for x, intensity in enumerate(hist_vals):
            cv2.line(img_hist, (x, img.shape[0]), (x, img.shape[0]-int(intensity)//255//region), (255, 0, 255), 1)
            cv2.circle(img_hist, (base_point, img.shape[0]), 20, (0,255,255), cv2.FILLED)
        return base_point, img_hist

    return base_point


def stackImages(scale, imgArray):
    rows = len(imgArray)
    cols = len(imgArray[0])
    rowsAvailable = isinstance(imgArray[0], list)
    width = imgArray[0][0].shape[1]
    height = imgArray[0][0].shape[0]
    if rowsAvailable:
        for x in range ( 0, rows):
            for y in range(0, cols):
                if imgArray[x][y].shape[:2] == imgArray[0][0].shape [:2]:
                    imgArray[x][y] = cv2.resize(imgArray[x][y], (0, 0), None, scale, scale)
                else:
                    imgArray[x][y] = cv2.resize(imgArray[x][y], (imgArray[0][0].shape[1], imgArray[0][0].shape[0]), None, scale, scale)
                if len(imgArray[x][y].shape) == 2: imgArray[x][y]= cv2.cvtColor( imgArray[x][y], cv2.COLOR_GRAY2BGR)
        imageBlank = np.zeros((height, width, 3), np.uint8)
        hor = [imageBlank]*rows
        hor_con = [imageBlank]*rows
        for x in range(0, rows):
            hor[x] = np.hstack(imgArray[x])
        ver = np.vstack(hor)
    else:
        for x in range(0, rows):
            if imgArray[x].shape[:2] == imgArray[0].shape[:2]:
                imgArray[x] = cv2.resize(imgArray[x], (0, 0), None, scale, scale)
            else:
                imgArray[x] = cv2.resize(imgArray[x], (imgArray[0].shape[1], imgArray[0].shape[0]), None,scale, scale)
            if len(imgArray[x].shape) == 2: imgArray[x] = cv2.cvtColor(imgArray[x], cv2.COLOR_GRAY2BGR)
        hor= np.hstack(imgArray)
        ver = hor
    return ver


def get_lane_curve(img, display=0):
    global timer
    img_copy = img.copy()
    img_result = img.copy()

    # ## Step 1
    img_thresh = thresholding(img)

    # ## Step 2
    h_t, w_t, ch = img.shape
    points = val_trackbars()
    img_warp = warp_image(img_thresh, points, w_t, h_t)
    img_warp_points = draw_points(img_copy, points)

    # ## Step 3
    mid_point, img_hist = get_histogram(img_warp, display=True, min_per=0.5, region=4)
    curve_avg_point, img_hist = get_histogram(img_warp, display=True, min_per=0.9)
    curve_raw = curve_avg_point - mid_point
    # print(curve_avg_point-mid_point)

    # ## Step 4
    curve_list.append(curve_raw)
    if len(curve_list) > avg_val:
        curve_list.pop(0)
    curve = int(sum(curve_list) // len(curve_list))

    # ## Step 5
    if display != 0:
        imgInvWarp = warp_image(img_warp, points, w_t, h_t, inv=True)
        imgInvWarp = cv2.cvtColor(imgInvWarp, cv2.COLOR_GRAY2BGR)
        imgInvWarp[0:h_t // 3, 0:w_t] = 0, 0, 0
        imgLaneColor = np.zeros_like(img)
        imgLaneColor[:] = 0, 255, 0
        imgLaneColor = cv2.bitwise_and(imgInvWarp, imgLaneColor)
        img_result = cv2.addWeighted(img_result, 1, imgLaneColor, 1, 0)
        # midY = 450//2
        midY = int(img_result.shape[1]*0.46)
        cv2.putText(img_result, str(curve/100), (w_t // 2 - 80, 85), cv2.FONT_HERSHEY_COMPLEX, 2, (255, 0, 255), 3)
        # print("(", w_t // 2, ",", midY, ") (", (w_t // 2 + (curve * 3)), ",", midY, ") ", img_result.shape)  # DEBUG
        # cv2.line(img_result, (50, 225), (200, 225), (255,0,0), 5) # DEBUG
        cv2.line(img_result, (w_t // 2, midY), (w_t // 2 + (curve * 3), midY), (255, 0, 255), 5)
        cv2.line(img_result, ((w_t // 2 + (curve * 3)), midY - 25), (w_t // 2 + (curve * 3), midY + 25), (0, 255, 0), 5)
        for x in range(-30, 30):
            w = w_t // 20
            cv2.line(img_result, (w * x + int(curve // 50), midY - 10),
                     (w * x + int(curve // 50), midY + 10), (0, 0, 255), 2)
        fps = cv2.getTickFrequency() / (cv2.getTickCount() - timer)
        timer = cv2.getTickCount()
        cv2.putText(img_result, 'FPS ' + str(int(fps)), (20, 40), cv2.FONT_HERSHEY_SIMPLEX, 1, (230, 50, 50), 3);
    if display == 2:
        imgStacked = stackImages(1, ([img, img_warp_points, img_warp],
                                             [img_hist, imgLaneColor, img_result]))
        cv2.imshow('ImageStack', imgStacked)
    elif display == 1:
        cv2.imshow('Result', img_result)

    # Normalization
    curve = curve/100
    if curve > 1: curve == 1
    if curve < -1: curve == -1

    # cv2.imshow("Threshold", img_thresh)
    # cv2.imshow("Image Warp", img_warp)
    # cv2.imshow("Warp Points", img_warp_points)
    # cv2.imshow("Historgram", img_hist)

    return curve


if __name__ == '__main__':
    init_trackbar_vals = [9, 56, 0, 113]  # DEFAULT: 0,0,0,200
    initialize_trackbars(init_trackbar_vals)
    frame_counter = 0

    while cap.isOpened():
        img = get_img(display=False)
        curve = get_lane_curve(img, display=2)
        # print(curve)
        #cv2.imshow("Video", img)   # temp DEBUG
        cv2.waitKey(1)
