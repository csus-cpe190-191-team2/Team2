import cv2
import numpy as np

cap = cv2.VideoCapture(0)   # input device id: 0-3
timer = 0                   # FPS timer


def empty(x): return x


def get_img(display=False, size=[480, 240]):
    _, img = cap.read()
    img = cv2.resize(img,(size[0], size[1]))
    if display:
        cv2.imshow('IMG', img)

    return img

def thresholding(img, points):
    img_hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    lower_blue = np.array(points[0])  # Blue: 81, 54, 140
    upper_blue = np.array(points[1])  # Blue: 128, 255, 255
    mask_blue = cv2.inRange(img_hsv, lower_blue, upper_blue)
    return mask_blue


def warp_image(img, points, w, h, inv=False):
    pts1 = np.float32(points)
    pts2 = np.float32([[0, 0], [w, 0], [0, h], [w, h]])
    if inv:
        matrix = cv2.getPerspectiveTransform(pts2, pts1)
    else:
        matrix = cv2.getPerspectiveTransform(pts1, pts2)
    img_warp = cv2.warpPerspective(img, matrix, (w,h))
    return img_warp

def draw_points(img, points):
    for x in range(4):
        cv2.circle(img,(int(points[x][0]), int(points[x][1])), 15, (0,0,255), cv2.FILLED)
    return img


def get_histogram(img, display=False, region=1):

    if region == 1:
        hist_vals = np.sum(img, axis=0)
    else:
        hist_vals = np.sum(img[img.shape[0]//region:,:], axis=0)

    # # Normalize values to max out at 255
    # for i, val in enumerate(hist_vals):
    #     if val > 255:
    #         hist_vals[i] = 255

    if display:
        img_hist = np.zeros((img.shape[0], img.shape[1], 3), np.uint8)
        for x, intensity in enumerate(hist_vals):
            cv2.line(img_hist, (x, img.shape[0]), (x, img.shape[0]-(int(intensity)//region//255)*2), (255, 0, 255), 1)
            # cv2.circle(img_hist, (base_point, img.shape[0]), 20, (0,255,255), cv2.FILLED)
        return hist_vals, img_hist
    else:
        return hist_vals


def initialize_trackbars(init_trackbar_vals, w_t=480, h_t=240): # w_t=360
    # Image warp sliders
    cv2.namedWindow("Warp Sliders")
    cv2.resizeWindow("Warp Sliders", w_t, h_t)
    cv2.createTrackbar("Width Top", "Warp Sliders", init_trackbar_vals[0], w_t//2, empty)
    cv2.createTrackbar("Height Top", "Warp Sliders", init_trackbar_vals[1], h_t, empty)
    cv2.createTrackbar("Width Bottom", "Warp Sliders", init_trackbar_vals[2], w_t // 2, empty)
    cv2.createTrackbar("Height Bottom", "Warp Sliders", init_trackbar_vals[3], h_t, empty)

    # Image threshold sliders
    cv2.namedWindow("HSV Sliders")
    cv2.resizeWindow("HSV Sliders", w_t, h_t)
    cv2.createTrackbar("HUE Min", "HSV Sliders", init_trackbar_vals[4], 179, empty)
    cv2.createTrackbar("HUE Max", "HSV Sliders", init_trackbar_vals[5], 179, empty)
    cv2.createTrackbar("SAT Min", "HSV Sliders", init_trackbar_vals[6], 255, empty)
    cv2.createTrackbar("SAT Max", "HSV Sliders", init_trackbar_vals[7], 255, empty)
    cv2.createTrackbar("VALUE Min", "HSV Sliders", init_trackbar_vals[8], 255, empty)
    cv2.createTrackbar("VALUE Max", "HSV Sliders", init_trackbar_vals[9], 255, empty)

    cv2.namedWindow("Tools")
    cv2.resizeWindow("Tools", w_t, h_t)
    cv2.createTrackbar("Hough Threshold", "Tools", init_trackbar_vals[10], 250, empty)
    cv2.createTrackbar("Hough Max Gap", "Tools", init_trackbar_vals[11], 500, empty)
    cv2.createTrackbar("Save Settings", "Tools", 0, 1, empty)


def val_trackbars(w_t=480, h_t=240):
    width_top = cv2.getTrackbarPos("Width Top", "Warp Sliders")
    height_top = cv2.getTrackbarPos("Height Top", "Warp Sliders")
    width_bottom = cv2.getTrackbarPos("Width Bottom", "Warp Sliders")
    height_bottom = cv2.getTrackbarPos("Height Bottom", "Warp Sliders")

    h_min = cv2.getTrackbarPos("HUE Min", "HSV Sliders")
    h_max = cv2.getTrackbarPos("HUE Max", "HSV Sliders")
    s_min = cv2.getTrackbarPos("SAT Min", "HSV Sliders")
    s_max = cv2.getTrackbarPos("SAT Max", "HSV Sliders")
    v_min = cv2.getTrackbarPos("VALUE Min", "HSV Sliders")
    v_max = cv2.getTrackbarPos("VALUE Max", "HSV Sliders")
    lower = np.array([h_min, s_min, v_min])
    upper = np.array([h_max, s_max, v_max])

    hough_thresh = cv2.getTrackbarPos("Hough Threshold", "Tools")
    hough_max_gap = cv2.getTrackbarPos("Hough Max Gap", "Tools")

    warp = np.float32([(width_top, height_top), (w_t - width_top, height_top),
                         (width_bottom, height_bottom), (w_t - width_bottom, height_bottom)])
    thresh = np.int_([lower, upper])
    hough = np.int_([hough_thresh, hough_max_gap])
    save = cv2.getTrackbarPos("Save Settings", "Tools")

    return warp, thresh, hough, save


def stack_images(scale, img_array):
    rows = len(img_array)
    cols = len(img_array[0])
    rows_available = isinstance(img_array[0], list)
    width = img_array[0][0].shape[1]
    height = img_array[0][0].shape[0]
    if rows_available:
        for x in range( 0, rows):
            for y in range(0, cols):
                if img_array[x][y].shape[:2] == img_array[0][0].shape [:2]:
                    img_array[x][y] = cv2.resize(img_array[x][y], (0, 0), None, scale, scale)
                else:
                    img_array[x][y] = cv2.resize(img_array[x][y], (img_array[0][0].shape[1], img_array[0][0].shape[0]), None, scale, scale)
                if len(img_array[x][y].shape) == 2: img_array[x][y]= cv2.cvtColor( img_array[x][y], cv2.COLOR_GRAY2BGR)
        image_blank = np.zeros((height, width, 3), np.uint8)
        hor = [image_blank]*rows
        hor_con = [image_blank]*rows
        for x in range(0, rows):
            hor[x] = np.hstack(img_array[x])
        ver = np.vstack(hor)
    else:
        for x in range(0, rows):
            if img_array[x].shape[:2] == img_array[0].shape[:2]:
                img_array[x] = cv2.resize(img_array[x], (0, 0), None, scale, scale)
            else:
                img_array[x] = cv2.resize(img_array[x], (img_array[0].shape[1], img_array[0].shape[0]), None,scale, scale)
            if len(img_array[x].shape) == 2: img_array[x] = cv2.cvtColor(img_array[x], cv2.COLOR_GRAY2BGR)
        hor = np.hstack(img_array)
        ver = hor
    return ver


if __name__ == '__main__':
    if not cap.isOpened():
        raise IOError("Cannot open webcam")
    else:
        img = get_img()
        h_t, w_t, ch = img.shape

        # Saved variable paths
        warp_fn = 'vars/warp_points.txt'
        thresh_fn = 'vars/thresh_points.txt'
        hough_fn = 'vars/hough_points.txt'

    init_trackbar_vals = [50, 132, 0, 187, 81, 128, 54, 255, 140, 255, 10, 150]  # DEFAULT: 0,0,0,200
    initialize_trackbars(init_trackbar_vals, w_t, h_t)

    frame_counter = 0

    while cap.isOpened():
        img = get_img()
        img_result = img.copy()
        warp_points, thresh_points, hough_points, save_points = val_trackbars()

        img = draw_points(img, warp_points)

        if save_points:
            a = warp_points
            b = thresh_points
            c = hough_points
            np.savetxt(warp_fn, a, fmt='%s')
            np.savetxt(thresh_fn, b, fmt='%s')
            np.savetxt(hough_fn, c, fmt='%s')
            with open(warp_fn, 'r') as file:
                warp_points = np.loadtxt(file)
            with open(thresh_fn, 'r') as file:
                thresh_points = np.loadtxt(file, dtype=int)
            with open(hough_fn, 'r') as file:
                hough_points = np.loadtxt(file, dtype=int)


        # ## Threshold and warp image
        img_thresh = thresholding(img, thresh_points)
        img_warp = warp_image(img_thresh, warp_points, w_t, h_t)
        _, img_hist = get_histogram(img_warp, display=True, region=2)
        edges = cv2.Canny(img_warp, 0, 1000)
        lines = cv2.HoughLinesP(edges, 1, np.pi / 180, hough_points[0], maxLineGap=hough_points[1])
        img_lines = img.copy()
        img_lines = warp_image(img_lines, warp_points, w_t, h_t)
        if lines is not None:
            for line in lines:
                x1, y1, x2, y2 = line[0]
                cv2.line(img_lines, (x1, y1), (x2, y2), (0, 0, 255), 3)

        # Draw detected lanes over image
        img_inv_warp = warp_image(img_warp, warp_points, w_t, h_t, inv=True)
        img_inv_warp = cv2.cvtColor(img_inv_warp, cv2.COLOR_GRAY2BGR)
        img_inv_warp[0:h_t // 3, 0:w_t] = 0, 0, 0
        img_lane_color = np.zeros_like(img)
        img_lane_color[:] = 0, 255, 0
        img_lane_color = cv2.bitwise_and(img_inv_warp, img_lane_color)
        img_result = cv2.addWeighted(img_result, 1, img_lane_color, 1, 0)
        fps = cv2.getTickFrequency() / (cv2.getTickCount() - timer)
        timer = cv2.getTickCount()
        cv2.putText(img_result, 'FPS ' + str(int(fps)), (20, 40), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 3);


        # img_result = warp_image(img_warp, points, w_t, h_t, inv=True)
        # img_result = cv2.addWeighted(img, 0.6, img_result, 0.8, 0)

        # Display Image Stack
        imgStacked = stack_images(1, ([img, img_thresh, edges],
                                     [img_result, img_hist, img_lines]))
        cv2.imshow('ImageStack', imgStacked)
        # cv2.imshow("img", img)
        # cv2.imshow("Threshold", img_thresh)
        # cv2.imshow("Edges", edges)
        # cv2.imshow("Warp", img_lines)
        # cv2.imshow("Result", img_result)
        # cv2.imshow("Histogram", img_hist)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()
