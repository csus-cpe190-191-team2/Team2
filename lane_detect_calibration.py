import cv2
import numpy as np
import lane_detect as ld

# Saved variable paths
warp_fn = 'vars/warp_points.txt'
thresh_fn = 'vars/thresh_points.txt'
timer = 0   # FPS timer


def empty(x): return x


def draw_points(img, points):
    for x in range(4):
        cv2.circle(img, (int(points[x][0]), int(points[x][1])), 15, (0, 0, 255), cv2.FILLED)
    return img


def initialize_trackbars(init_trackbar_vals, w_t=480, h_t=240):
    init_warp = init_trackbar_vals[0].astype(int)
    init_hsv = init_trackbar_vals[1]

    # Image warp sliders
    cv2.namedWindow("Warp Sliders")
    cv2.resizeWindow("Warp Sliders", w_t, h_t)
    cv2.createTrackbar("Width Top", "Warp Sliders", init_warp[0, 0], w_t//2, empty)
    cv2.createTrackbar("Height Top", "Warp Sliders", init_warp[1, 1], h_t, empty)
    cv2.createTrackbar("Width Bottom", "Warp Sliders", init_warp[2, 0], w_t//2, empty)
    cv2.createTrackbar("Height Bottom", "Warp Sliders", init_warp[3, 1], h_t, empty)

    # Image threshold sliders
    cv2.namedWindow("HSV Sliders")
    cv2.resizeWindow("HSV Sliders", w_t, h_t)
    cv2.createTrackbar("HUE Min", "HSV Sliders", init_hsv[0, 0], 179, empty)
    cv2.createTrackbar("HUE Max", "HSV Sliders", init_hsv[1, 0], 179, empty)
    cv2.createTrackbar("SAT Min", "HSV Sliders", init_hsv[0, 1], 255, empty)
    cv2.createTrackbar("SAT Max", "HSV Sliders", init_hsv[1, 1], 255, empty)
    cv2.createTrackbar("VALUE Min", "HSV Sliders", init_hsv[0, 2], 255, empty)
    cv2.createTrackbar("VALUE Max", "HSV Sliders", init_hsv[1, 2], 255, empty)

    # Save sliders
    cv2.namedWindow("Save")
    cv2.resizeWindow("Save", w_t, h_t)

    cv2.createTrackbar("Save Settings", "Save", 0, 1, empty)
    cv2.createTrackbar("Reset Settings", "Save", 0, 1, empty)


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

    warp = np.float32([(width_top, height_top), (w_t - width_top, height_top),
                       (width_bottom, height_bottom), (w_t - width_bottom, height_bottom)])
    thresh = np.int_([lower, upper])

    save = cv2.getTrackbarPos("Save Settings", "Save")
    reset_val = cv2.getTrackbarPos("Reset Settings", "Save")

    return warp, thresh, save, reset_val


def init_trackbars():
    # Read in variable variable values saved to disk
    with open(warp_fn, 'r') as load_file:
        warp_points = np.loadtxt(load_file)
    with open(thresh_fn, 'r') as load_file:
        thresh_points = np.loadtxt(load_file, dtype=int)

    cv2.destroyAllWindows()

    init_trackbar_vals = [warp_points, thresh_points]  # DEFAULT: 0,0,0,200
    initialize_trackbars(init_trackbar_vals, lane.w_t, lane.h_t)

    return warp_points, thresh_points


# ## Driver
if __name__ == '__main__':  # Program start from here
    # Initialize Lane Detection
    lane = ld.LaneDetect()

    # Initialize Trackbars
    warp_points, thresh_points = init_trackbars()

    frame_counter = 0

    while True:
        warp_points, thresh_points, save_points, reset = val_trackbars()

        lane.points[0]['warp_points'] = warp_points
        lane.points[0]['thresh_points'] = thresh_points

        if reset:
            warp_points, thresh_points = init_trackbars()
            print("\n##############################")
            print("Reloaded Settings From File!\n", warp_fn, '\n', thresh_fn)
            print("##############################\n")

        # If save switch is on, write slider values to disk
        if save_points:
            a = warp_points
            b = thresh_points

            np.savetxt(warp_fn, a, fmt='%s')
            np.savetxt(thresh_fn, b, fmt='%s')

            print("\n#########################")
            print("Settings Saved!\n", warp_fn, '\n', thresh_fn)
            print("#########################\n")
            init_trackbars()

        lane_curve = lane.get_curve()
        lane_area, sliding_windows = lane.display(lane_curve[2], lane_curve[3], lane_curve[4])

        # Get histogram display
        if lane.error:
            region = 1
        else:
            region = 2
        hist_vals, img_hist = lane.get_histogram(display=True, region=region)

        # Draw detected Lines
        img_result = lane.img.copy()
        lane.warp_image(img_in=lane.img_warp, inv=True)
        img_inv_warp = lane.img_warp_inv
        img_inv_warp = cv2.cvtColor(img_inv_warp, cv2.COLOR_GRAY2BGR)
        img_inv_warp[0:lane.h_t // 3, 0:lane.w_t] = 0, 0, 0
        img_lane_color = np.zeros_like(lane.img)
        img_lane_color[:] = 0, 255, 0
        img_lane_color = cv2.bitwise_and(img_inv_warp, img_lane_color)
        img_result = cv2.addWeighted(img_result, 1, img_lane_color, 1, 0)

        # Draw FPS counter
        fps = cv2.getTickFrequency() / (cv2.getTickCount() - timer)
        timer = cv2.getTickCount()
        cv2.putText(img_result, 'FPS ' + str(int(fps)), (20, 40), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 3)

        # Draw warp points
        img = draw_points(lane.img, warp_points)  # Draw warp points

        # Stack frames into one window
        if lane_curve[5]:
            imgStacked = ld.stack_images(1, ([lane.img, lane.img_thresh],
                                             [img_result, img_hist]))
        else:
            imgStacked = ld.stack_images(1, ([lane.img, lane.img_thresh, sliding_windows],
                                             [img_result, img_hist, lane_area]))
        cv2.imshow('Image Stack', imgStacked)

        # Delay 1ms per frame; Quit via ctl+c in terminal  or 'q' in window
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
