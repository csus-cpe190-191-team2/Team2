import cv2
import numpy as np
# import matplotlib.pyplot as plt

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
    lowerBlue = np.array([81, 54, 140])    # Green: 27, 68, 178
    upperBlue = np.array([128, 255, 255])   # Green: 35, 112, 223
    maskBlue = cv2.inRange(imgHsv, lowerBlue, upperBlue)
    return maskBlue

def warp_image(img, points, w, h, inv=False):
    pts1 = np.float32(points)
    pts2 = np.float32([[0, 0], [w, 0], [0, h], [w, h]]) # np.float32([[0, 0], [w, 0], [0, h], [w, h]])
    if inv:
        matrix = cv2.getPerspectiveTransform(pts2, pts1)
    else:
        matrix = cv2.getPerspectiveTransform(pts1, pts2)
    img_warp = cv2.warpPerspective(img, matrix, (w,h))
    return img_warp

def get_histogram(img, display=False, region=1):

    if region == 1:
        hist_vals = np.sum(img, axis=0)
    else:
        hist_vals = np.sum(img[img.shape[0]//region:,:], axis=0)

    if display:
        img_hist = np.zeros((img.shape[0], img.shape[1], 3), np.uint8)
        for x, intensity in enumerate(hist_vals):
            cv2.line(img_hist, (x, img.shape[0]), (x, img.shape[0]-int(intensity)//255//region), (255, 0, 255), 1)
            # cv2.circle(img_hist, (base_point, img.shape[0]), 20, (0,255,255), cv2.FILLED)
        return hist_vals, img_hist
    else:
        return hist_vals


def inv_perspective_warp(img,
                     dst_size=(480,240),
                     src=np.float32([(0,0), (1, 0), (0,1), (1,1)]),
                     dst=np.float32([(0.43,0.65),(0.58,0.65),(0.1,1),(1,1)])):
    img_size = np.float32([(img.shape[1],img.shape[0])])
    src = src* img_size
    # For destination points, I'm arbitrarily choosing some points to be
    # a nice fit for displaying our warped result
    # again, not exact, but close enough for our purposes
    dst = dst * np.float32(dst_size)
    # Given src and dst points, calculate the perspective transform matrix
    M = cv2.getPerspectiveTransform(src, dst)
    # Warp the image using OpenCV warpPerspective()
    warped = cv2.warpPerspective(img, M, dst_size)
    return warped

left_a, left_b, left_c = [], [], []
right_a, right_b, right_c = [], [], []


def sliding_window(img, nwindows=9, margin=150, minpix=1, draw_windows=True):
    global left_a, left_b, left_c, right_a, right_b, right_c
    left_fit_ = np.empty(3)
    right_fit_ = np.empty(3)
    out_img = np.dstack((img, img, img)) * 255

    histogram = get_histogram(img, region=2)
    # find peaks of left and right halves
    midpoint = int(histogram.shape[0] / 2)
    # print(histogram.shape, ' ', histogram[:midpoint], ' ', histogram[midpoint:])
    while sum(histogram[:midpoint]) == 0 or sum(histogram[midpoint:]) == 0:
        img = get_img()
        # img_warp_copy = img
        # gray = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
        img_thresh = thresholding(img)

        h_t, w_t, ch = img.shape
        width_top = 3
        height_top = 62
        width_bottom = 0
        height_bottom = 96
        points = np.float32([(width_top, height_top), (w_t - width_top, height_top),
                             (width_bottom, height_bottom), (w_t - width_bottom, height_bottom)])
        img_thresh_warp = warp_image(img_thresh, points, w_t, h_t)
        # img_warp = warp_image(img, points, w_t, h_t)
        histogram = get_histogram(img_thresh_warp, region=2)
        # find peaks of left and right halves
        midpoint = int(histogram.shape[0] / 2)
        # print(histogram.shape, ' ', histogram[:midpoint], ' ', histogram[midpoint:])

    leftx_base = np.argmax(histogram[:midpoint])
    rightx_base = np.argmax(histogram[midpoint:]) + midpoint

    # Set height of windows
    window_height = np.int(img.shape[0] / nwindows)
    # Identify the x and y positions of all nonzero pixels in the image
    nonzero = img.nonzero()
    nonzeroy = np.array(nonzero[0])
    nonzerox = np.array(nonzero[1])
    # Current positions to be updated for each window
    leftx_current = leftx_base
    rightx_current = rightx_base

    # Create empty lists to receive left and right lane pixel indices
    left_lane_inds = []
    right_lane_inds = []

    # Step through the windows one by one
    for window in range(nwindows):
        # Identify window boundaries in x and y (and right and left)
        win_y_low = img.shape[0] - (window + 1) * window_height
        win_y_high = img.shape[0] - window * window_height
        win_xleft_low = leftx_current - margin
        win_xleft_high = leftx_current + margin
        win_xright_low = rightx_current - margin
        win_xright_high = rightx_current + margin
        # Draw the windows on the visualization image
        if draw_windows == True:
            cv2.rectangle(out_img, (win_xleft_low, win_y_low), (win_xleft_high, win_y_high),
                          (100, 255, 255), 3)
            cv2.rectangle(out_img, (win_xright_low, win_y_low), (win_xright_high, win_y_high),
                          (100, 255, 255), 3)
            # Identify the nonzero pixels in x and y within the window
        good_left_inds = ((nonzeroy >= win_y_low) & (nonzeroy < win_y_high) &
                          (nonzerox >= win_xleft_low) & (nonzerox < win_xleft_high)).nonzero()[0]
        good_right_inds = ((nonzeroy >= win_y_low) & (nonzeroy < win_y_high) &
                           (nonzerox >= win_xright_low) & (nonzerox < win_xright_high)).nonzero()[0]
        # Append these indices to the lists
        left_lane_inds.append(good_left_inds)
        right_lane_inds.append(good_right_inds)
        # If you found > minpix pixels, recenter next window on their mean position
        if len(good_left_inds) > minpix:
            leftx_current = np.int(np.mean(nonzerox[good_left_inds]))
        if len(good_right_inds) > minpix:
            rightx_current = np.int(np.mean(nonzerox[good_right_inds]))

    #        if len(good_right_inds) > minpix:
    #            rightx_current = np.int(np.mean([leftx_current +900, np.mean(nonzerox[good_right_inds])]))
    #        elif len(good_left_inds) > minpix:
    #            rightx_current = np.int(np.mean([np.mean(nonzerox[good_left_inds]) +900, rightx_current]))
    #        if len(good_left_inds) > minpix:
    #            leftx_current = np.int(np.mean([rightx_current -900, np.mean(nonzerox[good_left_inds])]))
    #        elif len(good_right_inds) > minpix:
    #            leftx_current = np.int(np.mean([np.mean(nonzerox[good_right_inds]) -900, leftx_current]))

    # Concatenate the arrays of indices
    left_lane_inds = np.concatenate(left_lane_inds)
    right_lane_inds = np.concatenate(right_lane_inds)

    # Extract left and right line pixel positions
    leftx = nonzerox[left_lane_inds]
    lefty = nonzeroy[left_lane_inds]
    rightx = nonzerox[right_lane_inds]
    righty = nonzeroy[right_lane_inds]

    # Fit a second order polynomial to each
    left_fit = np.polyfit(lefty, leftx, 2)
    right_fit = np.polyfit(righty, rightx, 2)

    left_a.append(left_fit[0])
    left_b.append(left_fit[1])
    left_c.append(left_fit[2])

    right_a.append(right_fit[0])
    right_b.append(right_fit[1])
    right_c.append(right_fit[2])

    left_fit_[0] = np.mean(left_a[-10:])
    left_fit_[1] = np.mean(left_b[-10:])
    left_fit_[2] = np.mean(left_c[-10:])

    right_fit_[0] = np.mean(right_a[-10:])
    right_fit_[1] = np.mean(right_b[-10:])
    right_fit_[2] = np.mean(right_c[-10:])

    # Generate x and y values for plotting
    ploty = np.linspace(0, img.shape[0] - 1, img.shape[0])
    left_fitx = left_fit_[0] * ploty ** 2 + left_fit_[1] * ploty + left_fit_[2]
    right_fitx = right_fit_[0] * ploty ** 2 + right_fit_[1] * ploty + right_fit_[2]

    out_img[nonzeroy[left_lane_inds], nonzerox[left_lane_inds]] = [255, 0, 100]
    out_img[nonzeroy[right_lane_inds], nonzerox[right_lane_inds]] = [0, 100, 255]

    return out_img, (left_fitx, right_fitx), (left_fit_, right_fit_), ploty


def get_curve(img, leftx, rightx):
    ploty = np.linspace(0, img.shape[0] - 1, img.shape[0])
    y_eval = np.max(ploty)
    ym_per_pix = 30.5 / 720  # meters per pixel in y dimension
    xm_per_pix = 3.7 / 720  # meters per pixel in x dimension

    # Fit new polynomials to x,y in world space
    left_fit_cr = np.polyfit(ploty * ym_per_pix, leftx * xm_per_pix, 2)
    right_fit_cr = np.polyfit(ploty * ym_per_pix, rightx * xm_per_pix, 2)
    # Calculate the new radii of curvature
    left_curverad = ((1 + (2 * left_fit_cr[0] * y_eval * ym_per_pix + left_fit_cr[1]) ** 2) ** 1.5) / np.absolute(
        2 * left_fit_cr[0])
    right_curverad = ((1 + (2 * right_fit_cr[0] * y_eval * ym_per_pix + right_fit_cr[1]) ** 2) ** 1.5) / np.absolute(
        2 * right_fit_cr[0])

    car_pos = img.shape[1] / 2
    l_fit_x_int = left_fit_cr[0] * img.shape[0] ** 2 + left_fit_cr[1] * img.shape[0] + left_fit_cr[2]
    r_fit_x_int = right_fit_cr[0] * img.shape[0] ** 2 + right_fit_cr[1] * img.shape[0] + right_fit_cr[2]
    lane_center_position = (r_fit_x_int + l_fit_x_int) / 2
    center = (car_pos - lane_center_position) * xm_per_pix / 10
    # Now our radius of curvature is in meters
    return (left_curverad, right_curverad, center)


def draw_lanes(img, left_fit, right_fit, warp_points):
    ploty = np.linspace(0, img.shape[0] - 1, img.shape[0])
    color_img = np.zeros_like(img)

    left = np.array([np.transpose(np.vstack([left_fit, ploty]))])
    right = np.array([np.flipud(np.transpose(np.vstack([right_fit, ploty])))])
    points = np.hstack((left, right))

    cv2.fillPoly(color_img, np.int_(points), (0, 200, 255))
    # inv_perspective = inv_perspective_warp(color_img)
    h_t, w_t, ch = img.shape
    inv_perspective = warp_image(color_img, warp_points, w_t, h_t, inv=True)
    #print(img.shape, ' ', inv_perspective.shape)
    inv_perspective = cv2.addWeighted(img, 1, inv_perspective, 0.7, 0)
    return inv_perspective

def thresholdingY(img):
    imgHsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    lowerYellow = np.array([0,200,255])    # Green: 27, 68, 178
    upperYellow = np.array([0,200,255])   # Green: 35, 112, 223
    maskBlue = cv2.inRange(imgHsv, lowerYellow, upperYellow)
    return maskBlue

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

    # hist_vals = savgol_filter(hist_vals, 51, 3) # Smooth out histogram values; reduce noise
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
    img_thresh = thresholdingY(img)

    # ## Step 2
    h_t, w_t, ch = img.shape
    points = val_trackbars()
    img_warp = warp_image(img_thresh, points, w_t, h_t)
    img_warp_points = draw_points(img_copy, points)

    # ## Step 2.5
    edges = cv2.Canny(img_warp, 0, 1000)
    lines = cv2.HoughLinesP(edges, 1, np.pi/180, 10, maxLineGap=9000)
    if lines is not None:
        for line in lines:
            x1,y1,x2,y2 = line[0]

            # rho,theta = line[0]
            # a = np.cos(theta)
            # b = np.sin(theta)
            # x0 = a * rho
            # y0 = b * rho
            # x1 = int(x0 + 1000 * (-b))
            # y1 = int(y0 + 1000 * (a))
            # x2 = int(x0 - 1000 * (-b))
            # y2 = int(y0 - 1000 * (a))
            cv2.line(img_result, (x1,y1), (x2,y2), (0,0,255), 3)



    print(img_warp.shape)

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
    while cap.isOpened():
        img = get_img()
        img_warp_copy = img
        # gray = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
        img_thresh = thresholding(img)

        h_t, w_t, ch = img.shape
        width_top = 67       # 3
        height_top = 112     # 62
        width_bottom = 0    # 0
        height_bottom = 180  # 96
        points = np.float32([(width_top, height_top), (w_t - width_top, height_top),
                             (width_bottom, height_bottom), (w_t - width_bottom, height_bottom)])
        img_thresh_warp = warp_image(img_thresh, points, w_t, h_t)
        img_warp = warp_image(img, points, w_t, h_t)

        edges = cv2.Canny(img_thresh_warp, 85, 150)

        lines = cv2.HoughLinesP(edges, 1, np.pi/180, 30, maxLineGap=150)
        # print(img_thresh_warp.shape)
        hist_vals, hist_image = get_histogram(img_thresh_warp, display=True, region=2)
        # print(hist_vals.shape, ' ', len(hist_vals.shape))

        if lines is not None and len(hist_vals.shape) == 1:
            for line in lines:
                x1, y1, x2, y2 = line[0]
                cv2.line(img_warp, (x1, y1), (x2, y2), (0, 255, 0), 3)



            out_img, curves, lanes, ploty = sliding_window(img_thresh_warp)
            cv2.imshow("Out IMG", out_img)
            # plt.plot(curves[0], ploty, color='yellow', linewidth=1)
            # plt.plot(curves[1], ploty, color='yellow', linewidth=1)
            # print(np.asarray(curves).shape)

            curverad = get_curve(img, curves[0], curves[1])
            lane_curve = np.mean([curverad[0], curverad[1]])
            print('*** ', curverad, ' + ', lane_curve, ' ***')
            img_ = draw_lanes(img, curves[0], curves[1], warp_points=points)
            # # plt.imshow(img_, cmap='hsv')
            # font = cv2.FONT_HERSHEY_SIMPLEX
            # fontColor = (0, 0, 0)
            # fontSize = 0.5
            # cv2.putText(img, 'Lane Curvature: {:.0f} m'.format(lane_curve), (570, 620), font, fontSize, fontColor, 2)
            # cv2.putText(img, 'Vehicle offset: {:.4f} m'.format(curverad[2]), (570, 650), font, fontSize, fontColor, 2)
            # cv2.imshow("Curve", img_)
            curve = get_lane_curve(img_,display=2)


        cv2.imshow("Image Hist", hist_image)

        cv2.imshow("Edges", edges)
        cv2.imshow("Gray", img_thresh)
        cv2.imshow("Image", img)
        cv2.imshow("Image Warp", img_warp)

        cv2.waitKey(1)

    cap.release()
    cv2.destroyAllWindows()
