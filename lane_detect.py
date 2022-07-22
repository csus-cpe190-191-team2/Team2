#   This file handles camera input to process and detect
#   Lane lines, their curvature in degrees, and the
#   relative center position

import cv2
import numpy as np


def stack_images(scale, img_array):
    rows = len(img_array)
    cols = len(img_array[0])
    rows_available = isinstance(img_array[0], list)
    width = img_array[0][0].shape[1]
    height = img_array[0][0].shape[0]
    if rows_available:
        for x in range(0, rows):
            for y in range(0, cols):
                if img_array[x][y].shape[:2] == img_array[0][0].shape[:2]:
                    img_array[x][y] = cv2.resize(img_array[x][y], (0, 0), None, scale, scale)
                else:
                    img_array[x][y] = cv2.resize(img_array[x][y],
                                                 (img_array[0][0].shape[1], img_array[0][0].shape[0]), None, scale,
                                                 scale)
                if len(img_array[x][y].shape) == 2: img_array[x][y] = cv2.cvtColor(img_array[x][y],
                                                                                   cv2.COLOR_GRAY2BGR)
        image_blank = np.zeros((height, width, 3), np.uint8)
        hor = [image_blank] * rows

        for x in range(0, rows):
            hor[x] = np.hstack(img_array[x])
        ver = np.vstack(hor)
    else:
        for x in range(0, rows):
            if img_array[x].shape[:2] == img_array[0].shape[:2]:
                img_array[x] = cv2.resize(img_array[x], (0, 0), None, scale, scale)
            else:
                img_array[x] = cv2.resize(img_array[x], (img_array[0].shape[1], img_array[0].shape[0]), None, scale,
                                          scale)
            if len(img_array[x].shape) == 2: img_array[x] = cv2.cvtColor(img_array[x], cv2.COLOR_GRAY2BGR)
        hor = np.hstack(img_array)
        ver = hor
    return ver


class LaneDetect:
    def __init__(self, cap_dev=0):
        self.cap = cv2.VideoCapture(cap_dev)  # input device id: 0-3

        self.img = []
        self.img_result = []
        self.points = []
        self.img_thresh = []
        self.img_warp = []
        self.img_warp_inv = []
        self.edges = []
        self.lines = []
        self.hist_vals = []
        self.hist_img = []
        self.error = 0

        # Saved variable paths
        self.warp_fn = 'vars/warp_points.txt'
        self.thresh_fn = 'vars/thresh_points.txt'
        self.hough_fn = 'vars/hough_points.txt'

        if self.cap.isOpened():
            self.get_points()
            self.get_img()
            self.out_img = self.img.copy()
            self.h_t, self.w_t, self.ch = self.img.shape
            self.thresholding()
            self.warp_image()

        else:
            self.error = 1
            print("Cannot bind to capture device: ", cap_dev)

    def destroy(self):
        self.cap.release()
        cv2.destroyAllWindows()

    def get_img(self, display=False, size=[480, 240]):
        ret, self.img = self.cap.read()
        self.img = cv2.resize(self.img, (size[0], size[1]))

        if not ret:
            self.error = 1
        else:
            # Undistort image
            dist_vars = np.load('vars/cam_dist_matrix.npz')
            self.img = cv2.undistort(self.img, dist_vars['mtx'], dist_vars['dist'], None, dist_vars['newcameramtx'])

        if display:
            print('Press \'q\' to quit.')
            while True:
                self.get_img()
                cv2.imshow('IMG', self.img)
                if cv2.waitKey(1) & 0xFF == ord('q'):
                    break

    def thresholding(self):
        thresh_pts = self.points[0]['thresh_points']
        img_copy = self.img.copy()
        img_hsv = cv2.cvtColor(img_copy, cv2.COLOR_BGR2HSV)
        lower_thresh = np.array(thresh_pts[0])   # [81, 54, 140]
        upper_thresh = np.array(thresh_pts[1])    # [128, 255, 255]
        mask = cv2.inRange(img_hsv, lower_thresh, upper_thresh)
        self.img_thresh = mask

    def get_points(self):
        with open(self.warp_fn, 'r') as file:
            warp_points = np.loadtxt(file)
        with open(self.thresh_fn, 'r') as file:
            thresh_points = np.loadtxt(file, dtype=int)
        with open(self.hough_fn, 'r') as file:
            hough_points = np.loadtxt(file, dtype=int)

        self.points = np.array([{'warp_points': warp_points,
                                 'thresh_points': thresh_points,
                                 'hough_points': hough_points}])

    def warp_image(self, img_in=None, inv=False):
        pts1 = np.float32(self.points[0]['warp_points'])
        pts2 = np.float32([[0, 0], [self.w_t, 0], [0, self.h_t], [self.w_t, self.h_t]])

        if img_in is None:
            img_in = self.img_thresh

        if inv:
            # Warp original image with inverse perspective for self.draw_lanes()
            matrix = cv2.getPerspectiveTransform(pts2, pts1)
            self.img_warp_inv = cv2.warpPerspective(img_in, matrix, (self.w_t, self.h_t))
        else:
            # Warp image after threshold filter with standard perspective for lane processing
            matrix = cv2.getPerspectiveTransform(pts1, pts2)
            self.img_warp = cv2.warpPerspective(img_in, matrix, (self.w_t, self.h_t))

    def get_histogram(self, display=False, region=1):
        if region == 1:
            hist_vals = np.sum(self.img_warp, axis=0)
        else:
            hist_vals = np.sum(self.img_warp[self.img.shape[0] // region:, :], axis=0)

        if display:
            img_hist = np.zeros((self.img_warp.shape[0], self.img.shape[1], 3), np.uint8)
            for x, intensity in enumerate(hist_vals):
                cv2.line(img_hist, (x, self.img_warp.shape[0]), (x, self.img_warp.shape[0] - int(intensity) // 255 // region),
                         (255, 0, 255), 1)
                # cv2.circle(self.img_hist, (base_point, self.img.shape[0]), 20, (0,255,255), cv2.FILLED)
            return hist_vals, img_hist
        else:
            return hist_vals

    def sliding_window(self, nwindows=10, margin=60, minpix=10, draw_windows=False):
        left_a, left_b, left_c = [], [], []
        right_a, right_b, right_c = [], [], []
        left_fit_ = np.empty(3)
        right_fit_ = np.empty(3)
        self.out_img = np.dstack((self.img_warp, self.img_warp, self.img_warp)) * 255

        histogram = self.get_histogram(region=2)

        # find peaks of left and right halves
        midpoint = int(histogram.shape[0] / 2)

        # ### EXPERIMENTAL FIX FOR LESS THAN TWO LINES IN VIEW #####################################
        # ### If one lane exists return a one for either left or right depending on which side the
        # ### line is on. If no line exists return zero for left and right.
        # ### Error flag is also set and curve values set to -1.

        if np.sum(histogram[:midpoint]) == 0 or np.sum(histogram[midpoint:]) == 0:

            histogram = self.get_histogram(region=1)    # Get new hist without splitting region

            print("ERROR: LESS THAN TWO LANES IN VIEW")
            leftx_base = np.argmax(histogram[:midpoint])
            rightx_base = np.argmax(histogram[midpoint:])
            color_img = np.zeros_like(self.img)

            if leftx_base > rightx_base:
                left = 1
                right = 0
            elif leftx_base < rightx_base:
                left = 0
                right = 1
            else:
                left = 0
                right = 0

            # Return: radii: (left, right), angle: (left, right), ploty ,error
            if draw_windows:
                return color_img, (-1, -1), (left, right), -1, 1
            else:
                return (-1, -1), (left, right), -1, 1

        # #####################################################################################

        # Grab position of left and right line along the x-axis
        leftx_base = np.argmax(histogram[:midpoint])
        rightx_base = np.argmax(histogram[midpoint:]) + midpoint

        # Set height of windows
        window_height = np.int32(self.img_warp.shape[0] / nwindows)

        # Identify the x and y positions of all nonzero pixels in the image
        nonzero = self.img_warp.nonzero()
        nonzeroy = np.array(nonzero[0])
        nonzerox = np.array(nonzero[1])

        # Current positions to be updated for each window
        leftx_current = leftx_base
        rightx_current = rightx_base

        # Create empty lists to receive left and right lane pixel indices
        left_lane_inds = []
        right_lane_inds = []

        # Step through the windows one by one
        bottom_one = []
        top_second = []
        top_nine = []
        window_count = 0
        for window in range(nwindows):
            # Identify window boundaries in x and y (and right and left)
            win_y_low = self.img_warp.shape[0] - (window + 1) * window_height
            win_y_high = self.img_warp.shape[0] - window * window_height
            win_xleft_low = leftx_current - margin
            win_xleft_high = leftx_current + margin
            win_xright_low = rightx_current - margin
            win_xright_high = rightx_current + margin
            upper_point = [int((win_xleft_high+win_xright_low)/2), int(win_y_high)]
            lower_point = [int((win_xleft_high+win_xright_low)/2), int(win_y_low)]
            img_shape = self.out_img.shape

            if window_count == 1:
                bottom_one = lower_point
            if window_count == 2:
                top_second = upper_point
            if window_count == 8:
                top_nine = upper_point

            # Draw the windows on the visualization image
            if draw_windows:
                cv2.rectangle(self.out_img, (win_xleft_low, win_y_low), (win_xleft_high, win_y_high),
                              (100, 255, 255), 3)
                cv2.rectangle(self.out_img, (win_xright_low, win_y_low), (win_xright_high, win_y_high),
                              (100, 255, 255), 3)
                #cv2.line(self.out_img, (upper_point[0],upper_point[1]), (lower_point[0],lower_point[1]), (0,0,255), 3)
                cv2.line(self.out_img, (int(img_shape[1]/2), int(img_shape[0])), (int(img_shape[1]/2), 0), (255, 255, 0), 2)
                # cv2.line(self.out_img, (0, 240), (480, 0), (0, 0, 200), 4)

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
                leftx_current = np.int32(np.mean(nonzerox[good_left_inds]))
            if len(good_right_inds) > minpix:
                rightx_current = np.int32(np.mean(nonzerox[good_right_inds]))
            window_count += 1
        #cv2.line(self.out_img, (upper_point[0],upper_point[1]), (lower_point[0],lower_point[1]), (0,0,255), 3)
        cv2.line(self.out_img, (top_second[0], top_second[1]), (bottom_one[0], bottom_one[1]), (79,252,17), 2) #short line
        cv2.line(self.out_img, (top_nine[0], top_nine[1]), (bottom_one[0], bottom_one[1]), (17,173,252), 2) #long line
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

        # Generate x and y values for plotting using the quadratic formula: ax^2 + bx + c = 0
        ploty = np.linspace(0, self.img_warp.shape[0] - 1, self.img_warp.shape[0])
        left_fitx = left_fit_[0] * ploty ** 2 + left_fit_[1] * ploty + left_fit_[2]
        right_fitx = right_fit_[0] * ploty ** 2 + right_fit_[1] * ploty + right_fit_[2]

        self.out_img[nonzeroy[left_lane_inds], nonzerox[left_lane_inds]] = [255, 0, 100]
        self.out_img[nonzeroy[right_lane_inds], nonzerox[right_lane_inds]] = [0, 100, 255]

        # Print fitted curve over sliding window overlay
        left = np.vstack((left_fitx, ploty)).T
        right = np.vstack((right_fitx, ploty)).T

        if draw_windows:
            self.out_img = cv2.polylines(img=self.out_img, pts=np.int32([left]),
                                         isClosed=False, color=(0, 255, 0), thickness=3)
            self.out_img = cv2.polylines(img=self.out_img, pts=np.int32([right]),
                                         isClosed=False, color=(0, 255, 0), thickness=3)
            return self.out_img, (left_fitx, right_fitx), (left_fit_, right_fit_), ploty, 0
        else:
            return (left_fitx, right_fitx), (left_fit_, right_fit_), ploty, 0

    def get_curve(self):
        self.get_img()
        self.thresholding()
        self.warp_image()
        curves, lanes, ploty, error = self.sliding_window()
        leftx = curves[0]
        rightx = curves[1]

        if error:
            # DEBUG output
            print("Curves:", curves)
            print("Lanes:", lanes)
            print("Ploty:", ploty)
            print("Error:", error, '\n')
            return leftx, rightx, lanes[0], lanes[1], ploty, error

        ploty = np.linspace(0, self.img_warp.shape[0] - 1, self.img_warp.shape[0])
        y_eval = np.max(ploty)

        # Calculate PPI
        ym_per_pix = 35 / 240  # 50; 35 / 240 meters per pixel in y dimension
        xm_per_pix = 13 / 480  # 13; inches per pixel in x dim  # 13 / 480 meters per pixel in x dimension

        # Fit new polynomials to x,y in world space
        left_fit_cr = np.polyfit(ploty * ym_per_pix, leftx * xm_per_pix, 2)
        right_fit_cr = np.polyfit(ploty * ym_per_pix, rightx * xm_per_pix, 2)

        # Using the equation calculated from the least squares polyfit,
        # Calculate the new radii of curvature R = (1+(dy/dx)^2)^3/2)/|d^2y/dx|
        left_curverad = ((1 + (2 * left_fit_cr[0] * y_eval * ym_per_pix + left_fit_cr[1]) ** 2) ** 1.5) / np.absolute(
            2 * left_fit_cr[0])
        right_curverad = ((1 + (
                    2 * right_fit_cr[0] * y_eval * ym_per_pix + right_fit_cr[1]) ** 2) ** 1.5) / np.absolute(
            2 * right_fit_cr[0])

        left_circum = 2 * np.pi * left_curverad
        right_circum = 2 * np.pi * right_curverad
        left_angle = (leftx.shape[0] * 360) / left_circum
        right_angle = (rightx.shape[0] * 360) / right_circum

        # Quadratic Formula
        car_pos = self.img_warp.shape[1] / 2
        l_fit_x_int = left_fit_cr[0] * self.img_warp.shape[0] ** 2 + left_fit_cr[1] * self.img_warp.shape[0] + left_fit_cr[2]
        r_fit_x_int = right_fit_cr[0] * self.img_warp.shape[0] ** 2 + right_fit_cr[1] * self.img_warp.shape[0] + right_fit_cr[2]
        lane_center_position = (r_fit_x_int + l_fit_x_int) / 2
        center = (car_pos - lane_center_position)
        center = (car_pos - center)//10

        # DEBUG output
        print("Radian:", left_curverad, '\t', right_curverad)
        print("Circumference:", left_circum, '\t', right_circum)
        print("Angle:", left_angle, '\t', right_angle)
        print("Center:", center)
        print("Error:", error, '\n')

        return left_curverad, right_curverad, left_angle, right_angle, center, error

    def draw_lanes(self, left_curve, right_curve, center):
        out_img, curves, lanes, ploty, error = self.sliding_window(draw_windows=True)
        left_fit = curves[0]
        right_fit = curves[1]

        color_img = np.zeros_like(self.img)

        if error:
            self.img_result = self.img
            return self.img, self.img

        left = np.array([np.transpose(np.vstack([left_fit, ploty]))])
        right = np.array([np.flipud(np.transpose(np.vstack([right_fit, ploty])))])
        points = np.hstack((left, right))

        # Draw inside lane
        cv2.fillPoly(color_img, np.int_(points), (0, 200, 255))
        self.warp_image(img_in=color_img, inv=True)
        inv_perspective = cv2.addWeighted(self.img, 1, self.img_warp_inv, 0.7, 0)

        # Write var text to image result then stack images
        mean_radii = np.mean([left_fit[0], right_fit[1]])/100
        font = cv2.FONT_HERSHEY_SIMPLEX
        font_color = (255, 0, 255)
        font_size = 0.6
        text_str1 = 'Left: ' + '{:.2f}'.format(left_curve) + ' Right: ' \
                    + '{:.2f}'.format(right_curve) + ' deg '
        text_str2 = '{:.2f}'.format(mean_radii) + ' rad/in'
        cv2.putText(inv_perspective, 'Lane Curve: {:}'.format(text_str1),
                    (inv_perspective.shape[1] // 2 - 220, inv_perspective.shape[0] - 215), font, font_size, font_color, 2)
        cv2.putText(inv_perspective, 'Lane Radii: {:}'.format(text_str2),
                    (inv_perspective.shape[1] // 2 - 220, inv_perspective.shape[0] - 190), font, font_size, font_color,2)
        cv2.putText(inv_perspective, 'Vehicle offset: {:.4f} in'.format(center),
                    (inv_perspective.shape[1] // 2 - 220, inv_perspective.shape[0] - 165), font, font_size, font_color, 2)
        self.img_result = stack_images(1, ([inv_perspective, out_img]))

        return inv_perspective, out_img

    def display(self, left_cr, right_cr, center, display=False):
        lane_area, sliding_windows = self.draw_lanes(left_cr, right_cr, center)
        if display:
            cv2.imshow('Image Result vs Sliding Window Curve Fit', self.img_result)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                lane.destroy()

        return lane_area, sliding_windows


# ## Test Driver
if __name__ == '__main__':  # Program start from here
    lane = LaneDetect()

    while not lane.error:
        lane_curve = lane.get_curve()
        lane.display(lane_curve[2], lane_curve[3], lane_curve[4], display=True)


