"""
#    Acquire camera input and apply
#    threshold and/or warp to the output image.
"""

import cv2
import numpy as np
import time


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


class Eyes:
    def __init__(self,cap_dev=0):
        self.cap = cv2.VideoCapture(cap_dev)  # input device id: 0-3
        self.img = []
        self.points = []
        self.img_thresh = []
        self.img_warp = []
        self.img_warp_inv = []
        self.img_grayscale = []

        # Saved variable paths
        self.warp_fn = 'vars/warp_points.txt'
        self.thresh_fn = 'vars/thresh_points.txt'

        # Load camera distortion matrix
        self.dist_vars = np.load('vars/cam_dist_matrix.npz')

        if self.cap.isOpened():
            self.get_points()

    def destroy(self):
        self.cap.release()
        cv2.destroyAllWindows()

    def get_points(self):
        with open(self.warp_fn, 'r') as file:
            warp_points = np.loadtxt(file)
        with open(self.thresh_fn, 'r') as file:
            thresh_points = np.loadtxt(file, dtype=int)

        self.points = np.array([{'warp_points': warp_points,
                                 'thresh_points': thresh_points}])

    def cap_img(self, size=[480, 240]):
        ret, self.img = self.cap.read()
        self.img = cv2.resize(self.img, (size[0], size[1]))

        if not ret:
            self.img = None
        else:
            # Undistort image
            self.img = cv2.undistort(self.img, self.dist_vars['mtx'], self.dist_vars['dist'], None, self.dist_vars['newcameramtx'])

    def get_img(self):
        self.cap_img()
        return self.img

    def get_thresh_img(self, img=None):
        if img is None:
            self.cap_img()
            self.thresholding()
        else:
            self.img = img
            self.thresholding()
        return self.img_thresh

    def get_grayscale_img(self, get_img=False):
        if get_img is True:
            self.cap_img()
        img_copy = self.img.copy()
        self.img_grayscale = cv2.cvtColor(img_copy, cv2.COLOR_BGR2GRAY)


    def camera_warm_up(self, warm_time=5):
        print("Warming Up Camera...")
        for x in range(warm_time, 0, -1):
            print(f"{x} seconds remain...")
            self.get_img()
            time.sleep(1)
        print("Done.")

    def thresholding(self):
        thresh_pts = self.points[0]['thresh_points']
        img_copy = self.img.copy()
        img_hsv = cv2.cvtColor(img_copy, cv2.COLOR_BGR2HSV)
        lower_thresh = np.array(thresh_pts[0])   # [81, 54, 140]
        upper_thresh = np.array(thresh_pts[1])    # [128, 255, 255]
        mask = cv2.inRange(img_hsv, lower_thresh, upper_thresh)
        self.img_thresh = mask


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

    def show_img(self, img, string="Thresh Img", wait_ms=8000):
        cv2.imshow(string, img)
        cv2.waitKey(wait_ms)

    def save_img(self, path):
        if self.img is not None:
            cv2.imwrite(path, self.img)

    def save_warp_img(self, path):
        if self.img_warp is not None:
            cv2.imwrite(path, self.img_warp)

    def save_warp_inv_img(self, path):
        if self.img_warp_inv is not None:
            cv2.imwrite(path, self.img_warp_inv)

    def save_thresh_img(self, path):
        if self.img_thresh is not None:
            cv2.imwrite(path, self.img_thresh)

    def save_bw_img(self, path):
        if self.img_grayscale is not None:
            cv2.imwrite(path, self.img_grayscale)

