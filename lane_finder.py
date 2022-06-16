import math
import cv2
import numpy as np

def stack_images():
    pass

class LaneFinder:
    def __init__(self, cap_dev=0):
        self.cap = cv2.VideoCapture(cap_dev)
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
        self.angle = 0

        self.warp_fn = 'vars/warp_points.txt'
        self.thresh_fn = 'vars/thresh_points.txt'

        if self.cap.isOpened():
            self.get_points()

    def get_points(self):
        with open(self.warp_fn, 'r') as file:
            warp_points = np.loadtxt(file)
        with open(self.thresh_fn, 'r') as file:
            thresh_points = np.loadtxt(file, dtype=int)
        self.points = np.array([{'warp_points': warp_points,
                                 'thresh_points': thresh_points}])

    ######
    def get_curve(self): #gets image stats
        self.get_img()
        self.thresholding()
        self.warp_image()
        curves, lanes, ploty, error = self.sliding_window()
        leftx = curves[0]  #might have to turn this into an array
        rightx = curves[1] #for dynamic lane tracking

        if error:
            return 0, 0, lanes[0], lanes[1], 0, 1

    def get_img(self, display=False, size=[480,240]):
        ret, self.img = self.cap.read()
        self.img = cv2.resize(self.img, size) #(size[0], size[1])
        self.h_t, self.w_t, self.ch = self.img.shape
        if not ret:
            self.error = 1
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
        lower_thresh = np.array(thresh_pts[0])
        upper_thresh = np.array(thresh_pts[1])
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

    def sliding_window(self, nwindows=9, margin=60, minpix=10, draw_windows=False): #polyfit and window drawing and finds angle
        pass

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
    ######
    def display(self, left_cr, right_cr, center):
        self.draw_lanes(left_cr, right_cr, center)
        font = cv2.FONT_HERSHEY_SIMPLEX
        fontColor = (255, 0, 255)
        fontSize = 0.5
        cv2.putText(self.img_result, 'Angle: {:.2f} in'.format(self.angle),
                    (self.img_result.shape[1] // 2 - 100, self.img_result.shape[0] - 35), font, fontSize, fontColor, 2)

        cv2.imshow('Image Result vs Sliding Window Curve Fit', self.img_result)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            self.error = 1
            self.destroy()

    def draw_lanes(self):
        pass

    def destroy(self):
        self.cap.release()
        cv2.destroyAllWindows()
