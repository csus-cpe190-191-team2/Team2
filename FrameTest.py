import cv2
import numpy as np

if __name__=='__main__':
    #init camera
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        raise IOError('Cannot open webcam')
    #capture frame
    ret, frame = cap.read()
    frame = cv2.resize(frame, (480,240))
    #grayscale and edge detect
    imghsv = cv2.cvtColor(frame,cv2.COLOR_BGR2HSV)
    lowerBlue = np.array([99,118,0])
    upperBlue = np.array([174,255,200])
    imghsv = cv2.inRange(imghsv, lowerBlue, upperBlue)
    #Warp the threshold
    # img_copy = imghsv.copy()
    # h,w,ch = imghsv.shape
    #Hough Transform
    edges = cv2.Canny(imghsv, 75, 150)
    lines = cv2.HoughLinesP(edges, 1, np.pi/180, 50, maxLineGap=10, minLineLength=35)
    for line in lines:
        x1,y1,x2,y2 = line[0]
        cv2.line(frame, (x1,y1), (x2,y2), (0,0,255), 2)
    #show frames
    cv2.imshow('Frame', frame)
    cv2.imshow('Gray Frame', edges)
    # cv2.imshow('Warp',img_warp)

    cv2.waitKey(0)

    cap.release()
    cv2.destroyAllWindows()

# def warp_image(img, points, w, h, inv=False):
#     pts1 = np.float32(points)
#     pts2 = np.float32([[0, 0], [w, 0], [0, h], [w, h]]) # np.float32([[0, 0], [w, 0], [0, h], [w, h]])
#     if inv:
#         matrix = cv2.getPerspectiveTransform(pts2, pts1)
#     else:
#         matrix = cv2.getPerspectiveTransform(pts1, pts2)
#     img_warp = cv2.warpPerspective(img, matrix, (w,h))
#     return img_warp
# def initialize_trackbars(init_trackbar_vals, w_t=480, h_t=240):
#     cv2.namedWindow("Trackbars")
#     cv2.resizeWindow("Trackbars", 360, 240)
#     cv2.createTrackbar("Width Top", "Trackbars", init_trackbar_vals[0], w_t//2, empty)
#     cv2.createTrackbar("Height Top", "Trackbars", init_trackbar_vals[1], h_t, empty)
#     cv2.createTrackbar("Width Bottom", "Trackbars", init_trackbar_vals[2], w_t // 2, empty)
#     cv2.createTrackbar("Height Bottom", "Trackbars", init_trackbar_vals[3], h_t, empty)
#
#
# def val_trackbars(w_t=480, h_t=240):
#     width_top = cv2.getTrackbarPos("Width Top", "Trackbars")
#     height_top = cv2.getTrackbarPos("Height Top", "Trackbars")
#     width_bottom = cv2.getTrackbarPos("Width Bottom", "Trackbars")
#     height_bottom = cv2.getTrackbarPos("Height Bottom", "Trackbars")
#     points = np.float32([(width_top, height_top), (w_t - width_top, height_top),
#                          (width_bottom, height_bottom), (w_t - width_bottom, height_bottom)])
#     return points