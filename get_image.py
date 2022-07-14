# Take a series of pictures using camera for camera_calibration.py
# Print and use /images/camera_calibration/test/Checkerboard-A1-75mm-8x6.svg
# Place image on flat surface
# (rotated 90 degrees or in vertical orientation to avoid dimensional conflict in camera_calibration.py)
# After each picture, move camera slightly at different angles and distances for best calibration results

import cv2
import time

cap = cv2.VideoCapture(0)   # video capture source camera
cap_num = 0
path = 'images/c' + str(cap_num) + '.png'

print("### Warming up camera...")
for x in range(5, 0, -1):
    print("Starting in: ", x)
    ret, frame = cap.read()
    frame = cv2.resize(frame, (480, 240))
    cv2.imshow('Warm-up', frame)  # display the captured image
    time.sleep(1)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        cv2.destroyAllWindows()

print("### Starting Capture... ###")
while True:
    for i in range(0, 10):
        for t in range(0, 5):
            print(t+1, "/5")
            time.sleep(1)
        print("### CAPTURE", i+1, " of ", 10, "###")
        ret, frame = cap.read()  # return a single frame in variable `frame`
        frame = cv2.resize(frame, (480, 240))
        cv2.imshow('img1', frame)   # display the captured image
        cv2.imwrite(path, frame)
        cap_num += 1
        path = 'images/c' + str(cap_num) + '.png'

    if cv2.waitKey(1) & 0xFF == ord('q'):
        cv2.destroyAllWindows()
        break

    break

cap.release()
