import cv2
import numpy as np

face_cascade = cv2.CascadeClassifier('cars.xml')
source = 'road.mp4'
vc = cv2.VideoCapture(source)

if vc.isOpened():
    rval, frame = vc.read()
else:
    rval = False
    print(rval)

while rval:
    rval, frame = vc.read()

    cars = face_cascade.detectMultiScale(frame, 1.1, 2)

    ncars = 0
    for (x, y, w, h) in cars:
        cv2.rectangle(frame,(x, y), (x+w, y+h), (0, 0, 255), 2)
        ncars = ncars + 1

    cv2.imshow("Result", frame)
    cv.waitKey(1);
vc.release()
