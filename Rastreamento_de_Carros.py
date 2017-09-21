import cv2
import numpy as np

face_cascade = cv2.CascadeClassifier('cars.xml')
source = 'road.mp4'
source2 = 'video.avi'
source3 = 'MOV_0022.mp4'
source4 = 'trafficRockSanteiro.mp4'
vc = cv2.VideoCapture(source2)

if vc.isOpened():
    rval, frame = vc.read()
else:
    rval = False
    print (rval)

while rval:
    rval, frame = vc.read()

    cars = face_cascade.detectMultiScale(frame, 1.1, 2)

    for (x, y, w, h) in cars:
        cv2.rectangle(frame,(x, y), (x+w, y+h), (0, 0, 255), 2)

    cv2.imshow("Result", frame)
    cv2.waitKey(1);
vc.release()