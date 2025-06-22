import cv2 as cv
import numpy as np

haar_cascade = cv.CascadeClassifier('D:\Education\Self\OpenCV\Face Detection and Recognition\haar_face.xml')
capture = cv.VideoCapture(0)
while True:
    isTrue,frame = capture.read()
    # gray = cv.cvtColor(frame,cv.COLOR_BGR2GRAY)
    # cv.imshow('LIVE',frame)
    faces_rect = haar_cascade.detectMultiScale(frame, scaleFactor=1.1, minNeighbors= 3)
    # print(len(faces_rect))
    for (x,y,w,h) in faces_rect:
        cv.rectangle(frame,(x,y),(x+h,y+h),(0,255,0),2)
        cv.imshow('LIVE',frame)

    if cv.waitKey(20) & 0xFF==ord('d'): # if letter d is pressed then the video will stop
        break
capture.release()
cv.destroyAllWindows()
