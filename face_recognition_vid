import numpy as np
import os 
import cv2 as cv

haar_cascade = cv.CascadeClassifier('haar_face.xml')

p =[]
for i in os.listdir(r'D:\\Education\\Self\\OpenCV\\Faces'):
    p.append(i)


feature =np.load('Feature.npy',allow_pickle=True)
label =np.load('Labels.npy',allow_pickle=True)



face_recognizer = cv.face.LBPHFaceRecognizer_create()
face_recognizer.read('face_trained.yml')

capture = cv.VideoCapture(0)
while True:
    isTrue, frame =capture.read()
    gray = cv.cvtColor(frame, cv.COLOR_BGR2GRAY) 
    cv.imshow('Gray', gray)
    faces_rect = haar_cascade.detectMultiScale(gray, 1.1,4)
    for(x,y,w,h) in faces_rect:
        faces_roi = gray[x:x+h, y:y+w]

        label , confidence = face_recognizer.predict(faces_roi)
        print(f'label = {p[label]} with a confidence of {confidence}')

        cv.putText(frame, str(p[label]),(20,20), cv.FONT_HERSHEY_COMPLEX,1.0,(0,255,0), thickness=2)
        cv.rectangle(frame,(x,y),(x+w,y+h),(0,255,0),2)
    cv.imshow('LIVE',frame)
    if cv.waitKey(20) & 0xFF==ord('d'): # if letter d is pressed then the video will stop
        break
capture.release()
cv.destroyAllWindows()    
