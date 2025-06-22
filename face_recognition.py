import numpy as np
import cv2 as cv
import os

haar_cascade = cv.CascadeClassifier('haar_face.xml')

# p = ['Ben Affleck', 'Henry Cavill','Jake Gyllenhaal','Margot Robbie']
p =[]
for i in os.listdir(r'D:\\Education\\Self\\OpenCV\\Faces'):
    p.append(i)


feature =np.load('Feature.npy',allow_pickle=True)
label =np.load('Labels.npy',allow_pickle=True)



face_recognizer = cv.face.LBPHFaceRecognizer_create()
face_recognizer.read('face_trained.yml')

img = cv.imread('D:\Education\Self\OpenCV\Photos\\megan.jpg')
gray = cv.cvtColor(img, cv.COLOR_BGR2GRAY)
cv.imshow('Person', gray)

# Detect Face
faces_rect = haar_cascade.detectMultiScale(gray, 1.1,4)

for(x,y,w,h) in faces_rect:
    faces_roi = gray[x:x+h, y:y+w]

    label , confidence = face_recognizer.predict(faces_roi)
    print(f'label = {p[label]} with a confidence of {confidence}')

    cv.putText(img, str(p[label]),(20,20), cv.FONT_HERSHEY_COMPLEX,1.0,(0,255,0), thickness=2)
    cv.rectangle(img,(x,y),(x+w,y+h),(0,255,0),2)
cv.imshow('Detected Face',img)
cv.waitKey(0)    
