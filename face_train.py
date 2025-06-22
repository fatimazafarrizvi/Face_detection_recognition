import os
import cv2 as cv
import numpy as np
 
#people = ['Ben Affleck', 'Henry Cavill','Jake Gyllenhaal','Margot Robbie']

p =[]
for i in os.listdir(r'D:\\Education\\Self\\OpenCV\\Faces'):
    p.append(i)

DIR = r'D:\Education\Self\OpenCV\Faces'

haar_cascade = cv.CascadeClassifier('haar_face.xml')

feature = [] # Image array of the faces
labels = [] # Corresponding label

def create_train(): #it will loop over every folder and every image and grab the face and add it to our training set
    for person in p:
        path = os.path.join(DIR,person)
        label = p.index(person)
        for img in os.listdir(path):
            img_path = os.path.join(path,img)
            img_array = cv.imread(img_path)
            print(img_path)
            gray = cv.cvtColor(img_array, cv.COLOR_BGR2GRAY)
            
            # Detect Faces
            faces_rect = haar_cascade.detectMultiScale(gray,scaleFactor=1.1,minNeighbors=4)
            
            # Cropping out of faces
            for (x,y,w,h) in faces_rect:
                faces_roi = gray[y:y+h, x:x+w] 
                feature.append(faces_roi)
                labels.append(label)
create_train()

print(' Training Done ----------')
face_recognizer = cv.face.LBPHFaceRecognizer_create()

feature=np.array(feature, dtype='object')
labels=np.array(labels)


# Train the recognizer on the feature list and the labels list
face_recognizer.train(feature,labels)

face_recognizer.save('face_trained.yml')
np.save('Feature.npy', feature)
np.save('Labels.npy',labels)
