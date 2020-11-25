import cv2
from cv2 import CascadeClassifier
import face_recognition
import os
import numpy as np

path = 'faces'
images = []
classNames = []
MyList = os.listdir(path)
#print(MyList)

for cl in MyList:
    curimg  = cv2.imread(f'{path}/{cl}')
    images.append(curimg)
    classNames.append(os.path.splitext(cl)[0])
print(classNames)

def findEncodings(images):
    encodeList = []
    for img in images:
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        encode = face_recognition.face_encodings(img)[0]
        encodeList.append(encode)
    return encodeList

encodeListKnown = findEncodings(images)
print("Encoding Complete")

# Initzalize VideoCapture

cap = cv2.VideoCapture(0)
classifier = CascadeClassifier('haarcascade_frontalface_default.xml')
# Finds a face in the Webcam
while True:
    ret, img = cap.read()
    imgs = cv2.resize(img, (0,0), None, 0.25, 0.25)
    imgs = cv2.cvtColor(imgs, cv2.COLOR_BGR2RGB)

    faceCurFrame = face_recognition.face_locations(imgs)
    encodeCurFrame = face_recognition.face_encodings(imgs, faceCurFrame)

# Finds and names the face
    for encodeface, faceLoc in zip(encodeCurFrame, faceCurFrame):
        matches = face_recognition.compare_faces(encodeListKnown, encodeface)

        faceDis = face_recognition.face_distance(encodeListKnown, encodeface)
        #print(faceDis)
        matchIndex = np.argmin(faceDis)

        if matches[matchIndex]:
            name = classNames[matchIndex].upper()
            #print(name)
            y1,x1,y2,x2 = faceLoc
            y1,x1,y2,x2 = y1*4,x1*4,y2*4,x2*4
            cv2.rectangle(img, (x1,y1), (x2,y2), (0, 255, 0), 1)
            cv2.rectangle(img, (x1,y2-35), (x2, y2), (0, 255, 0), cv2.FILLED)
            cv2.putText(img, name, (x1-150, y2-6), cv2.FONT_HERSHEY_COMPLEX, 1, (255, 255, 255), 1 )

        else:
            name = "Unknown"
            y1,x1,y2,x2 = faceLoc
            y1,x1,y2,x2 = y1*4,x1*4,y2*4,x2*4
            cv2.rectangle(img, (x1,y1), (x2,y2), (0, 0, 255), 1)
            cv2.rectangle(img, (x1,y2-35), (x2, y2), (0, 0, 255), cv2.FILLED)
            cv2.putText(img, name, (x1-150, y2-6), cv2.FONT_HERSHEY_COMPLEX, 1, (255, 255, 255), 1 )

# Initzalize Webcam
    cv2.imshow('webcam', img)
    cv2.waitKey(1)
