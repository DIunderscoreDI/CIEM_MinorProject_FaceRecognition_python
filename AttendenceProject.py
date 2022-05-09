# import cv2
import cv2 as cv
import numpy as np
import face_recognition
import os
from datetime import datetime


path ='imageAttendence'
images =[]
className=[]
myList=os.listdir(path)
print(myList)
for cl in myList:
    curImg =cv.imread(f'{path}/{cl}')
    images.append(curImg)
    className.append(os.path.splitext(cl)[0])
print(className)

def findEncodings(images):
    encodeList =[]
    for img in images:
        img=cv.cvtColor(img,cv.COLOR_BGR2RGB)
        encode = face_recognition.face_encodings(img)[0]
        encodeList.append(encode)
    return encodeList
#
def markAttendence(name):
    with open("Attendence.csv",'r+') as f:
        myDataList= f.readline()
        nameList=[]
        for line in myDataList:
            entry=line.split(',')
            nameList.append(entry[0])
        if name not in nameList:
            now =datetime.now()
            dtString =now.strftime('%H:%M:%S')
            f.writelines(f'\n{name},{dtString}')

encodeListKnown = findEncodings(images)
print('Encode Complete')

# cap = cv.VideoCapture(1)
# url ="http://192.168.0.126:8080/"
# cap.open(url)
cap = cv.VideoCapture(0) #access  web camera

while True:
    success, img = cap.read()
    imgS = cv.resize(img,(0,0),None,0.25,0.25)
    imgS = cv.cvtColor(imgS, cv.COLOR_BGR2RGB)

    facesCurFrame = face_recognition.face_locations(imgS)
    encodeCurFrame = face_recognition.face_encodings(imgS,facesCurFrame) #web: convert image to frame

    for encodeFace,facelog in zip(encodeCurFrame,facesCurFrame):
        matches = face_recognition.compare_faces(encodeListKnown,encodeFace)
        faceDis= face_recognition.face_distance(encodeListKnown,encodeFace)
        # print(faceDis)
        matchIndex=np.argmin(faceDis)

        if matches[matchIndex]:
            name = className[matchIndex].upper()
            # print(name)
            y1,x2,y2,x1=facelog
            y1, x2, y2, x1=y1*4,x2*4,y2*4,x1*4
            cv.rectangle(img,(x1,y1),(x2,y2),(0,255,0),2)
            cv.rectangle(img,(x1,y2-35),(x2,y2),(0,255,0),cv.FILLED)
            cv.putText(img,name,(x1+6,y2-6),cv.FONT_HERSHEY_COMPLEX,1,(255,255,260),2)
            markAttendence(name)

    cv.imshow('webcam',img)
    cv.waitKey(1)
