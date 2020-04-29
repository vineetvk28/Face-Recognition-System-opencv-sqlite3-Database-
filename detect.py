#DETECT
import cv2
import sys
import os
import numpy as np
import sqlite3
faceDetect=cv2.CascadeClassifier('haarcascade_frontalface_default.xml')
cam=cv2.VideoCapture(0);
rec=cv2.face.LBPHFaceRecognizer_create()
rec.read('/Users/vineetkargeti/Downloads/training/trainingData.yml')
def getProfile(id):
    conn=sqlite3.connect("/Users/vineetkargeti/Downloads/FaceBase.db")
    cmd="SELECT * FROM People WHERE ID="+str(id)
    cursor=conn.execute(cmd)
    profile=None
    for row in cursor:
        profile=row
    conn.close()
    return profile
id=0
font=cv2.FONT_HERSHEY_COMPLEX_SMALL
fontcolor =(0,255,0)
while(True):
    ret,img=cam.read()
    gray=cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
    faces=faceDetect.detectMultiScale(gray,1.3,5)
    for(x,y,w,h) in faces:
        cv2.rectangle(img,(x,y),(x+w,y+h),(0,0,255),2)
        #cv2.rectangle(img,(x,y),(x+w,y+h),(0,255,0),2)
        id,conf=rec.predict(gray[y:y+h,x:x+w])
        profile=getProfile(id)
        if(profile!=None):
            cv2.putText(img,str(profile[1]),(x,y+h+30),font,1,fontcolor,2)
            cv2.putText(img,str(profile[2]),(x,y+h+60),font,1,fontcolor,2)
            cv2.putText(img,str(profile[3]),(x,y+h+90),font,1,fontcolor,2)
            #cv2.putText(img,str(profile[4]),(x,y+h+120),font,10,fontcolor,8)
    cv2.imshow("Face",img)
    if(cv2.waitKey(1)==ord('q')):
        break
cam.release()
cv2.destroyAllWindows()        