import cv2
from random import randrange as r
# from cv2 import CAP_DSHOW

#dataset load

trainedData=cv2.CascadeClassifier('C:/Users/neele/OneDrive/Desktop/mini project/face.xml')

#start the webcam

webcam=cv2.VideoCapture(0)


while True:
  success,img = webcam.read()   #image==frameS
  
   #Conversion to black and white(grayscale)
  grayimg=cv2.cvtColor(img,  cv2.COLOR_BGR2GRAY)

    #detect faces
  faceCordinates=trainedData.detectMultiScale(grayimg)

  for x,y,w,h in faceCordinates:
      #cv2.rectangle(img,(cordinates),(width,height),(color in BGR),Thickness)
      cv2.rectangle(img,(x,y),(x+w,y+h),(r(0,255),r(0,255),r(0,255)),2)

  #show images
  cv2.imshow("face detection", img)
  key=cv2.waitKey(1)
  if(key==81 or key==113):
    break
webcam.release()
cv2.destroyAllWindows  








