import cv2 #images
import time #delay
import imutils #resize 

cam=cv2.VideoCapture(0)#video initilization
time.sleep(1)#wait for 1 second after the camera is initiliazed
firstframe=None#the first frame is initilized to none
area=500
while True:
  _,img=cam.read()#read frame from camera
  text="Normal"
  img=imutils.resize(img) #resize
  grayimage=cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)#color to gray scale image
  gaussianImg=cv2.GaussianBlur(grayimage,(21,21),0)#smoothened

  if firstframe is None:
    firstframe=gaussianImg#capturing 1st frame on 1st iterations
    continue
  imgDiff=cv2.absdiff(firstframe,gaussianImg)#absolute difference between firstframe and the lastframe
  threshImg=cv2.threshold(imgDiff,25,255,cv2.THRESH_BINARY)[1]
  threshImg=cv2.dilate(threshImg,None,iterations=2)
  cnts=cv2.findContours(threshImg.copy(),cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_SIMPLE)
  cnts=imutils.grab_contours(cnts)
  for c in cnts:
    if cv2.contourArea(c) < area:
      continue
    (x,y,w,h)=cv2.boundingRect(c) 
    cv2.rectangle(img,(x,y),(x+w,y+h),(0,255,0),2)
    text="moving object detected"
    print(text)
    cv2.putText(img,text,(10,20),
    cv2.FONT_HERSHEY_SIMPLEX,0.5,(0,0,255),2)
    cv2.imshow("camerafeed",img)
    key=cv2.waitKey(1)&0xff
    if key==ord("q"):
      break
  cam.release()
  cv2.destroyAllWindows()


