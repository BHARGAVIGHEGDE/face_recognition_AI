#bgr to hsv 
#dst =cv2.cvtColor(blurred,cv2.COLOR_BGR2HSV)

#minimum enclosing  circle
#((x,y),radius)=cv2.minEnclosingCircle(CountourArea) 

#moments to find the centre of area
#center(int(M["m10"]/M[m"00"]),int(M["m01"]/M["m00"]))

#drawing Circle
#cv2.Circle(src,(x,y),int(radius),color,thickness)
#cv2.circle(frame,(int(x),iny(y)),int(radius),(0,255,255),2)


import imutils#to rezize the image
import cv2#for image processing

redLower=(27,47,150)#to fetch hsv points(hue,saturation,value)
redUpper=(179,255,255)#to fetch hsv points(hue,saturation,value)



img=cv2.VideoCapture(0)#to initiate the video

while True:
    (grabbed,frame)=img.read()#to read the image
    frame=imutils.resize(frame,width=600)#to resize the image
    blurred=cv2.GaussianBlur(frame,(11,11),0)#to blur the image
    hsv=cv2.cvtColor(blurred,cv2.COLOR_BGR2HSV)#to convert the blurred image with rgb to hsv 



    mask=cv2.inRange(hsv,redLower,redUpper)
    mask=cv2.erode(mask,None,iterations=2)
    mask=cv2.dilate(mask,None,iterations=2)

    cnts=cv2.findContours(mask.copy(),cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_SIMPLE)#to find the contours of the image

    centre=None
    if len(cnts)>0:
        key=cv2.contourArea
        c=max(cnts,key)
        ((x,y),radius)=cv2.minEnclosingCircle(c)
        M=cv2.moments(c)#useful for image shape analysis
        center=(int(M["m10"]/M["m00"]),int(M["m01"]/M["m00"]))
        if radius>10:
            cv2.circle(frame,(int(x),int(y)),int(radius),(0,255,255),2)#for yellow circle
            cv2.circle(frame,center,5,(0,0,255),-1)#for red circle
            if radius>25:
                print("stop")
            else:
                if(center[0]<150):
                    print("left")
                elif(center[0]>450):
                    print("right")
                elif(radius<250):
                    print("front")
                else:
                    print("stop")     
    cv2.imshow("frame",frame)                   
    key=cv2.waitKey(1)&0xFF
    if(key==ord('q')):
        break
img.release()
cv2.destroyAllWindows()    

                
