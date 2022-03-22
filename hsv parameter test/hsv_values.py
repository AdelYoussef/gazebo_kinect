
import cv2
import numpy as np 
import imutils 
from std_msgs.msg import Int32


def nothing(x):
    pass


cv2.namedWindow("TRACK")
cv2.createTrackbar("L-H","TRACK",0,255,nothing)
cv2.createTrackbar("L-S","TRACK",0,255,nothing)
cv2.createTrackbar("L-V","TRACK",0,255,nothing)
cv2.createTrackbar("U-H","TRACK",255,255,nothing)
cv2.createTrackbar("U-S","TRACK",255,255,nothing)
cv2.createTrackbar("U-V","TRACK",255,255,nothing)

while True:
    frame = cv2.imread('/home/adel/vision/box.png')
    frame=cv2.resize(frame,None,fx=0.6,fy=0.6)
    hsv =cv2.cvtColor(frame,cv2.COLOR_BGR2HSV)
    LH=cv2.getTrackbarPos("L-H","TRACK")
    LS=cv2.getTrackbarPos("L-S","TRACK")
    LV=cv2.getTrackbarPos("L-V","TRACK")
    UH=cv2.getTrackbarPos("U-H","TRACK")
    US=cv2.getTrackbarPos("U-S","TRACK")
    UV=cv2.getTrackbarPos("U-V","TRACK")
    
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    low_red=np.array([LH,LS,LV])
    high_red=np.array([UH,US,UV])
    
    mask = cv2.inRange(hsv, low_red, high_red)
    
    cnts = cv2.findContours(mask, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    cnts = imutils.grab_contours(cnts)
    
    for c in cnts:
        area = cv2.contourArea(c)
        if area> 0:
            
            M = cv2.moments(c)
            
            cx = int(M["m10"]/ M["m00"])
            cy = int(M["m01"]/ M["m00"])
            ccx=str(cx)
            ccy=str(cy)

            cv2.drawContours(frame, [c], -1, (0,255,0), 1)
            cv2.circle(frame, (cx, cy), 7, (255,255,255), -1)
            cv2.putText(frame, "CENTER", (cx-20, cy-20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0,0,0), 2)

            cv2.putText(frame,ccx,(cx+20, cy+20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0,0,0),2)
            cv2.putText(frame,ccy,(cx+60, cy+20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0,0,0),2)
            
            cv2.imshow("Frame", frame)
            cv2.imshow("mask",mask)

    k = cv2.waitKey(1)
    if k == 27:
        break

cv2.destroyAllWindows()        
                      
            
