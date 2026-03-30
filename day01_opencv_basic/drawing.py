import numpy as np
import cv2 as cv
 
# Create a black image
img = np.zeros((512,512,3), np.uint8)
 
# Draw a diagonal blue line with thickness of 5 px
cv.line(img,(0,511),(511,0),(0,255,255),5)
cv.line(img,(0,0),(511,511),(0,255,255),5)
cv.rectangle(img,(384,0),(510,128),(255,0,255),3)

cv.circle(img,(447,63), 63, (0,255,0), -1)

cv.ellipse(img,(256,256),(100,50),0,0,180,255,-1)

cv.ellipse(img,(256,256),(150,80),0,0,360,(255,255,255),-1)

pts = np.array([[250,80],[200,150],[300,150]], np.int32)
pts = pts.reshape((-1,1,2))
cv.polylines(img,[pts],True,(255,0,0), 3)

pts = np.array([[10,5],[20,30],[70,20],[50,10]], np.int32)
pts = pts.reshape((-1,1,2))
cv.polylines(img,[pts],True,(0,0,255))

font = cv.FONT_HERSHEY_SIMPLEX
cv.putText(img,'OpenCV',(10,500), font, 4,(127,127,127),2,cv.LINE_AA)

# 화면에 표시
cv.imshow("hi", img)
cv.waitKey(0)
cv.destroyAllWindows()