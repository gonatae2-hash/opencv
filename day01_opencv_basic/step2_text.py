import numpy as np
import cv2 as cv

img = cv.imread("my_photo0.png")

h,w = img.shape[:2]
overlay = img.copy()
cv.rectangle(img,(0,350),(w,h),(0,0,0),-1)
# 사진 높이 480

cv.addWeighted(overlay, 0.5, img, 0.5, 0 ,img)

print(img.shape)

cv.putText(img,'Kwan-tae',(320,415), 0, 2,(255,255,255),2,cv.LINE_AA)
cv.putText(img,'Yeonhee',(500,445), 0, 1,(255,255,255),2,cv.LINE_AA)
cv.imshow("image", img)
cv.imwrite("my_id_card.png", img)
cv.waitKey(0)
cv.destroyAllWindows()
