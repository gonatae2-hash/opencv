import numpy as np
import cv2 as cv

img = cv.imread("my_id_card.png")
img_original = img.copy()

drawing = False
mode = True
ix,iy = -1,-1

def draw_circle(event,x,y,flags,param):
    global ix,iy,drawing,mode
 
    if event == cv.EVENT_LBUTTONDOWN:
        drawing = True
        ix,iy = x,y
 
    elif event == cv.EVENT_MOUSEMOVE:
        if drawing == True:
            if mode == True:
                img[:] = img_original[:]
                cv.rectangle(img,(ix,iy),(x,y),(0,255,0),2)

    elif event == cv.EVENT_LBUTTONUP:
        drawing = False
        img[:] = img_original[:] 
        cv.rectangle(img,(ix,iy),(x,y),(0,255,0),2)
        cv.putText(img,'FACE',((x+ix)//2-35,iy-10), 0, 1,(0,255,255),2,cv.LINE_AA)    

cv.namedWindow('image')
cv.setMouseCallback('image', draw_circle)

while True:
    cv.imshow('image', img)
    key = cv.waitKey(1)
    if key == ord('s'):
        cv.imwrite("my_id_card_final.png", img)
        break
    elif key == ord('q'):
        break