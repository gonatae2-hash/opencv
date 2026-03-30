import numpy as np
import cv2 as cv
 
cap = cv.VideoCapture(0)
count = 0
if not cap.isOpened():
    print("Cannot open camera")
    exit()
while True:
    # Capture frame-by-frame
    ret, frame = cap.read()
 
    # if frame is read correctly ret is True
    if not ret:
        print("Can't receive frame (stream end?). Exiting ...")
        break
    # Our operations on the frame come here

    frame = cv.flip(frame, 1) # 좌우 반전
    gray = cv.cvtColor(frame, cv.COLOR_BGR2GRAY)
    # Display the resulting frame
    key = cv.waitKey(1)

    cv.imshow('frame', frame)
    if key == ord('q'):
        break
    elif key == ord('c'):
        filename = f"my_photo{count}.jpg"
        cv.imwrite(filename, frame)
        count += 1

# When everything done, release the capture
cap.release()
cv.destroyAllWindows()