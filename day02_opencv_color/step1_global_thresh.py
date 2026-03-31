import cv2 as cv
import numpy as np
import urllib.request
import os

def get_sample(filename):
    if not os.path.exists(filename):
        url = f"https://raw.githubusercontent.com/opencv/opencv/master/samples/data/{filename}"
        urllib.request.urlretrieve(url, filename)
    return cv.imread(filename,0)

img = get_sample('sudoku.png')
img = cv.medianBlur(img,5)

cv.namedWindow('Th')

def nothing(x):
    pass

cv.createTrackbar('threshold','Th',127,255,nothing)
cv.createTrackbar('mode','Th',0,1,nothing)

while(1):
    Threshold = cv.getTrackbarPos('threshold', 'Th')
    mode = cv.getTrackbarPos('mode', 'Th')

    if mode == 0:
        ret, result = cv.threshold(img, Threshold, 255, cv.THRESH_BINARY)
    else:
        ret, result = cv.threshold(img, Threshold, 255, cv.THRESH_BINARY_INV)

    cv.putText(result, f'Threshold: {Threshold}', (10, 30),
               0, 1, (128,128,128), 2, cv.LINE_AA)

    combined = np.hstack([img, result])
    cv.imshow('Th', combined)

    if cv.waitKey(1) & 0xFF == ord('q'):
        break
cv.destroyAllWindows()