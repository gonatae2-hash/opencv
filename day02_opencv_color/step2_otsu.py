import cv2 as cv
import numpy as np
import urllib.request
import os

def get_sample(filename):
    if not os.path.exists(filename):
        url = f"https://raw.githubusercontent.com/opencv/opencv/master/samples/data/{filename}"
        urllib.request.urlretrieve(url, filename)
    return cv.imread(filename, 0)

# 이미지 읽기 (그레이스케일)
img = get_sample('sudoku.png')
img = cv.medianBlur(img, 5)

# 창 생성
cv.namedWindow('Otsu')

def nothing(x):
    pass

# 트랙바 생성
cv.createTrackbar('manual_thresh', 'Otsu', 127, 255, nothing)
cv.createTrackbar('mode', 'Otsu', 0, 1, nothing)

while(1):
    manual_thresh = cv.getTrackbarPos('manual_thresh', 'Otsu')
    mode          = cv.getTrackbarPos('mode', 'Otsu')

    ret_manual, manual_th = cv.threshold(img, manual_thresh, 255, cv.THRESH_BINARY)

    ret_otsu, otsu_th = cv.threshold(img, 0, 255, cv.THRESH_BINARY + cv.THRESH_OTSU)

    cv.putText(otsu_th, f'Otsu: {ret_otsu:.0f}', (10, 30),
               0, 1, (128, 128, 128), 2, cv.LINE_AA)
    if mode == 0:
        combined = np.hstack([img, otsu_th])
    else:
        combined = np.hstack([img, manual_th, otsu_th])

    cv.imshow('Otsu', combined)

    if cv.waitKey(1) & 0xFF == ord('q'):
        break

cv.destroyAllWindows()