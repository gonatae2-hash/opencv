import urllib.request
import os
import cv2

def download_sample(filename):
    if not os.path.exists(filename):
        url = f"https://raw.githubusercontent.com/opencv/opencv/master/samples/data/{filename}"
        urllib.request.urlretrieve(url, filename)
    return filename

# 사용할 샘플 이미지
img = cv2.imread(download_sample("pic1.png"), cv2.IMREAD_GRAYSCALE)

_, binary = cv2.threshold(img, 127, 255, cv2.THRESH_BINARY)

contours, hierarchy = cv2.findContours(binary,cv2.RETR_TREE,cv2.CHAIN_APPROX_SIMPLE)

img_color = cv2.cvtColor(binary, cv2.COLOR_GRAY2BGR)

cv2.drawContours(img_color, contours, -1, (0, 255, 0), 2)

for cnt in contours:
    area = cv2.contourArea(cnt)
    print(f"면적:{area:.0f} ", end="")
    if 700 < area < 20000:
        cv2.drawContours(img_color, [cnt], 0, (255, 0, 0), 2)

cv2.imshow('Filtred Contours', img_color)
cv2.waitKey(0)
cv2.destroyAllWindows()
