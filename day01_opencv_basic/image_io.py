import cv2 as cv
import urllib.request
import sys
import os

def get_sample(filename):
    if not os.path.exists(filename):
        url = f"https://raw.githubusercontent.com/opencv/opencv/master/samples/data/{filename}"
        urllib.request.urlretrieve(url, filename)
    return cv.imread(filename)
# 파일 읽기
# img = cv.imread("./samples/starry_night.jpg")
img = get_sample("starry_night.jpg")
img = get_sample("orange.jpg")
#print("==img==")
#print(img)
# img_gray = cv.imread("./samples/starry_night.jpg", cv.IMREAD_GRAYSCALE)
img_gray = cv.imread("orange.jpg", cv.IMREAD_GRAYSCALE)
if img is None:
    sys.exit("Could not read the image.")

cv.imshow("Display window_color", img)
cv.imshow("Display window_gray", img_gray)
# 이미지 정보 확인
print("===     컬러 이미지     ===")
print(f"shape: {img.shape}") # (높이, 너비, 채널)
print(f"dtype: {img.dtype}") # 파일의 형태
print(f"size: {img.size}")  # 전체의 픽셀 수 (높이 x 너비 x 채널)

print("=== 그레이스케일 이미지 ===")
print(f"shape: {img_gray.shape}") # (높이, 너비, 채널)
print(f"dtype: {img_gray.dtype}") # 파일의 형태
print(f"size: {img_gray.size}")  # 전체의 픽셀 수 (높이 x 너비 x 채널)

# 창 닫기 (아무 키)
k = cv.waitKey(0)

# 파일 저장
if k == ord("s"):
    cv.imwrite("starry_night.png", img)
    cv.imwrite("starry_night_gray.png", img_gray)