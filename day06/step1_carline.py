import cv2 as cv
import numpy as np

# 1단계: 이미지 로드
img = cv.imread('carline2.jpg')
scale_process = 0.3
scale_display = 0.5
img_resized = cv.resize(img, (int(img.shape[1]*scale_process), int(img.shape[0]*scale_process)))

print(f"축소된 이미지 크기 :{img_resized.shape}") # 실제 크기 확인 

# 2단계: 그레이스케일 + 히스토그램 평활화
gray = cv.cvtColor(img_resized, cv.COLOR_BGR2GRAY)
gray_eq = cv.equalizeHist(gray)
hsv = cv.cvtColor(img_resized, cv.COLOR_BGR2HSV)

# 3단계: 흰색 마스크
lower_white = np.array([0,0,140])
upper_white = np.array([200,50,255])
mask_white = cv.inRange(hsv, lower_white, upper_white)
mask_lane = mask_white

# 마스크 적용
lane_filtered = cv.bitwise_and(gray_eq, gray_eq, mask=mask_lane)

# Step 2: Canny 파라미터 튜닝
canny_params = [
    (50, 150),
    (100, 200),
    (150, 250),
]

for lower, upper in canny_params:
    blur = cv.GaussianBlur(lane_filtered, (5, 5), 0)
    edges = cv.Canny(blur, lower, upper)

# 4단계: 허프 직선 변환
lines = cv.HoughLinesP(edges, 1, np.pi/180, threshold=70, minLineLength=170, maxLineGap=15)

# 5단계: 검출된 직선을 이미지에 그리기
if lines is not None:
    for line in lines:
        x1, y1, x2, y2 = line[0]
        # 수직, 수평선 제거
        angle = np.degrees(np.arctan2((y2 - y1), (x2 - x1)))
        if not (abs(angle) < 10 or abs(angle) > 80):
            cv.line(img_resized, (x1, y1), (x2, y2), (0, 255, 0), 2)
    print(f"검출된 직선 개수: {len(lines)}")

else:
    print("검출된 직선이 없습니다")


# 화면 표시용 크기 조절
gray_display = cv.resize(gray, (int(gray.shape[1]*scale_display),
                               int(gray.shape[0]*scale_display)))

edges_display = cv.resize(edges, (int(edges.shape[1]*scale_display),
                                 int(edges.shape[0]*scale_display)))

display = cv.resize(img_resized, (int(img_resized.shape[1]*scale_display),
                                  int(img_resized.shape[0]*scale_display)))

cv.imshow('Original', gray_display)
cv.imshow('Edges', edges_display)
cv.imshow('Hough Lines', display)
cv.waitKey(0)
cv.destroyAllWindows()

# 결과 저장
cv.imwrite('car_line_detected.jpg', img_resized)
