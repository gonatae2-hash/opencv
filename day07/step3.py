import cv2 as cv
import numpy as np
import matplotlib.pyplot as plt

# ========== Step 1: 정지 표지판 이미지 로드 ==========
img = cv.imread('stop_sign.jpg')

if img is None:
    print("Image not found!")
    exit()

hsv = cv.cvtColor(img, cv.COLOR_BGR2HSV)
h, w = img.shape[:2]

# ========== Step 2: 빨간색 마스크 생성 ==========
lower_red1 = np.array([0, 180, 100])
upper_red1 = np.array([10, 255, 255])
lower_red2 = np.array([170, 180, 100])
upper_red2 = np.array([180, 255, 255])

mask1 = cv.inRange(hsv, lower_red1, upper_red1)
mask2 = cv.inRange(hsv, lower_red2, upper_red2)
red_mask = cv.bitwise_or(mask1, mask2)
print(f"Red pixels: {cv.countNonZero(red_mask)}")

# ========== Step 3: 노이즈 제거 ==========
kernel = cv.getStructuringElement(cv.MORPH_ELLIPSE, (5, 5))
red_mask = cv.morphologyEx(red_mask, cv.MORPH_CLOSE, kernel)
red_mask = cv.morphologyEx(red_mask, cv.MORPH_OPEN, kernel)

# ========== Step 4: 컨투어 검출 ==========
contours, _ = cv.findContours(red_mask, cv.RETR_EXTERNAL, cv.CHAIN_APPROX_SIMPLE)
print(f"Found {len(contours)} contours")

# ========== Step 5: 컨투어 필터링 ==========
min_area = 10000
detected_signs = []

for contour in contours:
    area = cv.contourArea(contour)
    if area < min_area:
        continue

    perimeter = cv.arcLength(contour, True)
    approx = cv.approxPolyDP(contour, 0.02 * perimeter, True)
    num_vertices = len(approx)

    x, y, w, h = cv.boundingRect(contour)
    aspect_ratio = float(w) / h if h > 0 else 0

    if num_vertices >= 6:
        if 0.9 <= aspect_ratio <= 1.1:
            detected_signs.append((x, y, w, h, num_vertices))

print(f"Detected stop signs: {len(detected_signs)}")

# ========== Step 6: 결과 시각화 ==========
result_img = img.copy()

for x, y, w, h, vertices in detected_signs:
    cv.rectangle(result_img, (x, y), (x+w, y+h), (0, 0, 255), 2)
    cv.putText(result_img, f'Stop ({vertices}v)', (x, y-10),
               cv.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)

plt.figure(figsize=(12, 6))
plt.subplot(121)
plt.imshow(red_mask, cmap='gray')
plt.title('Red Color Mask')
plt.axis('off')
plt.subplot(122)
plt.imshow(cv.cvtColor(result_img, cv.COLOR_BGR2RGB))
plt.title(f'Detected Stop Signs ({len(detected_signs)})')
plt.axis('off')
plt.tight_layout()
plt.show()