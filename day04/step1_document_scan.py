import cv2 as cv
import numpy as np
import urllib.request
import os
from pathlib import Path

def get_sample(filename, repo='insightbook'):

    if not os.path.exists(filename):
        if repo == 'insightbook':
            url = f"https://raw.githubusercontent.com/dltpdn/insightbook.opencv_project_python/master/img/{filename}"
        else:  # opencv 공식
            url = f"https://raw.githubusercontent.com/opencv/opencv/master/samples/data/{filename}"
        urllib.request.urlretrieve(url, filename)
    return filename

# ============================================================
# 전역 변수
# ============================================================
win_name = "Document Scanning"
draw = None
rows, cols = 0, 0
pts_cnt = 0
pts = np.zeros((4, 2), dtype=np.float32)

# ============================================================
# 마우스 콜백 함수
# ============================================================
def onMouse(event, x, y, flags, param):

    global pts_cnt, draw, pts, img
    
    if event == cv.EVENT_LBUTTONDOWN:
        # 1️⃣ 클릭한 위치에 원 표시
        cv.circle(draw, (x,y), 10, (0,255,0), -1)
        cv.imshow(win_name, draw)
        # 2️⃣ 좌표 저장
        pts[pts_cnt] = [x,y]
        pts_cnt+=1
        
        if pts_cnt == 4:
            sm = pts.sum(axis=1)
            topLeft = pts[np.argmin(sm)]
            bottomRight = pts[np.argmax(sm)]

        #     # 차 계산 (우상단: 최소, 좌하단: 최대)
            diff = np.diff(pts, axis = 1)
            topRight = pts[np.argmin(diff)]
            bottomLeft = pts[np.argmax(diff)]   

        #     # 변환 전 4개 좌표
            pts1 = np.float32([topLeft, topRight, bottomRight , bottomLeft])
        
        #     # 변환 후 서류 크기 계산
            w1 = abs(bottomRight[0] - bottomLeft[0])    
            w2 = abs(topRight[0] - topLeft[0])         
            h1 = abs(topRight[1] - bottomRight[1])   
            h2 = abs(topLeft[1] - bottomLeft[1])       
            width = int(max([w1, w2]))
            height = int(max([h1, h2]))

        #     # 변환 후 4개 좌표 (직사각형)
            pts2 = np.float32([[0,0], [width-1,0], 
                                [width-1,height-1], [0,height-1]])
        #     # 원근 변환 적용
            mtrx = cv.getPerspectiveTransform(pts1, pts2)
            result = cv.warpPerspective(img, mtrx, (width, height))
        
        #     # 결과 저장
            cv.imshow('scanned', result)
        #     # 초기화 (새로운 이미지 스캔 가능)
            pts_cnt = 0
            pts = np.zeros((4, 2), dtype=np.float32)
            draw = img.copy()
            cv.imshow(win_name, draw)

# ============================================================
# 메인 실행
# ============================================================

# 이미지 로드 (파일 또는 웹캠)
img = cv.imread(get_sample('paper.jpg', repo='insightbook'))

if img is None:
    print("❌ 이미지를 불러올 수 없습니다.")
    exit()

rows, cols = img.shape[:2]
draw = img.copy()

# 윈도우 표시 + 마우스 콜백 등록

cv.namedWindow(win_name)
cv.setMouseCallback(win_name, onMouse)
cv.imshow(win_name, draw)

print("📝 사용법:")
print("1. 이미지 위에 4개 점을 클릭하세요 (좌상단, 우상단, 우하단, 좌하단 순서 무관)")
print("2. 4번째 점 클릭 후 자동으로 문서 스캔이 실행됩니다.")
print("3. 'Scanned Document' 윈도우에서 결과를 확인하세요.")
cv.waitKey(0)
cv.destroyAllWindows()

# 웹캠에서 프레임 캡처
cap = cv.VideoCapture(0)

while True:
    ret, frame = cap.read()
    if not ret:
        break
    
    # 프레임 크기 조정 (보기 좋게)
    frame = cv.resize(frame, (800, 600))
    img = frame.copy()
    draw = frame.copy()
    
    cv.imshow(win_name, draw)
    cv.setMouseCallback(win_name, onMouse)
    
    if cv.waitKey(1) & 0xFF == ord('q'):  # 'q' 누르면 종료
        break

cap.release()
cv.destroyAllWindows()