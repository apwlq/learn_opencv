# Real-time Image Alignment and Stitching with SIFT
# by chatcpt

import cv2
import numpy as np

# 웹캠 초기화
cap1 = cv2.VideoCapture(1)  # 첫 번째 웹캠
cap2 = cv2.VideoCapture(2)  # 두 번째 웹캠

if not cap1.isOpened() or not cap2.isOpened():
    print("웹캠을 열 수 없습니다.")
    exit()

# SIFT 특징점 검출기 생성
sift = cv2.SIFT_create()
bf = cv2.BFMatcher()

while True:
    # 두 개의 프레임 읽기
    ret1, frame1 = cap1.read()
    ret2, frame2 = cap2.read()

    if not ret1 or not ret2:
        print("프레임을 읽을 수 없습니다.")
        break

    # 이미지를 그레이스케일로 변환
    gray1 = cv2.cvtColor(frame1, cv2.COLOR_BGR2GRAY)
    gray2 = cv2.cvtColor(frame2, cv2.COLOR_BGR2GRAY)

    # 특징점과 기술자 찾기
    kp1, des1 = sift.detectAndCompute(gray1, None)
    kp2, des2 = sift.detectAndCompute(gray2, None)

    # 특징점 매칭
    matches = bf.knnMatch(des1, des2, k=2)

    # 좋은 매칭점 선별
    good = []
    for m, n in matches:
        if m.distance < 0.75 * n.distance:
            good.append(m)

    if len(good) > 10:  # 충분한 매칭점이 있는 경우에만 진행
        # 좋은 매칭점의 좌표
        src_pts = np.float32([kp1[m.queryIdx].pt for m in good]).reshape(-1, 1, 2)
        dst_pts = np.float32([kp2[m.trainIdx].pt for m in good]).reshape(-1, 1, 2)

        # 호모그래피 계산
        M, mask = cv2.findHomography(src_pts, dst_pts, cv2.RANSAC, 5.0)

        # frame1을 frame2에 맞추어 변환
        h, w = frame1.shape[:2]
        aligned_frame1 = cv2.warpPerspective(frame1, M, (w, h))

        # frame2와 aligned_frame1을 가로로 합치기
        combined_frame = np.concatenate((aligned_frame1, frame2), axis=1)

        # 결과 출력
        cv2.imshow('Combined Frame', combined_frame)

    else:
        print("매칭점이 충분하지 않습니다.")
        cv2.imshow('Combined Frame', frame1)  # 매칭점이 충분하지 않으면 원본 프레임1 출력

    # q를 누르면 종료
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# 리소스 해제
cap1.release()
cap2.release()
cv2.destroyAllWindows()
