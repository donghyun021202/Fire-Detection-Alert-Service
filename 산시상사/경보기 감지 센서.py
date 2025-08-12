import cv2
import numpy as np
import time
from collections import deque

# 카메라 캡처
cap = cv2.VideoCapture(0)

# 빨간색 범위 (두 개)
lower_red1 = np.array([0, 150, 150])   
upper_red1 = np.array([5, 255, 255])
lower_red2 = np.array([170, 150, 150])  
upper_red2 = np.array([180, 255, 255])

# 알림 관련 변수
last_alert_time = 0
alert_interval = 2  # 초

# 깜빡임 체크용 버퍼 (최근 10프레임 저장)
blink_buffer = deque(maxlen=10)

while True:
    ret, frame = cap.read()
    if not ret:
        break

    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    mask1 = cv2.inRange(hsv, lower_red1, upper_red1)
    mask2 = cv2.inRange(hsv, lower_red2, upper_red2)
    mask_red = cv2.bitwise_or(mask1, mask2)

    # 윤곽선 탐지
    contours, _ = cv2.findContours(mask_red, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    red_detected = len(contours) > 0

    # 현재 프레임 결과 버퍼에 추가
    blink_buffer.append(red_detected)

    # 깜빡임 판단: 최근 프레임에 감지된 것과 감지되지 않은 것이 섞여 있으면 True
    blink_detected = True if (True in blink_buffer and False in blink_buffer) else False

    # 깜빡임 + 빨간색 감지되었을 때만 알림 출력
    current_time = time.time()
    if red_detected and blink_detected and (current_time - last_alert_time > alert_interval):
        print("⚠️ 화재경보가 발생하였습니다!")
        last_alert_time = current_time

    # 결과 이미지 출력
    result = cv2.bitwise_and(frame, frame, mask=mask_red)
    cv2.imshow('Red Detection (Blink Filtered)', result)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
