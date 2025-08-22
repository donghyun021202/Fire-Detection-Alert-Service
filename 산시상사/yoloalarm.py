# run_alarm_blink_debug.py
# 요구 패키지: pip install ultralytics opencv-python numpy

import time
import argparse
from collections import deque
import csv
import os

import cv2
import numpy as np
from ultralytics import YOLO

# -------------------- 기본 설정 (민감도 한 단계 더 낮춤) --------------------
MODEL_PATH = "yolov8n.pt"        # COCO 사전학습
CONF_TH = 0.03                   # ▼ 더 낮춤 (기존 0.10/0.05에서 추가 완화)
RED_RATIO_TH = 0.003             # ▼ 더 낮춤 (빨강 비율 아주 작아도 통과)
ON_TH, OFF_TH = 0.10, 0.06       # ▼ 깜빡임 히스테리시스 임계 (score가 0.00~0.15일 때 맞춤)
MIN_CHANGES = 1                  # on/off 1번만 있어도 깜빡임으로 간주
ALERT_INTERVAL = 2.0             # 알림 최소 간격(초)
IMG_SIZE = 1280                  # 작은 물체 인식 위해 해상도 크게
SOURCE_DEFAULT = "0"             # 기본 카메라

# -------------------- 유틸 --------------------
def class_is_traffic_light(name: str) -> bool:
    if not name:
        return False
    return name.lower().replace(" ", "") == "trafficlight"

def red_ratio(bgr_roi: np.ndarray):
    if bgr_roi is None or bgr_roi.size == 0:
        return 0.0, None
    hsv = cv2.cvtColor(bgr_roi, cv2.COLOR_BGR2HSV)
    # LED 환경 고려해 하한 완화
    lower1 = np.array([0,   70, 80], dtype=np.uint8)
    upper1 = np.array([10, 255,255], dtype=np.uint8)
    lower2 = np.array([170, 70, 80], dtype=np.uint8)
    upper2 = np.array([180,255,255], dtype=np.uint8)
    mask = cv2.inRange(hsv, lower1, upper1) | cv2.inRange(hsv, lower2, upper2)
    ratio = float((mask > 0).mean())
    return ratio, mask

def blink_by_state(prob_buf: deque, on_th=0.10, off_th=0.06, min_changes=1):
    if len(prob_buf) < 6:
        return False
    arr = list(prob_buf)
    state = None
    changes = 0
    for p in arr:
        if state is None:
            state = (p > on_th)
        else:
            if state and p < off_th:
                state = False; changes += 1
            elif (not state) and p > on_th:
                state = True; changes += 1
    return changes >= min_changes

# -------------------- 메인 --------------------
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--source", type=str, default=SOURCE_DEFAULT)
    parser.add_argument("--imgsz", type=int, default=IMG_SIZE)
    parser.add_argument("--conf", type=float, default=CONF_TH)
    args = parser.parse_args()

    model = YOLO(MODEL_PATH)
    names = model.names

    cap = cv2.VideoCapture(int(args.source) if args.source.isdigit() else args.source)
    if not cap.isOpened():
        raise RuntimeError(f"카메라를 열 수 없습니다: {args.source}")

    # ▼ 저 FPS 대응: 버퍼 길이 90으로 확대 (10~15fps 환경 권장)
    prob_buf = deque(maxlen=90)
    last_alert = 0.0

    # 예측 이벤트 CSV 초기화
    PRED_PATH = "pred_events.csv"
    with open(PRED_PATH, "w", newline="", encoding="utf-8") as f:
        csv.writer(f).writerow(["start_ms", "end_ms"])
    in_alarm = False
    pred_start = None

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        results = model.predict(source=frame, imgsz=args.imgsz, conf=args.conf, verbose=False)
        max_p = 0.0

        for r in results:
            for b in (r.boxes or []):
                cls_id = int(b.cls.item())
                conf = float(b.conf.item())
                cname = names.get(cls_id, "")
                if not class_is_traffic_light(cname):
                    continue

                x1, y1, x2, y2 = map(int, b.xyxy[0].tolist())
                roi = frame[y1:y2, x1:x2]
                r_ratio, _ = red_ratio(roi)

                # 점수 = 탐지신뢰도 * 빨강비율 (하한 미만이면 0)
                score = conf * (r_ratio if r_ratio >= RED_RATIO_TH else 0.0)
                max_p = max(max_p, score)

                # 시각화
                color = (0,255,0) if score > ON_TH else (0,0,255)
                cv2.rectangle(frame,(x1,y1),(x2,y2),color,2)
                cv2.putText(frame,f"Alarm p={score:.2f}",(x1,max(15,y1-6)),
                            cv2.FONT_HERSHEY_SIMPLEX,0.5,color,1,cv2.LINE_AA)

        # 버퍼/깜빡임 판정
        prob_buf.append(max_p)
        blink_ok = blink_by_state(prob_buf, on_th=ON_TH, off_th=OFF_TH, min_changes=MIN_CHANGES)

        # 이벤트 시작/종료 기록
        t_ms = int(time.time() * 1000)
        if blink_ok and not in_alarm:
            in_alarm = True
            pred_start = t_ms
        elif (not blink_ok) and in_alarm:
            with open(PRED_PATH, "a", newline="", encoding="utf-8") as f:
                csv.writer(f).writerow([pred_start, t_ms])
            in_alarm = False
            pred_start = None

        # 디버그 출력
        print(f"score={max_p:.3f}, blink_ok={blink_ok}, buf_len={len(prob_buf)}")

        # ▼ 알림 문턱도 현재 스코어 분포(0~0.15)에 맞춰 낮춤
        now = time.time()
        if (max_p > 0.06) and blink_ok and (now - last_alert > ALERT_INTERVAL):
            print("⚠️ 화재경보기 LED 깜빡임 감지!")
            last_alert = now

        cv2.imshow("Alarm Blink Detection", frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    # 열린 이벤트 마무리
    if in_alarm and pred_start is not None:
        with open(PRED_PATH, "a", newline="", encoding="utf-8") as f:
            csv.writer(f).writerow([pred_start, int(time.time()*1000)])

    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
