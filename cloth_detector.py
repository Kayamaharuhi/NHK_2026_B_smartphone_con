#!/usr/bin/env python3
import sys
import time
import json
import threading
from collections import defaultdict, deque
import cv2
import numpy as np

import urllib.request
import urllib.error

HAS_YOLO = False
try:
    from ultralytics import YOLOWorld
    HAS_YOLO = True
except ImportError:
    YOLOWorld = None

REAL_cloth_l = 0.300   # 雑巾の長辺 (m)
REAL_cloth_s = 0.200   # 雑巾の短辺 (m)
persentage = 0.1      # 信頼度閾値
BOMP_score = 0.85     # 凸性スコア (これ未満で窪み/よれ判定)
SHAPE_score = 0.85     # 矩形度スコア (これ未満で長方形崩れ判定)
corner = 4             # 想定コーナー数
ditect_frame = 5       # 安定化フレーム数
W, H = 720, 480        # カメラ解像度 

CAMERA_ID = 0        
SERVER_HTTP_URL = "http://localhost:8000/api/cloth_alert"

MIN_ALERT_INTERVAL = 1.0   # 同じNG状態での最短送信間隔 (秒)
last_sent_time = 0.0
last_sent_status = None


def send_cloth_alert(payload: dict):
    def _post():
        global last_sent_time, last_sent_status
        try:
            req = urllib.request.Request(
                SERVER_HTTP_URL,
                data=json.dumps(payload).encode("utf-8"),
                headers={"Content-Type": "application/json"},
                method="POST"
            )
            with urllib.request.urlopen(req, timeout=0.8) as resp:
                pass
        except Exception as e:
            # サーバー未起動時などは無視して検知を優先
            pass

    threading.Thread(target=_post, daemon=True).start()


def get_cloth_mask(color_roi):
    if color_roi is None or color_roi.size == 0:
        return np.zeros((10, 10), dtype=np.uint8)

    gray = cv2.cvtColor(color_roi, cv2.COLOR_BGR2GRAY)
    _, color_mask = cv2.threshold(
        gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU
    )

    kernel = np.ones((7, 7), np.uint8)
    combined_mask = cv2.morphologyEx(color_mask, cv2.MORPH_CLOSE, kernel)
    combined_mask = cv2.morphologyEx(combined_mask, cv2.MORPH_OPEN, kernel)

    contours, _ = cv2.findContours(
        combined_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
    )
    if contours:
        main_cnt = max(contours, key=cv2.contourArea)
        clean_mask = np.zeros_like(combined_mask)
        cv2.drawContours(clean_mask, [main_cnt], -1, 255, thickness=cv2.FILLED)
        return clean_mask

    return combined_mask


REASON_JA_MAP = {
    'Folded(Dent)': '折れ・凹み',
    'NotRectangular': '長方形の崩れ',
    'CornerFolded': '角のめくれ',
    'BadAspect(Ratio)': '縦横比の歪み',
    'TooSmallContour': '布が小さすぎ',
    'NoContour': '輪郭なし'
}

def analyze_shape(mask_u8, box_w, box_h):
    ng_reasons = []
    debug = {}

    contours, _ = cv2.findContours(
        mask_u8, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
    )
    if not contours:
        return ['輪郭なし'], debug

    cnt = max(contours, key=cv2.contourArea)
    area = cv2.contourArea(cnt)
    if area < 400:
        return ['布が小さすぎ'], debug

    hull = cv2.convexHull(cnt)
    hull_area = cv2.contourArea(hull)
    convexity = area / hull_area if hull_area > 0 else 0
    debug['convexity'] = convexity

    if convexity < BOMP_score:
        ng_reasons.append('折れ・凹み')

    rect = cv2.minAreaRect(cnt)
    (rw, rh) = rect[1]
    rect_area = rw * rh if rw > 0 and rh > 0 else 1
    rectangularity = area / rect_area if rect_area > 0 else 0
    debug['rectangularity'] = rectangularity
    debug['min_area_rect'] = rect

    if rectangularity < SHAPE_score:
        ng_reasons.append('長方形の崩れ')

    peri = cv2.arcLength(cnt, True)
    approx = cv2.approxPolyDP(cnt, 0.04 * peri, True)
    n_corners = len(approx)
    debug['n_corners'] = n_corners
    debug['approx'] = approx

    if n_corners > corner + 2:
        ng_reasons.append('角のめくれ')

    if rw > 0 and rh > 0:
        eff_long, eff_short = max(rw, rh), min(rw, rh)
        measured_aspect = eff_long / eff_short
        real_aspect = max(REAL_cloth_l, REAL_cloth_s) / min(REAL_cloth_l, REAL_cloth_s)
        debug['measured_aspect'] = measured_aspect
        debug['real_aspect'] = real_aspect

        aspect_tolerance = 0.25
        if abs(measured_aspect - real_aspect) > (real_aspect * aspect_tolerance):
            ng_reasons.append('縦横比の歪み')

    debug['contour'] = cnt
    return ng_reasons, debug


status_history = defaultdict(lambda: deque(maxlen=ditect_frame))


def stabilize_status(track_id, raw_status):
    status_history[track_id].append(raw_status)
    hist = status_history[track_id]

    if len(hist) < ditect_frame:
        return raw_status

    ng_list = [s for s in hist if s.startswith('NG')]
    if len(ng_list) > ditect_frame // 2:
        return max(set(ng_list), key=ng_list.count)
    else:
        return 'OK (正常)'


def main():
    global last_sent_time, last_sent_status

    model = None
    target_classes = ['towel', 'cloth', 'rag']

    if HAS_YOLO and YOLOWorld is not None:
        try:
            print(f"[ClothDetector] Initializing YOLO-World on Camera {CAMERA_ID}...")
            model = YOLOWorld('yolov8s-world.pt')
            model.set_classes(target_classes)
            print("[ClothDetector] YOLO-World loaded successfully.")
        except Exception as e:
            print(f"[ClothDetector] Warning: Could not load YOLO model: {e}")
            print("[ClothDetector] -> Pure OpenCV モードに切り替えます。")
            model = None
    else:
        print("[ClothDetector] 純粋 OpenCV 形状・凸性解析モードで実行します。")

    print(f"[ClothDetector] カメラ ID {CAMERA_ID} (/dev/video{CAMERA_ID}) を起動中...")
    
    def open_camera_stream(cam_index):
        cap_inst = cv2.VideoCapture(cam_index, cv2.CAP_V4L2)
        if cap_inst.isOpened():
            return cap_inst
        cap_inst = cv2.VideoCapture(cam_index)
        if cap_inst.isOpened():
            return cap_inst
        return None

    cap = open_camera_stream(CAMERA_ID)
    
    # 接続リトライ
    retry_count = 0
    while (cap is None or not cap.isOpened()) and retry_count < 3:
        retry_count += 1
        print(f"[ClothDetector] カメラ {CAMERA_ID} を開けませんでした。1秒後に再試行 ({retry_count}/3)...")
        time.sleep(1.0)
        cap = open_camera_stream(CAMERA_ID)

    # 見つからない場合、全カメラ番号を自動スキャン
    if cap is None or not cap.isOpened():
        print(f"[ClothDetector] カメラ {CAMERA_ID} が認識されません。接続されている他のカメラ番号を自動スキャンします...")
        found_cam_id = None
        for candidate_id in range(10):
            if candidate_id == CAMERA_ID:
                continue
            test_cap = open_camera_stream(candidate_id)
            if test_cap is not None and test_cap.isOpened():
                ret_test, frame_test = test_cap.read()
                if ret_test and frame_test is not None:
                    found_cam_id = candidate_id
                    cap = test_cap
                    print(f"[ClothDetector]  カメラ ID {candidate_id} を自動検出し、切り替えました！")
                    break
                test_cap.release()

        if cap is None or not cap.isOpened():
            print(f"[ClothDetector]  エラー: 利用可能なカメラが見つかりませんでした。")
            print("  ・USBカメラの接続を確認してください")
            print("  ・Ubuntu端末で 'ls -l /dev/video*' を実行して番号を確認してください")
            print("  ・権限エラーの場合: sudo chmod 666 /dev/video*")
            sys.exit(1)

    cap.set(cv2.CAP_PROP_FRAME_WIDTH, W)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, H)
    cap.set(cv2.CAP_PROP_FPS, 30.0)
    print(f"[ClothDetector] カメラのオープンに成功しました！(解像度: {W}x{H})")

    print("[ClothDetector] Running cloth inspection loop.")
    print("[ClothDetector] [キー操作] 'd': 歪み検知ON/OFF切替 | 'q': 終了")

    distortion_enabled = True

    try:
        while True:
            ret, frame = cap.read()
            if not ret or frame is None:
                time.sleep(0.03)
                continue

            current_h, current_w = frame.shape[:2]
            CX, CY = current_w / 2.0, current_h / 2.0

            best_target = None
            max_conf = -1.0

            if model is not None:
                try:
                    results = model.track(frame, conf=persentage, persist=True, verbose=False)
                    for result in results:
                        if result.boxes is None:
                            continue

                        for box in result.boxes:
                            conf = float(box.conf[0])
                            cls_id = int(box.cls[0])
                            detected_name = target_classes[cls_id] if cls_id < len(target_classes) else 'cloth'
                            track_id = int(box.id[0]) if box.id is not None else -1

                            x1, y1, x2, y2 = map(int, box.xyxy[0].cpu().numpy())
                            x1, y1 = max(0, x1), max(0, y1)
                            x2, y2 = min(current_w, x2), min(current_h, y2)
                            box_w = x2 - x1
                            box_h = y2 - y1

                            if box_w < 35 or box_h < 35:
                                continue

                            cx = int((x1 + x2) / 2.0)
                            cy = int((y1 + y2) / 2.0)

                            color_roi = frame[y1:y2, x1:x2]
                            if color_roi.size == 0:
                                continue

                            mask_u8 = get_cloth_mask(color_roi)
                            ng_reasons = []
                            shape_debug = {}

                            if distortion_enabled:
                                shape_ng, shape_debug = analyze_shape(mask_u8, box_w, box_h)
                                ng_reasons.extend(shape_ng)
                                raw_status = f"NG ({'/'.join(ng_reasons)})" if ng_reasons else 'OK (正常)'
                            else:
                                raw_status = 'OK (歪み検知OFF)'

                            status = stabilize_status(track_id, raw_status) if (track_id >= 0 and distortion_enabled) else raw_status
                            color = (0, 0, 255) if status.startswith('NG') else (0, 255, 0)

                            if conf > max_conf:
                                max_conf = conf
                                best_target = (
                                    x1, y1, x2, y2, cx, cy, conf, status, color,
                                    detected_name, shape_debug, ng_reasons
                                )
                except Exception as yolo_err:
                    print(f"[YOLO Error] {yolo_err}")
            else:
                mask_u8 = get_cloth_mask(frame)
                contours, _ = cv2.findContours(mask_u8, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
                if contours:
                    main_cnt = max(contours, key=cv2.contourArea)
                    area = cv2.contourArea(main_cnt)
                    if area > 1200: 
                        x, y, box_w, box_h = cv2.boundingRect(main_cnt)
                        cx = int(x + box_w / 2.0)
                        cy = int(y + box_h / 2.0)

                        roi_mask = mask_u8[y:y+box_h, x:x+box_w]
                        ng_reasons = []
                        shape_debug = {}

                        if distortion_enabled:
                            shape_ng, shape_debug = analyze_shape(roi_mask, box_w, box_h)
                            ng_reasons.extend(shape_ng)
                            raw_status = f"NG ({'/'.join(shape_ng)})" if shape_ng else 'OK (正常)'
                            status = stabilize_status(1, raw_status)
                        else:
                            raw_status = 'OK (歪み検知OFF)'
                            status = raw_status

                        color = (0, 0, 255) if status.startswith('NG') else (0, 255, 0)

                        best_target = (
                            x, y, x + box_w, y + box_h, cx, cy, 0.95, status, color,
                            'cloth (CV)', shape_debug, ng_reasons
                        )

            now = time.time()

            if best_target is not None:
                (
                    x1, y1, x2, y2, cx, cy, conf, status, color,
                    detected_name, shape_debug, ng_reasons
                ) = best_target

                is_ng = status.startswith('NG')


                status_changed = (status != last_sent_status)
                time_elapsed = (now - last_sent_time) >= MIN_ALERT_INTERVAL

                if status_changed or (is_ng and time_elapsed):
                    last_sent_time = now
                    last_sent_status = status

                    alert_payload = {
                        "detected": True,
                        "status": status,
                        "is_ng": is_ng,
                        "reasons": ng_reasons,
                        "conf": round(conf, 2),
                        "cx": cx,
                        "cy": cy,
                        "convexity": round(shape_debug.get('convexity', 0.0), 2),
                        "rectangularity": round(shape_debug.get('rectangularity', 0.0), 2),
                        "aspect": round(shape_debug.get('measured_aspect', 0.0), 2),
                        "timestamp": now
                    }
                    send_cloth_alert(alert_payload)
                    print(f"[{'ALERT' if is_ng else 'OK'}] Sent: {status} at ({cx}, {cy})")

                # 描画処理 (GUI用)
                cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
                cv2.circle(frame, (cx, cy), 5, (0, 0, 255), -1)

                if 'contour' in shape_debug:
                    shifted_cnt = shape_debug['contour'] + np.array([x1, y1])
                    cv2.drawContours(frame, [shifted_cnt], -1, (255, 200, 0), 2)

                info_text = f'[{status}] {detected_name}({conf:.2f}) Pos:({cx},{cy})'
                cv2.putText(
                    frame, info_text, (x1, max(20, y1 - 10)),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.55, color, 2
                )
            else:
                # 雑巾が見えなくなった場合、1回だけロスト状態を送信
                if last_sent_status is not None and (now - last_sent_time) >= 1.0:
                    last_sent_status = None
                    last_sent_time = now
                    send_cloth_alert({
                        "detected": False,
                        "status": "NOT_DETECTED",
                        "is_ng": False,
                        "reasons": [],
                        "conf": 0.0,
                        "cx": 0,
                        "cy": 0,
                        "timestamp": now
                    })

            cv2.line(frame, (int(CX), 0), (int(CX), current_h), (255, 0, 0), 1)


            toggle_text = "[DISTORTION: ON] (Press 'd' to toggle)" if distortion_enabled else "[DISTORTION: OFF] (Press 'd' to toggle)"
            toggle_color = (0, 255, 0) if distortion_enabled else (0, 165, 255)
            cv2.putText(frame, toggle_text, (10, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.6, toggle_color, 2)


            try:
                cv2.imshow('Cloth Straight/Folded Tracker', frame)
                key = cv2.waitKey(1) & 0xFF
                if key == ord('q'):
                    break
                elif key == ord('d') or key == ord('D'):
                    distortion_enabled = not distortion_enabled
                    state_msg = "ON (有効)" if distortion_enabled else "OFF (停止・スキップ中)"
                    print(f"\n>>> [ClothDetector] 歪み検知切替: {state_msg}\n")
            except Exception:
                pass

    finally:
        cap.release()
        try:
            cv2.destroyAllWindows()
        except Exception:
            pass


if __name__ == '__main__':
    main()
