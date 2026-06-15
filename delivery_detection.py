import cv2
import numpy as np
import json
import os
from tqdm import tqdm

def detect_deliveries_fast(video_path, out_json="data/deliveries.json", sample_every=1, min_gap=18, motion_threshold=0.015, green_threshold=0.30, onset_offset_secs=1.5):
    cap = cv2.VideoCapture(video_path)
    fps = cap.get(cv2.CAP_PROP_FPS)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    duration = total_frames/fps
    frame_interval = max(1, int(fps * sample_every))
    total_samples = total_frames // frame_interval

    print(f"Duration: {duration/3600:.2f}h  |  Sampling every {sample_every}s  |  ~{total_samples} frames to check")
    deliveries = []
    last_delivery_t = -999
    prev_gray_small = None
    frame_idx = 0

    pbar = tqdm(total=total_samples, desc="Detecting deliveries")
    os.makedirs(os.path.dirname(out_json), exist_ok=True)
    while True:
        for i in range(frame_interval - 1):
            if not cap.grab():
                cap.release()
                pbar.close()
                with open(out_json, "w") as f:
                    json.dump(deliveries, f, indent=2)
                print(f"\nDetected {len(deliveries)} deliveries")
                return deliveries
        
        ret, frame = cap.read()
        if not ret:
            break
        t = frame_idx / fps
        frame_idx += frame_interval
        pbar.update(1)

        if t - last_delivery_t < min_gap:
            continue

        h, w = frame.shape[:2]
        small = cv2.resize(frame, (160, 90))
        hsv = cv2.cvtColor(small, cv2.COLOR_BGR2HSV)
        gmask = cv2.inRange(hsv, np.array([25, 20, 20]), np.array([95, 255, 255]))
        sh, sw = small.shape[:2]

        center = gmask[int(sh*0.3):int(sh*0.8), int(sw*0.4):int(sw*0.6)]
        side_l = gmask[int(sh*0.3):int(sh*0.8), int(sw*0.1):int(sw*0.3)]
        side_r = gmask[int(sh*0.3):int(sh*0.8), int(sw*0.7):int(sw*0.9)]

        g_overall = np.mean(gmask > 0)
        g_center = np.mean(center > 0)
        g_side_l  = np.mean(side_l > 0)
        g_side_r  = np.mean(side_r > 0)

        is_delivery_angle = (g_overall > green_threshold and g_side_l > 0.60 and g_side_r > 0.60 and g_center < 0.85)
        if not is_delivery_angle:
            gray_small = cv2.cvtColor(cv2.resize(frame, (160, 90)), cv2.COLOR_BGR2GRAY)
            prev_gray_small = gray_small
            continue
        motion_score = 0.0
        gray_small = cv2.cvtColor(cv2.resize(frame, (160, 90)), cv2.COLOR_BGR2GRAY)
        if prev_gray_small is not None:
            diff = cv2.absdiff(prev_gray_small, gray_small)
            motion_score = float(np.mean(diff)) / 255.0
        prev_gray_small = gray_small

        if motion_score >= motion_threshold:
            delivery_t = max(0.0, t - onset_offset_secs)
            hour = int(delivery_t) // 3600
            mins = (int(delivery_t) % 3600) // 60
            sec = int(delivery_t) % 60

            deliveries.append({"delivery_id": len(deliveries) + 1, "timestamp": round(delivery_t, 2), "hms": f"{hour:02d}:{mins:02d}:{sec:02d}", "motion_score": round(motion_score, 4), "green_ratio":  round(g_overall, 3)})
            last_delivery_t = t
    pbar.close()
    cap.release()

    with open(out_json, "w") as f:
        json.dump(deliveries, f, indent = 2)

    print(f"\nDetected {len(deliveries)} deliveries → {out_json}")
    print(f"Expected ~300 for full ODI, ~120 for T20")
    return deliveries

if __name__ == "__main__":
    detect_deliveries_fast("match1_h264.mp4")