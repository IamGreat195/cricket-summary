import cv2
import numpy as np

timestamps = [30, 45, 60, 120, 150, 180, 240, 300, 311] # adding some random times
cap = cv2.VideoCapture('match1_h264.mp4')
fps = cap.get(cv2.CAP_PROP_FPS)

for t in timestamps:
    cap.set(cv2.CAP_PROP_POS_FRAMES, t * fps)
    ret, frame = cap.read()
    if ret:
        hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
        h, w = frame.shape[:2]
        
        green_mask = cv2.inRange(hsv, np.array([35, 40, 40]), np.array([85, 255, 255]))
        
        # Center column lower half - likely pitch
        center_col = green_mask[int(h*0.3):int(h*0.8), int(w*0.4):int(w*0.6)]
        side_col_L = green_mask[int(h*0.3):int(h*0.8), int(w*0.1):int(w*0.3)]
        side_col_R = green_mask[int(h*0.3):int(h*0.8), int(w*0.7):int(w*0.9)]
        
        green_center = np.mean(center_col > 0)
        green_side_l = np.mean(side_col_L > 0)
        green_side_r = np.mean(side_col_R > 0)
        green_overall = np.mean(green_mask > 0)
        
        print(f"T={t:3d}s: Green={green_overall:.2f} | L={green_side_l:.2f} R={green_side_r:.2f} | Center={green_center:.2f}")

cap.release()
