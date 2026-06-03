import cv2
import numpy as np

# We'll extract a few frames at known timestamps that might be deliveries
timestamps = [30, 45, 60, 120, 155, 180, 240, 300]
cap = cv2.VideoCapture('match1_h264.mp4')
fps = cap.get(cv2.CAP_PROP_FPS)

for t in timestamps:
    cap.set(cv2.CAP_PROP_POS_FRAMES, t * fps)
    ret, frame = cap.read()
    if ret:
        cv2.imwrite(f'sample_frame_{t}.jpg', frame)
        hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
        h, w = frame.shape[:2]
        
        # Center column lower half - likely pitch
        pitch_roi = hsv[int(h*0.4):int(h*0.8), int(w*0.45):int(w*0.55)]
        mean_hsv = np.mean(pitch_roi, axis=(0,1))
        
        # Whole screen green
        green_mask = cv2.inRange(hsv, np.array([35, 40, 40]), np.array([85, 255, 255]))
        green_ratio = np.sum(green_mask > 0) / (h * w)
        
        print(f"T={t}s: Green Ratio = {green_ratio:.2f}, Pitch ROI Mean HSV = {mean_hsv}")

cap.release()
