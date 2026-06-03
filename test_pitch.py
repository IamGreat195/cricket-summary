import cv2
import numpy as np

frame = cv2.imread('sample_frame.jpg')
if frame is None:
    # Just try to grab one from the video
    cap = cv2.VideoCapture('match1_h264.mp4')
    cap.set(cv2.CAP_PROP_POS_FRAMES, 5000)
    ret, frame = cap.read()
    cv2.imwrite('sample_frame.jpg', frame)

hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
h, w = frame.shape[:2]

# Crop the center where pitch usually is (roughly middle x, middle-ish y)
pitch_roi = hsv[int(h*0.3):int(h*0.7), int(w*0.4):int(w*0.6)]

# Average HSV in pitch ROI
mean_hsv = np.mean(pitch_roi, axis=(0,1))
print("Mean HSV in center ROI:", mean_hsv)

# Let's see some samples of HSV in the center
import pprint
samples = []
for i in range(10):
    for j in range(10):
        y = int(h*0.3) + i * (int(h*0.7) - int(h*0.3)) // 10
        x = int(w*0.4) + j * (int(w*0.6) - int(w*0.4)) // 10
        samples.append(hsv[y, x, :])
        
samples = np.array(samples)
print("Min HSV:", np.min(samples, axis=0))
print("Max HSV:", np.max(samples, axis=0))
