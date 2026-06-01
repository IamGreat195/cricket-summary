import json
import subprocess
import cv2
import os
import pprint
from paddleocr import PaddleOCR
from scoreboard import read_scoreboard, crop_scoreboard, preprocess_for_ocr

print("Starting PaddleOCR... (this takes a moment to load CUDA)")

ocr = PaddleOCR(use_angle_cls=False, lang='en', use_gpu=False, show_log=False)
video_path = "match1.mp4.webm"

with open("segments_rms.json", "r") as f:
    segments = json.load(f)

print("\n--- SAMPLE EXTRACTING (First 10 segments) ---")
tmp_img = "tmp_sample.jpg"

for i in range(10):
    s = segments[i]
    time_sec = s["start"] + 1.0
    
    subprocess.run([
        "ffmpeg", "-y", "-ss", str(time_sec), "-i", video_path,
        "-vframes", "1", "-q:v", "2", tmp_img
    ], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    
    frame = cv2.imread(tmp_img)
    if frame is not None:
        region = crop_scoreboard(frame)
        processed = preprocess_for_ocr(region)
        # PaddleOCR v2.x returns result list
        result = ocr.ocr(processed, cls=False)
        
        print(f"\n[ Segment {i+1} : {s['start']}s -> {s['end']}s ]")
        pprint.pprint(result)

if os.path.exists(tmp_img):
    os.remove(tmp_img)
