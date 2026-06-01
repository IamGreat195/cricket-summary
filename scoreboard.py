import cv2
import numpy as np
import re
import json
import sys
from tqdm import tqdm
import os
os.environ["OPENCV_LOG_LEVEL"] = "SILENT"  # suppress AV1 / codec warnings

# Inject CUDA/cuDNN libraries into LD_LIBRARY_PATH and restart if needed
cudnn_lib = os.path.abspath(os.path.join(os.path.dirname(__file__), ".venv/lib/python3.12/site-packages/nvidia/cudnn/lib"))
cublas_lib = os.path.abspath(os.path.join(os.path.dirname(__file__), ".venv/lib/python3.12/site-packages/nvidia/cublas/lib"))
current_ld = os.environ.get("LD_LIBRARY_PATH", "")
if cudnn_lib not in current_ld:
    os.environ["LD_LIBRARY_PATH"] = f"{cudnn_lib}:{cublas_lib}:{current_ld}".strip(":")
    os.execv(sys.executable, [sys.executable] + sys.argv)
from paddleocr import PaddleOCR

# Initialise once globally — loads GPU model into VRAM on first call
ocr = PaddleOCR(use_angle_cls=False, lang='en', use_gpu=True)

def crop_scoreboard(frame):
    h, w = frame.shape[:2]
    # bottom bar — starts at 85% height, full width to catch overflowing graphics
    y1 = int(h * 0.85)
    y2 = int(h * 1.00)
    x1 = 0
    x2 = w
    return frame[y1:y2, x1:x2]

def preprocess_for_ocr(region):
    # Scale up 2x — PaddleOCR works better on larger text
    scaled = cv2.resize(region, None, fx=2, fy=2,
                        interpolation=cv2.INTER_CUBIC)
    return scaled

def parse_ecb_scoreboard(ocr_text):
    result = {}
    
    # clean common OCR errors
    cleaned = ocr_text
    cleaned = cleaned.replace("'", "/")
    cleaned = cleaned.replace("‘", "/")
    cleaned = cleaned.replace("’", "/")
    cleaned = cleaned.replace("|", "1")
    cleaned = cleaned.replace("O", "0")
    
    # match score — runs-wickets with DASH e.g. "2-0", "156-3"
    # this comes BEFORE the over counter
    score_match = re.search(r'\b(\d{1,3})\s*-\s*(\d{1})\b', cleaned)
    if score_match:
        result['runs']    = int(score_match.group(1))
        result['wickets'] = int(score_match.group(2))
    
    # over counter — e.g. "1/50", "48.2/50" or noisy ones like "1 /50"
    over_match = re.search(r'\b(\d{1,2}(?:\.\d)?)\s*\/\s*50\b', cleaned)
    if over_match:
        result['overs'] = float(over_match.group(1))
    
    # batsman scores — e.g. "Roy 1(2)", "Topley24)", "C Overton 9 (17)"
    batsmen = re.findall(r'((?:[A-Z]\s+)?[A-Z][a-z]+)\s*(\d+)\s*[\(\[]?\s*(\d+)\s*[\)\]]', cleaned)
    if batsmen:
        result['batsmen'] = [
            {'name': b[0], 'runs': int(b[1]), 'balls': int(b[2])}
            for b in batsmen
        ]
    
    return result if 'runs' in result else None
    
def read_scoreboard(frame):
    region    = crop_scoreboard(frame)
    processed = preprocess_for_ocr(region)
    # Guard: skip if image is too small
    if processed.shape[0] < 10 or processed.shape[1] < 10:
        return None, ""
    # PaddleOCR v2.x returns: [[[[box], (text, confidence)], ...]] 
    result = ocr.ocr(processed, cls=False)
    # Flatten all detected text chunks into one string
    lines = []
    if result and result[0]:
        for item in result[0]:
            # item is [[x, y...], ("text", confidence)]
            if len(item) == 2 and isinstance(item[1], tuple):
                text_content = item[1][0]
                lines.append(text_content)
    text = " ".join(lines)
    return parse_ecb_scoreboard(text), text.strip()

# ── batch process segments ─────────────────────────────────────
def batch_process_scoreboards(video_path, json_path="segments_rms.json", out_path="segments_ocr.json"):
    with open(json_path, "r") as f:
        segments = json.load(f)
    
    # Resume support: skip segments already processed
    start_idx = 0
    if os.path.exists(out_path):
        print(f"Found existing {out_path}, resuming from checkpoint...")
        with open(out_path, "r") as f:
            existing = json.load(f)
        # Merge existing OCR results back in
        existing_map = {s["id"]: s for s in existing if "scoreboard" in s}
        for s in segments:
            if s["id"] in existing_map:
                s.update(existing_map[s["id"]])
        start_idx = sum(1 for s in segments if "scoreboard" in s)
        print(f"Resuming from segment {start_idx}/{len(segments)}")
    else:
        print(f"Processing OCR for all {len(segments)} segments...")
        print("Warning: This will take several hours to complete.")
    
    todo = [s for s in segments if "scoreboard" not in s]
    
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise RuntimeError(f"Failed to open video: {video_path}")
    
    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    
    for i, s in enumerate(tqdm(todo, desc="OCR Scoreboards", initial=start_idx, total=len(segments))):
        time_sec = s["start"] + 1.0
        target_msec = time_sec * 1000.0
        
        # Fast grab sequence instead of slow I-Frame set() searches
        ret = True
        frame = None
        while ret:
            msec = cap.get(cv2.CAP_PROP_POS_MSEC)
            if msec >= target_msec - (500.0 / fps): # within half a frame
                ret, frame = cap.read()
                break
            ret = cap.grab()
        
        if ret and frame is not None:
            try:
                parsed, raw = read_scoreboard(frame)
                s["scoreboard"] = parsed
                s["ocr_raw"]    = raw
            except Exception as e:
                s["scoreboard"] = None
                s["ocr_raw"]    = f"ERROR: {e}"
        else:
            s["scoreboard"] = None
            s["ocr_raw"]    = "ERROR: frame read failed"
        
        # Save checkpoint every 500 segments
        if (i + 1) % 500 == 0:
            with open(out_path, "w") as f:
                json.dump(segments, f)
            print(f"  [checkpoint saved at {start_idx + i + 1}/{len(segments)}]")
    
    cap.release()
    with open(out_path, "w") as f:
        json.dump(segments, f, indent=4)
    print(f"\nDone! Saved OCR results to {out_path}")

if __name__ == "__main__":
    batch_process_scoreboards("match1_h264.mp4")