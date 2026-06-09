import cv2
import numpy as np
import re
import json
import sys
from tqdm import tqdm
import os
os.environ["OPENCV_LOG_LEVEL"] = "SILENT" 

cudnn_lib = os.path.abspath(os.path.join(os.path.dirname(__file__), ".venv/lib/python3.12/site-packages/nvidia/cudnn/lib"))
cublas_lib = os.path.abspath(os.path.join(os.path.dirname(__file__), ".venv/lib/python3.12/site-packages/nvidia/cublas/lib"))
current_ld = os.environ.get("LD_LIBRARY_PATH", "")
if cudnn_lib not in current_ld:
    os.environ["LD_LIBRARY_PATH"] = f"{cudnn_lib}:{cublas_lib}:{current_ld}".strip(":")
    os.execv(sys.executable, [sys.executable] + sys.argv)
from paddleocr import PaddleOCR

ocr = PaddleOCR(use_angle_cls=False, lang='en', use_gpu=True)

def crop_scoreboard(frame):
    h, w = frame.shape[:2]
    y1 = int(h * 0.85)
    y2 = int(h * 1.00)
    x1 = 0
    x2 = w
    return frame[y1:y2, x1:x2]

def preprocess_for_ocr(region):
    scaled = cv2.resize(region, None, fx=2, fy=2,
                        interpolation=cv2.INTER_CUBIC)
    return scaled

def parse_ecb_scoreboard(ocr_text):
    result = {}

    cleaned = ocr_text
    cleaned = cleaned.replace("'", "/")
    cleaned = cleaned.replace("'", "/")
    cleaned = cleaned.replace("|", "1")

    score_match = re.search(r'\b(\d{1,3})\s*-\s*(\d{1})\b', cleaned)
    if score_match:
        result['runs']    = int(score_match.group(1))
        result['wickets'] = int(score_match.group(2))

    over_match = re.search(r'\b(\d{1,2}(?:\.\d)?)\s*\/\s*50\b', cleaned)
    if over_match:
        result['overs'] = float(over_match.group(1))

    innings_match = re.search(r'\bP([12])\b', cleaned)
    if innings_match:
        result['innings'] = int(innings_match.group(1))

    crr_match = re.search(
        r'Run\s+Rate\s+(\d{1,2}\.\d{1,2})', cleaned, re.IGNORECASE)
    if crr_match:
        result['run_rate'] = float(crr_match.group(1))

    rrr_match = re.search(
        r'(?:Req|RRR|Required)\s*(?:Rate)?\s*(\d{1,2}\.\d{1,2})',
        cleaned, re.IGNORECASE)
    if rrr_match:
        result['required_run_rate'] = float(rrr_match.group(1))

    target_match = re.search(r'Target\s+(\d{2,3})', cleaned, re.IGNORECASE)
    if target_match:
        result['target'] = int(target_match.group(1))

    batsmen = re.findall(
        r'((?:[A-Z]\s+)?[A-Z][a-z]+)\s+(\d+)\s*\((\d+)\)', cleaned)
    cricket_keywords = {
        'Run', 'Rate', 'Req', 'Target', 'Toss', 'Mph', 'Kph', 'Over'
    }
    if batsmen:
        result['batsmen'] = [
            {'name': b[0].strip(), 'runs': int(b[1]), 'balls': int(b[2])}
            for b in batsmen
            if b[0].strip() not in cricket_keywords
            and not re.match(r'^[A-Z][a-z]+$', b[0]) == None
        ]

    bowler_match = re.search(
        r'([A-Z][a-z]+)\s+(\d)-(\d{1,3})\s*\((\d{1,2}\.\d)\)', cleaned)
    if bowler_match:
        result['bowler'] = {
            'name':    bowler_match.group(1),
            'wickets': int(bowler_match.group(2)),
            'runs':    int(bowler_match.group(3)),
            'overs':   float(bowler_match.group(4))
        }

    return result if 'runs' in result else None
        
def read_scoreboard(frame):
    region    = crop_scoreboard(frame)
    processed = preprocess_for_ocr(region)
    if processed.shape[0] < 10 or processed.shape[1] < 10:
        return None, ""
    result = ocr.ocr(processed, cls=False)
    lines = []
    if result and result[0]:
        for item in result[0]:
            if len(item) == 2 and isinstance(item[1], tuple):
                text_content = item[1][0]
                lines.append(text_content)
    text = " ".join(lines)
    return parse_ecb_scoreboard(text), text.strip()

def batch_process_scoreboards(video_path, json_path="data/segments_rms.json", out_path="data/segments_ocr.json"):
    with open(json_path, "r") as f:
        segments = json.load(f)
    
    start_idx = 0
    if os.path.exists(out_path):
        print(f"Found existing {out_path}, resuming from checkpoint...")
        with open(out_path, "r") as f:
            existing = json.load(f)
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
        
        ret = True
        frame = None
        while ret:
            msec = cap.get(cv2.CAP_PROP_POS_MSEC)
            if msec >= target_msec - (500.0 / fps): 
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