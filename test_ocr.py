import os
import sys

# Inject CUDA/cuDNN libraries into LD_LIBRARY_PATH and restart if needed
cudnn_lib = os.path.abspath(os.path.join(os.path.dirname(__file__), ".venv/lib/python3.12/site-packages/nvidia/cudnn/lib"))
cublas_lib = os.path.abspath(os.path.join(os.path.dirname(__file__), ".venv/lib/python3.12/site-packages/nvidia/cublas/lib"))
current_ld = os.environ.get("LD_LIBRARY_PATH", "")
if cudnn_lib not in current_ld:
    os.environ["LD_LIBRARY_PATH"] = f"{cudnn_lib}:{cublas_lib}:{current_ld}".strip(":")
    os.execv(sys.executable, [sys.executable] + sys.argv)

import json
import random
import cv2
from tqdm import tqdm
from scoreboard import read_scoreboard

VIDEO_PATH  = "match1_h264.mp4"
SEGMENTS_IN = "segments_rms.json"
N_SAMPLES   = 20

print("Initialising PaddleOCR GPU...")
from paddleocr import PaddleOCR  # noqa: imported here so env is already set
# ocr is initialised inside scoreboard.py on import, so no need to duplicate it

with open(SEGMENTS_IN) as f:
    segments = json.load(f)

sample = random.sample(segments, N_SAMPLES)
sample.sort(key=lambda s: s["start"])

cap = cv2.VideoCapture(VIDEO_PATH)
if not cap.isOpened():
    raise RuntimeError(f"Cannot open {VIDEO_PATH}")

results = []
for s in tqdm(sample, desc="Quick OCR test"):
    time_sec = s["start"] + 1.0
    cap.set(cv2.CAP_PROP_POS_MSEC, time_sec * 1000)
    ret, frame = cap.read()

    entry = {k: s[k] for k in ("id", "start", "end", "rms_norm")}
    if ret and frame is not None:
        try:
            parsed, raw = read_scoreboard(frame)
            entry["scoreboard"] = parsed
            entry["ocr_raw"]    = raw
        except Exception as e:
            entry["scoreboard"] = None
            entry["ocr_raw"]    = f"ERROR: {e}"
    else:
        entry["scoreboard"] = None
        entry["ocr_raw"]    = "ERROR: frame read failed"

    results.append(entry)

cap.release()

out_path = "test_ocr_sample.json"
with open(out_path, "w") as f:
    json.dump(results, f, indent=4)

print(f"\nSaved {N_SAMPLES} sample results → {out_path}\n")
for r in results:
    print(f"  [{r['start']}s]  raw: {r['ocr_raw']}")
    print(f"        parsed: {r['scoreboard']}")
