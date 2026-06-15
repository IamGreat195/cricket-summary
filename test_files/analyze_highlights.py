import cv2
import clip
import torch
from PIL import Image
import numpy as np
import subprocess
import os
import json
from tqdm import tqdm

device = "cuda" if torch.cuda.is_available() else "cpu"
VIDEO_PATH = "vidssave.com Cricket World Cup 2023 Final_ Australia v India _ Match Highlights 1080p.mp4"

PROMPTS = [
    "cricket batsman hitting a six",
    "cricket batsman hitting a boundary four",
    "cricket wicket falling batsman out",
    "cricket catch taken fielder",
    "cricket celebration players",
    "cricket replay on screen",
    "cricket crowd celebrating",
    "boring cricket dot ball",
]

LABELED_JSON = "data/highlight_deliveries_detected.json"

def analyze_highlights(video_path):
    if not os.path.exists(video_path):
        print(f"File not found: {video_path}")
        return

    if not os.path.exists(LABELED_JSON):
        print(f"{LABELED_JSON} not found — running delivery detection to create it...")
        from delivery_detection import detect_deliveries_fast
        detect_deliveries_fast(video_path, out_json=LABELED_JSON, sample_every=1, min_gap=10)
        print(f"Saved to {LABELED_JSON}. Add 'events' labels to each entry, then re-run.")

    print(f"Loading deliveries from {LABELED_JSON}...")
    with open(LABELED_JSON) as f:
        raw_deliveries = json.load(f)

    # For a highlight reel deliveries are back-to-back — no buffer time.
    # Slice segment[i] from its own timestamp to segment[i+1]'s timestamp.
    segments = []
    for i, d in enumerate(raw_deliveries):
        t_start = d["timestamp"]
        if i + 1 < len(raw_deliveries):
            t_end = raw_deliveries[i + 1]["timestamp"]
        else:
            t_end = t_start + 15  # last segment: small trailing window
        label = d.get("events", [])
        segments.append((t_start, t_end, label))
            
    print(f"Generated {len(segments)} segments. Loading CLIP model on {device}...")
    model, preprocess = clip.load("ViT-B/32", device=device)
    text_tokens = clip.tokenize(PROMPTS).to(device)

    results = []
    print(f"Processing {len(segments)} segments...")

    for i, (t_start, t_end, label) in enumerate(tqdm(segments)):
        # Sample frame from 30% into the segment — avoids replay/slow-mo intro frames
        t_sample = t_start + (t_end - t_start) * 0.3

        # Use ffmpeg to jump directly to the timestamp and pull one frame
        cmd = ["ffmpeg", "-y", "-ss", str(t_sample), "-i", video_path, "-vframes", "1", "temp_highlight.jpg"]
        subprocess.run(cmd, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)

        if not os.path.exists("temp_highlight.jpg"):
            continue

        pil = Image.open("temp_highlight.jpg").convert("RGB")
        image_input = preprocess(pil).unsqueeze(0).to(device)

        with torch.no_grad():
            logits, _ = model(image_input, text_tokens)
            probs = logits.softmax(dim=-1).cpu().numpy()[0]

        res = {
            "segment_id": i,
            "start": t_start,
            "end": t_end,
            "true_label": label,
            "predictions": {PROMPTS[j]: float(probs[j]) for j in range(len(PROMPTS))}
        }
        results.append(res)

    with open("clip_highlight_predictions.json", "w") as f:
        json.dump(results, f, indent=2)

    print("\n" + "="*65)
    print(f"{'Time':<12} {'True Label':<18} {'CLIP Top Prediction':<35}")
    print("="*65)
    for res in results:
        top = sorted(res["predictions"].items(), key=lambda x: x[1], reverse=True)[0]
        time_str = f"{int(res['start'])}s-{int(res['end'])}s"
        true_lbl = ",".join(res["true_label"]) if res["true_label"] else "(unlabeled)"
        top_pred = f"{top[0]} ({top[1]*100:.1f}%)"
        print(f"  {time_str:<10} {true_lbl:<18} {top_pred}")
    print("="*65)

if __name__ == "__main__":
    analyze_highlights(VIDEO_PATH)
