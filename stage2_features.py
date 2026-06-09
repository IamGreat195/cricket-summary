"""
stage2_features.py
──────────────────
Extracts CLIP, optical flow, scoreboard, and commentary features
for each candidate delivery window. Saves feature vectors ready
for classifier training/inference.
"""

import cv2
import clip
import torch
import whisper
import numpy as np
import json
import os
import subprocess
from tqdm import tqdm

# ── setup ──────────────────────────────────────────────────────
device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Using device: {device}")

# load CLIP once
clip_model, clip_preprocess = clip.load("ViT-B/32", device=device)
clip_model.eval()

# ── CLIP prompts for zero-shot scoring ─────────────────────────
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
text_tokens   = clip.tokenize(PROMPTS).to(device)
with torch.no_grad():
    text_features = clip_model.encode_text(text_tokens)
    text_features /= text_features.norm(dim=-1, keepdim=True)

# ── commentary keywords ─────────────────────────────────────────
HIGH_SIGNAL = {
    "six":        1.0, "sixer":     1.0,
    "six!":       1.0, "maximum":   0.9,
    "four":       0.8, "boundary":  0.8,
    "four!":      0.8,
    "wicket":     1.0, "out":       0.8,
    "bowled":     1.0, "caught":    1.0,
    "lbw":        1.0, "stumped":   1.0,
    "runout":     1.0, "run-out":   1.0,
    "century":    0.9, "hundred":   0.9,
    "fifty":      0.7, "milestone": 0.6,
}

def keyword_score(text):
    words  = text.lower().replace("!", "").replace(",", "").split()
    scores = [HIGH_SIGNAL.get(w, 0.0) for w in words]
    return max(scores) if scores else 0.0


def extract_frames(video_path, t_start, t_end, fps_target=1):
    """
    Extract frames from [t_start, t_end] at fps_target using OpenCV.
    Returns list of (timestamp, frame) tuples.
    """
    cap    = cv2.VideoCapture(video_path)
    fps    = cap.get(cv2.CAP_PROP_FPS)
    frames = []

    interval = fps / fps_target  # every Nth frame
    start_f  = int(t_start * fps)
    end_f    = int(t_end   * fps)

    cap.set(cv2.CAP_PROP_POS_FRAMES, start_f)
    frame_idx = start_f

    while frame_idx <= end_f:
        ret, frame = cap.read()
        if not ret:
            break
        t = frame_idx / fps
        frames.append((t, frame))
        # skip to next sample
        skip = int(interval) - 1
        for _ in range(skip):
            if not cap.grab():
                break
        frame_idx += int(interval)

    cap.release()
    return frames


def clip_features(frames_1fps):
    """
    Run CLIP on 1fps frames.
    Returns: 512-d mean feature vector + 8 zero-shot similarity scores
    """
    if not frames_1fps:
        return np.zeros(512 + len(PROMPTS))

    from PIL import Image
    img_tensors = []
    for _, frame in frames_1fps:
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        pil = Image.fromarray(rgb)
        img_tensors.append(clip_preprocess(pil))

    batch = torch.stack(img_tensors).to(device)

    with torch.no_grad():
        img_feats = clip_model.encode_image(batch)
        img_feats /= img_feats.norm(dim=-1, keepdim=True)

    # mean pooled visual vector
    mean_vec = img_feats.mean(dim=0).cpu().numpy()  # 512-d

    # zero-shot similarity against each prompt
    sims = (img_feats @ text_features.T).cpu().numpy()  # (n_frames, 8)
    mean_sims = sims.mean(axis=0)  # 8-d

    return np.concatenate([mean_vec, mean_sims])  # 520-d total


def optical_flow_features(frames_2fps):
    """
    Compute optical flow between consecutive 2fps frame pairs.
    Returns: [mean_magnitude, max_magnitude, std_magnitude] — 3 numbers
    """
    if len(frames_2fps) < 2:
        return np.zeros(3)

    magnitudes = []
    prev_gray  = None

    for _, frame in frames_2fps:
        small = cv2.resize(frame, (320, 180))
        gray  = cv2.cvtColor(small, cv2.COLOR_BGR2GRAY)

        if prev_gray is not None:
            flow = cv2.calcOpticalFlowFarneback(
                prev_gray, gray,
                None, 0.5, 2, 12, 2, 5, 1.1, 0
            )
            mag = np.sqrt(flow[..., 0]**2 + flow[..., 1]**2)
            magnitudes.append(float(np.mean(mag)))

        prev_gray = gray

    if not magnitudes:
        return np.zeros(3)

    return np.array([
        np.mean(magnitudes),
        np.max(magnitudes),
        np.std(magnitudes)
    ])


def scoreboard_features(scoreboard):
    """
    Extract numerical features from scoreboard dict.
    Returns: 8-d vector
    """
    if scoreboard is None:
        return np.zeros(8)

    runs    = scoreboard.get("runs",    0)
    wickets = scoreboard.get("wickets", 0)
    overs   = scoreboard.get("overs",   0.0)
    innings = scoreboard.get("innings", 1)
    crr     = scoreboard.get("run_rate", 0.0) or 0.0
    rrr     = scoreboard.get("required_run_rate", 0.0) or 0.0
    target  = scoreboard.get("target",  0)

    balls_remaining = max(0, (50 - overs) * 6)
    runs_needed     = max(0, target - runs) if target else 0

    return np.array([
        runs    / 300.0,          # normalised runs
        wickets / 10.0,           # normalised wickets
        overs   / 50.0,           # match progress
        float(innings - 1),       # 0=1st innings, 1=2nd innings
        crr     / 12.0,           # normalised CRR
        rrr     / 12.0,           # normalised RRR
        runs_needed    / 300.0,   # how much still needed
        balls_remaining / 300.0,  # urgency
    ])


def get_commentary(whisper_segments, t_start, t_end, window=5):
    """
    Get commentary keyword score for this clip window.
    Whisper segments = list of {start, end, text} dicts
    """
    if not whisper_segments:
        return 0.0

    # grab all words spoken during or just around this clip
    words = []
    for seg in whisper_segments:
        if seg["start"] >= t_start - window and seg["end"] <= t_end + window:
            words.append(seg["text"])

    text = " ".join(words)
    return keyword_score(text)


def extract_all_features(video_path, delivery, whisper_segments):
    """
    Master function — extracts all features for one delivery.
    Returns flat numpy feature vector.
    """
    t_start = delivery["clip_start"]
    t_end   = delivery["clip_end"]

    # extract frames at two rates
    frames_1fps = extract_frames(video_path, t_start, t_end, fps_target=1)
    frames_2fps = extract_frames(video_path, t_start, t_end, fps_target=2)

    # run each extractor
    clip_vec   = clip_features(frames_1fps)          # 520-d
    flow_vec   = optical_flow_features(frames_2fps)  # 3-d
    score_vec  = scoreboard_features(                # 8-d
                     delivery.get("scoreboard"))
    comm_score = get_commentary(                     # 1-d
                     whisper_segments,
                     t_start, t_end)

    # concatenate → 532-d feature vector
    feature_vec = np.concatenate([
        clip_vec,
        flow_vec,
        score_vec,
        [comm_score]
    ])

    return feature_vec


# ── run on all deliveries ──────────────────────────────────────
if __name__ == "__main__":

    VIDEO_PATH = "match1_h264.mp4"

    with open("data/delivery_highlights.json") as f:
        deliveries = json.load(f)

    # load whisper transcript if available
    whisper_segments = []
    if os.path.exists("data/transcript.json"):
        with open("data/transcript.json") as f:
            whisper_segments = json.load(f)
        print(f"Loaded {len(whisper_segments)} whisper segments")
    else:
        print("No data/transcript.json found — commentary score will be 0")
        print("Run: whisper match1_audio.wav --model small "
              "--word_timestamps True --output_format json")

    results = []

    for delivery in tqdm(deliveries, desc="Extracting features"):
        feat = extract_all_features(VIDEO_PATH, delivery, whisper_segments)

        results.append({
            "delivery_id":        delivery["delivery_id"],
            "delivery_timestamp": delivery["delivery_timestamp"],
            "hms":                delivery["delivery_hms"],
            "clip_start":         delivery["clip_start"],
            "clip_end":           delivery["clip_end"],
            "primary_event":      delivery.get("primary_event", "unknown"),
            "events":             delivery.get("events", []),
            "scoreboard":         delivery.get("scoreboard"),
            "feature_vector":     feat.tolist(),  # save as list for JSON
            "feature_dim":        len(feat),
        })

    os.makedirs("data", exist_ok=True)
    with open("data/delivery_features.json", "w") as f:
        json.dump(results, f, indent=2)

    print(f"\nSaved {len(results)} feature vectors → data/delivery_features.json")
    print(f"Feature vector dimension: {results[0]['feature_dim']}")