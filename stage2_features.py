import cv2
import torch
import numpy as np
import json
import os
import subprocess
from tqdm import tqdm
from scipy.io import wavfile
from transformers import VideoMAEImageProcessor, VideoMAEModel
from sentence_transformers import SentenceTransformer

device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Using device: {device}")

print("Loading VideoMAE...")
videomae_processor = VideoMAEImageProcessor.from_pretrained("MCG-NJU/videomae-base")
videomae_model = VideoMAEModel.from_pretrained("MCG-NJU/videomae-base").to(device)
videomae_model.eval()

print("Loading SentenceTransformer...")
sent_model = SentenceTransformer('all-MiniLM-L6-v2', device=device)
sent_model.eval()

AUDIO_PATH = "match1_audio.wav"
if os.path.exists(AUDIO_PATH):
    sample_rate, audio_data = wavfile.read(AUDIO_PATH)
    if len(audio_data.shape) > 1:
        audio_data = audio_data.mean(axis=1)
else:
    print(f"Warning: {AUDIO_PATH} not found. Audio features will be 0.")
    sample_rate, audio_data = 16000, np.zeros(16000)

def audio_features(t_start, t_end):
    start_idx = int(t_start * sample_rate)
    end_idx = int(t_end * sample_rate)
    chunk = audio_data[start_idx:end_idx]
    if len(chunk) == 0:
        return np.zeros(2)
    chunk = chunk.astype(float)
    rms = np.sqrt(np.mean(chunk**2)) if len(chunk) > 0 else 0
    abs_max = np.max(np.abs(chunk)) if len(chunk) > 0 else 0
    return np.array([rms / 32768.0, abs_max / 32768.0])

def extract_frames(video_path, t_start, t_end, fps_target=2):
    cap = cv2.VideoCapture(video_path)
    fps = cap.get(cv2.CAP_PROP_FPS)
    frames = []
    if fps == 0:
        return frames
    interval = fps / fps_target
    start_f = int(t_start * fps)
    end_f = int(t_end * fps)
    cap.set(cv2.CAP_PROP_POS_FRAMES, start_f)
    frame_idx = start_f
    while frame_idx <= end_f:
        ret, frame = cap.read()
        if not ret:
            break
        t = frame_idx / fps
        frames.append((t, frame))
        skip = int(interval) - 1
        for i in range(skip):
            if not cap.grab():
                break
        frame_idx += int(interval)
    cap.release()
    return frames

def videomae_features(frames_video):
    if len(frames_video) == 0:
        return np.zeros(768)
        
    indices = np.linspace(0, len(frames_video) - 1, 16, dtype=int)
    sampled = [cv2.cvtColor(frames_video[i][1], cv2.COLOR_BGR2RGB) for i in indices]
    
    inputs = videomae_processor(list(sampled), return_tensors="pt").to(device)
    with torch.no_grad():
        outputs = videomae_model(**inputs)
    embed = outputs.last_hidden_state.mean(dim=1).squeeze(0).cpu().numpy()
    return embed

def optical_flow_features(frames_2fps):
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
        runs    / 300.0,        
        wickets / 10.0,         
        overs   / 50.0,          
        float(innings - 1),       
        crr     / 12.0,       
        rrr     / 12.0,      
        runs_needed    / 300.0, 
        balls_remaining / 300.0, 
    ])

def sentence_features(whisper_segments, t_start, t_end, window=5):
    if not whisper_segments:
        return np.zeros(384)
    words = []
    for seg in whisper_segments:
        if seg["start"] >= t_start - window and seg["end"] <= t_end + window:
            words.append(seg["text"])
    
    text = " ".join(words).strip()
    if not text:
        return np.zeros(384)
    with torch.no_grad():
        embed = sent_model.encode(text)
    return embed

def extract_all_features(video_path, delivery, whisper_segments):
    t_start = delivery["clip_start"]
    t_end   = delivery["clip_end"]
    
    frames_2fps = extract_frames(video_path, t_start, t_end, fps_target=2)
    
    vid_vec   = videomae_features(frames_2fps)          # 768-d
    text_vec  = sentence_features(whisper_segments,
                                  t_start, t_end)       # 384-d
    flow_vec  = optical_flow_features(frames_2fps)      # 3-d
    score_vec = scoreboard_features(                    # 8-d
                     delivery.get("scoreboard"))
    aud_vec   = audio_features(t_start, t_end)          # 2-d
    
    feature_vec = np.concatenate([
        vid_vec,
        text_vec,
        flow_vec,
        score_vec,
        aud_vec
    ])
    return feature_vec

if __name__ == "__main__":
    import os
    VIDEO_PATH = os.environ.get("VIDEO_PATH", "match1_h264.mp4")
    AUDIO_PATH = os.environ.get("AUDIO_PATH", "match1_audio.wav")
    data_dir   = os.environ.get("DATA_DIR", "data")

    with open(f"{data_dir}/delivery_highlights.json") as f:
        deliveries = json.load(f)

    whisper_segments = []
    if os.path.exists("data/transcript.json"):
        with open("data/transcript.json") as f:
            whisper_segments = json.load(f)
        print(f"Loaded {len(whisper_segments)} whisper segments")
    else:
        print("No data/transcript.json found — commentary score will be 0")

    results = []

    for delivery in tqdm(deliveries, desc="Extracting features"):
        feat = extract_all_features(VIDEO_PATH, delivery, whisper_segments)

        results.append({
            "delivery_id":        delivery["delivery_id"],
            "delivery_timestamp": delivery.get("delivery_timestamp", delivery.get("clip_start", 0)),
            "hms":                delivery.get("delivery_hms", delivery.get("clip_start_hms", "")),
            "clip_start":         delivery.get("clip_start", 0),
            "clip_end":           delivery.get("clip_end", 0),
            "events":             delivery.get("events", []),
            "scoreboard":         delivery.get("scoreboard"),
            "feature_vector":     feat.tolist(),  
            "feature_dim":        len(feat),
        })

    os.makedirs(data_dir, exist_ok=True)
    with open(f"{data_dir}/delivery_features.json", "w") as f:
        json.dump(results, f, indent=2)

    print(f"\nSaved {len(results)} feature vectors → {data_dir}/delivery_features.json")
    if len(results) > 0:
        print(f"Feature vector dimension: {results[0]['feature_dim']}")
    else:
        print("No deliveries to extract features for.")