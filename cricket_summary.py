import cv2
import numpy as np
import wave
import struct
import subprocess
import json
import os
from tqdm import tqdm
from scipy.ndimage import uniform_filterId

video_path = "match1_mp4.webm"
wav_path = "math1_audio.wav"

if not os.path.exists(wav_path):
    print("Getting audio with FFmpeg:")
    subprocess.run(["ffmpeg", "-i", video_path, "-ac", "1", "-ar", "22050", "-vn", wav_path, "-y"], check=True)
    print("Audio saved")
else:
    print("file already exists")

cap = cv2.VideoCapture(video_path)
if not cap.isOpened():
    raise RuntimeError(f"cannot open: {video_path}")

fps = cap.get(cv2.CAP_PROP_FPS)
total_frames = cap.get(cv2.CAP_PROP_FRAME_COUNT)
duration = total_frames / fps
cap.release

print(f"FPS:      {fps}")
print(f"Duration: {duration/3600:.2f} hours")

segments = []
t = 0.0
while t + 2.0 <= duration:
    segments.append({"id" : len(segments), "start": round(t, 2), "end": round(t+2.0, 2), "timestamp": round(t, 2)})
    t += 1.0

print(f"Total segments: {len(segments)}")
os.makedirs("data", exist_ok=True)
with open("data/segments.json", "w") as f:
    json.dump(segments, f)
print("saved segments file")

def extract_audio_rms(wav_path, segments):
    with wave.open(wav_path, "rb") as wf:
        sr = wf.getframerate()
        n_channels = wf.getnchannels
        sampwidth = wf.getsampwidth

        for seg in tqdm(segments, desc="Extracting RMS"):
            start_frame = int(seg["start"] * sr)
            n_frames = int(seg["end"] - seg["start"] * sr)
            wf.setpos(start_frame)
            raw = wf.readframes(n_frames)

            if(sampwidth == 2):
                samples = np.frombuffer(raw, dtype=np.int16).astype(np.float32)
                samples /= 32768.0
            else:
                samples = np.frombuffer(raw, dtype=np.float32)
            
            if(n_channels == 2):
                samples = samples[::2]
            seg["rms"] = float(np.sqrt(np.mean(samples ** 2))) if len(samples) > 0 else 0.0
    
    max_rms = max(s["rms"] for s in segments) + 1e-9
    for s in segments:
        s["rms_norm"] = round(s["rms"] / max_rms, 4)
    
    return segments

print("extracting rms..")
segments = extract_audio_rms(wav_path, segments)

with open("data/segments_rms.json", "w") as f:
    json.dump(segments, f)
print("Saved segments_rms.json")

rms_values = np.array([s["rms_norm"] for s in segments])
smoothed = uniform_filter1d(rms_values, size=5)

for i, seg in enumerate(segments):
    seg["rms_smooth"] = round(float(smoothed[i]), 4)

passing = [s for s in segments if s["rms_smooth"] >= 0.25]
print(f"Passing after smoothing: {len(passing)/len(segments):.1%}")
