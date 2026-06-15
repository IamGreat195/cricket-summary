import cv2
import clip
import torch
from PIL import Image
import numpy as np
import warnings
warnings.filterwarnings("ignore")

# Prompts from stage2_features.py
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

device = "cuda" if torch.cuda.is_available() else "cpu"

def analyze_video(video_path):
    print(f"\nLoading CLIP model on {device}...")
    model, preprocess = clip.load("ViT-B/32", device=device)
    text_tokens = clip.tokenize(PROMPTS).to(device)

    import subprocess
    import os
    # extract middle frame via ffmpeg to avoid cv2 av1 decoding errors
    cmd = ["ffmpeg", "-y", "-i", video_path, "-vf", "select='eq(n\,0)'", "-vframes", "1", "temp.jpg"]
    subprocess.run(cmd, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    
    if not os.path.exists("temp.jpg"):
        print("Failed to read frame")
        return
        
    pil = Image.open("temp.jpg").convert("RGB")
    image_input = preprocess(pil).unsqueeze(0).to(device)
    
    with torch.no_grad():
        logits_per_image, _ = model(image_input, text_tokens)
        probs = logits_per_image.softmax(dim=-1).cpu().numpy()
        
    print(f"\nAnalyzing a frame of {video_path}")
    print("-" * 50)
    
    # Sort by probability descending
    results = list(zip(PROMPTS, probs[0]))
    results.sort(key=lambda x: x[1], reverse=True)
    
    for prompt, prob in results:
        print(f"{prompt:<40}: {prob*100:6.2f}%")

if __name__ == '__main__':
    import glob
    import random
    clips = glob.glob("sample_clips/*.mp4")
    if clips:
        # test 2 random clips
        random.seed(42) # fixed seed for reproducible test demo
        for c in random.sample(clips, 2):
            analyze_video(c)
    else:
        print("No clips found")
