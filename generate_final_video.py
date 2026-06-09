import json
import os
import subprocess
import joblib
import numpy as np
from tqdm import tqdm

def generate_video(features_path="data/delivery_features.json", 
                   model_path="highlight_classifier.pkl", 
                   video_path="match1_h264.mp4",
                   output_path="final_highlights.mp4"):
    
    print("Loading model and features...")
    model = joblib.load(model_path)
    
    with open(features_path) as f:
        deliveries = json.load(f)
        
    X = []
    valid_deliveries = []
    
    for d in deliveries:
        if d.get("feature_vector"):
            X.append(d["feature_vector"])
            valid_deliveries.append(d)
            
    X = np.array(X)
    
    # Predict on all available deliveries
    print("Running predictions...")
    predictions = model.predict(X)
    
    # Filter for anything that isn't 'none'
    highlight_clips = []
    for pred, d in zip(predictions, valid_deliveries):
        if pred != "none":
            highlight_clips.append(d)
            print(f"  Selected Delivery {d['delivery_id']:>3} {d['hms']} -> Predicted: {pred}")
            
    if not highlight_clips:
        print("\nModel didn't predict ANY highlights! This might be due to the small training dataset.")
        print("Falling back to top 10 deliveries based on 'none' probability threshold...")
        probs = model.predict_proba(X)
        none_idx = list(model.classes_).index("none")
        
        # highest probability of NOT being none
        not_none_probs = 1.0 - probs[:, none_idx]
        top_indices = np.argsort(not_none_probs)[-10:]
        
        for idx in sorted(top_indices): # chronological
            d = valid_deliveries[idx]
            highlight_clips.append(d)
            print(f"  Selected Delivery {d['delivery_id']:>3} {d['hms']} -> Probability of highlight: {not_none_probs[idx]:.3f}")

    print(f"\nTotal highlight clips selected: {len(highlight_clips)}")
    
    # Ensure temporary directory exists
    os.makedirs("temp_clips", exist_ok=True)
    
    # Extract each clip
    concat_list = "concat_list.txt"
    clip_files = []
    
    print("\nExtracting video clips using FFmpeg...")
    for idx, d in enumerate(tqdm(highlight_clips, desc="Encoding")):
        t_start = d["clip_start"]
        t_end = d["clip_end"]
        duration = t_end - t_start
        
        clip_name = f"temp_clips/clip_{idx}.mp4"
        cmd = [
            "ffmpeg", "-y", 
            "-ss", str(t_start), 
            "-i", video_path, 
            "-t", str(duration),
            "-c:v", "libx264", "-preset", "fast", "-crf", "23",
            "-c:a", "aac", "-b:a", "128k",
            "-loglevel", "error", 
            clip_name
        ]
        subprocess.run(cmd, check=True)
        clip_files.append(f"file '{clip_name}'")
        
    # Write concat list
    with open(concat_list, "w") as f:
        f.write("\n".join(clip_files))
        
    print(f"\nStitching {len(clip_files)} clips together...")
    cmd = [
        "ffmpeg", "-y", 
        "-f", "concat", 
        "-safe", "0", 
        "-i", concat_list, 
        "-c", "copy", 
        "-loglevel", "error",
        output_path
    ]
    subprocess.run(cmd, check=True)
    
    print(f"\nCleaning up temp files...")
    os.remove(concat_list)
    for f in os.listdir("temp_clips"):
        os.remove(os.path.join("temp_clips", f))
    os.rmdir("temp_clips")
    
    print(f"Done! Final video saved to {output_path}")

if __name__ == "__main__":
    generate_video()
