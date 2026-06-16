import json
import os
import subprocess
import joblib
import numpy as np
from tqdm import tqdm

def generate_video(features_path="data/delivery_features.json", 
                   model_path="highlight_pipeline.pkl", 
                   video_path="match1_h264.mp4",
                   output_path="final_highlights.mp4"):
    
    print("Loading pipeline and features...")
    pipeline = joblib.load(model_path)
    pca_vid = pipeline["pca_vid"]
    pca_txt = pipeline["pca_txt"]
    le = pipeline["label_encoder"]
    clf = pipeline["model"]
    
    with open(features_path) as f:
        deliveries = json.load(f)
        
    X_raw = []
    valid_deliveries = []
    
    for d in deliveries:
        if d.get("feature_vector") and len(d["feature_vector"]) >= 1165:
            X_raw.append(d["feature_vector"])
            valid_deliveries.append(d)
            
    X_raw = np.array(X_raw)
    if len(X_raw) == 0:
        print("No valid deliveries found with full feature vectors.")
        return

    # Transform features through PCA pipeline
    vid_feats = X_raw[:, 0:768]
    txt_feats = X_raw[:, 768:1152]
    core_feats = X_raw[:, 1152:1165]

    vid_pca = pca_vid.transform(vid_feats)
    txt_pca = pca_txt.transform(txt_feats)

    X_final = np.concatenate([vid_pca, txt_pca, core_feats], axis=1)
    
    print("Running predictions...")
    y_pred_idx = clf.predict(X_final)
    predictions = le.inverse_transform(y_pred_idx)
    
    highlight_clips = []
    if len(valid_deliveries) > 0:
        first_ball = dict(valid_deliveries[0])
        first_ball["clip_start"] = 0.0
        highlight_clips.append(first_ball)
        print(f"  Selected Delivery {first_ball.get('delivery_id', '?'):>3} {first_ball.get('hms', '?')} -> Predicted: opening_and_first_ball")

    for i, (pred, d) in enumerate(zip(predictions, valid_deliveries)):
        if i == 0:
            continue
        if pred not in ["none", "four"]:
            highlight_clips.append(d)
            print(f"  Selected Delivery {d.get('delivery_id', '?'):>3} {d.get('hms', '?')} -> Predicted: {pred}")
            
    if not highlight_clips:
        print("\nModel didn't predict ANY highlights! Falling back to top 10 probabilities...")
        probs = clf.predict_proba(X_final)
        none_idx = list(le.classes_).index("none")
        
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
