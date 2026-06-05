"""
build_cricket_dataset.py
────────────────────────
Extracts labelled frames from your match video using
the scoreboard events you already detected as labels.
No manual labelling needed for most classes.
"""

import cv2
import json
import os
from tqdm import tqdm

# event → folder name
CLASS_MAP = {
    "six":          "six",
    "four":         "four",
    "wicket":       "wicket",
    "runs":         "dot_or_single",
    "chase_climax": "tense_moment",
}

def extract_labelled_frames(
        video_path,
        delivery_highlights_path="delivery_highlights.json",
        out_dir="cricket_dataset"):

    with open(delivery_highlights_path) as f:
        deliveries = json.load(f)

    cap = cv2.VideoCapture(video_path)
    fps = cap.get(cv2.CAP_PROP_FPS)

    # create class folders
    for cls in CLASS_MAP.values():
        os.makedirs(f"{out_dir}/{cls}", exist_ok=True)
    os.makedirs(f"{out_dir}/none", exist_ok=True)

    counts = {cls: 0 for cls in CLASS_MAP.values()}
    counts["none"] = 0

    for delivery in tqdm(deliveries, desc="Extracting frames"):
        events_list = delivery.get("events", [])
        primary = "none"
        if "wicket" in events_list:
            primary = "wicket"
        elif "six" in events_list:
            primary = "six"
        elif "four" in events_list:
            primary = "four"
        elif "chase_climax" in events_list:
            primary = "chase_climax"
        elif "runs" in events_list:
            primary = "runs"
            
        cls = CLASS_MAP.get(primary, "none")

        # extract 3 frames from the clip window
        t_start = delivery["clip_start"]
        t_end   = delivery["clip_end"]
        times   = [
            t_start + (t_end - t_start) * 0.25,  # early
            t_start + (t_end - t_start) * 0.50,  # middle
            t_start + (t_end - t_start) * 0.75,  # late
        ]

        for t in times:
            cap.set(cv2.CAP_PROP_POS_MSEC, t * 1000)
            ret, frame = cap.read()
            if not ret:
                continue

            fname = f"{out_dir}/{cls}/delivery_{delivery['delivery_id']}_{int(t)}.jpg"
            cv2.imwrite(fname, frame)
            counts[cls] += 1

    cap.release()

    print("\nDataset built:")
    for cls, cnt in counts.items():
        print(f"  {cls:<20} {cnt} images")

    return counts

if __name__ == "__main__":
    extract_labelled_frames("match1_h264.mp4")