import subprocess
import sys
import time
import os
import argparse
from pathlib import Path

def run_step(step_idx, total_steps, name, script_path, env, skip_if_exists=None):
    if skip_if_exists and all(os.path.exists(p) for p in skip_if_exists):
        print(f"\n[SKIP] Step {step_idx}/{total_steps}: {name}")
        print(f"       Already exists: {', '.join(skip_if_exists)}")
        return

    print(f"\n{'='*60}")
    print(f"Step {step_idx}/{total_steps}: {name}")
    print(f"{'='*60}")

    start = time.time()
    try:
        subprocess.run([sys.executable, "-u", script_path], check=True, env=env)
    except subprocess.CalledProcessError as e:
        print(f"\n[ERROR] '{script_path}' failed with exit code {e.returncode}.")
        sys.exit(1)

    duration = time.time() - start
    mins = int(duration // 60)
    secs = int(duration % 60)
    print(f"\n[SUCCESS] {name} completed in {mins}m {secs}s")

def main():
    parser = argparse.ArgumentParser(description="Cricket Highlight Pipeline")
    parser.add_argument("--video", default="match1_h264.mp4",
                        help="Path to input video file (default: match1_h264.mp4)")
    parser.add_argument("--output", default="final_highlights.mp4",
                        help="Path for the output highlights video (default: final_highlights.mp4)")
    args = parser.parse_args()

    video_path = args.video
    video_stem = Path(video_path).stem    # e.g. "match1_h264" or "t20i_england"
    audio_path = f"{video_stem}_audio.wav"
    data_dir   = f"data/{video_stem}"

    os.makedirs(data_dir, exist_ok=True)
    os.makedirs("data", exist_ok=True)

    # Build env with all pipeline variables, inheriting current env
    env = {**os.environ,
           "VIDEO_PATH": video_path,
           "AUDIO_PATH": audio_path,
           "DATA_DIR":   data_dir,
           "OUTPUT_PATH": args.output}

    print(f"\n CRICKET HIGHLIGHT PIPELINE ")
    print(f" Video:   {video_path}")
    print(f" Data:    {data_dir}/")
    print(f" Audio:   {audio_path}")
    print(f" Output:  {args.output}\n")

    steps = [
        ("Delivery Detection (CV)",             "delivery_detection.py",  [f"{data_dir}/deliveries.json"]),
        ("Audio RMS & Segment Building",         "cricket_summary.py",     [f"{data_dir}/segments_rms.json"]),
        ("Scoreboard OCR (long step)",           "scoreboard.py",          [f"{data_dir}/segments_ocr.json"]),
        ("First Gate — Event Tagging",           "first_gate.py",          [f"{data_dir}/highlights.json"]),
        ("Map Deliveries to Highlight Windows",  "map_deliveries.py",      [f"{data_dir}/delivery_highlights.json"]),
        ("Whisper Commentary Transcription",     "transcribe_deliveries.py",[f"{data_dir}/transcript.json"]),
        ("Feature Extraction (CLIP + Flow)",     "stage2_features.py",     [f"{data_dir}/delivery_features.json"]),
        ("Train Random Forest Classifier",       "train_classifier.py",    [f"{data_dir}/highlight_classifier.pkl"]),
        ("Generate Final Highlights Video",      "generate_final_video.py", None),
    ]

    overall_start = time.time()

    for i, (name, script, skip_artifacts) in enumerate(steps, 1):
        run_step(i, len(steps), name, script, env, skip_if_exists=skip_artifacts)

    overall_duration = time.time() - overall_start
    total_mins = int(overall_duration // 60)
    total_secs = int(overall_duration % 60)

    print(f"\n{'='*60}")
    print(f"PIPELINE FINISHED IN {total_mins}m {total_secs}s!")
    print(f"Output: '{args.output}'")
    print(f"{'='*60}\n")

if __name__ == "__main__":
    main()
