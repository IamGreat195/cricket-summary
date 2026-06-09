import json
import os
import subprocess
import random

video_path = "match1.mp4.webm"
json_path = "segments_rms.json"
output_dir = "sample_clips"

if not os.path.exists(json_path):
    print(f"Error: {json_path} does not exist. Please run cricket_summary.py first.")
    exit(1)

# Load the segments
with open(json_path, "r") as f:
    segments = json.load(f)

# Filter passing segments (you can adjust the threshold here)
passing = [s for s in segments if s.get("rms_norm", 0) >= 0.3]

os.makedirs(output_dir, exist_ok=True)

# Extract the first 10 clips for viewing
num_clips = 10
print(f"Total passing clips: {len(passing)}")
print(f"Extracting first {min(num_clips, len(passing))} clips to '{output_dir}' directory...")

random_clips = random.sample(passing[:1000], 50)

for i, s in enumerate(random_clips):
    start_time = s["start"]
    end_time = s["end"]
    duration = end_time - start_time
    # Create a nice filename with the clip info
    output_file = os.path.join(output_dir, f"clip_{i:03d}_id{s['id']}_rms{s['rms_norm']}.mp4")
    
    print(f"Extracting {output_file} from {start_time}s to {end_time}s")
    
    # Fast seek using -ss before -i and copy streams for instantaneous extraction
    subprocess.run([
        "ffmpeg", 
        "-ss", str(start_time), 
        "-i", video_path, 
        "-t", str(duration), 
        "-c", "copy",  # Copy codec for instantaneous extraction
        "-y", 
        output_file
    ], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)

print(f"\nExtraction complete! You can view the clips in the '{output_dir}' folder.")
