import json
import os
import subprocess
import whisper
from tqdm import tqdm
import warnings

def extract_audio_segment(input_wav, output_wav, start_time, end_time):
    duration = end_time - start_time
    cmd = [
        "ffmpeg", "-y", "-i", input_wav,
        "-ss", str(start_time),
        "-t", str(duration),
        "-ac", "1", "-ar", "16000", "-loglevel", "error",
        output_wav
    ]
    subprocess.run(cmd, check=True)

def transcribe_deliveries(delivery_highlights_path="delivery_highlights.json", 
                          full_audio_path="match1_audio.wav", 
                          out_json="transcript.json"):
    
    with open(delivery_highlights_path) as f:
        deliveries = json.load(f)
        
    print(f"Loading Whisper model (small)...")
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        model = whisper.load_model("small")
        
    os.makedirs("temp_audio", exist_ok=True)
    
    global_segments = []
    
    for delivery in tqdm(deliveries, desc="Transcribing deliveries"):
        t_start = delivery["clip_start"]
        t_end = delivery["clip_end"]
        d_id = delivery["delivery_id"]
        
        # Add a 2s margin around the clip window for context
        t_start_padded = max(0, t_start - 2)
        t_end_padded = t_end + 2
        
        chunk_path = f"temp_audio/chunk_{d_id}.wav"
        extract_audio_segment(full_audio_path, chunk_path, t_start_padded, t_end_padded)
        
        # Run Whisper on the chunk
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            result = model.transcribe(chunk_path, language="en")
            
        # Shift segments back to global video time
        for seg in result["segments"]:
            global_segments.append({
                "start": seg["start"] + t_start_padded,
                "end": seg["end"] + t_start_padded,
                "text": seg["text"],
                "delivery_id": d_id
            })
            
        # Cleanup
        os.remove(chunk_path)
        
    # We sort the aggregated global segments just in case
    global_segments.sort(key=lambda x: x["start"])

    # Save the output to transcript.json, which is what stage2_features expects
    with open(out_json, "w") as f:
        json.dump(global_segments, f, indent=2)
        
    print(f"\nTranscribed {len(deliveries)} deliveries.")
    print(f"Generated {len(global_segments)} global text segments.")
    print(f"Saved to {out_json}")
    
    try:
        os.rmdir("temp_audio")
    except OSError:
        pass

if __name__ == "__main__":
    transcribe_deliveries()
