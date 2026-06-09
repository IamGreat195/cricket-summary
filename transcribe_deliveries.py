import json
import os
import warnings
import whisper
import torch

def transcribe_deliveries(delivery_highlights_path="data/delivery_highlights.json",
                          full_audio_path="match1_audio.wav",
                          out_json="data/transcript.json"):

    with open(delivery_highlights_path) as f:
        deliveries = json.load(f)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Loading Whisper model (small) on {device}...")
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        model = whisper.load_model("small", device=device)

    # ── Run Whisper ONCE on the full audio ──────────────────────────────────
    # This is much faster because Whisper natively skips silence (VAD), 
    # doesn't have Python loop overhead 200 times, and doesn't pad every 
    # tiny clip to 30 seconds.
    print(f"Transcribing full audio file: {full_audio_path}")
    print("This usually takes 3-5 mins, but avoids all Python overhead.")
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        result = model.transcribe(
            full_audio_path,
            language="en",
            verbose=False,  # This will still print one master progress bar natively in some versions!
            fp16=(device == "cuda"),
        )

    all_segs = result["segments"]
    print(f"Whisper produced {len(all_segs)} raw segments from full audio.")

    # ── Filter segments to only those overlapping a highlight clip ───────────
    windows = [
        (
            max(0, d["clip_start"] - 2),
            d["clip_end"] + 2,
            d["delivery_id"],
        )
        for d in deliveries
    ]

    global_segments = []
    for seg in all_segs:
        s, e, text = seg["start"], seg["end"], seg["text"]
        for w_start, w_end, d_id in windows:
            if s < w_end and e > w_start:
                global_segments.append({
                    "start": round(s, 3),
                    "end": round(e, 3),
                    "text": text,
                    "delivery_id": d_id,
                })
                break  

    global_segments.sort(key=lambda x: x["start"])

    os.makedirs(os.path.dirname(out_json), exist_ok=True)
    with open(out_json, "w") as f:
        json.dump(global_segments, f, indent=2)

    print(f"\nDelivery clips:         {len(deliveries)}")
    print(f"Total Whisper segments: {len(all_segs)}")
    print(f"Segments kept (in clips): {len(global_segments)}")
    print(f"Saved → {out_json}")

if __name__ == "__main__":
    transcribe_deliveries()
