import os
import re
import argparse
import time
from datetime import timedelta
from pathlib import Path

# ── Highlight keywords (add/remove as needed) ──────────────────────────────────
KEYWORDS = [
    # Wickets
    "out", "wicket", "bowled", "caught", "lbw", "stumped", "run out",
    "dismissed", "gone", "that's out", "he's out", "she's out",
    # Big shots
    "six", "sixes", "maximum", "huge", "massive", "out of the ground",
    "over the boundary", "into the crowd",
    # Fours
    "four", "boundary",
    # Close calls / drama
    "no ball", "wide", "dropped", "review", "umpire's call",
    "DRS", "third umpire", "not out", "overturned",
    # Pace / skill
    "yorker", "bouncer", "brilliant", "unplayable", "sensational",
    "outstanding", "magnificent", "what a delivery", "superb",
    # Score milestones
    "fifty", "hundred", "century", "fifty up", "hundred up",
    "five wickets", "hat trick", "hattrick",
]
# ───────────────────────────────────────────────────────────────────────────────

# Whisper model sizes: tiny (fastest) → base → small → medium → large (most accurate)
# For an i7 laptop: "base" is the sweet spot — fast and accurate enough
DEFAULT_WHISPER_MODEL = "base"

# Extra seconds to add at the end of the last delivery (match has no "next" delivery)
LAST_DELIVERY_EXTRA_SEC = 30


def hms(seconds):
    return str(timedelta(seconds=int(seconds)))


def load_delivery_timestamps(timestamps_file):
    """Load delivery start times from deliveries.json saved by detect_deliveries.py"""
    import json
    with open(timestamps_file, "r") as f:
        data = json.load(f)

    # Handle all possible JSON shapes:
    # 1. {"deliveries": [{"timestamp_sec": 12.3, ...}, ...]}  ← dict with list of dicts
    # 2. {"deliveries": [12.3, 45.6, ...]}                    ← dict with plain list
    # 3. [{"timestamp_sec": 12.3, ...}, ...]                  ← bare list of dicts
    # 4. [12.3, 45.6, ...]                                     ← bare plain list
    raw = data["deliveries"] if isinstance(data, dict) else data

    if len(raw) == 0:
        raise ValueError("No deliveries found in JSON file.")

    if isinstance(raw[0], dict):
        # list of dicts — try known key names first, then any numeric value
        preferred = ("timestamp_sec", "seconds", "time", "start", "ts")
        key = next((k for k in preferred if k in raw[0]), None)
        if key is None:
            # fallback: first key whose value is a number
            key = next((k for k, v in raw[0].items() if isinstance(v, (int, float))), None)
        if key is None:
            raise ValueError(f"No numeric key found in delivery dict: {raw[0]}")
        print(f"  Using key '{key}' for timestamps")
        deliveries = [float(d[key]) for d in raw]
    else:
        # plain list of numbers
        deliveries = [float(v) for v in raw]

    deliveries.sort()
    print(f"  Loaded {len(deliveries)} delivery timestamps from '{timestamps_file}'")
    return deliveries


def find_delivery_for_time(keyword_time, deliveries):
    """
    Given a keyword timestamp, return (delivery_start, delivery_end).
    delivery_end = start of the next delivery (or keyword_time + extra for last).
    """
    # Find the delivery that started just before or at the keyword time
    idx = None
    for i, d in enumerate(deliveries):
        if d <= keyword_time:
            idx = i
        else:
            break

    if idx is None:
        # Keyword is before the first delivery — use first delivery
        idx = 0

    start = deliveries[idx]
    if idx + 1 < len(deliveries):
        end = deliveries[idx + 1]
    else:
        end = keyword_time + LAST_DELIVERY_EXTRA_SEC

    return idx + 1, start, end   # 1-based delivery number


def extract_audio(video_path, audio_path):
    """Extract mono 16kHz WAV audio from the video for Whisper."""
    import subprocess
    print(f"  Extracting audio → {audio_path} ...")
    cmd = [
        "ffmpeg", "-y", "-i", video_path,
        "-vn",                     # no video
        "-ac", "1",               # mono
        "-ar", "16000",           # 16kHz (Whisper's native rate)
        "-acodec", "pcm_s16le",   # 16-bit PCM WAV
        audio_path
    ]
    result = subprocess.run(cmd, capture_output=True, text=True)
    if result.returncode != 0:
        raise RuntimeError(f"ffmpeg audio extraction failed:\n{result.stderr}")
    print(f"  Audio extracted.")


def transcribe_audio(audio_path, model_name):
    """Transcribe audio with Whisper and return list of (start, end, text) segments."""
    import whisper
    print(f"\n  Loading Whisper model '{model_name}' ...")
    model = whisper.load_model(model_name)
    print(f"  Transcribing (this is the slow part — ~5-15 min for a full match) ...")
    t0 = time.time()
    result = model.transcribe(audio_path, verbose=False, word_timestamps=False)
    elapsed = time.time() - t0
    segments = [(s["start"], s["end"], s["text"].strip()) for s in result["segments"]]
    print(f"  Transcription done in {elapsed:.0f}s — {len(segments)} segments")
    return segments


def find_keyword_hits(segments, keywords):
    """
    Search each transcript segment for keywords.
    Returns list of (timestamp_seconds, keyword_found, segment_text).
    Deduplicates: if multiple keywords in the same segment, pick the strongest.
    """
    hits = []
    keywords_lower = [k.lower() for k in keywords]

    for start, end, text in segments:
        text_lower = text.lower()
        matched = [k for k in keywords_lower if k in text_lower]
        if matched:
            # Use midpoint of segment as the keyword timestamp
            hits.append((start, matched[0], text))

    print(f"  Found {len(hits)} keyword hit(s) in transcript")
    return hits


def cut_clip(video_path, start_sec, end_sec, out_path):
    """Cut a clip from the video using ffmpeg (fast stream copy, no re-encode)."""
    import subprocess
    duration = end_sec - start_sec
    cmd = [
        "ffmpeg", "-y",
        "-ss", str(start_sec),
        "-i", video_path,
        "-t", str(duration),
        "-c", "copy",           # stream copy = very fast, no quality loss
        "-avoid_negative_ts", "1",
        out_path
    ]
    result = subprocess.run(cmd, capture_output=True, text=True)
    if result.returncode != 0:
        raise RuntimeError(f"ffmpeg clip extraction failed:\n{result.stderr}")


def save_transcript(segments, path):
    with open(path, "w", encoding="utf-8") as f:
        f.write("# Full Match Transcript\n")
        f.write("# Format: [start --> end]  text\n\n")
        for start, end, text in segments:
            f.write(f"[{hms(start)} --> {hms(end)}]  {text}\n")
    print(f"  Full transcript saved to: {path}")


def extract():
    parser = argparse.ArgumentParser(
        description="Extract highlight clips based on commentary keywords."
    )
    parser.add_argument("--video",       default="match1.mp4",      help="Match video file")
    parser.add_argument("--timestamps",  default="deliveries.json",  help="Delivery timestamps file from detect_deliveries.py")
    parser.add_argument("--model",       default=DEFAULT_WHISPER_MODEL,
                        choices=["tiny", "base", "small", "medium", "large"],
                        help="Whisper model size (default: base)")
    parser.add_argument("--output_dir",  default="highlights",      help="Folder to save highlight clips")
    parser.add_argument("--keep_audio",  action="store_true",       help="Keep extracted WAV audio file")
    parser.add_argument("--save_transcript", action="store_true",   help="Save full transcript to transcript.txt")
    args = parser.parse_args()

    print(f"\n{'='*57}")
    print(f"  Cricket Highlight Extractor")
    print(f"{'='*57}")
    print(f"  Video        : {args.video}")
    print(f"  Timestamps   : {args.timestamps}")
    print(f"  Whisper model: {args.model}")
    print(f"  Output folder: {args.output_dir}/")
    print(f"{'='*57}\n")

    # ── Step 1: Load delivery timestamps ──────────────────────────────────────
    deliveries = load_delivery_timestamps(args.timestamps)
    video_duration = deliveries[-1] + LAST_DELIVERY_EXTRA_SEC if deliveries else 3600

    # ── Step 2: Extract audio ──────────────────────────────────────────────────
    audio_path = "match_audio.wav"
    extract_audio(args.video, audio_path)

    # ── Step 3: Transcribe ─────────────────────────────────────────────────────
    segments = transcribe_audio(audio_path, args.model)

    if args.save_transcript:
        save_transcript(segments, "transcript.txt")

    if not args.keep_audio:
        os.remove(audio_path)

    # ── Step 4: Find keyword hits ──────────────────────────────────────────────
    hits = find_keyword_hits(segments, KEYWORDS)
    if not hits:
        print("\n  No highlight keywords found. Try adding more keywords or using a larger Whisper model.")
        return

    # ── Step 5: Map each hit to a delivery and cut clips ──────────────────────
    os.makedirs(args.output_dir, exist_ok=True)
    print(f"\n  Cutting highlight clips → ./{args.output_dir}/\n")

    seen_deliveries = set()   # avoid duplicate clips for same delivery
    clips_saved = []
    summary_lines = []

    for keyword_time, keyword, segment_text in hits:
        delivery_num, clip_start, clip_end = find_delivery_for_time(keyword_time, deliveries)

        if delivery_num in seen_deliveries:
            continue   # already saved this delivery
        seen_deliveries.add(delivery_num)

        duration = clip_end - clip_start
        safe_keyword = re.sub(r"[^a-z0-9_]", "_", keyword.replace(" ", "_"))
        filename = f"highlight_d{delivery_num:04d}_{hms(clip_start).replace(':', '-')}_{safe_keyword}.mp4"
        out_path = os.path.join(args.output_dir, filename)

        print(f"  ✓ Clip {len(clips_saved)+1:>3}  Delivery {delivery_num:>4}  "
              f"{hms(clip_start)} → {hms(clip_end)}  ({duration:.0f}s)  "
              f"keyword='{keyword}'")
        print(f"        Commentary: \"{segment_text[:80]}\"")

        cut_clip(args.video, clip_start, clip_end, out_path)
        clips_saved.append(out_path)
        summary_lines.append(
            f"{len(clips_saved)}\t{delivery_num}\t{hms(clip_start)}\t{hms(clip_end)}"
            f"\t{keyword}\t{segment_text[:100]}\t{filename}"
        )

    # ── Step 6: Save summary ──────────────────────────────────────────────────
    summary_path = os.path.join(args.output_dir, "highlights_summary.txt")
    with open(summary_path, "w", encoding="utf-8") as f:
        f.write("# Highlights Summary\n")
        f.write("# clip\tdelivery\tstart\tend\tkeyword\tcommentary\tfile\n\n")
        for line in summary_lines:
            f.write(line + "\n")

    print(f"\n{'='*57}")
    print(f"  Done!  {len(clips_saved)} highlight clips saved to ./{args.output_dir}/")
    print(f"  Summary: {summary_path}")
    print(f"{'='*57}\n")


if __name__ == "__main__":
    extract()
