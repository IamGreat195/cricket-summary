import json
import os
import numpy as np
from collections import Counter

RMS_PERCENTILE = 80
CONTEXT_SECS = 30
CHASE_RUNS_MARGIN = 40
CHASE_OVERS_MIN = 46.0
TOTAL_OVERS = 50

def fmt_hms(seconds):
    s = int(seconds)
    return f"{s//3600:02d}:{(s%3600)//60:02d}:{s%60:02d}"

def detect_events(segments):
    last_valid_sb = None
    last_valid_t = None
    first_innings_score = None

    for seg in segments:
        seg["events"] = []
        sb = seg.get("scoreboard")
        if sb is None:
            continue
        
        runs = sb.get("runs", 0)
        wickets = sb.get("wickets", 0)
        overs = sb.get("overs", 0.0)
        innings = sb.get("innings", 1)
        t = seg["start"]

        if innings == 1:
            first_innings_score = runs
        
        if innings == 2 and not sb.get("target") and first_innings_score:
            sb["target"] = first_innings_score + 1
        target = sb.get("target")

        if last_valid_sb is not None and last_valid_t is not None:
            time_gap = t - last_valid_t
            runs_delta = runs - last_valid_sb.get("runs", 0)
            wkt_delta = wickets - last_valid_sb.get("wickets", 0)
            same_innings = (innings == last_valid_sb.get("innings", 1))

            if same_innings and time_gap < 60:
                if 1 <= runs_delta <= 7:
                    if runs_delta >= 6:
                        seg["events"].append("six")
                    elif runs_delta >= 4:
                        seg["events"].append("four")
                
                elif runs_delta > 7:
                    seg["events"].append("boundary_approx")
                if 0 < wkt_delta <= 2:
                    seg["events"].append("wicket")
            
            if innings == 2 and target is not None and overs > 0:
                runs_needed = target - runs
                if 0 < runs_needed <= CHASE_RUNS_MARGIN and overs >= CHASE_OVERS_MIN:
                    seg["events"].append("chase_climax")
            
            for b in sb.get("batsmen", []):
                r = b.get("runs", 0)
                if 48 <= r <= 52:
                    seg["events"].append("milestone_50")
                elif 96 <= r <= 102:
                    seg["events"].append("milestone_100")
        last_valid_sb = sb
        last_valid_t = t
    return segments


def build_highlights(segments):
    all_rms = np.array([s.get("rms_norm", 0) for s in segments])
    rms_threshold = float(np.percentile(all_rms, RMS_PERCENTILE))
    print(f"RMS threshold (p{RMS_PERCENTILE}): {rms_threshold:.4f}")
    event_times = np.array([s["start"] for s in segments if s.get("events")])
    print(f"event tagged segments: {len(event_times)}")
    
    for seg in segments:
        t = seg["start"]
        rms = seg.get("rms_norm", 0)

        if rms >= rms_threshold:
            seg["is_highlight"] = True
            seg["highlight_reason"] = "rms"
            continue

        if seg.get("events"):
            seg["is_highlight"] = True
            seg["highlight_reason"] = "event"
            continue

        if len(event_times) > 0:
            nearest = float(np.min(np.abs(event_times - t)))
            if nearest <= CONTEXT_SECS:
                seg["is_highlight"] = True
                seg["highlight_reason"] = f"context_{int(nearest)}s"
                continue
        seg["is_highlight"] = False
        seg["highlight_reason"] = None
    return segments


if __name__ == "__main__":
    with open("data/segments_ocr.json") as f:
        segments = json.load(f)
    print(f"Loaded {len(segments):,} segments")
    segments = detect_events(segments)
    segments = build_highlights(segments)
    highlights = [s for s in segments if s["is_highlight"]]
    pct = len(highlights) / len(segments) * 100
    print(f"\nHighlights: {len(highlights):,} / {len(segments):,} ({pct:.1f}%)")
    all_events = []
    for s in segments:
        all_events.extend(s.get("events", []))
    print("\nEvent breakdown:")
    for ev, cnt in Counter(all_events).most_common():
        print(f"  {ev:<20} {cnt}")

    reasons = Counter(s["highlight_reason"] for s in highlights)
    print("\nHighlight reason breakdown:")
    for r, cnt in reasons.most_common():
        print(f"  {str(r):<30} {cnt}")
    print(f"\n{'─'*72}")
    print(f"{'Time':<12} {'Events':<25} {'Score':<20} {'RMS':>5}")
    print(f"{'─'*72}")
    sample = [h for h in highlights if h.get("events")][:20]
    for h in sample:
        sb    = h.get("scoreboard") or {}
        score = (f"{sb.get('runs','?')}-{sb.get('wickets','?')} "
                 f"({sb.get('overs','?')}ov)")
        rms   = f"{h.get('rms_norm', 0):.3f}"
        evts  = ", ".join(h["events"][:2])
        print(f"  {fmt_hms(h['start']):<10} {evts:<25} {score:<20} {rms:>5}")

    out = [{
        "id":               h["id"],
        "start":            h["start"],
        "end":              h["end"],
        "hms":              fmt_hms(h["start"]),
        "events":           h.get("events", []),
        "highlight_reason": h.get("highlight_reason"),
        "rms_norm":         h.get("rms_norm", 0),
        "scoreboard":       h.get("scoreboard"),
    } for h in highlights]

    os.makedirs("data", exist_ok=True)
    with open("data/highlights.json", "w") as f:
        json.dump(out, f, indent=2)
    print(f"\nSaved {len(out):,} highlights → data/highlights.json")
