"""
delivery_detection.py
─────────────────────
Scans the OCR scoreboard data (segments_ocr.json) to find the timestamp of
every delivery by detecting when the scoreboard state changes.

Strategy
────────
A delivery "happened" whenever one of these changes between consecutive
valid scoreboard reads:
  • runs increase     → run(s) scored  (boundary, single, etc.)
  • wickets increase  → wicket fell
  • overs increase    → over completed (catches dot balls at end-of-over too)
    
Because OCR is noisy, we use a simple debounce: a new state has to appear
in at least MIN_CONFIRM consecutive segments before we accept it as real.
This kills one-frame glitches.

Output: deliveries.json — list of delivery dicts, e.g.
  {
    "delivery_id": 42,
    "timestamp":   1823.0,      ← seconds from video start
    "hms":         "00:30:23",
    "event":       "runs",      ← "runs" | "wicket" | "over" | "multi"
    "before":      {"runs": 88, "wickets": 2, "overs": 14},
    "after":       {"runs": 92, "wickets": 2, "overs": 14}
  }
"""

import json
import os
from dataclasses import dataclass, field, asdict
from typing import Optional

# ── tuneable params ───────────────────────────────────────────────────────────
OCR_JSON    = "segments_ocr.json"
OUT_JSON    = "deliveries.json"
MIN_CONFIRM = 2   # how many consecutive segments must agree on new state
# ─────────────────────────────────────────────────────────────────────────────


@dataclass
class ScoreState:
    runs:    int
    wickets: int
    overs:   int

    def __eq__(self, other):
        if other is None:
            return False
        return (self.runs == other.runs and
                self.wickets == other.wickets and
                self.overs == other.overs)

    def to_dict(self):
        return asdict(self)


def seg_to_state(seg: dict) -> Optional[ScoreState]:
    """Return a ScoreState from a segment dict, or None if OCR failed."""
    sb = seg.get("scoreboard")
    if not sb:
        return None
    runs    = sb.get("runs")
    wickets = sb.get("wickets")
    overs   = sb.get("overs")
    if runs is None or wickets is None:
        return None
    # overs may be absent on some frames — default 0
    return ScoreState(runs=runs, wickets=wickets, overs=overs or 0)


def classify_event(before: ScoreState, after: ScoreState) -> str:
    """Label what kind of delivery event occurred."""
    tags = []
    if after.wickets > before.wickets:
        tags.append("wicket")
    if after.runs > before.runs:
        tags.append("runs")
    if after.overs > before.overs:
        tags.append("over")
    if not tags:
        return "unknown"
    return tags[0] if len(tags) == 1 else "multi"


def fmt_hms(seconds: float) -> str:
    s = int(seconds)
    return f"{s//3600:02d}:{(s%3600)//60:02d}:{s%60:02d}"


def detect_deliveries(ocr_json=OCR_JSON, out_json=OUT_JSON, min_confirm=MIN_CONFIRM):
    # ── load data ─────────────────────────────────────────────────────────────
    if not os.path.exists(ocr_json):
        raise FileNotFoundError(f"{ocr_json} not found — run scoreboard.py first")

    with open(ocr_json) as f:
        segments = json.load(f)

    print(f"Loaded {len(segments):,} segments from {ocr_json}")

    # ── debounced state machine ───────────────────────────────────────────────
    deliveries = []
    confirmed_state: Optional[ScoreState] = None   # last "locked in" state
    candidate_state: Optional[ScoreState] = None   # state we're evaluating
    candidate_count  = 0
    candidate_start_ts = 0.0   # timestamp of first segment showing candidate

    for seg in segments:
        state = seg_to_state(seg)
        if state is None:
            continue  # skip unreadable frames

        if confirmed_state is None:
            # bootstrap: accept first valid read immediately
            confirmed_state = state
            candidate_state = None
            candidate_count = 0
            continue

        if state == confirmed_state:
            # steady state — reset any pending candidate
            candidate_state = None
            candidate_count = 0
            continue

        # state differs from confirmed
        if state == candidate_state:
            candidate_count += 1
        else:
            # new candidate
            candidate_state    = state
            candidate_count    = 1
            candidate_start_ts = seg["start"]

        if candidate_count >= min_confirm:
            # ── delivery confirmed ────────────────────────────────────────
            event = classify_event(confirmed_state, candidate_state)

            deliveries.append({
                "delivery_id": len(deliveries) + 1,
                "timestamp":   candidate_start_ts,
                "hms":         fmt_hms(candidate_start_ts),
                "event":       event,
                "before":      confirmed_state.to_dict(),
                "after":       candidate_state.to_dict(),
            })

            confirmed_state = candidate_state
            candidate_state = None
            candidate_count = 0

    # ── save ──────────────────────────────────────────────────────────────────
    with open(out_json, "w") as f:
        json.dump(deliveries, f, indent=2)

    print(f"\nDetected {len(deliveries)} deliveries → saved to {out_json}")

    # ── quick summary ─────────────────────────────────────────────────────────
    from collections import Counter
    event_counts = Counter(d["event"] for d in deliveries)
    print("\nEvent breakdown:")
    for event, count in sorted(event_counts.items(), key=lambda x: -x[1]):
        print(f"  {event:<10} {count}")

    if deliveries:
        print(f"\nFirst delivery: {deliveries[0]['hms']}  ({deliveries[0]['event']})")
        print(f"Last  delivery: {deliveries[-1]['hms']} ({deliveries[-1]['event']})")

    return deliveries


if __name__ == "__main__":
    detect_deliveries()
