import json

def fmt_hms(seconds):
    s = int(seconds)
    return f"{s//3600:02d}:{(s%3600)//60:02d}:{s%60:02d}"

def map_deliveries_to_highlights(deliveries_path="data/deliveries.json", highlights_path="data/highlights.json", out_path="data/delivery_highlights.json"):
    with open(deliveries_path) as f:
        deliveries = json.load(f)
        
    with open(highlights_path) as f:
        highlights = json.load(f)

    deliveries.sort(key = lambda x: x["timestamp"])
    delivery_highlights = []
    for i, delivery in enumerate(deliveries):
        t_start = delivery["timestamp"]
        t_end = deliveries[i+1]["timestamp"] if i + 1 < len(deliveries) else t_start + 45.0
        if t_end - t_start > 60:
            t_end = t_start + 60
        matching_hl = [h for h in highlights if t_start <= h["start"] < t_end]
        if matching_hl:
            events = set()
            for h in matching_hl:   
                events.update(h.get("events", []))
            reasons = set()
            for h in matching_hl:
                r = h.get("highlight_reason")
                if r:
                    if str(r).startswith("context"):
                        reasons.add("context")
                    else:
                        reasons.add(str(r))
            max_rms = max((h.get("rms_norm", 0) for h in matching_hl), default=0)

            scoreboards = [h.get("scoreboard") for h in matching_hl if h.get("scoreboard")]
            final_sb = scoreboards[-1] if scoreboards else None
            clip_start = max(0, t_start - 5)
            clip_end = max(t_start + 15, matching_hl[-1]["end"] + 2)

            dh = {
                "delivery_id": delivery["delivery_id"],
                "delivery_timestamp": delivery["timestamp"],
                "delivery_hms": delivery["hms"],
                "clip_start": clip_start,
                "clip_end": clip_end,
                "clip_start_hms": fmt_hms(clip_start),
                "clip_end_hms": fmt_hms(clip_end),
                "events": list(events),
                "reasons": list(reasons),
                "max_rms": round(max_rms, 3),
                "scoreboard": final_sb
            }
            delivery_highlights.append(dh)
    with open(out_path, "w") as f:
        json.dump(delivery_highlights, f, indent=2)
        
    print(f"Mapped {len(deliveries)} deliveries to {len(delivery_highlights)} highlight deliveries.")
    print(f"Saved to {out_path}\n")
    # Print a summary
    print(f"{'ID':>4} | {'Delivery':<8} | {'Clip window':<19} | {'Events':<30} | {'RMS':>5} | {'Reasons'}")
    print("-" * 90)
    for dh in delivery_highlights[:30]:
        ev_str = ", ".join(dh["events"])
        rs_str = ", ".join(dh["reasons"])
        print(f"{dh['delivery_id']:>4} | {dh['delivery_hms']:<8} | {dh['clip_start_hms']} - {dh['clip_end_hms']} | {ev_str:<30} | {dh['max_rms']:.2f} | {rs_str}")

if __name__ == "__main__":
    map_deliveries_to_highlights()

