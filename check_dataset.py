import json

files = [
    "segments.json",
    "segments_rms.json",
    "segments_ocr.json",
    "deliveries.json",
    "highlights.json",
    "delivery_highlights.json",
    "delivery_features.json"
]

for fname in files:
    try:
        with open(fname) as f:
            data = json.load(f)
            
        print(f"--- {fname} ---")
        print(f"Length: {len(data)}")
        
        if len(data) > 0:
            first = data[0]
            last = data[-1]
            
            # Find timestamp fields
            ts_key = next((k for k in ["start", "timestamp", "delivery_timestamp", "clip_start"] if k in first), None)
            
            if ts_key:
                print(f"First ts: {first[ts_key]:.2f} (approx {first[ts_key]/3600:.2f}h)")
                print(f"Last ts:  {last[ts_key]:.2f} (approx {last[ts_key]/3600:.2f}h)")
                
                # if there is an innings key in scoreboard, check it
                # try to get the 'scoreboard' key from elements
                first_innings = []
                second_innings = []
                for item in data:
                    sb = item.get("scoreboard")
                    if sb:
                        innings = sb.get("innings")
                        if innings == 1:
                            first_innings.append(item)
                        elif innings == 2:
                            second_innings.append(item)
                
                if first_innings or second_innings:
                    print(f"Has scoreboard data. Innings 1 items: {len(first_innings)}, Innings 2 items: {len(second_innings)}")
            else:
                print("No standard timestamp key found.")
    except Exception as e:
        print(f"Error {fname}: {e}")
    print()
