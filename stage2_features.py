if __name__ == "__main__":

    VIDEO_PATH = "match1_h264.mp4"

    with open("data/delivery_highlights.json") as f:
        deliveries = json.load(f)

    # load whisper transcript if available
    whisper_segments = []
    if os.path.exists("data/transcript.json"):
        with open("data/transcript.json") as f:
            whisper_segments = json.load(f)
        print(f"Loaded {len(whisper_segments)} whisper segments")
    else:
        print("No data/transcript.json found — commentary score will be 0")
        print("Run: whisper match1_audio.wav --model small "
              "--word_timestamps True --output_format json")

    results = []

    for delivery in tqdm(deliveries, desc="Extracting features"):
        feat = extract_all_features(VIDEO_PATH, delivery, whisper_segments)

        results.append({
            "delivery_id":        delivery["delivery_id"],
            "delivery_timestamp": delivery["delivery_timestamp"],
            "hms":                delivery["delivery_hms"],
            "clip_start":         delivery["clip_start"],
            "clip_end":           delivery["clip_end"],
            "primary_event":      delivery.get("primary_event", "unknown"),
            "events":             delivery.get("events", []),
            "scoreboard":         delivery.get("scoreboard"),
            "feature_vector":     feat.tolist(),  # save as list for JSON
            "feature_dim":        len(feat),
        })

    os.makedirs("data", exist_ok=True)
    with open("data/delivery_features.json", "w") as f:
        json.dump(results, f, indent=2)

    print(f"\nSaved {len(results)} feature vectors → data/delivery_features.json")
    print(f"Feature vector dimension: {results[0]['feature_dim']}")