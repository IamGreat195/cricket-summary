#!/usr/bin/env python3
"""
Generate a comprehensive PDF report explaining the Cricket Highlight Pipeline.
Uses fpdf2 -- no external system dependencies needed.
"""

from fpdf import FPDF

class PipelineReport(FPDF):
    MARGIN = 15
    BODY_SIZE = 10.5
    H1_SIZE = 20
    H2_SIZE = 15
    H3_SIZE = 12.5
    CODE_SIZE = 8.5
    
    def __init__(self):
        super().__init__()
        self.set_auto_page_break(auto=True, margin=20)
        self.set_margins(self.MARGIN, self.MARGIN, self.MARGIN)
        # Core font (built-in, no TTF needed)
        self.add_page()

    # ── helpers ──────────────────────────────────────────────────────────────
    def _heading(self, text, level=1):
        sizes = {1: self.H1_SIZE, 2: self.H2_SIZE, 3: self.H3_SIZE}
        sz = sizes.get(level, self.BODY_SIZE)
        self.ln(4 if level > 1 else 8)
        self.set_font("Helvetica", "B", sz)
        self.set_text_color(20, 60, 120)
        self.multi_cell(0, sz * 0.55, text)
        self.set_text_color(0, 0, 0)
        if level == 1:
            self.set_draw_color(20, 60, 120)
            self.set_line_width(0.6)
            self.line(self.MARGIN, self.get_y(), self.w - self.MARGIN, self.get_y())
            self.ln(3)
        self.ln(2)

    def _body(self, text):
        self.set_font("Helvetica", "", self.BODY_SIZE)
        self.multi_cell(0, 5.5, text)
        self.ln(1)

    def _bold_body(self, text):
        self.set_font("Helvetica", "B", self.BODY_SIZE)
        self.multi_cell(0, 5.5, text)
        self.set_font("Helvetica", "", self.BODY_SIZE)
        self.ln(1)

    def _bullet(self, text):
        self.set_font("Helvetica", "", self.BODY_SIZE)
        indent = 8
        self.set_x(self.MARGIN + indent)
        self.multi_cell(self.w - self.MARGIN * 2 - indent, 5.5, "- " + text)
        self.ln(0.5)

    def _code_block(self, text):
        self.set_font("Courier", "", self.CODE_SIZE)
        self.set_fill_color(240, 240, 240)
        self.multi_cell(0, 4.5, text, fill=True)
        self.set_font("Helvetica", "", self.BODY_SIZE)
        self.ln(2)

    def _table_row(self, cols, widths, bold=False):
        self.set_font("Helvetica", "B" if bold else "", 9)
        h = 6
        for i, (col, w) in enumerate(zip(cols, widths)):
            self.cell(w, h, str(col)[:int(w/1.8)], border=1)
        self.ln(h)

    # ── title page ──────────────────────────────────────────────────────────
    def title_page(self):
        self.ln(50)
        self.set_font("Helvetica", "B", 28)
        self.set_text_color(20, 60, 120)
        self.cell(0, 15, "Cricket Highlight Generation", align="C", new_x="LMARGIN", new_y="NEXT")
        self.cell(0, 15, "Pipeline", align="C", new_x="LMARGIN", new_y="NEXT")
        self.ln(8)
        self.set_font("Helvetica", "", 14)
        self.set_text_color(80, 80, 80)
        self.cell(0, 10, "A Comprehensive Technical Report", align="C", new_x="LMARGIN", new_y="NEXT")
        self.ln(20)
        self.set_draw_color(20, 60, 120)
        self.set_line_width(1)
        x_center = self.w / 2
        self.line(x_center - 40, self.get_y(), x_center + 40, self.get_y())
        self.ln(10)
        self.set_font("Helvetica", "", 11)
        self.set_text_color(100, 100, 100)
        self.cell(0, 8, "Multimodal Machine Learning  |  Computer Vision  |  NLP", align="C", new_x="LMARGIN", new_y="NEXT")
        self.cell(0, 8, "VideoMAE  |  XGBoost  |  PaddleOCR  |  OpenAI Whisper", align="C", new_x="LMARGIN", new_y="NEXT")
        self.ln(30)
        self.set_font("Helvetica", "I", 10)
        self.cell(0, 8, "Author: Premesh", align="C", new_x="LMARGIN", new_y="NEXT")
        self.cell(0, 8, "June 2026", align="C", new_x="LMARGIN", new_y="NEXT")
        self.set_text_color(0, 0, 0)

    # ── content sections ────────────────────────────────────────────────────
    def section_overview(self):
        self.add_page()
        self._heading("1. Project Overview")
        self._body(
            "This project implements an end-to-end automated pipeline that takes a full-length "
            "cricket match video (T20I or ODI) as input and produces a condensed highlight reel "
            "containing only the most exciting moments -- sixes, wickets, milestones, chase climaxes, "
            "and other high-impact deliveries."
        )
        self._body(
            "The system is entirely automated: no manual annotation is required. It uses a combination "
            "of computer vision (green-field detection, optical flow), audio analysis (RMS energy), "
            "optical character recognition (PaddleOCR for scoreboard reading), speech-to-text "
            "(OpenAI Whisper for commentary transcription), and deep learning (VideoMAE for visual "
            "understanding, SentenceTransformers for commentary embeddings) to build a rich multimodal "
            "feature vector for every detected delivery. An XGBoost classifier then predicts whether "
            "each delivery is a highlight-worthy event, and FFmpeg stitches the selected clips into "
            "the final video."
        )

        self._heading("1.1 Technology Stack", level=2)
        techs = [
            ("Language", "Python 3.12"),
            ("Video Processing", "OpenCV, FFmpeg"),
            ("OCR", "PaddleOCR (GPU-accelerated)"),
            ("Speech-to-Text", "OpenAI Whisper (small model)"),
            ("Visual Features", "VideoMAE (MCG-NJU/videomae-base)"),
            ("Text Embeddings", "SentenceTransformer (all-MiniLM-L6-v2)"),
            ("Classifier", "XGBoost (multi-class softmax)"),
            ("Dimensionality Reduction", "Scikit-learn PCA"),
            ("Deep Learning", "PyTorch + CUDA"),
            ("Orchestration", "Custom Python pipeline runner"),
        ]
        ws = [55, 120]
        self._table_row(["Component", "Technology"], ws, bold=True)
        for k, v in techs:
            self._table_row([k, v], ws)

    def section_architecture(self):
        self.add_page()
        self._heading("2. Pipeline Architecture")
        self._body(
            "The pipeline is orchestrated by run_pipeline.py, which sequentially executes nine "
            "stages. Each stage reads its inputs from a per-match data directory and writes its "
            "outputs there, enabling checkpoint-based resumption if any stage fails."
        )
        self._body("The nine stages are executed in the following order:")

        stages = [
            ("1", "Delivery Detection (CV)", "delivery_detection.py", "deliveries.json"),
            ("2", "Audio RMS & Segment Building", "cricket_summary.py", "segments_rms.json"),
            ("3", "Scoreboard OCR", "scoreboard.py", "segments_ocr.json"),
            ("4", "First Gate - Event Tagging", "first_gate.py", "highlights.json"),
            ("5", "Map Deliveries to Highlights", "map_deliveries.py", "delivery_highlights.json"),
            ("6", "Whisper Commentary Transcription", "transcribe_deliveries.py", "transcript.json"),
            ("7", "Feature Extraction", "stage2_features.py", "delivery_features.json"),
            ("8", "Train XGBoost Classifier", "train_classifier.py", "highlight_pipeline.pkl"),
            ("9", "Generate Final Video", "generate_final_video.py", "final_highlights.mp4"),
        ]
        ws = [12, 58, 50, 55]
        self._table_row(["#", "Stage Name", "Script", "Output Artifact"], ws, bold=True)
        for row in stages:
            self._table_row(row, ws)
        self.ln(3)
        self._body(
            "Each step can be individually skipped if its output artifact already exists, "
            "allowing incremental re-runs. Environment variables VIDEO_PATH, AUDIO_PATH, "
            "DATA_DIR, and OUTPUT_PATH are passed to every sub-process."
        )

    def section_delivery_detection(self):
        self.add_page()
        self._heading("3. Stage 1: Delivery Detection")
        self._heading("3.1 Purpose", level=2)
        self._body(
            "This stage scans the entire match video to automatically detect the approximate "
            "timestamp of every ball bowled. It is the foundation of the pipeline -- all subsequent "
            "stages operate on the delivery windows identified here."
        )

        self._heading("3.2 Algorithm", level=2)
        self._body("The detection uses a two-filter heuristic on downscaled frames (160x90):")

        self._heading("Filter 1: Green-Field Ratio (Spatial)", level=3)
        self._body(
            "Each sampled frame is converted to HSV colour space. A mask isolates pixels in the "
            "green range (H: 25-95, S: 20-255, V: 20-255). The frame is divided into three "
            "vertical zones -- left side, centre, and right side. A delivery camera angle is "
            "characterised by:\n"
            "  - Overall green ratio > 30%\n"
            "  - Left and right side green > 60% (outfield grass visible)\n"
            "  - Centre green < 85% (the pitch strip is brown/grey, not all green)\n\n"
            "This rejects replays, crowd shots, graphics overlays, and close-ups."
        )

        self._heading("Filter 2: Motion Score (Temporal)", level=3)
        self._body(
            "For frames that pass the green filter, a motion score is computed as the mean "
            "absolute pixel difference between the current and previous greyscale frame, "
            "normalised to [0, 1]. If motion >= 0.015, the frame is considered a live delivery "
            "(the bowler is running in or the ball is in play). A minimum gap of 18 seconds "
            "between consecutive detections prevents double-counting the same delivery."
        )

        self._heading("3.3 Output", level=2)
        self._body(
            "deliveries.json: An array of objects, each with delivery_id, timestamp (seconds), "
            "hms (human-readable time), motion_score, and green_ratio. Typical counts: ~120 "
            "deliveries for a T20, ~300 for an ODI."
        )

    def section_audio_rms(self):
        self.add_page()
        self._heading("4. Stage 2: Audio RMS & Segment Building")
        self._heading("4.1 Purpose", level=2)
        self._body(
            "This stage extracts the audio track from the video (if not already done) and divides "
            "the entire match into overlapping 2-second segments. For each segment, it computes "
            "the RMS (Root Mean Square) energy of the audio waveform, providing a proxy for crowd "
            "noise intensity -- an excellent indicator of exciting moments."
        )

        self._heading("4.2 How It Works", level=2)
        self._bullet(
            "Audio Extraction: FFmpeg extracts a mono, 22050 Hz WAV file from the video."
        )
        self._bullet(
            "Segmentation: The match is divided into 2-second windows with a 1-second hop "
            "(50% overlap), producing ~2x the number of seconds as segments."
        )
        self._bullet(
            "RMS Calculation: For each segment, 16-bit PCM samples are normalised to [-1, 1], "
            "and RMS = sqrt(mean(samples^2)) is computed."
        )
        self._bullet(
            "Normalisation: All RMS values are divided by the maximum RMS across all segments "
            "to produce rms_norm in [0, 1]."
        )
        self._bullet(
            "Smoothing: A uniform 1D filter with window size 5 smooths the normalised RMS "
            "values, reducing transient spikes."
        )

        self._heading("4.3 Output", level=2)
        self._body(
            "segments_rms.json: An array of segment objects with id, start, end, rms, "
            "rms_norm, and rms_smooth fields."
        )

    def section_scoreboard_ocr(self):
        self.add_page()
        self._heading("5. Stage 3: Scoreboard OCR")
        self._heading("5.1 Purpose", level=2)
        self._body(
            "This stage reads the on-screen scoreboard overlay from each segment's mid-frame "
            "using PaddleOCR (GPU-accelerated). It extracts structured match data: runs, "
            "wickets, overs, innings, run rate, required run rate, target, batsman stats, "
            "and bowler figures. This is the longest-running stage (several hours for a full match)."
        )

        self._heading("5.2 How It Works", level=2)
        self._bullet(
            "Frame Cropping: The bottom 15% of each frame is extracted -- this is where "
            "broadcast scoreboards are typically overlaid."
        )
        self._bullet(
            "Preprocessing: The cropped region is upscaled 2x with cubic interpolation "
            "to improve OCR accuracy on small text."
        )
        self._bullet(
            "OCR: PaddleOCR runs on the preprocessed region to extract raw text."
        )
        self._bullet(
            "Regex Parsing: A custom parser (parse_ecb_scoreboard) applies multiple regular "
            "expressions to extract:\n"
            "  - Score: pattern 'NNN-W' for runs and wickets\n"
            "  - Overs: pattern 'NN.N/50'\n"
            "  - Innings: pattern 'P1' or 'P2'\n"
            "  - Run rate, required run rate, and target\n"
            "  - Batsman names, runs, and balls faced\n"
            "  - Bowler name, wickets, runs conceded, and overs bowled"
        )
        self._bullet(
            "Checkpointing: Results are saved every 500 segments, allowing resumption "
            "if the process is interrupted."
        )

        self._heading("5.3 Output", level=2)
        self._body(
            "segments_ocr.json: The segments array enriched with scoreboard (parsed dict) "
            "and ocr_raw (raw OCR text) fields."
        )

    def section_first_gate(self):
        self.add_page()
        self._heading("6. Stage 4: First Gate - Event Tagging")
        self._heading("6.1 Purpose", level=2)
        self._body(
            "This stage performs rule-based event detection and highlight filtering. It combines "
            "scoreboard deltas (detecting fours, sixes, wickets from score changes), audio energy "
            "(RMS thresholding), and contextual proximity to produce a filtered set of highlight segments."
        )

        self._heading("6.2 Event Detection Logic", level=2)
        self._body("For each pair of consecutive valid scoreboard readings within the same innings:")
        self._bullet("Runs delta of 6+ within 60s -> tagged as 'six'")
        self._bullet("Runs delta of 4-5 within 60s -> tagged as 'four'")
        self._bullet("Wickets delta of 1-2 -> tagged as 'wicket'")
        self._bullet("Runs delta > 7 -> tagged as 'boundary_approx'")
        self._bullet(
            "Chase Climax: In the 2nd innings, if runs_needed <= 40 and overs >= 46, "
            "tagged as 'chase_climax'."
        )
        self._bullet(
            "Milestones: Batsman runs in [48-52] -> 'milestone_50'; in [96-102] -> 'milestone_100'."
        )

        self._heading("6.3 Highlight Selection", level=2)
        self._body("A segment becomes a highlight if any of these conditions is true:")
        self._bullet(
            "RMS Gate: Its normalised RMS energy exceeds the 80th percentile threshold "
            "(crowd is loud)."
        )
        self._bullet("Event Gate: It has at least one tagged event (four, six, wicket, etc.).")
        self._bullet(
            "Context Gate: It is within 30 seconds of any event-tagged segment "
            "(captures build-up and reactions)."
        )

        self._heading("6.4 Output", level=2)
        self._body(
            "highlights.json: An array of highlight segments with id, start, end, hms, events, "
            "highlight_reason, rms_norm, and scoreboard."
        )

    def section_map_deliveries(self):
        self.add_page()
        self._heading("7. Stage 5: Map Deliveries to Highlight Windows")
        self._heading("7.1 Purpose", level=2)
        self._body(
            "This stage bridges the delivery-level timeline (from Stage 1) with the segment-level "
            "highlight decisions (from Stage 4). For each delivery, it collects all overlapping "
            "highlight segments and aggregates their events, reasons, scoreboard data, and RMS "
            "values into a single delivery-highlight record with clip boundaries."
        )

        self._heading("7.2 How It Works", level=2)
        self._bullet(
            "For each delivery at timestamp T, the window is [T, T_next] where T_next is "
            "the next delivery's timestamp (capped at T + 60s maximum)."
        )
        self._bullet(
            "All highlight segments whose start time falls within this window are collected."
        )
        self._bullet(
            "Events and reasons are unioned. The latest valid scoreboard is taken."
        )
        self._bullet(
            "Clip boundaries are set as: clip_start = T - 5 seconds, "
            "clip_end = max(T + 15, last_segment_end + 2)."
        )

        self._heading("7.3 Output", level=2)
        self._body(
            "delivery_highlights.json: An array of delivery-highlight objects with delivery_id, "
            "delivery_timestamp, clip_start/end, events, reasons, max_rms, and scoreboard."
        )

    def section_transcription(self):
        self._heading("8. Stage 6: Whisper Commentary Transcription")
        self._heading("8.1 Purpose", level=2)
        self._body(
            "This stage transcribes the match commentary using OpenAI's Whisper speech-to-text "
            "model. Rather than processing each delivery clip individually (which incurs heavy "
            "Python overhead), it transcribes the entire audio file in a single pass and then "
            "assigns segments to delivery windows by temporal overlap."
        )

        self._heading("8.2 How It Works", level=2)
        self._bullet(
            "Model Loading: Whisper 'small' model is loaded on GPU (CUDA) if available."
        )
        self._bullet(
            "Full-Audio Transcription: The entire match audio WAV is transcribed in one pass. "
            "This typically takes 3-5 minutes on GPU vs. hours if done clip-by-clip."
        )
        self._bullet(
            "Segment Assignment: Each Whisper segment is checked against delivery windows "
            "(clip_start - 2s to clip_end + 2s). If a segment overlaps any window, it's "
            "assigned to that delivery."
        )

        self._heading("8.3 Output", level=2)
        self._body(
            "transcript.json: An array of segment objects with start, end, text, and "
            "delivery_id. These are used by Stage 7 for sentence embeddings."
        )

    def section_feature_extraction(self):
        self.add_page()
        self._heading("9. Stage 7: Multimodal Feature Extraction")
        self._heading("9.1 Purpose", level=2)
        self._body(
            "This is the core machine learning stage. For each delivery, it constructs a "
            "1165-dimensional feature vector by extracting and concatenating five different "
            "modalities: video, text, optical flow, scoreboard, and audio."
        )

        self._heading("9.2 Feature Components", level=2)

        features = [
            ("Video (VideoMAE)", "768", "MCG-NJU/videomae-base",
             "16 frames uniformly sampled at 2 fps are passed through VideoMAE. "
             "The last hidden state is mean-pooled across the sequence dimension."),
            ("Text (Sentence)", "384", "all-MiniLM-L6-v2",
             "Commentary text within a +/-5 second window around the delivery is "
             "encoded into a dense embedding using SentenceTransformer."),
            ("Optical Flow", "3", "OpenCV Farneback",
             "Dense optical flow is computed between consecutive 2-fps frames "
             "downscaled to 320x180. Features: mean, max, and std of flow magnitudes."),
            ("Scoreboard", "8", "Regex + PaddleOCR",
             "Normalised scoreboard state: runs/300, wickets/10, overs/50, innings-1, "
             "crr/12, rrr/12, runs_needed/300, balls_remaining/300."),
            ("Audio", "2", "scipy wavfile",
             "RMS energy and absolute max amplitude of the audio chunk, each divided "
             "by 32768 for normalisation."),
        ]

        for name, dim, source, desc in features:
            self._heading(f"{name} - {dim} dimensions", level=3)
            self._bold_body(f"Source: {source}")
            self._body(desc)

        self._heading("9.3 Total Feature Vector", level=2)
        self._body(
            "The five components are concatenated in order: "
            "[768 Video | 384 Text | 3 Flow | 8 Scoreboard | 2 Audio] = 1165 dimensions."
        )

        self._heading("9.4 Output", level=2)
        self._body(
            "delivery_features.json: Each delivery object now includes a feature_vector "
            "(1165-element list), along with metadata (delivery_id, clip times, events, scoreboard)."
        )

    def section_training(self):
        self.add_page()
        self._heading("10. Stage 8: Train XGBoost Classifier")
        self._heading("10.1 Purpose", level=2)
        self._body(
            "This stage trains a multiclass classifier to predict the type of cricket event "
            "for each delivery based on its 1165-dimensional feature vector. The model learns "
            "to distinguish between: none, four, six, wicket, run-out, milestone, and win."
        )

        self._heading("10.2 Dimensionality Reduction (PCA)", level=2)
        self._body(
            "Before training, the high-dimensional video and text features are compressed "
            "using Principal Component Analysis (PCA) to reduce overfitting and training time:"
        )
        self._bullet("Video features: 768 dims -> 32 principal components")
        self._bullet("Text features: 384 dims -> 16 principal components")
        self._bullet("Core features (flow + scoreboard + audio): 13 dims (kept as-is)")
        self._body(
            "This produces a final 61-dimensional vector per delivery: "
            "[32 Video PCA + 16 Text PCA + 3 Flow + 8 Scoreboard + 2 Audio]."
        )

        self._heading("10.3 XGBoost Configuration", level=2)
        self._body("The classifier is configured with:")
        self._bullet("n_estimators = 100 (number of boosting rounds)")
        self._bullet("max_depth = 4 (tree depth limit)")
        self._bullet("learning_rate = 0.1")
        self._bullet("objective = multi:softmax (multiclass classification)")
        self._bullet("eval_metric = mlogloss (multiclass log loss)")

        self._heading("10.4 Output", level=2)
        self._body(
            "highlight_pipeline.pkl: A joblib-serialised dictionary containing the fitted "
            "PCA transformers (pca_vid, pca_txt), the LabelEncoder, and the trained XGBoost model."
        )

    def section_video_generation(self):
        self.add_page()
        self._heading("11. Stage 9: Generate Final Highlights Video")
        self._heading("11.1 Purpose", level=2)
        self._body(
            "The final stage loads the trained pipeline, predicts event classes for all "
            "deliveries, applies intelligent clip-spacing logic, extracts and encodes the "
            "selected clips, and stitches them together with FFmpeg."
        )

        self._heading("11.2 Clip Selection Logic", level=2)
        self._body("The selection rules are designed to produce a balanced, watchable highlight reel:")
        self._bullet(
            "Always Included: Wickets, run-outs, milestones, and the match-winning delivery "
            "are always included regardless of spacing."
        )
        self._bullet(
            "Spaced Sixes: A six is included only if at least 3 minutes have passed since "
            "the last included six, preventing clusters of repetitive maximums."
        )
        self._bullet(
            "Spaced Fours: A four is included only if at least 10 minutes have passed since "
            "the last included four, keeping only the most spread-out boundaries."
        )
        self._bullet(
            "Opening Ball: The first delivery of the match is always included for context."
        )
        self._bullet(
            "Fallback: If the model predicts zero highlights, the system falls back to the "
            "top 10 deliveries ranked by (1 - P(none)), ensuring the output is never empty."
        )

        self._heading("11.3 Video Encoding", level=2)
        self._body(
            "Each selected clip is extracted from the source video using FFmpeg with:\n"
            "  - Video codec: libx264, preset fast, CRF 23\n"
            "  - Audio codec: AAC at 128 kbps\n"
            "All clips are then concatenated using FFmpeg's concat demuxer with stream copy "
            "(no re-encoding of the final stitch)."
        )

        self._heading("11.4 Output", level=2)
        self._body(
            "final_highlights.mp4: The finished highlight reel, typically 15-25 minutes "
            "for a T20 match."
        )

    def section_pipeline_runner(self):
        self.add_page()
        self._heading("12. Pipeline Orchestration (run_pipeline.py)")
        self._body(
            "The orchestrator script manages the entire workflow end-to-end. Key features include:"
        )
        self._bullet(
            "CLI Interface: Accepts --video and --output arguments via argparse, "
            "defaulting to match1_h264.mp4 and final_highlights.mp4."
        )
        self._bullet(
            "Environment Propagation: Derives the audio path and data directory from the "
            "video filename stem. All paths are passed to sub-scripts via environment variables "
            "(VIDEO_PATH, AUDIO_PATH, DATA_DIR, OUTPUT_PATH), making the pipeline video-agnostic."
        )
        self._bullet(
            "Skip Logic: Each step specifies an output artifact. If the artifact already exists, "
            "the step is skipped with a [SKIP] message, enabling incremental re-runs."
        )
        self._bullet(
            "Timing: Each step and the overall pipeline are timed, with human-readable "
            "duration output."
        )
        self._bullet(
            "Error Handling: If any step fails (non-zero exit code), the pipeline halts "
            "immediately with an error message."
        )

        self._heading("12.1 Running the Pipeline", level=2)
        self._code_block(
            "# Default (processes match1_h264.mp4)\n"
            "python run_pipeline.py\n\n"
            "# Custom video\n"
            "python run_pipeline.py --video t20i_england_india.mp4 --output t20i_highlights.mp4"
        )

    def section_commentary_detection(self):
        self._heading("13. Commentary Detection (Legacy/Standalone)")
        self._body(
            "commentory_detection.py is an earlier standalone highlight extraction script "
            "that predates the multimodal pipeline. It uses a simpler keyword-matching "
            "approach on Whisper transcripts to find highlights. While not part of the "
            "main 9-stage pipeline, it served as a prototype. Key differences:"
        )
        self._bullet(
            "Uses keyword matching (e.g., 'six', 'wicket', 'bowled', 'caught', 'brilliant') "
            "instead of learned features."
        )
        self._bullet("Cuts individual clips per keyword hit rather than a single reel.")
        self._bullet(
            "Maps keyword timestamps back to delivery windows for clip boundaries."
        )

    def section_data_flow(self):
        self.add_page()
        self._heading("14. Data Flow Summary")
        self._body(
            "Each stage reads from and writes to the data directory (data/<video_stem>/). "
            "The full data flow is:"
        )
        self._body(
            "  Video File\n"
            "    |-> [Stage 1] -> deliveries.json (delivery timestamps)\n"
            "    |-> [Stage 2] -> segments_rms.json (2s audio segments + RMS)\n"
            "    |-> [Stage 3] -> segments_ocr.json (segments + scoreboard OCR)\n"
            "    |-> [Stage 4] -> highlights.json (filtered highlight segments)\n"
            "    |-> [Stage 5] -> delivery_highlights.json (delivery-level highlights)\n"
            "    |-> [Stage 6] -> transcript.json (Whisper commentary)\n"
            "    |-> [Stage 7] -> delivery_features.json (1165-D vectors)\n"
            "    |-> [Stage 8] -> highlight_pipeline.pkl (trained model)\n"
            "    |-> [Stage 9] -> final_highlights.mp4 (output video)"
        )

    def section_feature_table(self):
        self._heading("15. Feature Vector Reference")
        self._body(
            "The 1165-dimensional feature vector is structured as follows:"
        )
        ws = [35, 25, 25, 90]
        self._table_row(["Component", "Dimensions", "Index Range", "Source"], ws, bold=True)
        self._table_row(["Video (VideoMAE)", "768", "[0, 768)", "videomae-base"], ws)
        self._table_row(["Text (Sentence)", "384", "[768, 1152)", "all-MiniLM-L6-v2"], ws)
        self._table_row(["Optical Flow", "3", "[1152, 1155)", "Farneback Flow"], ws)
        self._table_row(["Scoreboard", "8", "[1155, 1163)", "PaddleOCR + Regex"], ws)
        self._table_row(["Audio", "2", "[1163, 1165)", "WAV RMS + Max"], ws)

    def section_class_labels(self):
        self.ln(5)
        self._heading("16. Classification Labels")
        self._body(
            "The XGBoost classifier targets 7 event classes:"
        )
        ws = [30, 145]
        self._table_row(["Label", "Description"], ws, bold=True)
        labels = [
            ("none", "Standard delivery with no significant event"),
            ("four", "Boundary hit (ball reaches the rope)"),
            ("six", "Maximum (ball clears the boundary on the full)"),
            ("wicket", "Batsman dismissed (caught, bowled, LBW, etc.)"),
            ("run-out", "Batsman run out during the delivery"),
            ("milestone", "Batsman reaches 50 or 100 runs"),
            ("win", "Match-winning delivery"),
        ]
        for label, desc in labels:
            self._table_row([label, desc], ws)

    def section_metrics(self):
        self.add_page()
        self._heading("17. Pipeline Evaluation Metrics")
        self._body(
            "The project leverages standard machine learning and natural language processing metrics "
            "to evaluate the quality of highlight classification and commentary transcriptions."
        )

        self._heading("17.1 Classification Metrics", level=2)
        self._body(
            "The XGBoost event classifier is evaluated natively during training using standard scikit-learn metrics. "
            "Because the training set may have class imbalances (e.g., standard deliveries vs 'six' or 'wicket'), "
            "weighted averages are computed:\n"
        )
        self._bullet(
            "Accuracy: The overall percentage of correctly classified deliveries across all 7 classes."
        )
        self._bullet(
            "Precision: The proportion of predicted positive highlights that are actually correct (minimizes false alarms)."
        )
        self._bullet(
            "Recall: The proportion of actual highlights that the model successfully detects (minimizes missed events)."
        )
        self._bullet(
            "F1 Score: The harmonic mean of Precision and Recall, providing a balanced metric for imbalanced data."
        )
        self._body(
            "These metrics are automatically computed in train_classifier.py and saved to data/metrics_output.json."
        )

        self._heading("17.2 Generative Text Evaluation (ROUGE)", level=2)
        self._body(
            "To evaluate any generated standard text (like transcribed commentary from Whisper, or AI-generated match summaries), "
            "the pipeline uses ROUGE (Recall-Oriented Understudy for Gisting Evaluation). This is available in the evaluate_metrics.py script."
        )
        self._bullet(
            "ROUGE-1: Measures unigram (single-word) overlap between the generated text and a ground-truth reference."
        )
        self._bullet(
            "ROUGE-2: Measures bigram (two-word phrase) overlap, ensuring the sentences have similar grammatical structures."
        )
        self._bullet(
            "ROUGE-L: Measures the Longest Common Subsequence (LCS), evaluating sentence-level structure similarity naturally."
        )

    # ── footer ──────────────────────────────────────────────────────────────
    def footer(self):
        self.set_y(-15)
        self.set_font("Helvetica", "I", 8)
        self.set_text_color(140, 140, 140)
        self.cell(0, 10, f"Cricket Highlight Pipeline Report  |  Page {self.page_no()}/{{nb}}",
                  align="C")


def main():
    pdf = PipelineReport()
    pdf.alias_nb_pages()

    pdf.title_page()
    pdf.section_overview()
    pdf.section_architecture()
    pdf.section_delivery_detection()
    pdf.section_audio_rms()
    pdf.section_scoreboard_ocr()
    pdf.section_first_gate()
    pdf.section_map_deliveries()
    pdf.section_transcription()
    pdf.section_feature_extraction()
    pdf.section_training()
    pdf.section_video_generation()
    pdf.section_pipeline_runner()
    pdf.section_commentary_detection()
    pdf.section_data_flow()
    pdf.section_feature_table()
    pdf.section_class_labels()
    pdf.section_metrics()

    out_path = "Cricket_Highlight_Pipeline_Report.pdf"
    pdf.output(out_path)
    print(f"\\nPDF report generated: {out_path}")
    print(f"Pages: {pdf.pages_count}")


if __name__ == "__main__":
    main()
