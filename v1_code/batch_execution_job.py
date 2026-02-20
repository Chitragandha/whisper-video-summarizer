import os
import json
import time
from pathlib import Path
from concurrent.futures import ProcessPoolExecutor, as_completed

from tools.env_setup import setup_environment
from tools.transcript_ingestor import TranscriptIngestor

# -------------------------
# CONFIG
# -------------------------
VIDEO_DIR = "data/videos"
TRANSCRIPT_DIR = "data/transcripts"

WHISPER_MODEL_SIZE = "base"
MAX_WORKERS = 2  # increase based on CPU/GPU

# -------------------------
# Transcription Worker
# -------------------------
def transcribe_video(video_file):
    ingestor = TranscriptIngestor(model_size=WHISPER_MODEL_SIZE)

    video_path = os.path.join(VIDEO_DIR, video_file)
    video_id = Path(video_file).stem
    transcript_path = os.path.join(TRANSCRIPT_DIR, f"{video_id}.json")

    if os.path.exists(transcript_path):
        return f"⏭ Skipped (exists): {video_file}"

    try:
        start = time.time()

        segments = ingestor.transcribe(video_path)

        with open(transcript_path, "w", encoding="utf-8") as f:
            json.dump(segments, f, indent=2)

        duration = time.time() - start

        return f"✅ Transcribed {video_file} in {duration:.2f}s"

    except Exception as e:
        return f"❌ Failed {video_file} — {str(e)}"


# -------------------------
# MAIN BATCH JOB
# -------------------------
def main():
    print("🔧 Setting up environment...")
    setup_environment()

    os.makedirs(TRANSCRIPT_DIR, exist_ok=True)

    video_files = [
        f for f in os.listdir(VIDEO_DIR)
        if f.lower().endswith((".mp4", ".mkv", ".avi", ".mov"))
    ]

    if not video_files:
        print("❌ No videos found.")
        return

    print(f"🎬 Found {len(video_files)} videos")
    print(f"🚀 Starting batch transcription with {MAX_WORKERS} workers\n")

    with ProcessPoolExecutor(max_workers=MAX_WORKERS) as executor:
        futures = [executor.submit(transcribe_video, v) for v in video_files]

        for future in as_completed(futures):
            print(future.result())

    print("\n✅ Batch transcription completed.")


if __name__ == "__main__":
    main()
