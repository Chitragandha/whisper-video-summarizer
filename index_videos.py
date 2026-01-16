import os
import json
from pathlib import Path
from tools.env_setup import setup_environment
from tools.transcript_ingestor import TranscriptIngestor
from tools.chunker import chunk_transcript
from tools.embedding_tool import embed
from tools.vector_store import VectorStore
import time

# -------------------------
# CONFIG
# -------------------------
VIDEO_DIR = "data/videos"
TRANSCRIPT_DIR = "data/transcripts"
FAISS_DIR = "data/faiss"

EMBEDDING_DIM = 384          # all-MiniLM-L6-v2
CHUNK_SIZE_SEC = 60          # 60s chunks


# -------------------------
# MAIN
# -------------------------
def main():
    print("🔧 Setting up environment...")
    setup_environment()

    os.makedirs(TRANSCRIPT_DIR, exist_ok=True)
    os.makedirs(FAISS_DIR, exist_ok=True)

    print("🎙️ Initializing Whisper model...")
    ingestor = TranscriptIngestor(model_size="base")

    vector_store = VectorStore(dim=EMBEDDING_DIM)

    all_embeddings = []
    all_metadata = []

    video_files = [
        f for f in os.listdir(VIDEO_DIR)
        if f.lower().endswith((".mp4", ".mkv", ".avi", ".mov"))
    ]

    if not video_files:
        raise RuntimeError("❌ No video files found in data/videos")

    for video_file in video_files:
        video_path = os.path.join(VIDEO_DIR, video_file)
        video_id = Path(video_file).stem

        print(f"\n📹 Processing video: {video_file}")

        # -------------------------
        # 1. Transcribe
        # -------------------------
        transcript_json_path = os.path.join(
            TRANSCRIPT_DIR, f"{video_id}.json"
        )

        if os.path.exists(transcript_json_path):
            print("   ↪ Transcript exists, loading...")
            with open(transcript_json_path, "r", encoding="utf-8") as f:
                segments = json.load(f)
        else:
            print("   🎧 Transcribing with Whisper...")
            start_time = time.time()
            segments = ingestor.transcribe(video_path)
            end_time = time.time()
            transcription_time = end_time - start_time
            print(f"   ⏱️ Transcription time: {transcription_time:.2f} seconds")
            with open(transcript_json_path, "w", encoding="utf-8") as f:
                json.dump(segments, f, indent=2)

        # -------------------------
        # 2. Chunk
        # -------------------------
        chunks = chunk_transcript(
            segments,
            chunk_size=CHUNK_SIZE_SEC
        )

        print(f"   ✂️ Created {len(chunks)} chunks")

        # -------------------------
        # 3. Embed + Store
        # -------------------------
        for idx, chunk in enumerate(chunks):
            embedding = embed(chunk["text"])

            all_embeddings.append(embedding)
            all_metadata.append({
                "video_id": video_id,
                "video_file": video_file,
                "chunk_id": idx,
                "start": chunk["start"],
                "end": chunk["end"],
                "text": chunk["text"]
            })

    print("\n📦 Adding embeddings to FAISS index...")
    vector_store.add(all_embeddings, all_metadata)

    # -------------------------
    # 4. Persist index
    # -------------------------
    index_path = os.path.join(FAISS_DIR, "video_index.faiss")
    meta_path = os.path.join(FAISS_DIR, "metadata.json")

    vector_store.save(index_path, meta_path)

    print("\n✅ Indexing completed successfully!")
    print(f"   FAISS index  → {index_path}")
    print(f"   Metadata    → {meta_path}")


if __name__ == "__main__":
    main()
