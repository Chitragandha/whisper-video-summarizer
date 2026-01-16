from collections import defaultdict
from tools.embedding_tool import embed
from tools.vector_store import VectorStore

EMBEDDING_DIM = 384
TOP_K_CHUNKS = 15
TOP_K_VIDEOS = 3


# -------------------------
# Vector store
# -------------------------

def load_vector_store(index_path, meta_path):
    vs = VectorStore(dim=EMBEDDING_DIM)
    vs.load(index_path, meta_path)
    return vs


# -------------------------
# Retrieval
# -------------------------

def retrieve_top_chunks(query, vector_store):
    query_embedding = embed(query)
    return vector_store.search(query_embedding, top_k=TOP_K_CHUNKS)


# -------------------------
# Grouping (FIXED)
# -------------------------

def group_chunks_by_video(chunk_results):
    """
    Group FULL chunk metadata by video_id.
    This is REQUIRED for duration calculation.
    """
    grouped = defaultdict(list)

    for meta, _ in chunk_results:
        grouped[meta["video_id"]].append({
            "text": meta["text"],
            "start": meta["start"],
            "end": meta["end"]
        })

    return dict(grouped)


# -------------------------
# Helpers
# -------------------------

def get_chunks_for_video(video_id, chunk_results):
    return [
        meta["text"]
        for meta, _ in chunk_results
        if meta["video_id"] == video_id
    ]


def get_video_duration_from_chunks(chunks):
    """
    Compute duration in minutes using transcript timestamps.
    """
    if not chunks:
        return 0.0

    valid_ends = [c["end"] for c in chunks if "end" in c and c["end"] > 0]

    if not valid_ends:
        return 0.0

    return round(max(valid_ends) / 60, 1)



# -------------------------
# Ranker output parsing (UPDATED)
# -------------------------
def parse_ranked_videos(ranker_output, grouped_chunks, min_score=4):
    raw_text = ranker_output.raw.strip()

    if raw_text == "NO_RELEVANT_VIDEOS":
        return []

    ranked_videos = []
    blocks = raw_text.split("\n\n")

    for block in blocks:
        lines = block.strip().split("\n")
        if len(lines) < 2:
            continue

        header = lines[0]
        parts = [p.strip() for p in header.split("|")]

        # EXPECT: video_id | score/10
        if len(parts) < 2:
            continue

        video_id = parts[0]

        try:
            score = int(parts[1].replace("/10", "").strip())
        except ValueError:
            continue

        if score < min_score:
            continue

        # ✅ duration computed safely from transcript chunks
        chunks = grouped_chunks.get(video_id, [])
        duration = get_video_duration_from_chunks(chunks)

        summary = lines[1].strip()
        highlights = "\n".join(lines[2:]).strip()

        ranked_videos.append({
            "video_id": video_id,
            "score": score,
            "duration": duration,
            "summary": f"{summary}\n{highlights}"
        })

    return ranked_videos
