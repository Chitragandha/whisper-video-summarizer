import os
os.environ["CREWAI_TELEMETRY_DISABLED"] = "true"

import streamlit as st
import pandas as pd

from tools.env_setup import setup_environment
setup_environment()

from crew import build_ranking_crew
from main import (
    load_vector_store,
    retrieve_top_chunks,
    group_chunks_by_video,
    parse_ranked_videos
)

FAISS_INDEX_PATH = "data/faiss/video_index.faiss"
METADATA_PATH = "data/faiss/metadata.json"

st.set_page_config(
    page_title="CrewAI Video Ranking",
    layout="wide"
)

@st.cache_resource
def load_index():
    return load_vector_store(
        FAISS_INDEX_PATH,
        METADATA_PATH
    )

vector_store = load_index()

# -------------------------
# UI
# -------------------------
st.title("🎥 Video Topic Relevance Ranking")

query = st.text_input(
    "Enter a topic",
    placeholder="e.g. Agentic AI, CrewAI, LLM agents"
)

if query:
    # -------------------------
    # Step 1: Vector retrieval
    # -------------------------
    with st.spinner("🔍 Retrieving relevant transcript chunks..."):
        chunk_results = retrieve_top_chunks(query, vector_store)
        grouped_chunks = group_chunks_by_video(chunk_results)

    if not grouped_chunks:
        st.warning("❌ No videos match this topic.")
        st.stop()

    # -------------------------
    # Step 2: Rank videos
    # -------------------------
    with st.spinner("🤖 Ranking videos using AI..."):
        ranking_crew = build_ranking_crew(
            topic=query,
            retrieved_chunks=grouped_chunks
        )
        ranking_output = ranking_crew.kickoff()
        ranked_videos = parse_ranked_videos(
            ranking_output,
            grouped_chunks
        )


    if not ranked_videos:
        st.warning(
            "❌ No uploaded videos are relevant to this topic."
        )
        st.stop()


    #-------------------------
    #Step 3: Display tabular output
    #-------------------------
    st.subheader("🎯 Top Relevant Videos")
    df = pd.DataFrame([
        {
            "Video ID": v["video_id"],
            "Relevance Score(out of 10)": v["score"],
            "Duration (min)": v["duration"],
            "Summary & Key Highlights": v["summary"]
        }
        for v in ranked_videos
    ])


    st.table(df)