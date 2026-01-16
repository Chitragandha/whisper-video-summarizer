# tasks/rank_task.py
from crewai import Task
from agents.ranker_agent import ranker_agent


def create_rank_task(topic, grouped_chunks):
    return Task(
        description=f"""
Topic:
{topic}

Grouped transcript chunks by video:
{grouped_chunks}

Instructions:
- Evaluate how relevant each video is to the given topic.
- Assign a relevance score from 0 to 10 (10 = highly relevant).
- Videos scoring below 4 should be treated as NOT relevant.
- Do NOT force ranking.

- If NO video scores 4 or above, return EXACTLY:
NO_RELEVANT_VIDEOS

- For each relevant video, generate:
  1. video_id
  2. relevance score (out of 10)
  3. a short descriptive summary
  4. exactly 5 key highlights in bullet points

- Rank videos by relevance (highest first).
- Return ONLY the top 3 relevant videos.

Output format (STRICT, one video per block):

video_id | score/10
short summary
- highlight 1
- highlight 2
- highlight 3
- highlight 4
- highlight 5
""",
        expected_output="Top 3 relevant videos with scores, summaries, and highlights",
        agent=ranker_agent
    )
