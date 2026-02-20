# tasks/rank_task.py
from crewai import Task
from agents.ranker_agent import ranker_agent


def create_rank_task(topic, grouped_chunks):
    video_ids = list(grouped_chunks.keys())
    return Task(
        description=f"""
Topic:
{topic}

Below are transcript excerpts grouped by video_id.
You MUST rank ONLY from the provided video IDs.

Available video IDs:
{video_ids}

Grouped Chunks:
{grouped_chunks}

EVALUATION INSTRUCTIONS:
- Evaluate how relevant each video is to the topic
- Assign a relevance score from 0 to 10
- Videos scoring below 4 are NOT relevant
- Do NOT force ranking

OUTPUT RULES:
- Use ONLY the video IDs listed above
- If NO video scores 4 or above, output EXACTLY:
NO_RELEVANT_VIDEOS

- Rank videos by relevance (highest first)
- Return ONLY the TOP 3 relevant videos

STRICT OUTPUT FORMAT (ONE VIDEO PER BLOCK):

<video_id> | <score>/10
<one-line relevance summary>
- <highlight 1>
- <highlight 2>
- <highlight 3>
- <highlight 4>
- <highlight 5>

IMPORTANT:
- The <video_id> MUST match EXACTLY one of the provided IDs
- Do NOT use placeholders like "video id"
- Do NOT add explanations outside the specified format
""",
        expected_output="Top 3 relevant video IDs with scores, summaries, and highlights",
        agent=ranker_agent
    )
