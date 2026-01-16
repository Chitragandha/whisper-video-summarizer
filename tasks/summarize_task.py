# tasks/summarize_task.py
from crewai import Task
from agents.summarizer_agent import summarizer_agent

def create_summarize_task(selected_video_id, video_chunks):
    return Task(
        description=f"""
        Video ID:
        {selected_video_id}

        Transcript:
        {video_chunks}

        Instructions:
        - Summarize ONLY the content of this video.
        - then list the key points discussed in the video as bullet points.
        - Provide a clear, descriptive, and detailed summary in paragraph form at least 10 lines.
        """,
        
        expected_output="Key points as bullet points and a concise summary paragraph.",
        agent=summarizer_agent
    )
