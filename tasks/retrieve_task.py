# tasks/retrieve_task.py
from crewai import Task
from agents.retriever_agent import retriever_agent

def create_retrieve_task(topic, retrieved_chunks):
    return Task(
        description=f"""
        Topic:
        {topic}

        Transcript chunks retrieved via vector search:
        {retrieved_chunks}

        Instructions:
        - Remove weak or unrelated chunks
        - Keep only highly relevant chunks
        - Return chunks grouped by video_id

        Output format:
        {{
          "video_id": [
            "relevant chunk 1",
            "relevant chunk 2"
          ]
        }}
        """,
        expected_output="Filtered chunks grouped by video_id",
        agent=retriever_agent
    )
