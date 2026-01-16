# agents/summarizer_agent.py
from crewai import Agent, LLM

summarizer_agent = Agent(
    role="Video Summarizer",
    goal="""
    Provide a clear, descriptive, and detailed summary in paragraph form at least 10 lines.
    Then list the key points discussed in the video as bullet points.

    Rules:
    - Focus ONLY on provided transcript
    - No hallucination
    - No repetition
    - Each bullet ≤ 20 words
    """,
    backstory="Expert in technical and educational summarization",
    llm=LLM(model="gpt-4.1", temperature=0.15),
    verbose=True
)
