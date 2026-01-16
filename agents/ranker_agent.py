# agents/ranker_agent.py
from crewai import Agent, LLM

ranker_agent = Agent(
    role="Video Ranker",
    goal="""
    Rank videos by overall semantic relevance
    to the provided topic.
    """,
    backstory="""
    You specialize in comparing grouped transcript content
    and identifying the strongest topical match.
    """,
    llm=LLM(model="gpt-4o", temperature=0.1),
    verbose=True
)
