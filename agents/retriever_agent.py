# agents/retriever_agent.py
from crewai import Agent, LLM

retriever_agent = Agent(
    role="Transcript Relevance Filter",
    goal="""
    Filter transcript chunks to keep only those
    that are strongly relevant to the given topic.
    """,
    backstory="""
    You are an expert at semantic relevance judgment.
    You do not perform searches or embeddings.
    You only evaluate provided transcript chunks.
    """,
    llm=LLM(model="gpt-4o", temperature=0.1),
    verbose=True
)
