from crewai import Crew, Process

from agents.retriever_agent import retriever_agent
from agents.ranker_agent import ranker_agent
from agents.summarizer_agent import summarizer_agent

from tasks.retrieve_task import create_retrieve_task
from tasks.rank_task import create_rank_task
from tasks.summarize_task import create_summarize_task


def build_crew(topic, retrieved_chunks, selected_video_id, video_chunks):
    retrieve_task = create_retrieve_task(
        topic=topic,
        retrieved_chunks=retrieved_chunks
    )

    rank_task = create_rank_task(
        topic=topic,
        grouped_chunks=retrieved_chunks
    )

    summarize_task = create_summarize_task(
        selected_video_id=selected_video_id,
        video_chunks=video_chunks
    )

    return Crew(
        agents=[
            retriever_agent,
            ranker_agent,
            summarizer_agent
        ],
        tasks=[
            retrieve_task,
            rank_task,
            summarize_task
        ],
        process=Process.sequential,
        verbose=False
    )

def build_ranking_crew(topic, retrieved_chunks):
    retrieve_task = create_retrieve_task(
        topic=topic,
        retrieved_chunks=retrieved_chunks
    )

    rank_task = create_rank_task(
        topic=topic,
        grouped_chunks=retrieved_chunks
    )

    return Crew(
        agents=[retriever_agent, ranker_agent],
        tasks=[retrieve_task, rank_task],
        process=Process.sequential,
        verbose=False
    )
