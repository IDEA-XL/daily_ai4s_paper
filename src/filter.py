import logging
import asyncio
from typing import List
from langchain_core.prompts import ChatPromptTemplate
from pydantic import BaseModel, Field
from src.data_models import PaperCandidate
from src.llm_client import LLMClient, LLMConfig
from dotenv import load_dotenv, find_dotenv

# Load environment variables from .env file
load_dotenv(find_dotenv())

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

class RelevanceResponse(BaseModel):
    """
    Pydantic model for the structured output from the relevance filter LLM.
    """
    is_relevant: bool = Field(description="True if the paper is relevant, False otherwise.")
    reason: str = Field(description="A brief reason for the relevance decision.")

class RelevanceFilter:
    """
    Filters papers to identify those relevant to 'AI4Science'.
    """

    def __init__(self, llm_client: LLMClient):
        """
        Initializes the RelevanceFilter.

        Args:
            llm_client (LLMClient): An instance of the LLM client.
        """
        self.llm = llm_client.llm
        self.prompt = ChatPromptTemplate.from_messages(
            [
                (
                    "system",
                    "You are an expert in scientific research and AI. Your task is to determine if a paper is relevant to the field of 'AI for Science'. "
                    "This means the paper should be about applying AI, machine learning, or data science techniques to a scientific domain like physics, biology, chemistry, materials science, or medicine. "
                    "A paper that is purely about AI theory or computer science without a clear scientific application is not relevant. "
                    "Respond with a boolean and a brief reason for your decision."
                ),
                (
                    "user",
                    "Paper Title: {title}\n\nAbstract: {abstract}"
                ),
            ]
        )
        self.structured_llm = self.prompt | self.llm.with_structured_output(RelevanceResponse)

    async def is_relevant(self, paper: PaperCandidate) -> bool:
        """
        Uses an LLM to perform a rapid classification based on the title and abstract.

        Args:
            paper (PaperCandidate): The paper to classify.

        Returns:
            bool: True if the paper is relevant, False otherwise.
        """
        try:
            response = await self.structured_llm.ainvoke({
                "title": paper.title,
                "abstract": paper.abstract
            })
            logging.info(f"Paper '{paper.title}' relevance check: {response.is_relevant}. Reason: {response.reason}")
            return response.is_relevant
        except Exception as e:
            logging.error(f"Error classifying paper {paper.id}: {e}")
            return False

    async def filter_papers(self, papers: List[PaperCandidate]) -> List[PaperCandidate]:
        """
        Filters a list of papers for relevance asynchronously.

        Args:
            papers (List[PaperCandidate): The list of papers to filter.

        Returns:
            List[PaperCandidate]: A new list containing only the relevant papers.
        """
        logging.info(f"Starting relevance filtering for {len(papers)} papers.")
        
        relevance_tasks = [self.is_relevant(paper) for paper in papers]
        results = await asyncio.gather(*relevance_tasks)
        
        relevant_papers = [paper for paper, is_relevant in zip(papers, results) if is_relevant]
        
        logging.info(f"Found {len(relevant_papers)} relevant papers.")
        return relevant_papers