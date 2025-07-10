import logging
import re
import httpx
import fitz  # PyMuPDF
import asyncio
from typing import Dict, List
from langchain_core.prompts import ChatPromptTemplate
from pydantic import BaseModel, Field
from src.data_models import PaperCandidate, AnalyzedPaper
from src.llm_client import LLMClient, LLMConfig
from dotenv import load_dotenv, find_dotenv

# Load environment variables
load_dotenv(find_dotenv())

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# --- Constants ---
ANALYTICAL_QUESTIONS = [
    "1. What is the main research question or problem the paper addresses?",
    "2. What is the key innovation or contribution of this paper?",
    "3. What is the methodology or approach used in this paper?",
    "4. What were the main results of the experiments or analysis?",
    "5. What are the main limitations of the work described in the paper?",
    "6. How does this work compare to previous research in the field?",
    "7. What are the potential future directions for this research?",
    "8. What are the practical applications or implications of this work?",
    "9. What datasets were used in this study, and are they publicly available?",
    "10. Is the code or software used in this study available, and if so, where?"
]

# --- Pydantic Models for Structured LLM Output ---

class QAResponse(BaseModel):
    """Pydantic model for a single question-answer pair."""
    question: str = Field(description="The question that was asked.")
    answer: str = Field(description="The answer to the question based on the paper's content.")

class AnalysisResult(BaseModel):
    """Pydantic model for the full analysis of a paper."""
    analysis_qa: List[QAResponse] = Field(description="List of question-answer pairs.")
    keywords: List[str] = Field(description="A list of relevant keywords for the paper.")
    summary: str = Field(description="A concise, one-paragraph summary of the paper's contribution.")


class PaperAnalysisAgent:
    """
    The core analysis engine. It takes a single relevant paper and extracts all required information.
    """

    def __init__(self, llm_client: LLMClient):
        """
        Initializes the PaperAnalysisAgent.

        Args:
            llm_client (LLMClient): An instance of the LLM client.
        """
        self.llm = llm_client.llm
        self.qa_prompt = self._build_qa_prompt()
        structured_llm = self.llm.with_structured_output(AnalysisResult)
        self.analysis_chain = self.qa_prompt | structured_llm

    def _build_qa_prompt(self):
        """Builds the prompt for Q&A, keyword, and summary extraction."""
        question_list = "\n".join(ANALYTICAL_QUESTIONS)
        return ChatPromptTemplate.from_messages(
            [
                (
                    "system",
                    "You are a highly skilled research assistant specializing in AI for Science. "
                    "Your task is to read the provided scientific paper text and perform a detailed analysis. "
                    "You must answer a specific set of questions, generate relevant keywords, and provide a concise summary. "
                    "Base your answers strictly on the content of the paper. If the paper does not provide an answer to a question, state that clearly."
                ),
                (
                    "user",
                    "Please analyze the following paper text and provide the answers to the questions, keywords, and a summary.\n\n"
                    "**Paper Text:**\n\n{paper_text}\n\n"
                    "**Questions to Answer:**\n{questions}"
                ),
            ]
        )

    async def _download_and_parse_pdf(self, pdf_url: str) -> str:
        """
        Fetches a PDF from a URL and parses its text content asynchronously.
        """
        logging.info(f"Downloading and parsing PDF from: {pdf_url}")
        async with httpx.AsyncClient() as client:
            try:
                response = await client.get(pdf_url, timeout=30, follow_redirects=True)
                response.raise_for_status()
                
                pdf_content = await response.aread()

                # fitz.open is synchronous, so we run it in a thread
                def parse_text(content):
                    with fitz.open(stream=content, filetype="pdf") as doc:
                        text = "".join(page.get_text() for page in doc)
                    return re.sub(r'\s+', ' ', text).strip()

                text = await asyncio.to_thread(parse_text, pdf_content)
                
                logging.info(f"Successfully parsed PDF. Text length: {len(text)} characters.")
                return text
            except httpx.RequestError as e:
                logging.error(f"Failed to download PDF from {pdf_url}: {e}")
            except Exception as e:
                logging.error(f"Failed to parse PDF from {pdf_url}: {e}")
            return ""

    def _extract_resource_links(self, text: str) -> Dict[str, str]:
        """
        Scans text for URLs pointing to GitHub, Hugging Face, or project websites.
        This is a CPU-bound task, so it remains synchronous.
        """
        links = {
            "github": "",
            "huggingface": "",
            "project_page": ""
        }
        
        github_matches = re.findall(r'https?://(?:www\.)?github\.com/[\w\-]+/[\w\-]+', text)
        huggingface_matches = re.findall(r'https?://(?:www\.)?huggingface\.co/[\w\-]+', text)
        
        if github_matches:
            links["github"] = github_matches[0]
            logging.info(f"Found GitHub link: {links['github']}")
        if huggingface_matches:
            links["huggingface"] = huggingface_matches[0]
            logging.info(f"Found Hugging Face link: {links['huggingface']}")
            
        return links

    async def analyze_paper(self, paper: PaperCandidate) -> AnalyzedPaper | None:
        """
        Runs the full analysis pipeline on a single paper asynchronously.
        """
        logging.info(f"Starting analysis for paper: {paper.title}")
        
        paper_text = await self._download_and_parse_pdf(paper.pdf_url)
        if not paper_text:
            return None
        
        resource_links = self._extract_resource_links(paper_text)
        
        max_text_length = 30000 # TODO: make this configurable
        if len(paper_text) > max_text_length:
            logging.warning(f"Paper text is too long ({len(paper_text)} chars). Truncating to {max_text_length}.")
            paper_text = paper_text[:max_text_length]

        try:
            logging.info(f"Invoking LLM for deep analysis for paper: {paper.title}")
            analysis_result = await self.analysis_chain.ainvoke({
                "paper_text": paper_text,
                "questions": "\n".join(ANALYTICAL_QUESTIONS)
            })
            
            analysis_qa_dict = {item.question: item.answer for item in analysis_result.analysis_qa}

            analyzed_paper = AnalyzedPaper(
                metadata=paper,
                analysis_qa=analysis_qa_dict,
                keywords=analysis_result.keywords,
                summary=analysis_result.summary,
                resource_links=resource_links
            )
            logging.info(f"Successfully completed analysis for paper: {paper.title}")
            return analyzed_paper

        except Exception as e:
            logging.error(f"An error occurred during LLM analysis for paper {paper.id}: {e}")
            return None