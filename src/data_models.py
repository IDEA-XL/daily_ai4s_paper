from pydantic import BaseModel, Field
from typing import List, Dict

class PaperCandidate(BaseModel):
    """A lightweight object containing the initial data fetched from the source API."""
    id: str = Field(description="Unique identifier from the source.")
    url: str = Field(description="Link to the abstract page.")
    pdf_url: str = Field(description="Direct link to the PDF.")
    title: str = Field(description="Paper title.")
    abstract: str = Field(description="Paper abstract.")
    authors: List[str] = Field(description="List of authors.")
    source: str = Field(description="e.g., 'arXiv', 'bioRxiv'.")

class AnalyzedPaper(BaseModel):
    """A comprehensive object containing all extracted information."""
    metadata: PaperCandidate = Field(description="PaperCandidate object.")
    keywords: List[str] = Field(description="List of keywords/topics.")
    analysis_qa: Dict[str, str] = Field(description="A dictionary containing answers to the 10 key questions.")
    resource_links: Dict[str, str] = Field(description="Dictionary of found links (github, huggingface, project_page).")
    summary: str = Field(description="A short, LLM-generated summary of the paper's contribution.")
