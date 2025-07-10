import logging
from datetime import datetime
import asyncio
from typing import List
import os
from dotenv import load_dotenv, find_dotenv

# if python version is <3.12, use TypedDict
import sys
if sys.version_info < (3, 12):
    from typing_extensions import TypedDict
else:
    from typing import TypedDict

from langgraph.graph import StateGraph, END
from langgraph.graph.state import CompiledStateGraph
from src.data_models import PaperCandidate, AnalyzedPaper
from src.fetcher import PaperFetcher
from src.filter import RelevanceFilter
from src.analysis import PaperAnalysisAgent
from src.synthesizer import MarkdownSynthesizer
from src.llm_client import LLMClient, LLMConfig
from src.cache import load_processed_ids, save_processed_ids

# Load environment variables
load_dotenv(find_dotenv())

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# --- Graph State ---
class GraphState(TypedDict):
    """The state of the graph."""
    paper_candidates: List[PaperCandidate]
    relevant_papers: List[PaperCandidate]
    analyzed_papers: List[AnalyzedPaper]
    markdown_report: str
    error: str | None

# --- Node Definitions ---

async def fetch_papers_node(state: GraphState) -> GraphState:
    """Node to fetch paper candidates asynchronously."""
    logging.info("--- Running Fetch Papers Node ---")
    try:
        fetcher = PaperFetcher()
        candidates = await fetcher.fetch_papers()
        return {**state, "paper_candidates": candidates, "error": None}
    except Exception as e:
        logging.error(f"Error in fetch_papers_node: {e}")
        return {**state, "error": "Failed to fetch papers."}

async def filter_papers_node(state: GraphState) -> GraphState:
    """Node to filter papers for relevance and check against the cache."""
    logging.info("--- Running Filter Papers Node ---")
    if state.get("error"):
        return state
    try:
        # 1. Filter by relevance
        llm_client = LLMClient(LLMConfig(temperature=0))
        relevance_filter = RelevanceFilter(llm_client=llm_client)
        candidates = state["paper_candidates"]
        relevant_papers = await relevance_filter.filter_papers(candidates)
        
        # 2. Filter by cache
        processed_ids = load_processed_ids()
        if processed_ids:
            logging.info(f"Loaded {len(processed_ids)} processed paper IDs from cache.")
            papers_to_analyze = [
                paper for paper in relevant_papers if paper.id not in processed_ids
            ]
            num_skipped = len(relevant_papers) - len(papers_to_analyze)
            if num_skipped > 0:
                logging.info(f"Skipping {num_skipped} papers already present in the cache.")
            relevant_papers = papers_to_analyze
        else:
            logging.info("No cache found or cache is empty. Processing all relevant papers.")

        return {**state, "relevant_papers": relevant_papers}
    except Exception as e: 
        logging.error(f"Error in filter_papers_node: {e}")
        return {**state, "error": "Failed to filter papers for relevance."}

async def analyze_papers_node(state: GraphState) -> GraphState:
    """Node to analyze relevant papers and update the cache."""
    logging.info("--- Running Analyze Papers Node ---")
    if state.get("error") or not state.get("relevant_papers"):
        return state
    try:
        llm_client = LLMClient(LLMConfig(temperature=0.2))
        analysis_agent = PaperAnalysisAgent(llm_client=llm_client)
        relevant_papers = state["relevant_papers"]
        
        analysis_tasks = [analysis_agent.analyze_paper(paper) for paper in relevant_papers]
        results = await asyncio.gather(*analysis_tasks, return_exceptions=True)
        
        analyzed_papers = [res for res in results if isinstance(res, AnalyzedPaper)]
        errors = [res for res in results if isinstance(res, Exception)]
        
        if errors:
            logging.error(f"Encountered {len(errors)} errors during paper analysis.")
            for err in errors:
                logging.error(err)
        
        # Update cache with successfully analyzed papers
        successfully_analyzed_ids = {paper.metadata.id for paper in analyzed_papers}
        if successfully_analyzed_ids:
            save_processed_ids(successfully_analyzed_ids)

        return {**state, "analyzed_papers": analyzed_papers}
    except Exception as e:
        logging.error(f"Error in analyze_papers_node: {e}")
        return {**state, "error": "Failed during paper analysis."}

async def synthesize_report_node(state: GraphState, synthesizer: MarkdownSynthesizer) -> GraphState:
    """
    Node to synthesize the final Markdown report and save it asynchronously.
    """
    logging.info("--- Running Synthesize Report Node ---")
    if state.get("error"):
        return state
    
    try:
        # Step 1: Use the provided synthesizer instance
        # synthesizer = MarkdownSynthesizer()
        analyzed_papers = state["analyzed_papers"]
        
        # Step 2: Generate the report content (this is a fast, synchronous, in-memory operation)
        report_content = synthesizer.synthesize(analyzed_papers)
        
        # Step 3: Define a filename and save the report asynchronously
        # This is the non-blocking file write operation.
        report_filename = f"AI4Science_Report_{datetime.now().strftime('%Y-%m-%d')}.md"
        await synthesizer.save_report_async(report_content, report_filename)
        
        logging.info(f"Report has been successfully synthesized and saved to {report_filename}")

        # Step 4: Update the state with the report content and the filename
        return {
            **state,
            "markdown_report": report_content,
            "report_filename": report_filename
        }
        
    except Exception as e:
        logging.error(f"Error in synthesize_report_node: {e}", exc_info=True)
        return {**state, "error": "Failed to synthesize and save the report."}

# --- Graph Builder ---

def get_graph() -> CompiledStateGraph:
    """
    Builds and compiles the LangGraph workflow.
    This function is the entry point for the 'langgraph' CLI.
    """
    report_saving_dir = os.getenv("REPORT_SAVING_DIR", ".")
    logging.info(f"Report saving directory: {report_saving_dir}")
    """
    Builds and compiles the LangGraph workflow.
    This function is the entry point for the 'langgraph' CLI.
    """
    workflow = StateGraph(GraphState)

    workflow.add_node("fetch_papers", fetch_papers_node)
    workflow.add_node("filter_papers", filter_papers_node)
    workflow.add_node("analyze_papers", analyze_papers_node)
    synthesizer = MarkdownSynthesizer(report_saving_dir=report_saving_dir)
    workflow.add_node("synthesize_report", lambda state: synthesize_report_node(state, synthesizer))

    workflow.set_entry_point("fetch_papers")
    workflow.add_edge("fetch_papers", "filter_papers")
    workflow.add_edge("filter_papers", "analyze_papers")
    workflow.add_edge("analyze_papers", "synthesize_report")
    workflow.add_edge("synthesize_report", END)

    return workflow.compile()

async def main():
    """Main execution function."""
    graph = get_graph()
    
    print("--- Starting AI4Science Daily Paper Graph Execution ---")
    initial_state = {
        "paper_candidates": [],
        "relevant_papers": [],
        "analyzed_papers": [],
        "markdown_report": "",
        "error": None
    }
    
    final_state = await graph.ainvoke(initial_state)
    
    print("--- Graph Execution Finished ---")

    if final_state.get("error"):
        print(f"\n--- Workflow failed: {final_state['error']} ---")
    else:
        print("\n--- Workflow Succeeded ---")
        report_path = "daily_report.md"
        with open(report_path, "w", encoding="utf-8") as f:
            f.write(final_state["markdown_report"])
        print(f"Report saved to {report_path}")

if __name__ == '__main__':
    asyncio.run(main())