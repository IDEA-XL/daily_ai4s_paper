import pytest
from unittest.mock import MagicMock, patch, AsyncMock
from src.graph import (
    fetch_papers_node,
    filter_papers_node,
    analyze_papers_node,
    synthesize_report_node,
    get_graph,
    GraphState
)
from src.data_models import PaperCandidate, AnalyzedPaper

@pytest.fixture
def sample_paper_candidate():
    return PaperCandidate(id='1', url='url', pdf_url='pdf_url', title='t', abstract='a', authors=['a'], source='s')

@pytest.fixture
def sample_analyzed_paper(sample_paper_candidate):
    return AnalyzedPaper(
        metadata=sample_paper_candidate,
        keywords=[],
        analysis_qa={},
        resource_links={},
        summary='summary'
    )

@pytest.fixture
def initial_state() -> GraphState:
    return {
        "paper_candidates": [],
        "relevant_papers": [],
        "analyzed_papers": [],
        "markdown_report": "",
        "error": None,
    }

@pytest.mark.asyncio
async def test_fetch_papers_node_success(initial_state, sample_paper_candidate):
    with patch('src.graph.PaperFetcher') as MockFetcher:
        mock_fetcher_instance = MockFetcher.return_value
        mock_fetcher_instance.fetch_papers = AsyncMock(return_value=[sample_paper_candidate])
        
        result_state = await fetch_papers_node(initial_state)
        
        assert result_state['paper_candidates'] == [sample_paper_candidate]
        assert result_state['error'] is None
        mock_fetcher_instance.fetch_papers.assert_called_once()

@pytest.mark.asyncio
async def test_filter_papers_node_success(initial_state, sample_paper_candidate):
    with patch('src.graph.RelevanceFilter', new_callable=MagicMock) as MockFilter, \
         patch('src.graph.LLMClient', new_callable=MagicMock):
        
        mock_filter_instance = MockFilter.return_value
        mock_filter_instance.filter_papers = AsyncMock(return_value=[sample_paper_candidate])
        
        state = {**initial_state, "paper_candidates": [sample_paper_candidate]}
        result_state = await filter_papers_node(state)
        
        assert result_state['relevant_papers'] == [sample_paper_candidate]
        mock_filter_instance.filter_papers.assert_called_once_with([sample_paper_candidate])

@pytest.mark.asyncio
async def test_analyze_papers_node_success(initial_state, sample_paper_candidate, sample_analyzed_paper):
    with patch('src.graph.PaperAnalysisAgent') as MockAnalyzer, \
         patch('src.graph.LLMClient'):

        mock_analyzer_instance = MockAnalyzer.return_value
        # Mock the concurrent analysis
        mock_analyzer_instance.analyze_paper = AsyncMock(return_value=sample_analyzed_paper)
        
        state = {**initial_state, "relevant_papers": [sample_paper_candidate, sample_paper_candidate]}
        result_state = await analyze_papers_node(state)
        
        assert len(result_state['analyzed_papers']) == 2
        assert result_state['analyzed_papers'][0] == sample_analyzed_paper
        assert mock_analyzer_instance.analyze_paper.call_count == 2

def test_synthesize_report_node_success(initial_state, sample_analyzed_paper):
    # This node remains synchronous
    with patch('src.graph.MarkdownSynthesizer') as MockSynthesizer:
        mock_synthesizer_instance = MockSynthesizer.return_value
        mock_synthesizer_instance.synthesize.return_value = "# Report"
        
        state = {**initial_state, "analyzed_papers": [sample_analyzed_paper]}
        result_state = synthesize_report_node(state)
        
        assert result_state['markdown_report'] == "# Report"
        mock_synthesizer_instance.synthesize.assert_called_once_with([sample_analyzed_paper])

def test_get_graph_builds_and_compiles():
    """Tests if the graph is built and compiled without errors."""
    with patch('src.graph.StateGraph') as MockStateGraph:
        mock_graph_instance = MockStateGraph.return_value
        
        graph = get_graph()
        
        assert mock_graph_instance.add_node.call_count == 4
        mock_graph_instance.set_entry_point.assert_called_once_with("fetch_papers")
        assert mock_graph_instance.add_edge.call_count == 4
        mock_graph_instance.compile.assert_called_once()
        assert graph is not None
