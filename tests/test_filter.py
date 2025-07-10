import pytest
from unittest.mock import MagicMock, AsyncMock
from src.filter import RelevanceFilter, RelevanceResponse
from src.data_models import PaperCandidate
from src.llm_client import LLMClient

@pytest.fixture
def mock_llm_client():
    """Pytest fixture for a mocked LLMClient."""
    return LLMClient()

@pytest.fixture
def relevance_filter(mock_llm_client, mocker):
    """Pytest fixture for a RelevanceFilter instance."""
    mock_llm_instance = MagicMock()
    structured_llm_mock = AsyncMock()
    
    mock_llm_instance.with_structured_output.return_value = structured_llm_mock
    
    mocker.patch.object(mock_llm_client, 'llm', mock_llm_instance)
    
    return RelevanceFilter(llm_client=mock_llm_client), structured_llm_mock

@pytest.fixture
def paper_candidate():
    """Pytest fixture for a sample PaperCandidate."""
    return PaperCandidate(
        id="test001",
        url="http://example.com/test001",
        pdf_url="http://example.com/test001.pdf",
        title="A Test Paper",
        abstract="This paper tests things.",
        authors=["Tester"],
        source="Test"
    )

@pytest.mark.asyncio
async def test_is_relevant_true(relevance_filter, paper_candidate):
    """Test is_relevant returns True for relevant papers."""
    relevance_filter_instance, structured_llm_mock = relevance_filter
    
    structured_llm_mock.ainvoke.return_value = RelevanceResponse(
        is_relevant=True, 
        reason="It's relevant."
    )
    
    result = await relevance_filter_instance.is_relevant(paper_candidate)
    
    assert result is True
    structured_llm_mock.ainvoke.assert_called_once_with({
        "title": paper_candidate.title,
        "abstract": paper_candidate.abstract
    })

@pytest.mark.asyncio
async def test_is_relevant_false(relevance_filter, paper_candidate):
    """Test is_relevant returns False for irrelevant papers."""
    relevance_filter_instance, structured_llm_mock = relevance_filter
    
    structured_llm_mock.ainvoke.return_value = RelevanceResponse(
        is_relevant=False, 
        reason="It's not relevant."
    )
    
    result = await relevance_filter_instance.is_relevant(paper_candidate)
    
    assert result is False

@pytest.mark.asyncio
async def test_filter_papers(relevance_filter, paper_candidate):
    """Test the concurrent filtering of a list of papers."""
    paper1 = paper_candidate
    paper2 = paper_candidate.copy(update={"title": "Another Paper"})
    papers = [paper1, paper2]
    
    # Mock the async is_relevant method
    async def side_effect(paper):
        return paper.title != "Another Paper"
        
    relevance_filter.is_relevant = AsyncMock(side_effect=side_effect)
    
    relevant_papers = await relevance_filter.filter_papers(papers)
    
    assert len(relevant_papers) == 1
    assert relevant_papers[0].title == "A Test Paper"
    assert relevance_filter.is_relevant.call_count == 2

@pytest.mark.asyncio
async def test_is_relevant_handles_exception(relevance_filter, paper_candidate, caplog):
    """Test that is_relevant handles exceptions and returns False."""
    relevance_filter_instance, structured_llm_mock = relevance_filter
    
    structured_llm_mock.ainvoke.side_effect = Exception("API Error")
    
    result = await relevance_filter_instance.is_relevant(paper_candidate)
    
    assert result is False
    assert "Error classifying paper" in caplog.text