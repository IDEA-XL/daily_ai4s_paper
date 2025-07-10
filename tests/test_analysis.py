import pytest
from unittest.mock import MagicMock, patch, AsyncMock
from src.analysis import PaperAnalysisAgent, AnalysisResult, QAResponse
from src.data_models import PaperCandidate, AnalyzedPaper
from src.llm_client import LLMClient

@pytest.fixture
def mock_llm_client(mocker):
    """Pytest fixture for a mocked LLMClient."""
    mock_llm = MagicMock()
    mocker.patch('langchain_openai.ChatOpenAI', return_value=mock_llm)
    
    client = LLMClient()
    
    structured_llm_mock = AsyncMock()
    client.llm.with_structured_output.return_value = structured_llm_mock
    
    return client, structured_llm_mock

@pytest.fixture
def analysis_agent(mock_llm_client):
    """Pytest fixture for a PaperAnalysisAgent instance."""
    client, _ = mock_llm_client
    return PaperAnalysisAgent(llm_client=client)

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
async def test_download_and_parse_pdf(mocker, analysis_agent):
    """Test async PDF downloading and parsing."""
    mock_aread = AsyncMock(return_value=b'fake-pdf-content')
    mock_response = AsyncMock(aread=mock_aread)
    mock_response.raise_for_status.return_value = None
    
    mock_async_client = AsyncMock()
    mock_async_client.__aenter__.return_value.get.return_value = mock_response
    mocker.patch('httpx.AsyncClient', return_value=mock_async_client)

    # Mock the synchronous part that runs in a thread
    mocker.patch('fitz.open', MagicMock())
    mocker.patch('asyncio.to_thread', new_callable=AsyncMock, return_value="This is the text from the PDF.")
    
    text = await analysis_agent._download_and_parse_pdf("http://example.com/fake.pdf")
    
    assert "This is the text from the PDF." in text

def test_extract_resource_links(analysis_agent):
    """Test the extraction of resource links (synchronous method)."""
    text = "Code: https://github.com/test/repo. Model: https://huggingface.co/test/model."
    links = analysis_agent._extract_resource_links(text)
    assert links['github'] == 'https://github.com/test/repo'
    assert links['huggingface'] == 'https://huggingface.co/test/model'

@pytest.mark.asyncio
async def test_analyze_paper_full_success(analysis_agent, mock_llm_client, paper_candidate):
    """Test the full async analysis pipeline for a paper."""
    _, structured_llm_mock = mock_llm_client
    
    with patch.object(analysis_agent, '_download_and_parse_pdf', new_callable=AsyncMock, return_value="Full paper text.") as mock_parse:
        
        mock_llm_response = AnalysisResult(
            analysis_qa=[QAResponse(question="Q1", answer="A1")],
            keywords=["kw1"],
            summary="summary"
        )
        structured_llm_mock.ainvoke.return_value = mock_llm_response
        
        result = await analysis_agent.analyze_paper(paper_candidate)
        
        mock_parse.assert_called_once_with(paper_candidate.pdf_url)
        structured_llm_mock.ainvoke.assert_called_once()
        
        assert isinstance(result, AnalyzedPaper)
        assert result.summary == "summary"
        assert result.analysis_qa == {"Q1": "A1"}

@pytest.mark.asyncio
async def test_analyze_paper_pdf_failure(analysis_agent, paper_candidate, caplog):
    """Test that analysis returns None if async PDF download/parse fails."""
    with patch.object(analysis_agent, '_download_and_parse_pdf', new_callable=AsyncMock, return_value="") as mock_parse:
        result = await analysis_agent.analyze_paper(paper_candidate)
        assert result is None
        mock_parse.assert_called_once_with(paper_candidate.pdf_url)

@pytest.mark.asyncio
async def test_analyze_paper_llm_failure(analysis_agent, mock_llm_client, paper_candidate, caplog):
    """Test that analysis returns None if the async LLM call fails."""
    _, structured_llm_mock = mock_llm_client
    
    with patch.object(analysis_agent, '_download_and_parse_pdf', new_callable=AsyncMock, return_value="Text"):
        structured_llm_mock.ainvoke.side_effect = Exception("LLM API Error")
        
        result = await analysis_agent.analyze_paper(paper_candidate)
        
        assert result is None
        assert f"An error occurred during LLM analysis for paper {paper_candidate.id}" in caplog.text