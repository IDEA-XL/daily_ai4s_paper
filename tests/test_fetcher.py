import pytest
from unittest.mock import MagicMock, patch, AsyncMock
from datetime import datetime, timezone
from src.fetcher import PaperFetcher
from src.data_models import PaperCandidate
import asyncio
import sys
import io
import logging

@pytest.fixture
def fetcher():
    """Pytest fixture for a PaperFetcher instance."""
    return PaperFetcher(sources=['arXiv', 'bioRxiv', 'chemRxiv'])

@pytest.mark.asyncio
async def test_fetch_from_arxiv(mocker):
    """Test fetching papers from arXiv asynchronously."""
    mock_result = MagicMock()
    mock_result.entry_id = 'http://arxiv.org/abs/2401.0001v1'
    mock_result.title = 'Test ArXiv Paper'
    mock_result.summary = 'This is a test abstract.'
    mock_author = MagicMock()
    mock_author.name = 'Dr. Test'
    mock_result.authors = [mock_author]
    mock_result.pdf_url = 'http://arxiv.org/pdf/2401.0001v1.pdf'
    mock_result.published = datetime.now(timezone.utc)

    # Mock the synchronous search function that will be run in a thread
    def mock_search_results(*args, **kwargs):
        return [mock_result]

    mock_search = MagicMock()
    mock_search.results.side_effect = mock_search_results
    mocker.patch('arxiv.Search', return_value=mock_search)
    
    fetcher_instance = PaperFetcher(sources=['arXiv'])
    candidates = await fetcher_instance._fetch_from_arxiv()
    
    assert len(candidates) == 1
    paper = candidates[0]
    assert paper.title == 'Test ArXiv Paper'
    assert paper.source == 'arXiv'

@pytest.mark.asyncio
async def test_fetch_from_biorxiv(mocker):
    """Test fetching papers from bioRxiv asynchronously."""
    mock_response_json = {
        "collection": [
            {
                "doi": "10.1101/2024.01.01.123456",
                "title": "Test BioRxiv Paper",
                "abstract": "An abstract from bioRxiv.",
                "authors": [{"author": "Bio Tester"}],
                "version": "1"
            }
        ]
    }
    
    # Mock the async client
    mock_response = MagicMock()
    mock_response.json.return_value = mock_response_json
    mock_response.raise_for_status.return_value = None

    mock_async_client_instance = AsyncMock()
    mock_async_client_instance.get.return_value = mock_response

    mocker.patch('httpx.AsyncClient', return_value=mock_async_client_instance)
    mock_async_client_instance.__aenter__.return_value = mock_async_client_instance
    
    fetcher_instance = PaperFetcher(sources=['bioRxiv'])
    candidates = await fetcher_instance._fetch_from_rxiv('biorxiv', '2024-01-01')
    
    assert len(candidates) == 1
    paper = candidates[0]
    assert paper.title == 'Test BioRxiv Paper'
    assert paper.source == 'Biorxiv'

@pytest.mark.asyncio
async def test_fetch_from_chemrxiv(mocker):
    """Test fetching papers from chemRxiv asynchronously."""
    mock_entry = MagicMock()
    mock_entry.id = '10.26434/chemrxiv-2024-abcde'
    mock_entry.link = 'http://chemrxiv.org/engage/chemrxiv/article-details/60c72b2f9b0e9b001f3e4a5b'
    mock_entry.title = 'Test ChemRxiv Paper'
    mock_entry.summary = 'A chemRxiv abstract.'
    mock_entry.authors = [{'name': 'Chem Tester'}]
    mock_entry.published_parsed = (datetime.now(timezone.utc)).timetuple()
    mock_feed = MagicMock(entries=[mock_entry], bozo=0)
    
    # Mock the synchronous feedparser and the async scraping
    mocker.patch('feedparser.parse', return_value=mock_feed)
    mocker.patch.object(PaperFetcher, '_scrape_pdf_link_from_page', new_callable=AsyncMock, return_value='http://chemrxiv.org/pdf.pdf')

    fetcher_instance = PaperFetcher(sources=['chemRxiv'])
    candidates = await fetcher_instance._fetch_from_chemrxiv()

    assert len(candidates) == 1
    paper = candidates[0]
    assert paper.title == 'Test ChemRxiv Paper'
    assert paper.source == 'chemRxiv'

@pytest.mark.asyncio
async def test_fetch_all_sources(mocker):
    """Test the main fetch_papers method calls all source-specific methods."""
    fetcher_instance = PaperFetcher(sources=['arXiv', 'bioRxiv', 'chemRxiv'])
    
    # Mock the individual async fetch methods
    mocker.patch.object(fetcher_instance, '_fetch_from_arxiv', new_callable=AsyncMock, return_value=[MagicMock(spec=PaperCandidate)])
    mocker.patch.object(fetcher_instance, '_fetch_from_rxiv', new_callable=AsyncMock, return_value=[MagicMock(spec=PaperCandidate)])
    mocker.patch.object(fetcher_instance, '_fetch_from_chemrxiv', new_callable=AsyncMock, return_value=[MagicMock(spec=PaperCandidate)])
    
    papers = await fetcher_instance.fetch_papers()
    
    # Total papers should be 4 (1 from arxiv, 2 from biorxiv/medrxiv, 1 from chemrxiv)
    assert len(papers) == 4
    
    fetcher_instance._fetch_from_arxiv.assert_called_once()
    assert fetcher_instance._fetch_from_rxiv.call_count == 2
    fetcher_instance._fetch_from_chemrxiv.assert_called_once()

@pytest.mark.asyncio
async def test_main_execution(mocker):
    """Test the execution of the main block in fetcher.py."""
    mock_paper_fetcher_instance = MagicMock()
    mock_paper_fetcher_instance.fetch_papers = AsyncMock(return_value=[
        PaperCandidate(id='1', url='http://example.com/1', pdf_url='http://example.com/1.pdf', title='Test Paper 1', abstract='Abstract 1', authors=['Author A'], source='arXiv'),
        PaperCandidate(id='2', url='http://example.com/2', pdf_url='http://example.com/2.pdf', title='Test Paper 2', abstract='Abstract 2', authors=['Author B'], source='bioRxiv')
    ])
    mocker.patch('src.fetcher.PaperFetcher', return_value=mock_paper_fetcher_instance)

    # Capture stdout and stderr
    old_stdout = sys.stdout
    old_stderr = sys.stderr
    sys.stdout = new_stdout = io.StringIO()
    sys.stderr = new_stderr = io.StringIO()

    try:
        # Import src.fetcher here to ensure the logging configuration is applied
        # and then the main function is called.
        import src.fetcher
        # Re-configure logging to output to the captured stderr
        for handler in logging.root.handlers[:]:
            logging.root.removeHandler(handler)
        logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s', stream=new_stderr)

        await src.fetcher.main()
    finally:
        sys.stdout = old_stdout
        sys.stderr = old_stderr
        # Restore original logging handlers to avoid interfering with other tests
        for handler in logging.root.handlers[:]:
            logging.root.removeHandler(handler)
        logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

    captured_stdout = new_stdout.getvalue()
    captured_stderr = new_stderr.getvalue()
    
    mock_paper_fetcher_instance.fetch_papers.assert_called_once()
    assert "[arXiv] Test Paper 1" in captured_stdout
    assert "[bioRxiv] Test Paper 2" in captured_stdout
    assert "Fetched a total of 2 paper candidates from all sources." in captured_stderr