import arxiv
import logging
import requests
import feedparser
from lxml import html
from datetime import datetime, timedelta, timezone
from typing import List
from src.data_models import PaperCandidate
import asyncio
import httpx

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

class PaperFetcher:
    """
    Connects to public APIs of academic paper sources to fetch new publications.
    """

    def __init__(self, sources: List[str] = None):
        """
        Initializes the PaperFetcher.
        
        Args:
            sources (List[str], optional): A list of sources to fetch from. 
                                          Defaults to ['arXiv', 'bioRxiv', 'chemRxiv'].
        """
        if sources is None:
            self.sources = ['arXiv', 'bioRxiv', 'chemRxiv']
        else:
            self.sources = sources
        
        self.arxiv_categories = ['cs.AI', 'cs.LG', 'cs.CL', 'cs.CV', 'cs.NE', 'stat.ML']
        self.biorxiv_servers = ['biorxiv', 'medrxiv']


    async def fetch_papers(self) -> List[PaperCandidate]:
        """
        Fetches papers from the configured sources asynchronously.

        Returns:
            List[PaperCandidate]: A list of paper candidates.
        """
        yesterday_str = (datetime.now(timezone.utc) - timedelta(days=1)).strftime("%Y-%m-%d")
        
        tasks = []
        if 'arXiv' in self.sources:
            tasks.append(self._fetch_from_arxiv())
        
        if 'bioRxiv' in self.sources:
            for server in self.biorxiv_servers:
                 tasks.append(self._fetch_from_rxiv(server, yesterday_str))

        if 'chemRxiv' in self.sources:
            tasks.append(self._fetch_from_chemrxiv())
            
        results = await asyncio.gather(*tasks, return_exceptions=True)
        
        all_candidates = []
        for result in results:
            if isinstance(result, list):
                all_candidates.extend(result)
            elif isinstance(result, Exception):
                logging.error(f"An error occurred in a fetcher task: {result}")

        logging.info(f"Fetched a total of {len(all_candidates)} paper candidates from all sources.")
        return all_candidates

    async def _fetch_from_arxiv(self) -> List[PaperCandidate]:
        """
        Fetches recent publications from arXiv in specified categories.
        Runs the synchronous arxiv library in a thread to avoid blocking.
        """
        logging.info(f"Fetching papers from arXiv for categories: {self.arxiv_categories}")
        
        def search_arxiv():
            yesterday = datetime.now(timezone.utc) - timedelta(days=1)
            query = " OR ".join([f"cat:{cat}" for cat in self.arxiv_categories])
            
            search = arxiv.Search(
                query=query,
                max_results=100,
                sort_by=arxiv.SortCriterion.SubmittedDate
            )
            
            results = list(search.results())
            logging.info(f"Found {len(results)} results from arXiv.")

            candidates = []
            for result in results:
                if result.published.astimezone(timezone.utc) > yesterday:
                    pdf_url = result.pdf_url or next((link.href for link in result.links if link.title == 'pdf'), None)
                    if not pdf_url:
                        logging.warning(f"Could not find PDF URL for arXiv paper: {result.title}")
                        continue

                    candidates.append(PaperCandidate(
                        id=result.entry_id.split('/')[-1],
                        url=result.entry_id,
                        pdf_url=pdf_url,
                        title=result.title,
                        abstract=result.summary,
                        authors=[author.name for author in result.authors],
                        source='arXiv'
                    ))
            return candidates

        try:
            candidates = await asyncio.to_thread(search_arxiv)
            logging.info(f"Filtered to {len(candidates)} arXiv candidates from the last 24 hours.")
            return candidates
        except Exception as e:
            logging.error(f"An error occurred while fetching from arXiv: {e}")
            return []

    async def _fetch_from_rxiv(self, server: str, from_date: str) -> List[PaperCandidate]:
        """
        Fetches recent publications from bioRxiv/medRxiv asynchronously.
        """
        logging.info(f"Fetching papers from {server} from date: {from_date}")
        to_date = (datetime.now(timezone.utc)).strftime("%Y-%m-%d")
        url = f"https://api.biorxiv.org/details/{server}/{from_date}/{to_date}"
        
        async with httpx.AsyncClient() as client:
            try:
                response = await client.get(url, timeout=30)
                response.raise_for_status()
                data = response.json()
                
                candidates = []
                for paper in data.get('collection', []):
                    # Construct the URL and PDF URL from the DOI
                    doi = paper.get('doi')
                    if not doi:
                        continue
                    
                    paper_url = f"https://www.biorxiv.org/content/{doi}"
                    pdf_url = f"{paper_url}v{paper.get('version')}.full.pdf"

                    authors = []
                    for author_data in paper.get('authors', []):
                        if isinstance(author_data, dict) and 'author' in author_data:
                            authors.append(author_data['author'])
                        elif isinstance(author_data, str):
                            authors.append(author_data)
                    authors = ''.join(authors).split(";")
                    candidates.append(PaperCandidate(
                        id=doi,
                        url=paper_url,
                        pdf_url=pdf_url,
                        title=paper.get('title', ''),
                        abstract=paper.get('abstract', ''),
                        authors=authors,
                        source=server.capitalize()
                    ))
                
                logging.info(f"Fetched {len(candidates)} candidates from {server.capitalize()}.")
                return candidates
            except httpx.RequestError as e:
                logging.error(f"Failed to fetch data from {server}: {e}")
                return []

    async def _fetch_from_chemrxiv(self) -> list['PaperCandidate']:
        """
        Fetches recent publications from ChemRxiv via its Public API asynchronously.
        """
        logging.info("Fetching papers from ChemRxiv Public API.")
        api_url = "https://chemrxiv.org/engage/chemrxiv/public-api/v1/items"
        
        # Define the start date for the API query (yesterday)
        yesterday = datetime.now(timezone.utc) - timedelta(days=1)
        start_date_str = yesterday.strftime("%Y-%m-%d")

        all_items = []
        page_size = 50  # Or another suitable batch size
        page = 0
        
        try:
            async with httpx.AsyncClient() as client:
                while True:
                    params = {
                        "limit": page_size,
                        "skip": page * page_size,
                        "searchDateFrom": start_date_str,
                    }
                    
                    response = await client.get(api_url, params=params, timeout=30)
                    response.raise_for_status()
                    data = response.json()
                    
                    item_hits = data.get("itemHits", [])
                    if not item_hits:
                        # No more results, break the loop
                        break
                    
                    all_items.extend(item_hits)
                    page += 1

                # Important: The structure of 'item' from the API is different from
                # the 'entry' object from an RSS feed. You will need to adapt
                # your _scrape_and_create_chemrxiv_candidate method accordingly.
                #
                # For example, instead of entry.link, you would use item['doi'].
                # Instead of entry.published_parsed, you would parse item['publishedDate'].
                scrape_tasks = [
                    self._scrape_and_create_chemrxiv_candidate(item) for item in all_items
                ]
                
                candidates = await asyncio.gather(*scrape_tasks, return_exceptions=True)
                
                # Filter out any exceptions that occurred during scraping
                final_candidates = [c for c in candidates if isinstance(c, PaperCandidate)]
                
                logging.info(f"Fetched {len(final_candidates)} candidates from ChemRxiv.")
                return final_candidates

        except httpx.HTTPStatusError as e:
            logging.error(f"HTTP error occurred while fetching from ChemRxiv API: {e.response.status_code} - {e.response.text}")
            return []
        except Exception as e:
            logging.error(f"An error occurred while fetching from ChemRxiv: {e}")
            return []

    async def _scrape_and_create_chemrxiv_candidate(self, item: dict) -> PaperCandidate | None:
        """
        Creates a PaperCandidate object from a ChemRxiv API item dictionary.
        
        This function no longer scrapes the page, as all necessary data, including
        the PDF URL, is available in the API response.
        """
        try:
            # The main item data is nested under the 'item' key in the original response,
            # but the previous function loops over 'itemHits', so we expect the item itself.
            
            # Extract direct PDF URL. Using .get() provides safety against missing keys.
            if 'item' in item:
                item = item['item']
            pdf_url = item.get('asset', {}).get('original', {}).get('url')
            if not pdf_url:
                logging.warning(f"Could not find PDF URL for ChemRxiv paper: {item.get('title')}")
                return None

            # Construct a persistent URL from the DOI.
            page_url = f"https://chemrxiv.org/engage/chemrxiv/article-details/{item['id']}"
            # Format author names from the list of author objects.
            authors_list = item.get('authors', [])
            authors = [f"{author.get('firstName', '')} {author.get('lastName', '')}".strip() for author in authors_list]

            return PaperCandidate(
                id=item['id'],
                url=page_url,
                pdf_url=pdf_url,
                title=item['title'],
                abstract=item.get('abstract'),
                authors=authors,
                source='chemRxiv'
            )
        except KeyError as e:
            logging.error(f"Failed to parse ChemRxiv API item due to missing key: {e}. Item: {item}")
            return None
        except Exception as e:
            logging.error(f"An unexpected error occurred while creating a chemRxiv candidate: {e}")
            return None

async def main():
    fetcher = PaperFetcher(sources=['chemRxiv'])
    papers = await fetcher.fetch_papers()
    for paper in papers[:2]:
        print(f"Abstract: {paper.abstract}")
        print(f"Authors: {paper.authors}")
        print(f"Source: {paper.source}")
        print(f"ID: {paper.id}")
        print(f"URL: {paper.url}")
        print(f"PDF URL: {paper.pdf_url}")
        print(f"Source: {paper.source}")
        print("-"*100)

if __name__ == '__main__':
    asyncio.run(main())
