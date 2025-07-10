### **System Design: AI4Science Daily Paper Digestion Agent**

---

#### **1. System Overview**

The proposed system is a multi-agent, event-driven architecture designed to autonomously collect, analyze, and summarize daily academic papers on AI for Science. It leverages a series of specialized agents orchestrated by LangGraph to ensure a modular, efficient, and scalable workflow.

The system will perform the following sequence of operations daily:
1.  **Fetch:** Scrape new pre-print publications from specified sources (arXiv, bioRxiv, etc.).
2.  **Filter:** Triage the fetched papers to identify those relevant to "AI4Science".
3.  **Analyze:** For each relevant paper, a dedicated agent will parse the content and extract key information by answering a predefined set of analytical questions.
4.  **Synthesize:** Aggregate the analyses into a single, structured Markdown report.
5.  **Publish:** The final Markdown is made available for distribution to social platforms.

**Core Technology:**
*   **Orchestration:** LangGraph
*   **Language Models:** A powerful LLM (e.g., GPT-4, Claude 3, Gemini) for analysis and triage tasks.

---

#### **2. Core Components & Data Models**

The system is composed of several specialized components, each responsible for a distinct task. They communicate via structured data objects.

**2.1. Data Models**

*   `PaperCandidate`: A lightweight object containing the initial data fetched from the source API.
    *   `id`: Unique identifier from the source.
    *   `url`: Link to the abstract page.
    *   `pdf_url`: Direct link to the PDF.
    *   `title`: Paper title.
    *   `abstract`: Paper abstract.
    *   `authors`: List of authors.
    *   `source`: e.g., 'arXiv', 'bioRxiv'.

*   `AnalyzedPaper`: A comprehensive object containing all extracted information.
    *   `metadata`: `PaperCandidate` object.
    *   `keywords`: List of keywords/topics.
    *   `analysis_qa`: A dictionary containing answers to the 10 key questions.
    *   `resource_links`: Dictionary of found links (`github`, `huggingface`, `project_page`).
    *   `summary`: A short, LLM-generated summary of the paper's contribution.

**2.2. Component Descriptions**

1.  **Data Ingestion (`PaperFetcher`)**
    *   **Function:** Connects to the public APIs of arXiv, bioRxiv, medRxiv, and chemRxiv.
    *   **Logic:**
        *   Executes once daily on a schedule (e.g., via a cron job).
        *   Queries the APIs for publications submitted in the last 24 hours in relevant categories (e.g., `cs.AI`, `cs.LG`, `cond-mat`, `q-bio`).
        *   Transforms the API responses into a list of `PaperCandidate` objects.
    *   **Output:** `list[PaperCandidate]`

2.  **Relevance Filter (`RelevanceFilter`)**
    *   **Function:** Quickly determines if a paper is relevant to "AI4Science".
    *   **Logic:**
        *   Takes a `PaperCandidate` as input.
        *   Uses an LLM to perform a rapid classification based on the `title` and `abstract`.
        *   The prompt will ask for a simple "Yes" or "No" answer to the question: "Is this paper about applying AI, machine learning, or data science techniques to a scientific domain like physics, biology, chemistry, or materials science?"
    *   **Output:** A boolean decision (`is_relevant`).

3.  **Paper Analysis Agent (`PaperAnalysisAgent`)**
    *   **Function:** The core analysis engine. It takes a single relevant paper and extracts all required information. This is best implemented as a sub-graph in LangGraph.
    *   **Logic (Internal Steps):**
        1.  **PDF Parser:** Fetches the PDF from `pdf_url` and parses its text content. Tools like `PyMuPDF` can be used.
        2.  **Resource Link Extractor:** Scans the text for URLs pointing to GitHub, Hugging Face, or project websites using regex and keyword heuristics.
        3.  **Q&A Extractor:** This is the most critical step. It uses an LLM to answer the 10 analytical questions. To ensure accuracy, this is not a single call. It should be a chain of calls or a function-calling agent that "reads" sections of the paper to answer each question. For example, to answer "Q6: How were the experiments designed?", the agent is prompted to find and summarize the "Methods" or "Experiments" section of the parsed text.
        4.  **Keyword Extractor:** An LLM call to generate relevant keywords and topics based on the full text.
    *   **Output:** An `AnalyzedPaper` object.

4.  **Report Synthesis (`MarkdownSynthesizer`)**
    *   **Function:** Compiles the final daily report.
    *   **Logic:**
        *   Receives a list of all `AnalyzedPaper` objects for the day.
        *   Iterates through the list, formatting each paper's information into a standardized Markdown template. The template should be aesthetically pleasing and easy to read for platforms like WeChat.
    *   **Output:** A single string containing the full Markdown content.

---

#### **3. Orchestration with LangGraph**

The entire workflow is defined as a graph, allowing for clear visualization, state management, and robust error handling.

**Graph Definition:**

*   **State:** The graph's state object will manage the list of paper candidates, the filtered list of relevant papers, and the final list of analyzed papers.

*   **Nodes:**
    1.  `fetch_papers`: Executes the `PaperFetcher`.
    2.  `filter_papers`: This node iterates through the candidates from `fetch_papers` and runs the `RelevanceFilter` on each. It populates a list of relevant papers to be processed.
    3.  `analyze_papers`: This node triggers the `PaperAnalysisAgent` sub-graph *in parallel* for each paper deemed relevant. This is a key efficiency gain.
    4.  `synthesize_report`: Executes the `MarkdownSynthesizer` once all analyses are complete.

*   **Edges:**
    1.  **Entry Point -> `fetch_papers`**: The graph starts here.
    2.  **`fetch_papers` -> `filter_papers`**: The list of candidates is passed for filtering.
    3.  **`filter_papers` -> `analyze_papers`**: The filtered list is passed for deep analysis.
    4.  **`analyze_papers` -> `synthesize_report`**: The collected `AnalyzedPaper` objects are passed for final report generation.
    5.  **`synthesize_report` -> END**: The graph execution finishes, outputting the Markdown file.

**Visual Flow:**

```
[START]
   |
   v
[fetch_papers] --(list of candidates)--> [filter_papers]
   |
   `--(list of relevant papers)--> [analyze_papers] (Parallel execution for each paper)
                                       |
                                       `--(list of analyzed papers)--> [synthesize_report]
                                                                          |
                                                                          v
                                                                       [END] (Output: Markdown file)
```

---

#### **4. Deployment & Operations**

1.  **Environment:** The system can be packaged in a Docker container for portability.
2.  **Execution:** A lightweight scheduler (like `cron` on a VM, or a scheduled trigger in a cloud environment like AWS Lambda or Google Cloud Functions) will invoke the LangGraph application once every 24 hours.
3.  **Output & Publishing:** The generated Markdown file is saved to a persistent location (e.g., an S3 bucket, a local directory). From there, separate, simple scripts can be used to handle the API calls for posting to WeChat, Xiaohongshu, and Twitter, keeping the core agent system decoupled from the specific publishing platforms.
4.  **Monitoring:** LangGraph's built-in tracing (e.g., with LangSmith) is crucial for debugging, monitoring costs, and evaluating the performance of the LLM-driven steps.