import pytest
from src.synthesizer import MarkdownSynthesizer
from src.data_models import AnalyzedPaper, PaperCandidate

@pytest.fixture
def synthesizer():
    """Pytest fixture for a MarkdownSynthesizer instance."""
    return MarkdownSynthesizer()

@pytest.fixture
def analyzed_paper():
    """Pytest fixture for a sample AnalyzedPaper."""
    candidate = PaperCandidate(
        id="test001",
        url="http://example.com/test001",
        pdf_url="http://example.com/test001.pdf",
        title="A Test Paper on AI for Science",
        abstract="This paper tests things.",
        authors=["Dr. Tester"],
        source="arXiv"
    )
    return AnalyzedPaper(
        metadata=candidate,
        keywords=["AI", "Testing", "Science"],
        analysis_qa={
            "What is the key innovation?": "A novel testing framework.",
            "What are the limitations?": "It only works in tests."
        },
        resource_links={"github": "http://github.com/test/repo"},
        summary="This paper introduces a groundbreaking method for testing."
    )

def test_synthesize_empty_list(synthesizer):
    """Test that an empty list of papers produces the correct empty report."""
    report = synthesizer.synthesize([])
    assert "No relevant papers found today" in report

def test_synthesize_single_paper(synthesizer, analyzed_paper):
    """Test that a single paper is formatted correctly into Markdown."""
    report = synthesizer.synthesize([analyzed_paper])
    
    # Check for key elements
    assert "AI4Science Daily Paper Report" in report
    assert "## 1. A Test Paper on AI for Science" in report
    assert "**Authors:** *Dr. Tester*" in report
    assert "**Source:** [arXiv](http://example.com/test001) | **PDF:** [Link](http://example.com/test001.pdf)" in report
    assert "**Resources:** [GitHub](http://github.com/test/repo)" in report
    assert "**Keywords:** `AI, Testing, Science`" in report
    assert "### Summary\nThis paper introduces a groundbreaking method for testing." in report
    assert "<details>" in report
    assert "**Q:** What is the key innovation?" in report
    assert "**A:** A novel testing framework." in report

def test_synthesize_multiple_papers(synthesizer, analyzed_paper):
    """Test that multiple papers are formatted correctly."""
    paper2 = analyzed_paper.copy(update={
        "metadata": analyzed_paper.metadata.copy(update={"title": "Another Test Paper"})
    })
    
    report = synthesizer.synthesize([analyzed_paper, paper2])
    
    assert "## 1. A Test Paper on AI for Science" in report
    assert "## 2. Another Test Paper" in report
    assert report.count("### Summary") == 2
    assert report.count("<details>") == 2

def test_formatting_no_resource_links(synthesizer, analyzed_paper):
    """Test that the resources line is omitted if no links are found."""
    analyzed_paper.resource_links = {}
    report = synthesizer.synthesize([analyzed_paper])
    
    assert "**Resources:**" not in report