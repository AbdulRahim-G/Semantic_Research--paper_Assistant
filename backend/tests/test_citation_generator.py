# backend/tests/test_citation_generator.py
from app.services.citation_generator import CitationGenerator

def test_generate_citations_single_author():
    citations = CitationGenerator.generate_citations(
        title="Attention Is All You Need",
        authors_list=["Ashish Vaswani"],
        year=2017,
        journal="Advances in Neural Information Processing Systems",
        doi="10.1234/nips.2017"
    )
    
    # Verify APA
    assert "Vaswani, A. (2017)" in citations["apa"]
    assert "Attention Is All You Need" in citations["apa"]
    assert "Advances in Neural Information Processing Systems" in citations["apa"]
    
    # Verify MLA
    assert "Vaswani, Ashish." in citations["mla"]
    assert '"Attention Is All You Need."' in citations["mla"]
    
    # Verify IEEE
    assert "A. Vaswani" in citations["ieee"]

def test_generate_citations_multiple_authors():
    citations = CitationGenerator.generate_citations(
        title="Attention Is All You Need",
        authors_list=["Ashish Vaswani", "Noam Shazeer", "Niki Parmar"],
        year=2017,
        journal="NIPS"
    )
    
    # Verify APA multiple authors separator
    assert "Vaswani, A., Shazeer, N., & Parmar, N. (2017)" in citations["apa"]
    
    # Verify MLA et al
    assert "Vaswani, Ashish, et al." in citations["mla"]
    
    # Verify IEEE et al
    assert "A. Vaswani et al." in citations["ieee"]
