# backend/tests/test_pdf_processor.py
from app.services.pdf_processor import PDFProcessor

def test_clean_text():
    # Test line break hyphens and spacing normalization
    dirty_text = "This is a semi-\nconductor processor.\n\n   It is very fast."
    cleaned = PDFProcessor.clean_text(dirty_text)
    assert cleaned == "This is a semiconductor processor. It is very fast."

def test_extract_metadata_heuristics():
    sample_text = (
        "Deep Learning for Sequence Classification\n"
        "John Doe, Jane Smith\n"
        "Department of Computer Science, University of Technology\n"
        "Proceedings of the AI Conference, 2021\n"
        "Abstract\n"
        "This paper introduces a new sequence model."
    )
    meta = PDFProcessor.extract_metadata_heuristics(sample_text, "sequence_classification.pdf")
    
    assert meta["title"] == "Deep Learning for Sequence Classification"
    assert "John Doe" in meta["authors"]
    assert "Jane Smith" in meta["authors"]
    assert meta["year"] == 2021

def test_extract_references():
    sample_text = (
        "Some contents of the paper here.\n"
        "References\n"
        "[1] Vaswani et al. Attention Is All You Need. NIPS 2017.\n"
        "[2] Devlin et al. BERT: Pre-training of Deep Bidirectional Transformers. NAACL 2019."
    )
    references = PDFProcessor.extract_references(sample_text)
    
    assert len(references) == 2
    assert "Vaswani" in references[0]
    assert "BERT" in references[1]
