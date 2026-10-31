"""
Unit tests for SansKrypt reference books dataset ingestion module.
Tests:
- Sanskrit text normalization and Devanagari character verification
- Verse and passage boundary segmentation
- Idempotent translation caching (avoiding repeated translation)
- ChromaDB indexing with provenance metadata
- Prevention of duplicate records on repeated ingestion
- Semantic retrieval compatibility with domain and source filtering
"""

import os
import json
import tempfile
import pytest
import numpy as np
from unittest.mock import patch, MagicMock

from ingest_reference_books import (
    clean_sanskrit_text,
    compute_devanagari_metrics,
    segment_sanskrit_verses,
    translate_passages_with_caching,
    index_reference_passages_into_chromadb,
    ingest_pdf_book,
    load_persistent_reference_corpus,
    save_persistent_reference_corpus
)
from rag_service import (
    get_chroma_collection,
    reset_chroma_client,
    retrieve_relevant_passages
)


class MockSentenceTransformer:
    """Deterministic mock embedding model for isolated unit testing."""
    def __init__(self, dimension: int = 16):
        self.dimension = dimension

    def encode(self, texts, normalize_embeddings=True, show_progress_bar=False, **kwargs):
        vectors = []
        for text in texts:
            vec = np.zeros(self.dimension, dtype=np.float32)
            for i, word in enumerate(text.lower().split()[:self.dimension]):
                vec[i % self.dimension] += hash(word) % 100
            norm = np.linalg.norm(vec)
            if norm > 0:
                vec = vec / norm
            else:
                vec[0] = 1.0
            vectors.append(vec)
        return np.array(vectors)


@pytest.fixture
def mock_model():
    return MockSentenceTransformer(dimension=16)


@pytest.fixture
def temp_chroma_dir():
    temp_dir = tempfile.mkdtemp(prefix="sanskrypt_test_ingest_chroma_")
    yield temp_dir
    reset_chroma_client()
    import shutil
    shutil.rmtree(temp_dir, ignore_errors=True)


def test_devanagari_metrics_and_cleaning():
    """Verifies Unicode character detection and text cleaning."""
    raw_verse = "  वायुः  पित्तं   कफश्चेति त्रयो दोषाः समासतः || 1 || \n\n"
    cleaned = clean_sanskrit_text(raw_verse)
    assert "॥" in cleaned
    assert "||" not in cleaned

    metrics = compute_devanagari_metrics(cleaned)
    assert metrics["devanagari_chars"] > 15
    assert metrics["devanagari_ratio"] > 0.70

    # Non-Devanagari text
    eng_metrics = compute_devanagari_metrics("Chapter 1: The Three Doshas")
    assert eng_metrics["devanagari_chars"] == 0
    assert eng_metrics["devanagari_ratio"] == 0.0


def test_segment_sanskrit_verses():
    """Verifies that verses are properly segmented at danda boundaries."""
    page_text = (
        "Page 10\n"
        "धर्मक्षेत्रे कुरुक्षेत्रे समवेता युयुत्सवः । मामकाः पाण्डवाश्चैव किमकुर्वत सञ्जय ॥\n"
        "दृष्ट्वा तु पाण्डवानीकं व्यूढं दुर्योधनस्तदा । आचार्यमुपसङ्गम्य राजा वचनमब्रवीत् ॥\n"
    )
    passages = segment_sanskrit_verses(
        page_text=page_text,
        page_num=10,
        domain="Philosophy",
        book_title="Bhagavad Gita Study"
    )

    assert len(passages) >= 2
    for p in passages:
        assert p["domain"] == "Philosophy"
        assert p["source"] == "Bhagavad Gita Study"
        assert p["page"] == 10
        assert p["source_type"] == "reference_book"
        assert p["passage_id"].startswith("REF-PHIL-BHAGAV-P10-V")
        assert len(p["sanskrit"]) > 15


def test_translation_caching_avoids_repeated_work():
    """Verifies that already-translated passages reuse cached results."""
    mock_passages = [
        {
            "passage_id": "REF-PHIL-TEST-P1-V1",
            "sanskrit": "सत्यमेव जयते नानृतम्",
            "domain": "Philosophy",
            "source": "Mundaka Study",
            "reference": "Page 1, Verse 1",
            "needs_review": False
        }
    ]

    existing_corpus = {
        "REF-PHIL-TEST-P1-V1": {
            "passage_id": "REF-PHIL-TEST-P1-V1",
            "sanskrit": "सत्यमेव जयते नानृतम्",
            "translation": "Truth alone triumphs, not untruth.",
            "verification_status": "ai_generated",
            "needs_review": False
        }
    }

    # Should reuse cache and NOT call translate_text
    with patch("ingest_reference_books.translate_text") as mock_trans:
        enriched, trans_count = translate_passages_with_caching(mock_passages, existing_corpus)
        assert trans_count == 0
        assert enriched[0]["translation"] == "Truth alone triumphs, not untruth."
        mock_trans.assert_not_called()

    # If new passage, translate_text should be called
    new_passage = [{
        "passage_id": "REF-PHIL-TEST-P1-V2",
        "sanskrit": "अहं ब्रह्मास्मि",
        "domain": "Philosophy",
        "source": "Upanishad Study",
        "reference": "Page 1, Verse 2",
        "needs_review": False
    }]
    with patch("ingest_reference_books.translate_text", return_value="I am Brahman.") as mock_trans:
        enriched_new, trans_count_new = translate_passages_with_caching(new_passage, existing_corpus)
        assert trans_count_new == 1
        assert enriched_new[0]["translation"] == "I am Brahman."
        mock_trans.assert_called_once()


def test_idempotent_chromadb_indexing(temp_chroma_dir, mock_model):
    """Verifies that indexing reference passages into ChromaDB creates zero duplicates."""
    col = get_chroma_collection(persist_dir=temp_chroma_dir, collection_name="test_ref_books_col")

    sample_passages = [
        {
            "passage_id": "REF-AYUR-CS-P1-V1",
            "domain": "Ayurveda",
            "source": "Charaka Samhita Reference",
            "reference": "Charaka Samhita, Page 1, Verse 1",
            "page": 1,
            "sanskrit": "वायुः पित्तं कफश्चेति त्रयो दोषाः समासतः॥",
            "translation": "Vata, Pitta, and Kapha are the three bodily humors in brief.",
            "verification_status": "ai_generated",
            "keywords": "Ayurveda, Dosha, Vata, Pitta, Kapha"
        },
        {
            "passage_id": "REF-ASTRO-AB-P2-V1",
            "domain": "Astronomy & Mathematics",
            "source": "Aryabhatiya Reference",
            "reference": "Aryabhatiya, Page 2, Verse 1",
            "page": 2,
            "sanskrit": "अनुलोमगतिर्नौस्थः पश्यत्यचलं विलोमगं यद्वत्॥",
            "translation": "Just as a person in a moving boat sees fixed objects moving in the opposite direction.",
            "verification_status": "ai_generated",
            "keywords": "Astronomy, Rotation, Earth, Motion"
        }
    ]

    # First indexing run
    count1 = index_reference_passages_into_chromadb(sample_passages, collection=col, model=mock_model)
    assert count1 == 2
    assert col.count() == 2

    # Verify metadata fields
    res = col.get(ids=["REF-AYUR-CS-P1-V1"], include=["metadatas"])
    meta = res["metadatas"][0]
    assert meta["source_type"] == "reference_book"
    assert meta["is_uploaded"] is False
    assert meta["domain"] == "Ayurveda"
    assert meta["source"] == "Charaka Samhita Reference"

    # Second indexing run (rerun should be 100% idempotent)
    count2 = index_reference_passages_into_chromadb(sample_passages, collection=col, model=mock_model)
    assert count2 == 2
    assert col.count() == 2, "ChromaDB count must remain 2 after rerun (no duplicates)"


def test_retrieval_includes_reference_book_passages(temp_chroma_dir, mock_model):
    """Verifies that retrieve_relevant_passages retrieves reference book passages under 'all' and 'reference'."""
    col = get_chroma_collection(persist_dir=temp_chroma_dir, collection_name="test_ref_retrieval")

    passages = [
        {
            "passage_id": "REF-PHIL-TB-P5-V1",
            "domain": "Philosophy",
            "source": "Tattvabodha Reference",
            "reference": "Page 5, Verse 1",
            "page": 5,
            "sanskrit": "विवेकः कः? आत्मा सत्यः तदन्यत् सर्वं मिथ्या इति॥",
            "translation": "What is discernment? The Self is real, and everything else is transient.",
            "verification_status": "ai_generated",
            "keywords": "Philosophy, Discrimination, Atman, Satya, Viveka"
        }
    ]
    index_reference_passages_into_chromadb(passages, collection=col, model=mock_model)

    # Retrieval with source_filter='all'
    results_all = retrieve_relevant_passages(
        query="What is discernment and truth?",
        domain="Philosophy",
        source_filter="all",
        top_k=2,
        min_similarity=0.0,
        collection=col,
        model=mock_model
    )
    assert len(results_all) == 1
    assert results_all[0]["passage_id"] == "REF-PHIL-TB-P5-V1"
    assert results_all[0]["source_type"] == "reference_book"

    # Retrieval with source_filter='reference'
    results_ref = retrieve_relevant_passages(
        query="What is discernment and truth?",
        domain="Philosophy",
        source_filter="reference",
        top_k=2,
        min_similarity=0.0,
        collection=col,
        model=mock_model
    )
    assert len(results_ref) == 1

    # Retrieval with source_filter='uploaded' should NOT return reference book
    results_up = retrieve_relevant_passages(
        query="What is discernment and truth?",
        domain="Philosophy",
        source_filter="uploaded",
        top_k=2,
        min_similarity=0.0,
        collection=col,
        model=mock_model
    )
    assert len(results_up) == 0


def test_ingest_pdf_book_end_to_end_mocked(temp_chroma_dir, mock_model, tmp_path):
    """End-to-End test of ingest_pdf_book with mocked PDF pages and translator."""
    mock_pages = [
        (1, "Title Page: Sanskrit Astronomy Treatise\nAuthor: Classical Sage"),
        (2, "गोलपादः । वृत्तपरिधौ षष्ट्यधिकत्रिशतं भागाः ॥ १ ॥ खकक्ष्यायां ग्रहाणां भ्रमः ॥ २ ॥")
    ]
    temp_corpus = str(tmp_path / "temp_ref_corpus.json")

    with patch("ingest_reference_books.extract_raw_pages_from_pdf", return_value=mock_pages), \
         patch("ingest_reference_books.translate_text", return_value="The circumference of a circle has 360 degrees."):

        col = get_chroma_collection(persist_dir=temp_chroma_dir, collection_name="test_ingest_e2e")
        res = ingest_pdf_book(
            pdf_path="dummy_path.pdf",
            domain="Astronomy & Mathematics",
            book_title="Classical Ganita",
            dry_run=False,
            corpus_path=temp_corpus,
            collection=col,
            model=mock_model
        )

        assert res["status"] == "success"
        assert res["book_title"] == "Classical Ganita"
        assert res["passages_extracted"] >= 1
        assert res["translations_generated"] >= 1
        assert col.count() >= 1

        # Re-running ingestion must be idempotent (zero new translations, same count)
        res_rerun = ingest_pdf_book(
            pdf_path="dummy_path.pdf",
            domain="Astronomy & Mathematics",
            book_title="Classical Ganita",
            dry_run=False,
            corpus_path=temp_corpus,
            collection=col,
            model=mock_model
        )
        assert res_rerun["translations_generated"] == 0
        assert col.count() == 1
