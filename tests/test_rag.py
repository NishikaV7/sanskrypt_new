"""
Tests for SansKrypt RAG and Semantic Retrieval Module
Covers:
1. Empty query validation
2. Corpus parsing and validation of missing/corrupt fields
3. Duplicate indexing prevention and stable passage IDs
4. Query embedding and retrieval using a deterministic mock embedding model
5. Domain-specific filtering
6. Insufficient evidence handling
7. Gemini unavailable fallback
8. Answer references correspond to retrieved passages
9. Flask /ask route handling (GET, POST, and errors)
"""
import os
import json
import shutil
import tempfile
import pytest
import numpy as np
from unittest.mock import MagicMock, patch

from rag_service import (
    load_corpus,
    index_corpus,
    retrieve_relevant_passages,
    generate_grounded_answer,
    query_rag_pipeline,
    get_chroma_collection,
    reset_chroma_client,
    get_embedding_model_status,
    get_active_embedding_model_name,
    detect_query_domains,
    MIN_EVIDENCE_SIMILARITY
)


class MockSentenceTransformer:
    """Deterministic mock embedding model that requires zero network access."""
    def __init__(self, dimension: int = 16):
        self.dimension = dimension

    def encode(self, texts, normalize_embeddings=True, show_progress_bar=False, **kwargs):
        vectors = []
        for text in texts:
            # Deterministic hash-based vector
            vec = np.zeros(self.dimension, dtype=np.float32)
            for i, word in enumerate(text.lower().split()[:self.dimension]):
                vec[i % self.dimension] += hash(word) % 100
            # Normalize to unit vector
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
    temp_dir = tempfile.mkdtemp(prefix="test_chroma_")
    reset_chroma_client()
    yield temp_dir
    reset_chroma_client()
    shutil.rmtree(temp_dir, ignore_errors=True)


@pytest.fixture
def sample_test_corpus():
    return [
        {
            "passage_id": "TEST-PHIL-001",
            "sanskrit": "सत्यमेव जयते नानृतम्",
            "translation": "Truth alone triumphs, not falsehood.",
            "domain": "Philosophy",
            "source": "Mundaka Upanishad",
            "reference": "3.1.6",
            "verification_status": "verified_canonical",
            "keywords": "Truth, Satya, Dharma"
        },
        {
            "passage_id": "TEST-AYUR-001",
            "sanskrit": "वायुः पित्तं कफश्चेति त्रयो दोषाः समासतः",
            "translation": "Vata, Pitta, and Kapha are the three bodily humors (doshas) stated in brief.",
            "domain": "Ayurveda",
            "source": "Ashtanga Hridaya",
            "reference": "Sutrasthana 1.6",
            "verification_status": "verified_canonical",
            "keywords": "Vata, Pitta, Kapha, Dosha, Health"
        },
        {
            "passage_id": "TEST-ASTRO-001",
            "sanskrit": "मखि भखि फखि धखि णखि",
            "translation": "Aryabhata's sine difference table in astronomical computation.",
            "domain": "Astronomy",
            "source": "Aryabhatiya",
            "reference": "Gitikapada 12",
            "verification_status": "verified_canonical",
            "keywords": "Aryabhata, Sine, Astronomy, Mathematics"
        }
    ]


# ---------------------------------------------------------------------------
# TEST 1: Empty Query Validation
# ---------------------------------------------------------------------------
def test_empty_query_validation():
    res1 = query_rag_pipeline("")
    assert res1["passages"] == []
    assert res1["sufficient_evidence"] is False
    assert "valid" in res1["answer"].lower()

    res2 = query_rag_pipeline("   ")
    assert res2["passages"] == []
    assert res2["sufficient_evidence"] is False


# ---------------------------------------------------------------------------
# TEST 2: Corpus Parsing & Missing Fields
# ---------------------------------------------------------------------------
def test_corpus_parsing_and_missing_fields(tmp_path):
    corpus_file = tmp_path / "test_corpus.json"
    data = [
        # Valid item
        {
            "passage_id": "VALID-01",
            "sanskrit": "अहं ब्रह्मास्मि",
            "translation": "I am Brahman.",
            "domain": "Philosophy",
            "source": "Upanishad",
            "reference": "1.4.10",
            "verification_status": "verified_canonical"
        },
        # Missing translation
        {
            "passage_id": "INVALID-01",
            "sanskrit": "सत्यम्",
            "domain": "Philosophy"
        },
        # Missing domain
        {
            "passage_id": "INVALID-02",
            "sanskrit": "वायुः",
            "translation": "Wind"
        },
        # Not a dict
        "invalid string record"
    ]
    corpus_file.write_text(json.dumps(data), encoding="utf-8")

    records = load_corpus(str(corpus_file))
    assert len(records) == 1
    assert records[0]["passage_id"] == "VALID-01"
    assert records[0]["domain"] == "Philosophy"


# ---------------------------------------------------------------------------
# TEST 3: Duplicate Indexing & Stable Passage IDs
# ---------------------------------------------------------------------------
def test_duplicate_indexing_and_stable_ids(temp_chroma_dir, mock_model, sample_test_corpus):
    col = get_chroma_collection(persist_dir=temp_chroma_dir, collection_name="test_col_dedup")

    # First index pass
    count1 = index_corpus(
        corpus_records=sample_test_corpus,
        collection=col,
        model=mock_model
    )
    assert count1 == 3
    assert col.count() == 3

    # Re-indexing the exact same corpus should update (upsert), NOT duplicate
    count2 = index_corpus(
        corpus_records=sample_test_corpus,
        collection=col,
        model=mock_model
    )
    assert count2 == 3
    assert col.count() == 3

    # Check stable IDs
    item = col.get(ids=["TEST-PHIL-001"])
    assert len(item["ids"]) == 1
    assert item["metadatas"][0]["source"] == "Mundaka Upanishad"


# ---------------------------------------------------------------------------
# TEST 4: Query Embedding & Retrieval
# ---------------------------------------------------------------------------
def test_query_retrieval_mocked(temp_chroma_dir, mock_model, sample_test_corpus):
    col = get_chroma_collection(persist_dir=temp_chroma_dir, collection_name="test_col_retrieval")
    index_corpus(corpus_records=sample_test_corpus, collection=col, model=mock_model)

    passages = retrieve_relevant_passages(
        query="What are the three doshas Vata Pitta Kapha?",
        top_k=2,
        min_similarity=0.0,  # accept mock similarities
        collection=col,
        model=mock_model
    )

    assert len(passages) >= 1
    top = passages[0]
    assert "passage_id" in top
    assert "sanskrit" in top
    assert "translation" in top
    assert "domain" in top
    assert "source" in top
    assert "reference" in top
    assert "similarity_score" in top
    assert "verification_status" in top


# ---------------------------------------------------------------------------
# TEST 5: Domain Filtering
# ---------------------------------------------------------------------------
def test_domain_filtering(temp_chroma_dir, mock_model, sample_test_corpus):
    col = get_chroma_collection(persist_dir=temp_chroma_dir, collection_name="test_col_domains")
    index_corpus(corpus_records=sample_test_corpus, collection=col, model=mock_model)

    # Filter to Ayurveda only
    ayur_passages = retrieve_relevant_passages(
        query="medical health and humors",
        domain="Ayurveda",
        top_k=3,
        min_similarity=0.0,
        collection=col,
        model=mock_model
    )
    for p in ayur_passages:
        assert p["domain"] == "Ayurveda"

    # Filter to Astronomy only
    astro_passages = retrieve_relevant_passages(
        query="astronomical calculations and sine",
        domain="Astronomy",
        top_k=3,
        min_similarity=0.0,
        collection=col,
        model=mock_model
    )
    for p in astro_passages:
        assert p["domain"] == "Astronomy"


# ---------------------------------------------------------------------------
# TEST 6: Missing or Insufficient Evidence
# ---------------------------------------------------------------------------
def test_insufficient_evidence_handling():
    # Case A: No passages retrieved
    ans_empty = generate_grounded_answer(
        query="What is quantum mechanics?",
        passages=[]
    )
    assert ans_empty["sufficient_evidence"] is False
    assert ans_empty["has_generated_answer"] is False
    assert "insufficient" in ans_empty["answer"].lower()

    # Case B: Passages below similarity threshold
    low_sim_passages = [{
        "passage_id": "LOW-01",
        "sanskrit": "काव्यम्",
        "translation": "Poetry",
        "domain": "Literature",
        "source": "Kavya",
        "reference": "1.1",
        "verification_status": "unverified",
        "similarity_score": 0.15  # below MIN_EVIDENCE_SIMILARITY
    }]
    ans_low = generate_grounded_answer(
        query="What are the planetary orbits?",
        passages=low_sim_passages
    )
    assert ans_low["sufficient_evidence"] is False
    assert "threshold" in ans_low["answer"].lower() or "insufficient" in ans_low["answer"].lower()


# ---------------------------------------------------------------------------
# TEST 7: Gemini Unavailable Fallback
# ---------------------------------------------------------------------------
def test_gemini_unavailable_fallback():
    good_passages = [{
        "passage_id": "TEST-AYUR-001",
        "sanskrit": "वायुः पित्तं कफश्चेति त्रयो दोषाः समासतः",
        "translation": "Vata, Pitta, and Kapha are the three bodily humors in brief.",
        "domain": "Ayurveda",
        "source": "Ashtanga Hridaya",
        "reference": "Sutrasthana 1.6",
        "verification_status": "verified_canonical",
        "similarity_score": 0.85
    }]

    with patch.dict(os.environ, {"GEMINI_API_KEY": "", "GOOGLE_API_KEY": ""}):
        res = generate_grounded_answer(
            query="What are the doshas?",
            passages=good_passages,
            gemini_key=""
        )
        assert res["sufficient_evidence"] is True
        assert res["has_generated_answer"] is False
        assert "unavailable" in res["answer"].lower()
        assert "TEST-AYUR-001" in res["cited_passages"]


# ---------------------------------------------------------------------------
# TEST 8: Answer References Correspond to Retrieved Passages
# ---------------------------------------------------------------------------
def test_answer_references_correspond_to_retrieved_passages():
    good_passages = [{
        "passage_id": "TEST-PHIL-001",
        "sanskrit": "सत्यमेव जयते नानृतम्",
        "translation": "Truth alone triumphs, not falsehood.",
        "domain": "Philosophy",
        "source": "Mundaka Upanishad",
        "reference": "Mundaka 3.1.6",
        "verification_status": "verified_canonical",
        "similarity_score": 0.90
    }]

    mock_client = MagicMock()
    mock_response = MagicMock()
    mock_response.text = (
        "According to [TEST-PHIL-001] Mundaka Upanishad (3.1.6), truth alone triumphs, "
        "and untruth does not."
    )
    mock_client.models.generate_content.return_value = mock_response

    with patch("google.genai.Client", return_value=mock_client):
        res = generate_grounded_answer(
            query="Does truth triumph?",
            passages=good_passages,
            gemini_key="dummy_mock_key"
        )
        assert res["sufficient_evidence"] is True
        assert res["has_generated_answer"] is True
        assert "TEST-PHIL-001" in res["cited_passages"]
        assert "Mundaka Upanishad" in res["answer"]


# ---------------------------------------------------------------------------
# TEST 9: Flask Route /ask (GET, POST, and Error Handling)
# ---------------------------------------------------------------------------
def test_flask_ask_route_get_and_post(client):
    # GET /ask initial view
    get_res = client.get("/ask")
    assert get_res.status_code == 200
    assert b"Ask the Sanskrit Knowledge Base" in get_res.data
    assert b"NATURAL-LANGUAGE QUESTION" in get_res.data

    # POST /ask with empty query redirects with warning
    empty_post = client.post("/ask", data={"query": ""}, follow_redirects=True)
    assert empty_post.status_code == 200

    # Mock query_rag_pipeline for route testing
    mock_rag_result = {
        "query": "What are doshas?",
        "domain": "Ayurveda",
        "passages": [{
            "passage_id": "AYUR-001",
            "sanskrit": "वायुः पित्तं कफः",
            "translation": "Vata, Pitta, Kapha.",
            "domain": "Ayurveda",
            "source": "Ashtanga Hridaya",
            "reference": "1.6",
            "verification_status": "verified_canonical",
            "similarity_score": 0.88
        }],
        "answer": "Vata, Pitta, and Kapha are the primary humors in Ayurveda [AYUR-001].",
        "sufficient_evidence": True,
        "has_generated_answer": True,
        "cited_passages": ["AYUR-001"]
    }

    with patch("app.query_rag_pipeline", return_value=mock_rag_result):
        post_res = client.post(
            "/ask",
            data={"query": "What are doshas?", "domain": "Ayurveda"}
        )
        assert post_res.status_code == 200
        assert b"Grounded Answer" in post_res.data
        assert b"Vata, Pitta, and Kapha" in post_res.data
        assert b"AYUR-001" in post_res.data

    # GET /ask?query=... with query parameters
    with patch("app.query_rag_pipeline", return_value=mock_rag_result):
        param_res = client.get("/ask?query=What+are+doshas&domain=Ayurveda")
        assert param_res.status_code == 200
        assert b"Grounded Answer" in param_res.data


# ---------------------------------------------------------------------------
# TEST 10: Model Transparency and Fallback Reporting
# ---------------------------------------------------------------------------
def test_model_transparency_and_fallback_status(client):
    status = get_embedding_model_status()
    assert "configured_primary" in status
    assert "active_model" in status
    assert "is_fallback" in status
    assert "index_dimension" in status
    assert status["index_dimension"] == 384
    assert status["is_fallback"] is False

    # Check GET /ask displays the active model matching the 384-dim index
    res = client.get("/ask")
    assert res.status_code == 200
    assert b"Active Embedding Engine:" in res.data
    assert b"all-MiniLM-L6-v2" in res.data


# ---------------------------------------------------------------------------
# TEST 11: Real Persistent ChromaDB Index Verification (Non-Mocked)
# ---------------------------------------------------------------------------
def test_real_persistent_chroma_index():
    """Directly verifies that the existing persistent vector index is intact and queryable."""
    reset_chroma_client()
    try:
        col = get_chroma_collection()
        count = col.count()
        assert count >= 15, f"Expected at least 15 indexed passages, found {count}"

        sample = col.get(ids=["PHIL-UP-001"], include=["embeddings", "metadatas", "documents"])
        assert len(sample["ids"]) == 1
        assert sample["ids"][0] == "PHIL-UP-001"
        assert len(sample["embeddings"][0]) == 384
        assert sample["metadatas"][0]["domain"] == "Philosophy"
        assert sample["metadatas"][0]["source"] == "Mundaka Upanishad"
        assert sample["metadatas"][0]["verification_status"] == "verified_canonical"
    finally:
        reset_chroma_client()


# ---------------------------------------------------------------------------
# TEST 12: Query Domain Understanding & Intent Routing
# ---------------------------------------------------------------------------
def test_query_domain_intent_detection():
    """Verifies that natural-language queries are correctly routed to their knowledge domains."""
    # Ayurveda query
    ayur = detect_query_domains("What are the three doshas in Ayurveda?")
    assert ayur["primary_domain"] == "Ayurveda"
    assert "Ayurveda" in ayur["target_domains"]

    # Astronomy & Mathematics query
    astro = detect_query_domains("Explain the planetary calculations described in the manuscript.")
    assert astro["primary_domain"] == "Astronomy & Mathematics"
    assert "Astronomy & Mathematics" in astro["target_domains"]

    # Philosophy query
    phil = detect_query_domains("Explain the concept of Atman in the Bhagavad Gita.")
    assert phil["primary_domain"] == "Philosophy"
    assert "Philosophy" in phil["target_domains"]

    # Genuine mixed-domain query
    mixed = detect_query_domains("How does Samkhya philosophy explain the gunas and doshas in health?")
    assert mixed["is_mixed"] is True
    assert "Ayurveda" in mixed["target_domains"]
    assert "Philosophy" in mixed["target_domains"]

    # Unrelated/generic query falls back to 'all'
    unrelated = detect_query_domains("How do I bake a chocolate cake?")
    assert unrelated["primary_domain"] == "all"
    assert unrelated["target_domains"] == ["all"]


# ---------------------------------------------------------------------------
# TEST 13: Removal of Unconditional Upload Boost
# ---------------------------------------------------------------------------
def test_no_unconditional_upload_boost(temp_chroma_dir, mock_model):
    """Verifies that an irrelevant uploaded chunk is not artificially boosted above a relevant canonical passage."""
    col = get_chroma_collection(persist_dir=temp_chroma_dir, collection_name="test_boost_calib")

    # Seed one authoritative reference passage and one weak uploaded chunk
    col.add(
        ids=["REF-AYUR-CANONICAL", "UPLOAD-M99-C1"],
        documents=[
            "Domain: Ayurveda | Source: Ashtanga Hridaya\nTranslation: Vata, Pitta, and Kapha are the three doshas.",
            "Domain: Ayurveda | Source: Uploaded #99\nTranslation: Random herbal leaf without mention of doshas."
        ],
        metadatas=[
            {
                "passage_id": "REF-AYUR-CANONICAL",
                "source_type": "reference_canonical",
                "domain": "Ayurveda",
                "source": "Ashtanga Hridaya",
                "reference": "1.6",
                "verification_status": "verified_canonical",
                "sanskrit": "वायुः पित्तं कफश्चेति त्रयो दोषाः समासतः",
                "translation": "Vata, Pitta, and Kapha are the three doshas.",
                "keywords": "Vata, Pitta, Kapha, Tridosha"
            },
            {
                "passage_id": "UPLOAD-M99-C1",
                "source_type": "uploaded_manuscript",
                "domain": "Ayurveda",
                "source": "Manuscript #99",
                "reference": "Chunk 1",
                "verification_status": "uploaded_manuscript",
                "sanskrit": "पत्रं पुष्पम्",
                "translation": "A leaf and a flower.",
                "keywords": "leaf, flower"
            }
        ],
        embeddings=mock_model.encode([
            "Vata Pitta Kapha doshas Ashtanga Hridaya",
            "xyz unrelated forest flower leaf"
        ]).tolist()
    )

    passages = retrieve_relevant_passages(
        query="Vata Pitta Kapha doshas Ashtanga Hridaya",
        domain="Ayurveda",
        top_k=2,
        min_similarity=0.0,
        collection=col,
        model=mock_model
    )

    assert len(passages) >= 1
    # Authoritative canonical passage must rank first
    assert passages[0]["passage_id"] == "REF-AYUR-CANONICAL"


# ---------------------------------------------------------------------------
# TEST 14: Cross-Domain Isolation with domain='all'
# ---------------------------------------------------------------------------
def test_cross_domain_isolation_with_domain_all(temp_chroma_dir, mock_model, sample_test_corpus):
    """Verifies that query domain routing scopes retrieval to the intended domain when domain='all'."""
    col = get_chroma_collection(persist_dir=temp_chroma_dir, collection_name="test_col_isolation")
    index_corpus(corpus_records=sample_test_corpus, collection=col, model=mock_model)

    # Ayurveda question with domain='all'
    ayur_passages = retrieve_relevant_passages(
        query="What are the three doshas in Ayurveda?",
        domain="all",
        top_k=3,
        min_similarity=0.0,
        collection=col,
        model=mock_model
    )
    assert len(ayur_passages) >= 1
    for p in ayur_passages:
        assert p["domain"] == "Ayurveda"
        assert p["domain"] != "Astronomy"
        assert p["domain"] != "Philosophy"

    # Astronomy question with domain='all'
    astro_passages = retrieve_relevant_passages(
        query="Explain the planetary calculations and sine table in the manuscript.",
        domain="all",
        top_k=3,
        min_similarity=0.0,
        collection=col,
        model=mock_model
    )
    assert len(astro_passages) >= 1
    for p in astro_passages:
        assert p["domain"] in ("Astronomy", "Mathematics", "Astronomy & Mathematics")
        assert p["domain"] != "Ayurveda"


# ---------------------------------------------------------------------------
# TEST 15: Uploaded Manuscript Retrieval by Semantic Query
# ---------------------------------------------------------------------------
def test_uploaded_manuscript_retrieval(temp_chroma_dir, mock_model):
    """Verifies that uploaded manuscript chunks are retrievable by content-specific queries."""
    col = get_chroma_collection(persist_dir=temp_chroma_dir, collection_name="test_col_upload_retrieval")

    col.add(
        ids=["UPLOAD-M4-C3", "REF-AYUR-GENERIC"],
        documents=[
            "Domain: Ayurveda | Source: Manuscript #4\nTranslation: Kunduru resin is sharp, pacifier of phlegm.",
            "Domain: Ayurveda | Source: General Reference\nTranslation: General guidelines on seasons."
        ],
        metadatas=[
            {
                "passage_id": "UPLOAD-M4-C3",
                "manuscript_id": 4,
                "source_type": "uploaded_manuscript",
                "domain": "Ayurveda",
                "source": "Manuscript #4",
                "reference": "Chunk 3",
                "verification_status": "uploaded_manuscript",
                "sanskrit": "कुन्दुरुः कफशामकः",
                "translation": "Kunduru resin is sharp, pacifier of phlegm.",
                "keywords": "Kunduru, resin, phlegm"
            },
            {
                "passage_id": "REF-AYUR-GENERIC",
                "source_type": "reference_canonical",
                "domain": "Ayurveda",
                "source": "General Reference",
                "reference": "1.1",
                "verification_status": "verified_canonical",
                "sanskrit": "ऋतुचर्या",
                "translation": "General guidelines on seasons.",
                "keywords": "seasons, ritu"
            }
        ],
        embeddings=mock_model.encode([
            "Kunduru resin is sharp, pacifier of phlegm.",
            "General guidelines on seasons."
        ]).tolist()
    )

    passages = retrieve_relevant_passages(
        query="What are the properties of Kunduru resin in the manuscript?",
        domain="Ayurveda",
        top_k=2,
        min_similarity=0.0,
        collection=col,
        model=mock_model
    )

    assert len(passages) >= 1
    top = passages[0]
    assert top["passage_id"] == "UPLOAD-M4-C3"
    assert top["is_uploaded"] is True
    assert top["source_type"] == "uploaded_manuscript"
    assert top["manuscript_id"] == 4



