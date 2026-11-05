"""
SansKrypt RAG Service Module
Provides natural language query processing, vector embeddings using EmbeddingGemma 2,
dense semantic retrieval via ChromaDB, hybrid lexical re-ranking, and evidence-grounded
RAG answer synthesis using Google Gemini.
"""
import os
import sys
import re
import json
import logging
from pathlib import Path
from typing import List, Dict, Any, Optional, Tuple

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")

logger = logging.getLogger("SansKrypt.RAG")

BASE_DIR = Path(__file__).resolve().parent
DEFAULT_CORPUS_PATH = os.path.join(BASE_DIR, "data", "canonical_corpus.json")
CHROMA_PERSIST_DIR = os.path.join(BASE_DIR, "chroma_db")
COLLECTION_NAME = "sanskrypt_rag_collection"

# Auto-load .env file if present
_env_path = BASE_DIR / ".env"
if _env_path.is_file():
    try:
        with open(_env_path, "r", encoding="utf-8") as _f:
            for _line in _f:
                _line = _line.strip()
                if _line and not _line.startswith("#") and "=" in _line:
                    _k, _v = _line.split("=", 1)
                    if _k.strip() not in os.environ:
                        os.environ[_k.strip()] = _v.strip().strip("\"'")
    except Exception:
        pass

# Embedding model configurations
PRIMARY_MODEL_NAME = os.getenv("EMBEDDING_MODEL", "sentence-transformers/all-MiniLM-L6-v2")
FALLBACK_MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"

# Task-specific instruction prefix for asymmetric models (e.g., EmbeddingGemma)
QUERY_INSTRUCTION_PREFIX = (
    "task: Given a web search query, retrieve relevant passages that answer the query\nquery: "
)

def format_query_for_embedding(clean_query: str, model_name: Optional[str] = None) -> str:
    """
    Formats the user query for dense vector encoding.
    Only prepends task instruction prefix if the active embedding model is explicitly
    an asymmetric model like Google EmbeddingGemma. For symmetric sentence transformers
    like all-MiniLM-L6-v2, returns the clean query directly to prevent semantic vector drift.
    """
    target = (model_name or get_active_embedding_model_name() or "").lower()
    if "gemma" in target:
        return f"{QUERY_INSTRUCTION_PREFIX}{clean_query}"
    return clean_query

# Minimum cosine similarity threshold to consider evidence relevant
MIN_EVIDENCE_SIMILARITY = float(os.getenv("MIN_EVIDENCE_SIMILARITY", "0.20"))

# Singleton holder for lazy model loading
_embedding_model = None
_active_model_name = None
_chroma_client = None
_chroma_collection = None
_current_persist_dir = None


def get_active_embedding_model_name() -> str:
    """Returns the name of the currently active embedding model for diagnostics."""
    global _active_model_name
    if _active_model_name:
        return _active_model_name
    return PRIMARY_MODEL_NAME


def get_embedding_model_status() -> Dict[str, Any]:
    """
    Returns diagnostic details about configured, active, and fallback embedding models.
    Does not trigger model downloads or heavy imports.
    """
    global _active_model_name

    index_dim = 384
    index_count = 0
    try:
        col = get_chroma_collection()
        index_count = col.count()
        if index_count > 0:
            sample = col.get(limit=1, include=["embeddings"])
            if sample and sample.get("embeddings") and len(sample["embeddings"]) > 0:
                index_dim = len(sample["embeddings"][0])
    except Exception as e:
        logger.debug(f"Could not inspect ChromaDB collection status: {e}")

    active = _active_model_name or PRIMARY_MODEL_NAME
    is_fallback = (
        PRIMARY_MODEL_NAME != FALLBACK_MODEL_NAME
        and ("fallback" in str(active).lower() or active == FALLBACK_MODEL_NAME)
    )

    fallback_reason = (
        f"Primary model '{PRIMARY_MODEL_NAME}' is unsupported by the current 'transformers' environment. "
        f"Dense retrieval is active via '{FALLBACK_MODEL_NAME}' matching the 384-dimensional persistent ChromaDB index ({index_count} passages)."
    ) if is_fallback else None

    return {
        "configured_primary": PRIMARY_MODEL_NAME,
        "fallback_model": FALLBACK_MODEL_NAME,
        "active_model": active,
        "is_fallback": is_fallback,
        "index_dimension": index_dim,
        "indexed_passages": index_count,
        "fallback_reason": fallback_reason
    }



def get_embedding_model(model_name: Optional[str] = None):
    """
    Lazily loads the SentenceTransformer embedding model into CPU memory.
    Never downloads or loads models during application import.
    """
    global _embedding_model, _active_model_name
    if _embedding_model is not None:
        return _embedding_model

    from sentence_transformers import SentenceTransformer

    target_name = model_name or PRIMARY_MODEL_NAME
    logger.info(f"Loading embedding model '{target_name}' on CPU...")

    try:
        _embedding_model = SentenceTransformer(target_name, device="cpu")
        _active_model_name = target_name
        logger.info(f"Successfully loaded embedding model: {target_name}")
        return _embedding_model
    except Exception as primary_err:
        logger.warning(
            f"Failed to load primary embedding model '{target_name}': {primary_err}. "
            f"Attempting local fallback model '{FALLBACK_MODEL_NAME}'..."
        )
        try:
            _embedding_model = SentenceTransformer(FALLBACK_MODEL_NAME, device="cpu")
            _active_model_name = f"{FALLBACK_MODEL_NAME} (Fallback; {PRIMARY_MODEL_NAME} architecture unsupported by current transformers environment)"
            logger.info(f"Loaded fallback embedding model: {FALLBACK_MODEL_NAME}")
            return _embedding_model
        except Exception as fallback_err:
            logger.error(f"Fallback embedding model failed: {fallback_err}")
            raise RuntimeError(
                f"Could not load embedding model '{target_name}' or fallback '{FALLBACK_MODEL_NAME}': {fallback_err}"
            )


def set_mock_embedding_model(mock_model, name: str = "MockSentenceTransformer"):
    """Allows test fixtures to inject a mock embedding model without internet access."""
    global _embedding_model, _active_model_name
    _embedding_model = mock_model
    _active_model_name = name


def get_chroma_collection(persist_dir: Optional[str] = None, collection_name: Optional[str] = None):
    """
    Initializes or retrieves the persistent ChromaDB collection.
    Uses a separate folder and collection, leaving the SQLite database completely untouched.
    """
    global _chroma_client, _chroma_collection, _current_persist_dir
    target_dir = os.path.abspath(persist_dir or CHROMA_PERSIST_DIR)
    target_col = collection_name or COLLECTION_NAME

    # If the target directory changed (e.g. across test fixtures), reset previous client
    if _chroma_client is not None and _current_persist_dir != target_dir:
        reset_chroma_client()

    if _chroma_client is None:
        import chromadb
        from chromadb.config import Settings
        os.makedirs(target_dir, exist_ok=True)
        try:
            _chroma_client = chromadb.PersistentClient(
                path=target_dir,
                settings=Settings(anonymized_telemetry=False)
            )
            _current_persist_dir = target_dir
        except Exception as e:
            _chroma_client = None
            _current_persist_dir = None
            logger.error(f"Failed to initialize ChromaDB PersistentClient at '{target_dir}': {e}", exc_info=True)
            raise

    _chroma_collection = _chroma_client.get_or_create_collection(
        name=target_col,
        metadata={"hnsw:space": "cosine"}
    )
    return _chroma_collection


def reset_chroma_client():
    """Resets client reference and releases underlying SQLite file locks cleanly."""
    global _chroma_client, _chroma_collection, _current_persist_dir
    if _chroma_client is not None:
        try:
            _chroma_client.close()
        except Exception:
            pass
    _chroma_client = None
    _chroma_collection = None
    _current_persist_dir = None


def load_corpus(corpus_path: Optional[str] = None) -> List[Dict[str, Any]]:
    """
    Loads and validates manuscript passages from the canonical corpus JSON file.
    Ensures every record contains required fields without fabricating data.
    """
    path = corpus_path or DEFAULT_CORPUS_PATH
    if not os.path.exists(path):
        logger.warning(f"Corpus file not found at path: {path}")
        return []

    try:
        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)
    except Exception as e:
        logger.error(f"Failed to parse corpus file {path}: {e}")
        return []

    valid_records = []
    required_keys = ["passage_id", "sanskrit", "translation", "domain"]

    for item in data:
        if not isinstance(item, dict):
            continue
        if all(item.get(k) and str(item[k]).strip() for k in required_keys):
            valid_records.append({
                "passage_id": str(item["passage_id"]).strip(),
                "sanskrit": str(item["sanskrit"]).strip(),
                "translation": str(item["translation"]).strip(),
                "domain": str(item["domain"]).strip(),
                "source": str(item.get("source", "Unknown Classical Source")).strip(),
                "reference": str(item.get("reference", "Unspecified Verse")).strip(),
                "verification_status": str(item.get("verification_status", "unverified")).strip(),
                "keywords": str(item.get("keywords", "")).strip()
            })
        else:
            logger.warning(f"Skipping incomplete corpus item missing required keys: {item.get('passage_id')}")

    logger.info(f"Loaded {len(valid_records)} validated passages from {path}")
    return valid_records


def build_passage_text_for_embedding(passage: Dict[str, Any]) -> str:
    """
    Formats passage metadata and content into a structured textual representation
    for embedding generation, allowing cross-lingual semantic matching.
    """
    return (
        f"Domain: {passage.get('domain', '')} | Source: {passage.get('source', '')} ({passage.get('reference', '')})\n"
        f"Translation: {passage.get('translation', '')}\n"
        f"Sanskrit Text: {passage.get('sanskrit', '')}\n"
        f"Keywords: {passage.get('keywords', '')}"
    )


def index_corpus(
    corpus_records: Optional[List[Dict[str, Any]]] = None,
    collection=None,
    model=None,
    persist_dir: Optional[str] = None
) -> int:
    """
    Generates and persists embeddings for the provided corpus records.
    Uses ChromaDB upsert to ensure stable updates without creating duplicate vectors.
    """
    records = corpus_records if corpus_records is not None else load_corpus()
    if not records:
        logger.warning("No records available to index.")
        return 0

    col = collection if collection is not None else get_chroma_collection(persist_dir=persist_dir)
    emb_model = model if model is not None else get_embedding_model()

    ids = []
    documents = []
    metadatas = []
    texts_to_embed = []

    for r in records:
        pid = r["passage_id"]
        doc_text = build_passage_text_for_embedding(r)
        ids.append(pid)
        documents.append(doc_text)
        metadatas.append({
            "passage_id": pid,
            "domain": r["domain"],
            "source": r["source"],
            "reference": r["reference"],
            "verification_status": r["verification_status"],
            "sanskrit": r["sanskrit"],
            "translation": r["translation"],
            "keywords": r["keywords"]
        })
        texts_to_embed.append(doc_text)

    logger.info(f"Generating embeddings for {len(texts_to_embed)} passages...")
    vectors = emb_model.encode(texts_to_embed, normalize_embeddings=True, show_progress_bar=False)

    # Convert to standard Python lists of floats for ChromaDB
    embeddings_list = [v.tolist() if hasattr(v, "tolist") else list(v) for v in vectors]

    col.upsert(
        ids=ids,
        documents=documents,
        metadatas=metadatas,
        embeddings=embeddings_list
    )

    logger.info(f"Successfully upserted {len(ids)} passage vectors into ChromaDB (total count: {col.count()}).")
    return len(ids)


def chunk_manuscript_text(
    sanskrit_text: str,
    translated_text: str,
    max_chunk_chars: int = 400
) -> List[Dict[str, Any]]:
    """
    Creates meaningful chunks from manuscript Sanskrit text and English translation.
    Preserves verse integrity (using danda '॥' or newline boundaries).
    """
    s_clean = (sanskrit_text or "").strip()
    t_clean = (translated_text or "").strip()

    if not s_clean and not t_clean:
        return []

    # If text is concise or single verse, return as a single cohesive chunk
    if len(s_clean) <= max_chunk_chars:
        return [{
            "chunk_index": 1,
            "sanskrit": s_clean,
            "translation": t_clean
        }]

    # Split by double danda '॥' or double newline
    raw_verses = [v.strip() for v in re.split(r"(?<=॥)|(?:\r?\n\s*\r?\n)", s_clean) if v.strip()]
    if len(raw_verses) <= 1:
        raw_verses = [v.strip() for v in s_clean.splitlines() if v.strip()]

    if len(raw_verses) <= 1:
        return [{
            "chunk_index": 1,
            "sanskrit": s_clean,
            "translation": t_clean
        }]

    t_lines = [tl.strip() for tl in re.split(r"(?<=[.!?])\s+|\r?\n+", t_clean) if tl.strip()]

    chunks = []
    current_s = []
    current_chars = 0

    for verse in raw_verses:
        if current_chars + len(verse) > max_chunk_chars and current_s:
            chunks.append(" ".join(current_s))
            current_s = [verse]
            current_chars = len(verse)
        else:
            current_s.append(verse)
            current_chars += len(verse)

    if current_s:
        chunks.append(" ".join(current_s))

    result = []
    num_chunks = len(chunks)
    for idx, s_chunk in enumerate(chunks, start=1):
        if len(t_lines) == num_chunks:
            chunk_t = t_lines[idx - 1]
        elif t_lines:
            start_i = (idx - 1) * len(t_lines) // num_chunks
            end_i = idx * len(t_lines) // num_chunks
            chunk_t = " ".join(t_lines[start_i:max(end_i, start_i + 1)])
        else:
            chunk_t = t_clean

        result.append({
            "chunk_index": idx,
            "sanskrit": s_chunk,
            "translation": chunk_t
        })

    return result


def delete_manuscript_from_index(manuscript_id: int, collection=None) -> int:
    """Removes all indexed chunks for a manuscript from ChromaDB."""
    try:
        col = collection if collection is not None else get_chroma_collection()
        res = col.get(where={"manuscript_id": int(manuscript_id)})
        if res and res.get("ids"):
            col.delete(ids=res["ids"])
            logger.info(f"Deleted {len(res['ids'])} chunks from ChromaDB for manuscript #{manuscript_id}")
            return len(res["ids"])
    except Exception as e:
        logger.warning(f"Could not delete ChromaDB chunks for manuscript #{manuscript_id}: {e}")
    return 0


def index_manuscript(
    manuscript_id: int,
    sanskrit_text: str,
    translation: str,
    domain: str,
    title: Optional[str] = None,
    status: str = "completed",
    ocr_confidence: float = 0.0,
    keywords: Optional[str] = None,
    collection=None,
    model=None
) -> int:
    """
    Automatically chunks, embeds, and indexes an uploaded manuscript into ChromaDB.
    Attaches provenance metadata: manuscript_id, source_type='uploaded_manuscript', domain, etc.
    Guarantees idempotency by clearing any previous chunks for this manuscript_id.
    """
    if not (sanskrit_text or "").strip() and not (translation or "").strip():
        logger.warning(f"No text content to index for manuscript #{manuscript_id}")
        return 0

    col = collection if collection is not None else get_chroma_collection()
    emb_model = model if model is not None else get_embedding_model()

    # Ensure idempotency: remove any prior chunks for this manuscript_id
    delete_manuscript_from_index(manuscript_id, collection=col)

    chunks = chunk_manuscript_text(sanskrit_text, translation)
    if not chunks:
        return 0

    from explain_classify import normalize_to_three_domains
    normalized_domain = normalize_to_three_domains(domain)

    ids = []
    documents = []
    metadatas = []
    texts_to_embed = []

    clean_title = title.strip() if title and title.strip() else f"Uploaded Manuscript #{manuscript_id}"

    for chunk in chunks:
        c_idx = chunk["chunk_index"]
        pid = f"UPLOAD-M{manuscript_id}-C{c_idx}"

        doc_text = (
            f"Domain: {normalized_domain} | Source: {clean_title} (Manuscript #{manuscript_id}, Chunk {c_idx})\n"
            f"Translation: {chunk['translation']}\n"
            f"Sanskrit Text: {chunk['sanskrit']}\n"
            f"Keywords: {keywords or ''}"
        )

        ids.append(pid)
        documents.append(doc_text)
        metadatas.append({
            "passage_id": pid,
            "manuscript_id": int(manuscript_id),
            "source_type": "uploaded_manuscript",
            "domain": normalized_domain,
            "source": clean_title,
            "reference": f"Manuscript #{manuscript_id}, Chunk {c_idx} (OCR: {round(ocr_confidence, 1)}%)",
            "verification_status": "uploaded_manuscript" if status == "completed" else "needs_review",
            "sanskrit": chunk["sanskrit"],
            "translation": chunk["translation"],
            "keywords": str(keywords or "")
        })
        texts_to_embed.append(doc_text)

    vectors = emb_model.encode(texts_to_embed, normalize_embeddings=True, show_progress_bar=False)
    embeddings_list = [v.tolist() if hasattr(v, "tolist") else list(v) for v in vectors]

    col.upsert(
        ids=ids,
        documents=documents,
        metadatas=metadatas,
        embeddings=embeddings_list
    )
    logger.info(f"Indexed {len(ids)} chunk(s) into ChromaDB for manuscript #{manuscript_id}.")
    return len(ids)


ENGLISH_STOP_WORDS = {
    "a", "about", "above", "after", "again", "against", "all", "am", "an", "and", "any", "are",
    "aren't", "as", "at", "be", "because", "been", "before", "being", "below", "between", "both",
    "but", "by", "can't", "cannot", "could", "couldn't", "did", "didn't", "do", "does", "doesn't",
    "doing", "don't", "down", "during", "each", "few", "for", "from", "further", "had", "hadn't",
    "has", "hasn't", "have", "haven't", "having", "he", "he'd", "he'll", "he's", "her", "here",
    "here's", "hers", "herself", "him", "himself", "his", "how", "how's", "i", "i'd", "i'll",
    "i'm", "i've", "if", "in", "into", "is", "isn't", "it", "it's", "its", "itself", "let's",
    "me", "more", "most", "mustn't", "my", "myself", "no", "nor", "not", "of", "off", "on",
    "once", "only", "or", "other", "ought", "our", "ours", "ourselves", "out", "over", "own",
    "same", "shan't", "she", "she'd", "she'll", "she's", "should", "shouldn't", "so", "some",
    "such", "than", "that", "that's", "the", "their", "theirs", "them", "themselves", "then",
    "there", "there's", "these", "they", "they'd", "they'll", "they're", "they've", "this",
    "those", "through", "to", "too", "under", "until", "up", "very", "was", "wasn't", "we",
    "we'd", "we'll", "we're", "we've", "were", "weren't", "what", "what's", "when", "when's",
    "where", "where's", "which", "while", "who", "who's", "whom", "why", "why's", "with",
    "won't", "would", "wouldn't", "you", "you'd", "you'll", "you're", "you've", "your",
    "yours", "yourself", "yourselves"
}


# ---------------------------------------------------------------------------
# Domain Terminology Lexicons & Query Intent Understanding
# ---------------------------------------------------------------------------

DOMAIN_INTENT_LEXICON = {
    "Ayurveda": {
        "tokens": {
            "dosha", "doshas", "vata", "pitta", "kapha", "tridosha", "tridoshas",
            "ayurveda", "ayurvedic", "svastha", "swastha", "health", "disease",
            "medicine", "cure", "healing", "humor", "humors", "bodily",
            "sushruta", "charaka", "vagbhata", "ashtanga", "hridaya", "samhita",
            "rasayana", "elixir", "dhatu", "dhatus", "mala", "agni", "phlegm", "bile", "wind",
            "herbs", "herbal", "medicinal", "treatment", "nighantu", "guggulu", "resin",
            "kunduru", "sallaki", "haritaki", "triphala", "bhasma", "pathya", "chikitsa"
        },
        "devanagari": {
            "दोष", "दोषाः", "वात", "पित्त", "कफ", "त्रिदोष", "आयुर्वेद", "स्वास्थ्य",
            "चरक", "सुश्रुत", "वाग्भट", "अष्टाङ्ग", "हृदय", "संहिता", "रसायन", "अग्नि",
            "धातु", "मल", "भैषज्य", "चिकित्सा", "निघण्टु", "गुग्गुलु", "कुन्दुरु", "पथ्य"
        }
    },
    "Astronomy & Mathematics": {
        "tokens": {
            "astronomy", "astronomical", "mathematics", "mathematical", "math",
            "planet", "planets", "planetary", "orbit", "orbits", "eclipse",
            "celestial", "star", "stars", "sun", "moon", "calculation", "calculations",
            "ganita", "sine", "sines", "cosine", "jya", "kotijya", "trigonometry",
            "zero", "shunya", "aryabhata", "aryabhatiya", "surya", "siddhanta",
            "brahmagupta", "bhaskara", "nakshatra", "graha", "grahas", "rashi",
            "lagna", "equinox", "solstice", "epicycle", "retrograde", "horoscope",
            "astrology", "astrological", "drekkana", "navamsha", "hora", "house",
            "houses", "ascendant", "longitude", "latitude", "yuga", "kalpa"
        },
        "devanagari": {
            "गणित", "ज्योतिष", "ग्रह", "ग्रहाः", "नक्षत्र", "सूर्यसिद्धान्त", "आर्यभट",
            "आर्यभटीय", "शून्य", "ज्या", "कोटिज्या", "राशि", "लग्न", "ग्रहण", "होरा",
            "द्रेष्काण", "नवांश", "भास्कर", "ब्रह्मगुप्त", "सूर्य", "चन्द्र"
        }
    },
    "Philosophy": {
        "tokens": {
            "philosophy", "philosophical", "atman", "brahman", "soul", "self",
            "karma", "dharma", "moksha", "liberation", "gita", "bhagavad",
            "upanishad", "upanishads", "darshana", "samkhya", "vedanta", "yoga",
            "maya", "illusion", "purusha", "prakriti", "avidya", "jiva",
            "satyameva", "jayate", "truth", "righteousness", "rebirth",
            "consciousness", "meditation", "krishna", "arjuna", "brahma",
            "advaita", "buddhi", "manas", "adhyatma", "jnana", "bhakti"
        },
        "devanagari": {
            "आत्मन्", "आत्मा", "ब्रह्मन्", "ब्रह्म", "कर्म", "धर्म", "मोक्ष",
            "गीता", "भगवद्", "उपनिषद्", "दर्शन", "सांख्य", "वेदान्त", "योग",
            "माया", "पुरुष", "प्रकृति", "अविद्या", "जीव", "सत्यमेव", "जयते",
            "ज्ञान", "भक्ति", "अध्यात्म"
        }
    }
}


def detect_query_domains(query: str) -> Dict[str, Any]:
    """
    Analyzes the user's natural language or Sanskrit query to identify its intended
    knowledge domain(s) before retrieval, preventing cross-domain noise.
    Supports single-domain routing, genuine cross-domain intent, and unconstrained fallback.
    """
    clean_q = query.strip()
    if not clean_q:
        return {"primary_domain": "all", "target_domains": ["all"], "confidence": 0.0, "is_mixed": False}

    q_lower = clean_q.lower()
    words = set(re.findall(r"\w+", q_lower))

    scores = {}
    for dom, lex in DOMAIN_INTENT_LEXICON.items():
        score = 0.0
        for tok in lex["tokens"]:
            if tok in words:
                score += 2.0
            elif len(tok) >= 5 and tok in q_lower:
                score += 1.0
        for dev in lex["devanagari"]:
            if dev in clean_q:
                score += 3.0
        scores[dom] = score

    sorted_domains = sorted(scores.items(), key=lambda x: x[1], reverse=True)
    top_dom, top_score = sorted_domains[0]
    second_dom, second_score = sorted_domains[1]

    if top_score < 1.5:
        return {
            "primary_domain": "all",
            "target_domains": ["all"],
            "confidence": 0.0,
            "is_mixed": False,
            "scores": scores
        }

    # Detect genuine cross-domain queries (e.g. Samkhya & Ayurveda / Yoga & Health)
    if second_score >= 3.0 and second_score >= 0.55 * top_score:
        return {
            "primary_domain": top_dom,
            "target_domains": [top_dom, second_dom],
            "confidence": min(1.0, top_score / 4.0),
            "is_mixed": True,
            "scores": scores
        }

    return {
        "primary_domain": top_dom,
        "target_domains": [top_dom],
        "confidence": min(1.0, top_score / 3.0),
        "is_mixed": False,
        "scores": scores
    }


def get_existing_manuscript_ids() -> Optional[set]:
    """
    Retrieves set of valid manuscript IDs from database to prevent orphan vector citations.
    Fails safely if running standalone without Flask context or database.
    """
    try:
        from database import db, Manuscript
        return set(r[0] for r in db.session.query(Manuscript.id).all())
    except Exception:
        return None


def calculate_keyword_overlap(query: str, passage_meta: Dict[str, Any]) -> float:
    """
    Calculates lexical overlap ratio between the query and passage text/keywords
    to enable hybrid retrieval alongside dense semantic similarity.
    Filters common English stop words and transliterates Romanized Sanskrit query terms
    into Devanagari to match directly against Sanskrit text.
    Handles transliteration errors gracefully without breaking normal English queries.
    """
    clean_q = re.findall(r"\w+", query.lower())
    if not clean_q:
        return 0.0

    content_tokens = [t for t in clean_q if t not in ENGLISH_STOP_WORDS and len(t) > 2]
    eval_tokens = content_tokens if content_tokens else [t for t in clean_q if len(t) > 2]
    if not eval_tokens:
        eval_tokens = clean_q

    unique_tokens = set(eval_tokens)
    haystack = (
        f"{passage_meta.get('translation', '')} "
        f"{passage_meta.get('keywords', '')} "
        f"{passage_meta.get('source', '')} "
        f"{passage_meta.get('sanskrit', '')}"
    ).lower()
    sanskrit_raw = passage_meta.get("sanskrit", "")

    matches = 0.0
    for token in unique_tokens:
        if token in haystack:
            matches += 1.0
        elif re.match(r"^[a-zA-Z]+$", token) and len(token) >= 4 and sanskrit_raw:
            try:
                from indic_transliteration import sanscript
                for scheme in (sanscript.ITRANS, sanscript.HK):
                    try:
                        dev = sanscript.transliterate(token, scheme, sanscript.DEVANAGARI)
                        if dev and len(dev) >= 3 and (dev in sanskrit_raw or dev[:3] in sanskrit_raw):
                            matches += 1.5
                            break
                    except Exception:
                        continue
            except Exception as trans_err:
                logger.debug(f"Transliteration lookup failed for token '{token}': {trans_err}")

    return min(matches / max(len(unique_tokens), 1), 1.0)


def retrieve_relevant_passages(
    query: str,
    domain: Optional[str] = None,
    source_filter: Optional[str] = "all",
    manuscript_id: Optional[int] = None,
    top_k: int = 4,
    min_similarity: float = MIN_EVIDENCE_SIMILARITY,
    collection=None,
    model=None
) -> List[Dict[str, Any]]:
    """
    Embeds the user query and retrieves the most relevant manuscript passages using
    dense cosine similarity and hybrid scoring across both uploaded manuscripts
    and reference knowledge.
    Applies domain understanding to filter out unrelated domains while ensuring
    relevant uploaded manuscripts and canonical treatises are properly ranked.
    """
    if not query or not query.strip():
        return []

    clean_query = query.strip()
    col = collection if collection is not None else get_chroma_collection()

    if col.count() == 0:
        logger.warning("Vector collection is empty. Please run index_corpus() first.")
        return []

    emb_model = model if model is not None else get_embedding_model()

    # 1. Determine effective query domains (user-specified or auto-detected)
    detected = detect_query_domains(clean_query)
    effective_domains = []

    if domain and domain.lower() != "all":
        effective_domains = [domain.strip()]
    elif detected["target_domains"] != ["all"]:
        effective_domains = detected["target_domains"]

    # Expand domain aliases
    expanded_domains = []
    for d in effective_domains:
        if d in ("Astronomy & Mathematics", "Astronomy and Mathematics", "Astronomy", "Mathematics"):
            expanded_domains.extend(["Astronomy", "Mathematics", "Astronomy & Mathematics"])
        else:
            expanded_domains.append(d)
    expanded_domains = list(set(expanded_domains))

    # Apply task-specific instruction prefix only if asymmetric model (e.g. EmbeddingGemma)
    active_name = getattr(emb_model, "_model_name", None) or get_active_embedding_model_name()
    query_text = format_query_for_embedding(clean_query, active_name)
    query_vector = emb_model.encode([query_text], normalize_embeddings=True, show_progress_bar=False)
    query_vector_list = query_vector[0].tolist() if hasattr(query_vector[0], "tolist") else list(query_vector[0])

    filter_clauses = []

    # Domain filter: allow target domains plus unassigned/uncertain uploaded manuscripts for content evaluation
    if expanded_domains:
        allowed_domains = expanded_domains + ["General/Other", "Uncertain", "General"]
        filter_clauses.append({"domain": {"$in": allowed_domains}})

    # Source filter ('uploaded' vs 'reference' vs 'all')
    if source_filter and source_filter.lower() in ("uploaded", "uploaded_manuscripts", "manuscripts"):
        filter_clauses.append({"source_type": "uploaded_manuscript"})
    elif source_filter and source_filter.lower() in ("reference", "canonical", "reference_knowledge"):
        filter_clauses.append({"source_type": {"$ne": "uploaded_manuscript"}})

    # Manuscript-specific filter
    if manuscript_id is not None:
        try:
            filter_clauses.append({"manuscript_id": int(manuscript_id)})
        except (ValueError, TypeError):
            pass

    if len(filter_clauses) == 1:
        where_filter = filter_clauses[0]
    elif len(filter_clauses) > 1:
        where_filter = {"$and": filter_clauses}
    else:
        where_filter = None

    # Query ChromaDB (fetch candidate pool for hybrid re-ranking, expanded to 30 while respecting collection size)
    fetch_n = min(max(top_k * 8, 30), col.count())
    try:
        results = col.query(
            query_embeddings=[query_vector_list],
            n_results=fetch_n,
            where=where_filter
        )
    except Exception as q_err:
        logger.warning(f"ChromaDB query with where={where_filter} failed: {q_err}. Falling back without where filter.")
        results = col.query(
            query_embeddings=[query_vector_list],
            n_results=fetch_n
        )

    if not results or not results.get("ids") or not results["ids"][0]:
        return []

    candidates = []
    ids = results["ids"][0]
    distances = results["distances"][0] if "distances" in results else [0.0] * len(ids)
    metadatas = results["metadatas"][0] if "metadatas" in results else [{}] * len(ids)

    # Optional orphan check to ensure uploaded manuscripts exist in DB
    existing_m_ids = get_existing_manuscript_ids()

    for pid, dist, meta in zip(ids, distances, metadatas):
        # Convert cosine distance to cosine similarity (1.0 - distance)
        cosine_sim = max(0.0, min(1.0, 1.0 - float(dist)))
        kw_score = calculate_keyword_overlap(clean_query, meta)

        is_uploaded = (meta.get("source_type") == "uploaded_manuscript")

        # Exclude orphan vectors of deleted manuscripts
        manuscript_id_val = meta.get("manuscript_id")
        if is_uploaded and manuscript_id_val is not None and existing_m_ids is not None:
            try:
                if int(manuscript_id_val) not in existing_m_ids:
                    logger.debug(f"Skipping orphan vector {pid} for deleted manuscript #{manuscript_id_val}")
                    continue
            except (ValueError, TypeError):
                pass

        # Domain boundary protection: eliminate candidates with explicitly conflicting domains
        cand_domain = meta.get("domain", "")
        if expanded_domains and cand_domain not in expanded_domains and cand_domain not in ("General/Other", "Uncertain", "General"):
            continue

        # Calibrated Hybrid Scoring:
        # 1. Semantic similarity dominates (85% weight)
        # 2. Keyword overlap provides supplementary precision (15% weight), but only if semantic similarity >= 0.35
        # 3. Relevance-conditioned upload boost: modest +0.03 only when semantic relevance is genuinely high (>= 0.48)
        effective_kw = kw_score if cosine_sim >= 0.35 else 0.0
        upload_boost = 0.03 if (is_uploaded and cosine_sim >= 0.48) else 0.0
        hybrid_score = round(min(1.0, 0.85 * cosine_sim + 0.15 * effective_kw + upload_boost), 4)

        candidates.append({
            "passage_id": pid,
            "manuscript_id": meta.get("manuscript_id"),
            "source_type": meta.get("source_type", "reference"),
            "is_uploaded": is_uploaded,
            "sanskrit": meta.get("sanskrit", ""),
            "translation": meta.get("translation", ""),
            "domain": meta.get("domain", ""),
            "source": meta.get("source", "Unknown Source"),
            "reference": meta.get("reference", "Unspecified"),
            "verification_status": meta.get("verification_status", "unverified"),
            "similarity_score": round(cosine_sim, 4),
            "hybrid_score": hybrid_score
        })

    # Sort primarily by hybrid score, favoring uploaded manuscripts on ties
    candidates.sort(key=lambda x: (x["hybrid_score"], 1 if x.get("is_uploaded") else 0), reverse=True)

    # Filter by minimum similarity score
    valid_candidates = [c for c in candidates if c["similarity_score"] >= min_similarity or c["hybrid_score"] >= min_similarity]

    # Balanced evidence selection when searching across all sources (source_filter == 'all'):
    # Ensures both top uploaded manuscript evidence and authoritative reference treatises are represented
    # while preserving strict hybrid score ranking
    if source_filter == "all" and len(valid_candidates) > top_k:
        uploaded_cands = [c for c in valid_candidates if c["is_uploaded"] and c["similarity_score"] >= 0.45]
        ref_cands = [c for c in valid_candidates if not c["is_uploaded"] and c["similarity_score"] >= 0.40]

        if uploaded_cands and ref_cands:
            selected_set = {uploaded_cands[0]["passage_id"], ref_cands[0]["passage_id"]}
            selected = [c for c in valid_candidates if c["passage_id"] in selected_set]
            for c in valid_candidates:
                if len(selected) >= top_k:
                    break
                if c["passage_id"] not in selected_set:
                    selected.append(c)
                    selected_set.add(c["passage_id"])
            # Sort selected items strictly by hybrid score
            selected.sort(key=lambda x: (x["hybrid_score"], 1 if x.get("is_uploaded") else 0), reverse=True)
        else:
            selected = valid_candidates[:top_k]
    else:
        selected = valid_candidates[:top_k]

    return selected


def build_evidence_context(passages: List[Dict[str, Any]]) -> str:
    """Formats retrieved passages into a structured evidence prompt block."""
    blocks = []
    for p in passages:
        if p.get("is_uploaded") or p.get("source_type") == "uploaded_manuscript":
            src_type_label = f"UPLOADED MANUSCRIPT EVIDENCE (Primary Source, ID #{p.get('manuscript_id', 'Unknown')})"
        elif p.get("source_type") == "reference_book":
            src_type_label = f"REFERENCE SCHOLARLY TREATISE [{p.get('source', 'Reference Book')}]"
        else:
            src_type_label = "CANONICAL REFERENCE KNOWLEDGE"

        block = (
            f"=== {src_type_label} [{p['passage_id']}] ===\n"
            f"Source: {p.get('source')} ({p.get('reference')})\n"
            f"Domain: {p.get('domain')}\n"
            f"Verification: {p.get('verification_status')}\n"
            f"Sanskrit Text: {p.get('sanskrit')}\n"
            f"English Translation: {p.get('translation')}\n"
        )
        blocks.append(block)
    return "\n".join(blocks)


def generate_grounded_answer(
    query: str,
    passages: List[Dict[str, Any]],
    gemini_key: Optional[str] = None
) -> Dict[str, Any]:
    """
    Synthesizes an evidence-grounded answer based strictly on the retrieved passages
    using Google Gemini API. Prioritizes uploaded manuscript evidence, supports canonical
    treatises, and avoids hallucinations.
    """
    if not passages:
        return {
            "answer": (
                "Insufficient manuscript evidence found in the indexed repository to answer this query reliably. "
                "No answer was generated to prevent hallucination."
            ),
            "sufficient_evidence": False,
            "has_generated_answer": False,
            "cited_passages": []
        }

    # Verify that the highest score (considering both dense similarity and hybrid score) meets the evidence threshold
    top_score = max(max(p.get("similarity_score", 0.0), p.get("hybrid_score", 0.0)) for p in passages)
    if top_score < MIN_EVIDENCE_SIMILARITY:
        return {
            "answer": (
                f"The closest available manuscript passages score below the evidence threshold "
                f"({top_score:.2f} < {MIN_EVIDENCE_SIMILARITY:.2f}). "
                f"Insufficient evidence is available to formulate a reliable grounded answer."
            ),
            "sufficient_evidence": False,
            "has_generated_answer": False,
            "cited_passages": [p["passage_id"] for p in passages]
        }

    key = (
        gemini_key
        or os.getenv("GEMINI_API_KEY", "").strip()
        or os.getenv("GOOGLE_API_KEY", "").strip()
    )

    if not key:
        logger.info("No Gemini API key available; returning evidence passages without LLM synthesis.")
        return {
            "answer": (
                "Gemini answer generation is currently unavailable (no active API key configured). "
                "The retrieved evidence passages and verified translations are presented below for direct reference."
            ),
            "sufficient_evidence": True,
            "has_generated_answer": False,
            "cited_passages": [p["passage_id"] for p in passages]
        }

    evidence_text = build_evidence_context(passages)
    prompt = (
        "You are an expert scholarly assistant for ancient Sanskrit manuscripts and classical Indian heritage.\n"
        "Synthesize a clear, accurate, and faithful answer to the user's question, relying STRICTLY and ONLY on the retrieved manuscript evidence provided below.\n\n"
        "STRICT GROUNDING & CITATION RULES:\n"
        "1. Base your answer strictly on the explicit statements, verses, and verified translations in the retrieved passages.\n"
        "2. When UPLOADED MANUSCRIPT EVIDENCE is present, address it directly as primary source evidence from the user's uploaded artifacts.\n"
        "3. When CANONICAL or REFERENCE TREATISE evidence is present, use it to provide authoritative scholarly context.\n"
        "4. Do NOT invent, extrapolate, or hallucinate verses, cures, calculations, or doctrines not present in the retrieved text.\n"
        "5. Explicitly cite the passage IDs and sources (e.g. '[UPLOAD-M4-C3]' or '[AYUR-AH-001] Ashtanga Hridaya') for every factual statement.\n"
        "6. If the retrieved passages only partially address the question, clearly state what the evidence shows and what remains unaddressed. Do NOT fabricate an answer.\n"
        "7. Format your response cleanly in clear English, quoting the original Sanskrit verse and translation where relevant.\n\n"
        f"USER QUESTION:\n{query}\n\n"
        f"RETRIEVED MANUSCRIPT EVIDENCE:\n{evidence_text}\n\n"
        "GROUNDED ANSWER:"
    )

    candidate_models = [
        "gemini-flash-lite-latest",
        "gemini-3.1-flash-lite",
        "gemini-3.5-flash",
        "gemini-3.5-flash-lite",
        "gemini-flash-latest",
        "gemini-3.8-flash"
    ]

    try:
        from google import genai
        client = genai.Client(api_key=key)

        for model_name in candidate_models:
            try:
                res = client.models.generate_content(
                    model=model_name,
                    contents=prompt
                )
                if res and res.text:
                    answer_text = res.text.strip()
                    logger.info(f"RAG answer generated successfully via {model_name}.")
                    return {
                        "answer": answer_text,
                        "sufficient_evidence": True,
                        "has_generated_answer": True,
                        "cited_passages": [p["passage_id"] for p in passages],
                        "model_used": model_name
                    }
            except Exception as model_err:
                logger.debug(f"Gemini model {model_name} failed: {model_err}")
                continue

    except Exception as api_err:
        logger.warning(f"Gemini API invocation failed: {api_err}")

    # Fallback if all Gemini models failed
    return {
        "answer": (
            "Gemini answer generation encountered a connection or service issue. "
            "The retrieved evidence passages and translations are provided below for direct scholarly reference."
        ),
        "sufficient_evidence": True,
        "has_generated_answer": False,
        "cited_passages": [p["passage_id"] for p in passages]
    }


def query_rag_pipeline(
    query: str,
    domain: Optional[str] = None,
    source_filter: Optional[str] = "all",
    manuscript_id: Optional[int] = None,
    top_k: int = 4,
    gemini_key: Optional[str] = None
) -> Dict[str, Any]:
    """
    End-to-end execution of the RAG pipeline across uploaded manuscripts and reference knowledge:
    1. Validates user query.
    2. Identifies question domain intent.
    3. Retrieves relevant passages via MiniLM + ChromaDB (with domain, source, and manuscript filters).
    4. Synthesizes a grounded answer via Gemini (if configured).
    5. Returns structured results with full provenance and detected domain information.
    """
    if not query or not query.strip():
        return {
            "query": "",
            "domain": domain,
            "detected_domain": "all",
            "target_domains": ["all"],
            "source_filter": source_filter or "all",
            "manuscript_id": manuscript_id,
            "passages": [],
            "answer": "Please provide a valid, non-empty query.",
            "sufficient_evidence": False,
            "has_generated_answer": False
        }

    clean_q = query.strip()
    detected_info = detect_query_domains(clean_q)

    passages = retrieve_relevant_passages(
        query=clean_q,
        domain=domain,
        source_filter=source_filter,
        manuscript_id=manuscript_id,
        top_k=top_k
    )

    answer_data = generate_grounded_answer(
        query=clean_q,
        passages=passages,
        gemini_key=gemini_key
    )

    return {
        "query": clean_q,
        "domain": domain or "All Domains",
        "detected_domain": detected_info.get("primary_domain", "all"),
        "target_domains": detected_info.get("target_domains", ["all"]),
        "is_mixed_domain": detected_info.get("is_mixed", False),
        "source_filter": source_filter or "all",
        "manuscript_id": manuscript_id,
        "passages": passages,
        "answer": answer_data["answer"],
        "sufficient_evidence": answer_data["sufficient_evidence"],
        "has_generated_answer": answer_data["has_generated_answer"],
        "cited_passages": answer_data["cited_passages"],
        "model_used": answer_data.get("model_used"),
        "embedding_model": get_active_embedding_model_name(),
        "model_status": get_embedding_model_status()
    }


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="SansKrypt RAG & Semantic Retrieval CLI")
    parser.add_argument("--index", action="store_true", help="Index the canonical Sanskrit corpus into ChromaDB")
    parser.add_argument("--corpus", type=str, default=DEFAULT_CORPUS_PATH, help="Path to corpus JSON file")
    parser.add_argument("--model", type=str, default=None, help="Embedding model name or identifier")
    parser.add_argument("--query", type=str, default=None, help="Query the RAG pipeline with a natural language question")
    parser.add_argument("--domain", type=str, default=None, help="Optional domain filter (Philosophy, Ayurveda, Astronomy)")
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO)

    if args.index:
        print(f"Indexing corpus from: {args.corpus}")
        count = index_corpus(corpus_records=load_corpus(args.corpus), model=get_embedding_model(args.model))
        print(f"Indexing complete! Successfully indexed {count} passages into ChromaDB.")

    if args.query:
        print(f"\nProcessing query: '{args.query}' (Domain: {args.domain or 'All'})")
        res = query_rag_pipeline(query=args.query, domain=args.domain)
        print("\n" + "="*60)
        print(f"ANSWER:\n{res['answer']}")
        print("="*60)
        print(f"\nRetrieved {len(res['passages'])} Evidence Passages:")
        for p in res["passages"]:
            print(f"- [{p['passage_id']}] {p['source']} ({p['reference']}) - Sim: {p['similarity_score']:.4f}")
            print(f"  Sanskrit: {p['sanskrit']}")
            print(f"  Translation: {p['translation']}\n")
