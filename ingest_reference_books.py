"""
SansKrypt Reference Books Dataset Ingestion Module
Extracts Sanskrit verses and passages from domain reference PDF books (Philosophy, Ayurveda,
Astronomy & Mathematics), translates them via the existing translation pipeline, preserves
scholarly provenance, and idempotently indexes them into the persistent ChromaDB collection.
"""

import os
import sys
import re
import json
import hashlib
import logging
import unicodedata
from pathlib import Path
from typing import List, Dict, Any, Optional, Tuple

import pypdf

# Internal imports from SansKrypt
from config import BASE_DIR
from translation import translate_text
from rag_service import (
    get_chroma_collection,
    get_embedding_model,
    COLLECTION_NAME,
    CHROMA_PERSIST_DIR,
    build_passage_text_for_embedding
)
from explain_classify import normalize_to_three_domains

logger = logging.getLogger("SansKrypt.Ingest")

# Persistent storage file for extracted reference book records
REFERENCE_CORPUS_FILE = os.path.join(BASE_DIR, "data", "reference_books_corpus.json")

# Devanagari Unicode range regex
DEVANAGARI_CHAR_REGEX = re.compile(r"[\u0900-\u097F]")
# Verse ending markers: double danda, single danda, or verse numbering
VERSE_SPLIT_REGEX = re.compile(r"(?<=[॥।])\s*(?=[^\s॥।])|(?:\r?\n\s*\r?\n)")


def is_devanagari_character(char: str) -> bool:
    """Checks if a character falls within the Devanagari script Unicode block."""
    code = ord(char)
    return 0x0900 <= code <= 0x097F


def compute_devanagari_metrics(text: str) -> Dict[str, Any]:
    """Calculates Devanagari character count and ratio for quality gating."""
    if not text:
        return {"devanagari_chars": 0, "total_chars": 0, "devanagari_ratio": 0.0}
    clean = re.sub(r"\s+", "", text)
    if not clean:
        return {"devanagari_chars": 0, "total_chars": 0, "devanagari_ratio": 0.0}
    dev_count = sum(1 for c in clean if is_devanagari_character(c))
    return {
        "devanagari_chars": dev_count,
        "total_chars": len(clean),
        "devanagari_ratio": dev_count / len(clean)
    }


def clean_sanskrit_text(text: str) -> str:
    """
    Cleans and normalizes extracted Sanskrit text:
    - Normalizes Unicode to NFC
    - Replaces odd quotes, bullets, and line noise
    - Standardizes danda marks (। and ॥)
    - Normalizes internal whitespace
    """
    if not text:
        return ""
    norm = unicodedata.normalize("NFC", text)
    # Standardize dandas
    norm = norm.replace("||", "॥").replace("|", "।")
    # Remove page header artifacts like solitary Latin page headers if mixed
    # Normalize excessive spaces
    norm = re.sub(r"[ \t]+", " ", norm)
    # Normalize multiple newlines
    norm = re.sub(r"\n{3,}", "\n\n", norm)
    return norm.strip()


def extract_raw_pages_from_pdf(pdf_path: str) -> List[Tuple[int, str]]:
    """
    Extracts text page by page from a reference PDF file using pypdf.
    Returns a list of (page_number_1_indexed, page_text).
    """
    if not os.path.exists(pdf_path):
        raise FileNotFoundError(f"PDF file not found: {pdf_path}")

    pages = []
    reader = pypdf.PdfReader(pdf_path)
    logger.info(f"Opening PDF '{os.path.basename(pdf_path)}' with {len(reader.pages)} pages.")

    for idx, page in enumerate(reader.pages, start=1):
        try:
            page_text = page.extract_text() or ""
            pages.append((idx, page_text))
        except Exception as e:
            logger.warning(f"Failed to extract text from page {idx} of {pdf_path}: {e}")
            pages.append((idx, ""))

    return pages


def segment_sanskrit_verses(
    page_text: str,
    page_num: int,
    domain: str,
    book_title: str,
    min_length: int = 15,
    max_length: int = 450
) -> List[Dict[str, Any]]:
    """
    Identifies meaningful Sanskrit verses and prose boundaries from extracted page text.
    Filters out standalone Latin noise, English annotations, or page numbers.
    """
    cleaned_page = clean_sanskrit_text(page_text)
    if not cleaned_page:
        return []

    # Split page by double danda '॥', single danda '।', or double newlines
    # while preserving the danda with the preceding verse
    raw_segments = []
    current_tokens = []

    # Split lines first
    lines = [l.strip() for l in cleaned_page.splitlines() if l.strip()]
    buffer = ""

    for line in lines:
        metrics = compute_devanagari_metrics(line)
        # Skip pure page numbers or lines with zero Devanagari
        if metrics["devanagari_chars"] == 0 and re.match(r"^\d+$|^page\s*\d+$|^[ivxlcdm]+$", line, re.I):
            continue

        if buffer:
            buffer += " " + line
        else:
            buffer = line

        # If line ends with danda or punctuation, consider it a segment boundary
        if buffer.endswith("॥") or buffer.endswith("।") or "॥" in line or len(buffer) >= max_length:
            raw_segments.append(buffer)
            buffer = ""

    if buffer:
        raw_segments.append(buffer)

    passages = []
    slug_domain = "PHIL" if "philo" in domain.lower() else "AYUR" if "ayur" in domain.lower() else "ASTRO"
    # Clean book title for ID slug
    slug_book = re.sub(r"[^a-zA-Z0-9]", "", book_title)[:6].upper() or "BOOK"

    verse_counter = 1
    for seg in raw_segments:
        seg_clean = seg.strip()
        metrics = compute_devanagari_metrics(seg_clean)

        # Quality gate: must contain at least min_length chars and meaningful Devanagari ratio
        if len(seg_clean) < min_length or metrics["devanagari_chars"] < 8:
            continue

        # If predominantly English/Latin commentary, check if Devanagari verse is embedded
        if metrics["devanagari_ratio"] < 0.35:
            # Try to extract the Devanagari substring
            dev_matches = re.findall(r"[\u0900-\u097F\s।॥]+", seg_clean)
            dev_block = " ".join(m.strip() for m in dev_matches if len(m.strip()) >= min_length)
            if len(dev_block) >= min_length:
                seg_clean = dev_block
                metrics = compute_devanagari_metrics(seg_clean)
            else:
                continue

        needs_review = (metrics["devanagari_ratio"] < 0.60 or len(seg_clean) < 20)

        # Unique, deterministic passage ID
        pid = f"REF-{slug_domain}-{slug_book}-P{page_num}-V{verse_counter}"

        passages.append({
            "passage_id": pid,
            "domain": domain,
            "source": book_title,
            "reference": f"{book_title}, Page {page_num}, Passage {verse_counter}",
            "page": page_num,
            "verse_index": verse_counter,
            "sanskrit": seg_clean,
            "source_type": "reference_book",
            "needs_review": needs_review,
            "verification_status": "needs_review" if needs_review else "ai_generated",
            "keywords": extract_simple_keywords(seg_clean, domain)
        })
        verse_counter += 1

    return passages


def extract_simple_keywords(sanskrit_text: str, domain: str) -> str:
    """Extracts thematic domain keywords from Sanskrit verse for hybrid retrieval."""
    keywords = [domain]
    # Common core terms across domains
    term_map = {
        "सत्य": "Truth/Satya",
        "धर्म": "Dharma",
        "ब्रह्म": "Brahman",
        "आत्म": "Atman",
        "कर्म": "Karma",
        "योग": "Yoga",
        "वायु": "Vata/Vayu",
        "पित्त": "Pitta",
        "कफ": "Kapha",
        "दोष": "Dosha",
        "धातु": "Dhatu",
        "अग्नि": "Agni",
        "स्वास्थ्य": "Health/Svastha",
        "सूर्य": "Surya/Sun",
        "चन्द्र": "Chandra/Moon",
        "ग्रह": "Graha/Planet",
        "गणित": "Ganita/Mathematics",
        "गोल": "Gola/Sphere",
        "राशि": "Rashi/Zodiac",
        "शून्य": "Shunya/Zero",
        "अहोरात्र": "Ahoratra/Day-Night"
    }
    for s_word, k_word in term_map.items():
        if s_word in sanskrit_text and k_word not in keywords:
            keywords.append(k_word)
    return ", ".join(keywords)


def load_persistent_reference_corpus(path: Optional[str] = None) -> Dict[str, Dict[str, Any]]:
    """Loads existing reference book records from disk to ensure idempotency and reuse."""
    target_path = path or REFERENCE_CORPUS_FILE
    if not os.path.exists(target_path):
        return {}
    try:
        with open(target_path, "r", encoding="utf-8") as f:
            records = json.load(f)
            if isinstance(records, list):
                return {r["passage_id"]: r for r in records if "passage_id" in r}
            elif isinstance(records, dict):
                return records
    except Exception as e:
        logger.warning(f"Could not load reference corpus from {target_path}: {e}")
    return {}


def save_persistent_reference_corpus(
    records_dict: Dict[str, Dict[str, Any]],
    path: Optional[str] = None
) -> None:
    """Saves reference book records into structured JSON format."""
    target_path = path or REFERENCE_CORPUS_FILE
    os.makedirs(os.path.dirname(os.path.abspath(target_path)), exist_ok=True)
    records_list = sorted(records_dict.values(), key=lambda x: x.get("passage_id", ""))
    with open(target_path, "w", encoding="utf-8") as f:
        json.dump(records_list, f, ensure_ascii=False, indent=2)
    logger.info(f"Saved {len(records_list)} reference records to {target_path}")


def translate_passages_with_caching(
    passages: List[Dict[str, Any]],
    existing_corpus: Dict[str, Dict[str, Any]]
) -> Tuple[List[Dict[str, Any]], int]:
    """
    Translates extracted Sanskrit passages using the existing Gemini translation pipeline.
    Avoids repeatedly translating passages that have already been processed and cached.
    """
    translated_count = 0
    enriched = []

    for p in passages:
        pid = p["passage_id"]
        # Check if already cached by passage_id or identical Sanskrit text
        cached_record = existing_corpus.get(pid)
        if not cached_record:
            # Check by sanskrit text matching
            for cr in existing_corpus.values():
                if cr.get("sanskrit") == p["sanskrit"] and cr.get("translation"):
                    cached_record = cr
                    break

        if cached_record and cached_record.get("translation"):
            p["translation"] = cached_record["translation"]
            p["verification_status"] = cached_record.get("verification_status", "ai_generated")
            p["needs_review"] = cached_record.get("needs_review", False)
            if "keywords" in cached_record:
                p["keywords"] = cached_record["keywords"]
            enriched.append(p)
            continue

        # Needs translation: call existing translation pipeline
        sanskrit_text = p["sanskrit"]
        logger.info(f"Translating passage [{pid}] ({len(sanskrit_text)} chars)...")
        try:
            translation = translate_text(sanskrit_text)
            if not translation or "No text available" in translation:
                translation = "Translation pending review."
                p["needs_review"] = True
                p["verification_status"] = "needs_review"
            else:
                p["verification_status"] = "ai_generated"

            p["translation"] = translation
            translated_count += 1
        except Exception as e:
            logger.error(f"Translation failed for passage {pid}: {e}")
            p["translation"] = "Translation unavailable (error during generation)."
            p["needs_review"] = True
            p["verification_status"] = "needs_review"

        enriched.append(p)

    return enriched, translated_count


def index_reference_passages_into_chromadb(
    passages: List[Dict[str, Any]],
    collection=None,
    model=None
) -> int:
    """
    Idempotently indexes reference book passages into ChromaDB using MiniLM.
    Preserves all existing reference and uploaded manuscript records.
    """
    if not passages:
        return 0

    col = collection if collection is not None else get_chroma_collection()
    emb_model = model if model is not None else get_embedding_model()

    ids = []
    documents = []
    metadatas = []
    texts_to_embed = []

    for p in passages:
        pid = p["passage_id"]
        doc_text = build_passage_text_for_embedding(p)

        ids.append(pid)
        documents.append(doc_text)
        metadatas.append({
            "passage_id": pid,
            "source_type": "reference_book",
            "is_uploaded": False,
            "domain": p["domain"],
            "source": p["source"],
            "reference": p["reference"],
            "page": int(p.get("page", 1)),
            "verification_status": p.get("verification_status", "ai_generated"),
            "sanskrit": p["sanskrit"],
            "translation": p["translation"],
            "keywords": p.get("keywords", "")
        })
        texts_to_embed.append(doc_text)

    logger.info(f"Generating MiniLM embeddings for {len(texts_to_embed)} reference book passages...")
    vectors = emb_model.encode(texts_to_embed, normalize_embeddings=True, show_progress_bar=False)
    embeddings_list = [v.tolist() if hasattr(v, "tolist") else list(v) for v in vectors]

    # Upsert ensures stable, idempotent indexing without duplicating records
    col.upsert(
        ids=ids,
        documents=documents,
        metadatas=metadatas,
        embeddings=embeddings_list
    )
    logger.info(f"Successfully indexed {len(ids)} reference passages into ChromaDB (total records: {col.count()}).")
    return len(ids)


def ingest_pdf_book(
    pdf_path: str,
    domain: str,
    book_title: Optional[str] = None,
    dry_run: bool = False,
    corpus_path: Optional[str] = None,
    collection=None,
    model=None
) -> Dict[str, Any]:
    """
    End-to-End Ingestion Pipeline for a Reference PDF Book:
    1. Extract Sanskrit pages
    2. Segment meaningful verses & boundaries
    3. Generate/reuse English translations via Gemini pipeline
    4. Save to persistent JSON corpus
    5. Index into ChromaDB
    """
    clean_domain = normalize_to_three_domains(domain)
    default_title = Path(pdf_path).stem.replace("_", " ").title()
    title = book_title.strip() if book_title and book_title.strip() else default_title

    logger.info(f"=== Starting Ingestion: '{title}' [{clean_domain}] ===")
    pages = extract_raw_pages_from_pdf(pdf_path)

    all_passages = []
    for page_num, text in pages:
        page_passages = segment_sanskrit_verses(
            page_text=text,
            page_num=page_num,
            domain=clean_domain,
            book_title=title
        )
        all_passages.extend(page_passages)

    logger.info(f"Extracted {len(all_passages)} raw Sanskrit passage(s) across {len(pages)} pages.")

    if not all_passages:
        return {
            "status": "warning",
            "message": "No Sanskrit passages meeting quality thresholds were extracted from the PDF.",
            "passages_extracted": 0,
            "translations_generated": 0,
            "passages_needing_review": 0,
            "indexed_count": 0
        }

    # Load persistent corpus cache to avoid redundant translations
    corpus_dict = load_persistent_reference_corpus(path=corpus_path)
    enriched_passages, translated_count = translate_passages_with_caching(all_passages, corpus_dict)

    review_count = sum(1 for p in enriched_passages if p.get("needs_review"))

    if dry_run:
        logger.info("Dry-run requested: skipping persistence and ChromaDB indexing.")
        return {
            "status": "dry_run",
            "passages_extracted": len(enriched_passages),
            "translations_generated": translated_count,
            "passages_needing_review": review_count,
            "indexed_count": 0,
            "passages": enriched_passages
        }

    # Update persistent corpus JSON
    for p in enriched_passages:
        corpus_dict[p["passage_id"]] = p
    save_persistent_reference_corpus(corpus_dict, path=corpus_path)

    # Index into ChromaDB
    indexed_count = index_reference_passages_into_chromadb(
        enriched_passages,
        collection=collection,
        model=model
    )

    return {
        "status": "success",
        "book_title": title,
        "domain": clean_domain,
        "passages_extracted": len(enriched_passages),
        "translations_generated": translated_count,
        "passages_needing_review": review_count,
        "indexed_count": indexed_count,
        "sample_passages": [p["passage_id"] for p in enriched_passages[:5]]
    }


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="SansKrypt Reference Books Ingestion Tool")
    parser.add_argument("--pdf", required=True, help="Path to reference PDF file")
    parser.add_argument("--domain", required=True, choices=["Philosophy", "Ayurveda", "Astronomy & Mathematics"], help="Knowledge domain")
    parser.add_argument("--title", default=None, help="Book title (defaults to PDF filename)")
    parser.add_argument("--dry-run", action="store_true", help="Extract and segment without indexing into ChromaDB")

    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")

    res = ingest_pdf_book(
        pdf_path=args.pdf,
        domain=args.domain,
        book_title=args.title,
        dry_run=args.dry_run
    )
    print(json.dumps(res, indent=2, ensure_ascii=False))
