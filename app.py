import os
import uuid
import logging
from typing import Tuple

import werkzeug
if not hasattr(werkzeug, "__version__"):
    try:
        import importlib.metadata
        werkzeug.__version__ = importlib.metadata.version("werkzeug")
    except Exception:
        werkzeug.__version__ = "3.1.3"

from flask import (
    Flask,
    request,
    session,
    render_template,
    redirect,
    url_for,
    flash,
    send_from_directory,
    abort
)
from werkzeug.utils import secure_filename

# Configuration imports
from config import (
    SECRET_KEY,
    UPLOAD_FOLDER,
    PROCESSED_FOLDER,
    ALLOWED_EXTENSIONS,
    MAX_CONTENT_LENGTH,
    LOW_CONFIDENCE_THRESHOLD,
    SQLALCHEMY_DATABASE_URI,
    SQLALCHEMY_TRACK_MODIFICATIONS,
    CANONICAL_CATEGORIES
)

# Pipeline modules
from preprocessing import preprocess_image
from ocr import extract_text, validate_source_image, evaluate_ocr_quality
from translation import translate_text
from explain_classify import (
    generate_explanation,
    extract_keywords,
    classify_content,
    normalize_to_three_domains
)
from database import (
    db,
    init_db,
    save_manuscript,
    get_manuscript,
    list_manuscripts,
    delete_manuscript,
    Manuscript
)
from search import (
    search_manuscripts,
    search_by_keyword,
    search_by_classification,
    search_by_text
)
from rag_service import (
    query_rag_pipeline,
    get_active_embedding_model_name,
    get_embedding_model_status,
    index_manuscript,
    delete_manuscript_from_index
)

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] [%(name)s] %(message)s"
)
logger = logging.getLogger("SansKrypt")


def create_app() -> Flask:
    """Factory function for creating and configuring the Flask application."""
    app = Flask(__name__)
    app.config["SECRET_KEY"] = SECRET_KEY
    app.config["SQLALCHEMY_DATABASE_URI"] = SQLALCHEMY_DATABASE_URI
    app.config["SQLALCHEMY_TRACK_MODIFICATIONS"] = SQLALCHEMY_TRACK_MODIFICATIONS
    app.config["MAX_CONTENT_LENGTH"] = MAX_CONTENT_LENGTH

    # Initialize database
    init_db(app)

    return app


app = create_app()


def is_allowed_file(filename: str) -> bool:
    """Validates uploaded image file extension."""
    if not filename or "." not in filename:
        return False
    ext = filename.rsplit(".", 1)[1].lower()
    return ext in ALLOWED_EXTENSIONS


# ---------------------------------------------------------------------------
# ROUTES
# ---------------------------------------------------------------------------

@app.route("/", methods=["GET"])
def index():
    """Renders the upload home page with recent manuscripts summary."""
    try:
        recent = list_manuscripts(limit=5)
    except Exception as e:
        logger.error(f"Failed to fetch recent manuscripts: {str(e)}")
        recent = []

    return render_template("index.html", recent_manuscripts=recent)


@app.route("/upload", methods=["POST"])
def upload_manuscript():
    """
    Handles manuscript image uploads and executes the end-to-end pipeline:
    1. Validate file
    2. Preprocess with OpenCV
    3. Extract Sanskrit OCR text and confidence score
    4. Translate to English
    5. Generate explanation, keywords, and domain classification
    6. Persist to SQLite only when all stages complete successfully
    7. Redirect to results page
    """
    if "image" not in request.files:
        flash("No image file was provided in the upload request.", "error")
        return redirect(url_for("index"))

    file = request.files["image"]
    user_title = request.form.get("title", "").strip()

    if file.filename == "":
        flash("No image file was selected.", "error")
        return redirect(url_for("index"))

    if not is_allowed_file(file.filename):
        allowed_str = ", ".join(sorted(ALLOWED_EXTENSIONS)).upper()
        flash(f"Invalid file type. Please upload an image format: {allowed_str}", "error")
        return redirect(url_for("index"))

    try:
        # Save raw uploaded image safely with unique filename
        filename = secure_filename(file.filename)
        unique_prefix = uuid.uuid4().hex[:8]
        safe_name = f"{unique_prefix}_{filename}"
        raw_image_path = os.path.join(UPLOAD_FOLDER, safe_name)
        file.save(raw_image_path)
        # Read optional Gemini API key for handwritten vision OCR and validation
        api_key = request.form.get("api_key", "").strip() or session.get("gemini_api_key", "")
        if api_key:
            session["gemini_api_key"] = api_key
            os.environ["GEMINI_API_KEY"] = api_key

        # Input Validation Gate: Inspect uploaded source image
        logger.info("[Validation Gate] Validating source image content...")
        is_valid, _ = validate_source_image(raw_image_path, api_key=api_key)
        if not is_valid:
            logger.warning(f"Upload rejected by input validation gate: {raw_image_path}")
            flash("No Sanskrit text detected.", "error")
            try:
                if os.path.exists(raw_image_path):
                    os.remove(raw_image_path)
            except Exception:
                pass
            return redirect(url_for("index"))

        # Stage 2: OpenCV Image Preprocessing
        logger.info("[Stage 2/5] Running OpenCV preprocessing...")
        processed_filename = f"{unique_prefix}_proc_{filename}"
        target_processed_path = os.path.join(PROCESSED_FOLDER, processed_filename)
        _, processed_image_path = preprocess_image(
            image_path=raw_image_path,
            output_path=target_processed_path
        )
        logger.info(f"[Stage 2/5] Preprocessed image saved to: {processed_image_path}")

        # Stage 3: Sanskrit OCR with confidence score
        logger.info("[Stage 3/5] Extracting Sanskrit text and confidence...")
        try:
            # First try with raw or processed image using Gemini Vision if key provided, or multi-pass Tesseract
            ocr_text, ocr_confidence = extract_text(raw_image_path, api_key=api_key)
            if not ocr_text or len(ocr_text.strip()) < 3:
                ocr_text, ocr_confidence = extract_text(processed_image_path, api_key=api_key)
        except RuntimeError as ocr_err:
            flash(f"OCR Error: {str(ocr_err)}", "error")
            return redirect(url_for("index"))

        if not ocr_text or not ocr_text.strip():
            ocr_text = "[No readable Sanskrit text characters could be identified by OCR]"
            ocr_confidence = 0.0

        # Assess OCR quality signals and determine status (completed vs low_confidence/Needs Review)
        quality_eval = evaluate_ocr_quality(ocr_text, ocr_confidence, image_input=raw_image_path)
        status = quality_eval["status"]
        if status == "low_confidence":
            reasons_summary = "; ".join(quality_eval["reasons"])
            logger.warning(
                f"Manuscript tagged as 'low_confidence' (Needs Review). "
                f"Confidence: {ocr_confidence}%. Reasons: {reasons_summary}"
            )

        # Stage 4: Sanskrit -> English Translation
        logger.info("[Stage 4/5] Translating Sanskrit to English...")
        translated_text = translate_text(ocr_text)

        # Stage 5: Contextual Explanation, Keyword Extraction & Classification
        logger.info("[Stage 5/5] Generating LLM explanation, keywords, and classification...")
        explanation = generate_explanation(ocr_text, translated_text)
        keywords = extract_keywords(ocr_text, translated_text)
        raw_classification = classify_content(ocr_text, translated_text)
        if raw_classification in ("Needs Review", "needs_review", "General/Other"):
            classification = "Needs Review"
        else:
            classification = normalize_to_three_domains(raw_classification)

        # Generate default title if none provided
        title = user_title if user_title else f"Manuscript ({classification}) - {unique_prefix}"

        # Persist to database only after all pipeline stages succeed
        manuscript = save_manuscript(
            title=title,
            original_image_path=raw_image_path,
            processed_image_path=processed_image_path,
            ocr_text=ocr_text,
            ocr_confidence=ocr_confidence,
            translated_text=translated_text,
            explanation=explanation,
            keywords=keywords,
            classification=classification,
            status=status
        )

        # Stage 6: Automatic Vector Indexing into ChromaDB for Dynamic RAG
        try:
            logger.info(f"[Stage 6] Indexing manuscript #{manuscript.id} into ChromaDB...")
            indexed_chunks = index_manuscript(
                manuscript_id=manuscript.id,
                sanskrit_text=ocr_text,
                translation=translated_text,
                domain=classification,
                title=title,
                status=status,
                ocr_confidence=ocr_confidence,
                keywords=keywords
            )
            logger.info(f"[Stage 6] Successfully indexed {indexed_chunks} chunks for manuscript #{manuscript.id}.")
        except Exception as idx_err:
            logger.error(f"Auto-indexing failed for manuscript #{manuscript.id}: {idx_err}", exc_info=True)

        flash("Manuscript processed and digitized successfully!", "success")
        return redirect(url_for("view_manuscript", manuscript_id=manuscript.id))

    except Exception as e:
        logger.error(f"Pipeline failure during manuscript processing: {str(e)}", exc_info=True)
        flash(f"An error occurred while processing the manuscript: {str(e)}", "error")
        return redirect(url_for("index"))


@app.route("/manuscript/<int:manuscript_id>", methods=["GET"])
def view_manuscript(manuscript_id: int):
    """Renders the detailed result view for a digitized manuscript artifact."""
    manuscript = get_manuscript(manuscript_id)
    if not manuscript:
        return render_template(
            "error.html",
            error_title="Manuscript Not Found",
            error_message=f"No manuscript artifact exists with ID #{manuscript_id}."
        ), 404

    return render_template("results.html", manuscript=manuscript)


@app.route("/search", methods=["GET"])
def search_view():
    """Renders the repository search interface."""
    return render_template("search.html", categories=CANONICAL_CATEGORIES)


@app.route("/search/results", methods=["GET"])
def search_query():
    """Executes multi-criteria search query and renders matching results."""
    query = request.args.get("query", "").strip()
    keyword = request.args.get("keyword", "").strip()
    classification = request.args.get("classification", "").strip()
    status = request.args.get("status", "").strip()
    min_confidence = request.args.get("min_confidence", None)

    try:
        results = search_manuscripts(
            query=query,
            keyword=keyword,
            classification=classification,
            status=status,
            min_confidence=min_confidence
        )
    except Exception as e:
        logger.error(f"Search query error: {str(e)}", exc_info=True)
        results = []
        flash(f"Search encountered an error: {str(e)}", "error")

    query_params = {
        "query": query,
        "keyword": keyword,
        "classification": classification,
        "status": status,
        "min_confidence": min_confidence
    }

    return render_template(
        "search_results.html",
        results=results,
        query_params=query_params,
        categories=CANONICAL_CATEGORIES
    )


@app.route("/ask", methods=["GET"])
def ask_view():
    """Renders the natural-language question and answer (RAG) interface."""
    query = request.args.get("query", "").strip()
    domain = request.args.get("domain", "all").strip()
    source_filter = request.args.get("source_filter", "all").strip()
    manuscript_id = request.args.get("manuscript_id", "").strip() or None

    rag_result = None
    if query:
        api_key = session.get("gemini_api_key", "")
        try:
            rag_result = query_rag_pipeline(
                query=query,
                domain=domain,
                source_filter=source_filter,
                manuscript_id=manuscript_id,
                gemini_key=api_key
            )
        except Exception as e:
            logger.error(f"RAG query error: {str(e)}", exc_info=True)
            flash("An error occurred during semantic retrieval or answer generation.", "error")

    return render_template(
        "ask.html",
        rag_result=rag_result,
        current_query=query,
        current_domain=domain,
        current_source_filter=source_filter,
        current_manuscript_id=manuscript_id,
        active_embedding_model=get_active_embedding_model_name(),
        model_status=get_embedding_model_status()
    )


@app.route("/ask", methods=["POST"])
def ask_query():
    """Processes natural-language questions and returns evidence-grounded answers."""
    query = request.form.get("query", "").strip()
    domain = request.form.get("domain", "all").strip()
    source_filter = request.form.get("source_filter", "all").strip()
    manuscript_id = request.form.get("manuscript_id", "").strip() or None
    api_key = session.get("gemini_api_key", "")

    if not query:
        flash("Please enter a question to search the repository.", "warning")
        return redirect(url_for("ask_view", manuscript_id=manuscript_id))

    try:
        rag_result = query_rag_pipeline(
            query=query,
            domain=domain,
            source_filter=source_filter,
            manuscript_id=manuscript_id,
            gemini_key=api_key
        )
    except Exception as e:
        logger.error(f"RAG pipeline failure: {str(e)}", exc_info=True)
        flash("An unexpected error occurred while processing your question.", "error")
        rag_result = None

    return render_template(
        "ask.html",
        rag_result=rag_result,
        current_query=query,
        current_domain=domain,
        current_source_filter=source_filter,
        current_manuscript_id=manuscript_id,
        active_embedding_model=get_active_embedding_model_name(),
        model_status=get_embedding_model_status()
    )


@app.route("/manuscripts", methods=["GET"])
def list_repository():
    """Lists all digitized manuscripts in the repository with optional status/domain filtering."""
    status = request.args.get("status", "all")
    classification = request.args.get("classification", "all")

    try:
        manuscripts = list_manuscripts(
            status=status if status != "all" else None,
            classification=classification if classification != "all" else None,
            limit=200
        )
    except Exception as e:
        logger.error(f"Failed to list repository manuscripts: {str(e)}")
        manuscripts = []

    return render_template(
        "manuscripts.html",
        manuscripts=manuscripts,
        current_status=status,
        current_classification=classification,
        categories=CANONICAL_CATEGORIES
    )


@app.route("/manuscript/<int:manuscript_id>/delete", methods=["POST"])
def delete_manuscript_route(manuscript_id: int):
    """Deletes a manuscript record and cleans up associated file assets."""
    manuscript = get_manuscript(manuscript_id)
    if not manuscript:
        flash(f"Manuscript #{manuscript_id} not found.", "error")
        return redirect(url_for("list_repository"))

    try:
        # Delete image files if they exist
        for path in [manuscript.original_image_path, manuscript.processed_image_path]:
            if path and os.path.isfile(path):
                try:
                    os.remove(path)
                except Exception as file_err:
                    logger.warning(f"Could not remove file {path}: {str(file_err)}")

        # Prune indexed chunks from ChromaDB for this manuscript
        delete_manuscript_from_index(manuscript_id)

        delete_manuscript(manuscript_id)
        flash(f"Manuscript #{manuscript_id} was deleted successfully.", "success")
    except Exception as e:
        logger.error(f"Failed to delete manuscript #{manuscript_id}: {str(e)}")
        flash(f"Could not delete manuscript: {str(e)}", "error")

    return redirect(url_for("list_repository"))


@app.route("/manuscript/<int:manuscript_id>/image/<img_type>")
def manuscript_image(manuscript_id: int, img_type: str):
    """Serves original or processed manuscript images safely by manuscript ID."""
    manuscript = get_manuscript(manuscript_id)
    if not manuscript:
        abort(404)

    target_path = manuscript.original_image_path if img_type == "original" else manuscript.processed_image_path
    if target_path and os.path.isfile(target_path):
        directory = os.path.dirname(os.path.abspath(target_path))
        filename = os.path.basename(target_path)
        return send_from_directory(directory, filename)

    # Check relative to base uploads
    filename = manuscript.original_filename if img_type == "original" else manuscript.processed_filename
    base_uploads = os.path.join(os.path.dirname(__file__), "uploads")
    for sub in ["", "original", "processed"]:
        candidate = os.path.join(base_uploads, sub, os.path.basename(filename)) if sub else os.path.join(base_uploads, filename)
        if candidate and os.path.isfile(candidate):
            return send_from_directory(os.path.dirname(os.path.abspath(candidate)), os.path.basename(candidate))

    # Fallback to dynamic SVG placeholder if image file is not on disk
    from flask import Response
    title_escaped = (manuscript.title or f"Manuscript #{manuscript.id}").replace("<", "&lt;").replace(">", "&gt;")
    svg = f"""<svg xmlns="http://www.w3.org/2000/svg" width="600" height="400" viewBox="0 0 600 400">
        <rect width="100%" height="100%" fill="#fef3c7" />
        <rect x="20" y="20" width="560" height="360" rx="12" fill="#fffbeb" stroke="#d97706" stroke-width="2" stroke-dasharray="8,8" />
        <text x="50%" y="42%" dominant-baseline="middle" text-anchor="middle" font-family="sans-serif" font-size="20" font-weight="bold" fill="#78350f">📜 {title_escaped}</text>
        <text x="50%" y="56%" dominant-baseline="middle" text-anchor="middle" font-family="sans-serif" font-size="14" fill="#92400e">ID #{manuscript.id} &bull; {manuscript.classification} &bull; {img_type.capitalize()} View</text>
    </svg>"""
    return Response(svg, mimetype="image/svg+xml")


@app.route("/uploads/<path:filename>")
def uploaded_file(filename: str):
    """Serves uploaded original and processed manuscript images securely across all platforms."""
    # Sanitize and normalize relative path
    clean_fn = filename.replace("\\", "/").lstrip("/")
    base_uploads = os.path.join(os.path.dirname(__file__), "uploads")
    
    # Check direct in base_uploads
    direct_path = os.path.join(base_uploads, clean_fn)
    if os.path.isfile(direct_path):
        return send_from_directory(base_uploads, clean_fn)
        
    # Check original subfolder
    orig_path = os.path.join(base_uploads, "original", os.path.basename(clean_fn))
    if os.path.isfile(orig_path):
        return send_from_directory(os.path.join(base_uploads, "original"), os.path.basename(clean_fn))

    # Check processed subfolder
    proc_path = os.path.join(base_uploads, "processed", os.path.basename(clean_fn))
    if os.path.isfile(proc_path):
        return send_from_directory(os.path.join(base_uploads, "processed"), os.path.basename(clean_fn))

    abort(404)


# ---------------------------------------------------------------------------
# ERROR HANDLERS
# ---------------------------------------------------------------------------

@app.errorhandler(404)
def not_found_error(error):
    return render_template(
        "error.html",
        error_title="404 - Page Not Found",
        error_message="The requested resource or page could not be located."
    ), 404


@app.errorhandler(500)
def internal_error(error):
    return render_template(
        "error.html",
        error_title="500 - Internal Server Error",
        error_message="An unexpected system error occurred. Please check server logs."
    ), 500


if __name__ == "__main__":
    # Create required upload folders
    os.makedirs(UPLOAD_FOLDER, exist_ok=True)
    os.makedirs(PROCESSED_FOLDER, exist_ok=True)
    
    print("\n" + "="*60)
    print("  SansKrypt: System for Translating Manuscripts")
    print("  Local Server Starting on http://127.0.0.1:5000")
    print("="*60 + "\n")
    
    # Configure reliable development server reloader
    # Using 'stat' reloader avoids watchdog recursively scanning Python standard library
    # and site-packages on Windows, preventing continuous restart loops.
    use_reloader = os.environ.get("FLASK_USE_RELOADER", "true").lower() in ("true", "1", "yes")
    reloader_type = os.environ.get("FLASK_RELOADER_TYPE", "stat")
    exclude_patterns = [
        "*site-packages*",
        "*Python313*",
        "*Lib*",
        "*.git*",
        "*.pytest_cache*",
        "*__pycache__*",
        "*chroma_db*"
    ]

    app.run(
        host="127.0.0.1",
        port=5000,
        debug=True,
        use_reloader=use_reloader,
        reloader_type=reloader_type,
        exclude_patterns=exclude_patterns
    )
