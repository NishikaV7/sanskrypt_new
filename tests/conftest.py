import os
import tempfile
import pytest
import numpy as np
import cv2
from PIL import Image

import werkzeug
if not hasattr(werkzeug, "__version__"):
    try:
        import importlib.metadata
        werkzeug.__version__ = importlib.metadata.version("werkzeug")
    except Exception:
        werkzeug.__version__ = "3.1.3"

import flask.globals
if not hasattr(flask.globals, "app_ctx") or flask.globals.app_ctx is None:
    class _AppCtxProxy:
        def _get_current_object(self):
            top = getattr(flask.globals, "_app_ctx_stack", None)
            if top is not None and top.top is not None:
                return top.top
            return self
    flask.globals.app_ctx = _AppCtxProxy()

from app import app as flask_app
from database import db, Manuscript


@pytest.fixture
def app():
    """Configures the Flask application with an in-memory SQLite database for testing."""
    from sqlalchemy import create_engine
    flask_app.config.update({
        "TESTING": True,
        "SQLALCHEMY_DATABASE_URI": "sqlite:///:memory:",
        "WTF_CSRF_ENABLED": False
    })

    sa = flask_app.extensions.get("sqlalchemy")
    orig_engines = sa._app_engines.get(flask_app) if sa else None
    mem_engine = create_engine("sqlite:///:memory:")
    if sa:
        sa._app_engines[flask_app] = {None: mem_engine}

    with flask_app.app_context():
        db.create_all()
        yield flask_app
        db.session.remove()
        db.drop_all()
        mem_engine.dispose()
        if sa and orig_engines:
            sa._app_engines[flask_app] = orig_engines


@pytest.fixture
def client(app):
    """Test HTTP client."""
    return app.test_client()


@pytest.fixture
def sample_image_path(tmp_path):
    """Creates a temporary synthetic Sanskrit manuscript image for testing."""
    img_path = str(tmp_path / "test_manuscript.png")
    
    # Create synthetic manuscript image: 300x200 with text-like strokes
    image = np.ones((200, 300, 3), dtype=np.uint8) * 230  # Parchment tone
    # Draw some text-like lines
    cv2.putText(image, "Dharma", (30, 80), cv2.FONT_HERSHEY_SIMPLEX, 1.2, (40, 20, 10), 2)
    cv2.putText(image, "Karma", (30, 140), cv2.FONT_HERSHEY_SIMPLEX, 1.2, (40, 20, 10), 2)
    
    cv2.imwrite(img_path, image)
    return img_path


@pytest.fixture
def sample_manuscripts(app):
    """Seeds sample manuscripts in the database for search and detail testing."""
    with app.app_context():
        m1 = Manuscript(
            title="Rigveda Creation Hymn",
            original_image_path="uploads/original/rigveda.png",
            processed_image_path="uploads/processed/rigveda_proc.png",
            ocr_text="नासदासीन्नो सदासीत्तदानीं नासीद्रजो नो व्योमा परो यत्",
            ocr_confidence=85.5,
            translated_text="Then was not non-existence nor existence; there was no realm of air, no sky beyond it.",
            explanation="The Nasadiya Sukta inquires into cosmological origin, existence, and the mystery of primordial creation.",
            keywords="Rigveda, Creation, Nasadiya, Cosmology, Brahman",
            classification="Philosophy",
            status="completed"
        )
        m2 = Manuscript(
            title="Charaka Samhita Tridosha",
            original_image_path="uploads/original/charaka.png",
            processed_image_path="uploads/processed/charaka_proc.png",
            ocr_text="वायुः पित्तं कफश्चेति त्रयो दोषाः समासतः",
            ocr_confidence=32.0,  # Below threshold
            translated_text="Vata, Pitta, and Kapha are the three bodily humors in brief.",
            explanation="A fundamental precept of Ayurveda detailing the three bio-energetic principles maintaining physiological homeostasis.",
            keywords="Vata, Pitta, Kapha, Dosha, Ayurveda, Healing",
            classification="Ayurveda",
            status="low_confidence"
        )
        m3 = Manuscript(
            title="Aryabhatiya Sine Table",
            original_image_path="uploads/original/aryabhata.png",
            processed_image_path="uploads/processed/aryabhata_proc.png",
            ocr_text="मखि भखि फखि धखि णखि",
            ocr_confidence=78.2,
            translated_text="The alphabetical table of sine differences calculated for mathematical astronomy.",
            explanation="Aryabhata's ingenious trigonometric verse expressing sine tables using Devanagari numeration.",
            keywords="Ganita, Aryabhata, Sine, Mathematics, Astronomy",
            classification="Mathematics",
            status="completed"
        )
        db.session.add_all([m1, m2, m3])
        db.session.commit()
        return [m1.id, m2.id, m3.id]
