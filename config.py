"""
SansKrypt Configuration Module
Central settings for file uploads, OCR, translation, LLM inference, and database.
"""
import os
from pathlib import Path

# Base directory
BASE_DIR = Path(__file__).resolve().parent

# Auto-load .env file if present
env_path = BASE_DIR / ".env"
if env_path.is_file():
    try:
        with open(env_path, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if line and not line.startswith("#") and "=" in line:
                    k, v = line.split("=", 1)
                    os.environ[k.strip()] = v.strip().strip("\"'")
    except Exception:
        pass

# File storage directories
UPLOAD_FOLDER = os.path.join(BASE_DIR, "uploads", "original")
PROCESSED_FOLDER = os.path.join(BASE_DIR, "uploads", "processed")

# Create necessary directories
os.makedirs(UPLOAD_FOLDER, exist_ok=True)
os.makedirs(PROCESSED_FOLDER, exist_ok=True)

# Upload constraints
ALLOWED_EXTENSIONS = {"png", "jpg", "jpeg", "webp", "tiff", "bmp"}
MAX_CONTENT_LENGTH = 16 * 1024 * 1024  # 16 MB max file size

# Database configuration
DATABASE_FILE = os.path.join(BASE_DIR, "sanskrypt.db")
SQLALCHEMY_DATABASE_URI = f"sqlite:///{Path(DATABASE_FILE).as_posix()}"
SQLALCHEMY_TRACK_MODIFICATIONS = False

# API Keys & Cloud Services
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY", os.getenv("GOOGLE_API_KEY", ""))

# OCR Settings
LOW_CONFIDENCE_THRESHOLD = 40.0  # Tunable threshold (0 - 100)
OCR_LANGUAGE = "san"             # Sanskrit Tesseract traineddata

# OCR Quality Evaluation Thresholds
# Minimum character length for usable manuscript text after whitespace normalization
OCR_MIN_TEXT_LENGTH = int(os.getenv("OCR_MIN_TEXT_LENGTH", "3"))

# Minimum count of Devanagari script characters required
OCR_MIN_DEVANAGARI_CHARS = int(os.getenv("OCR_MIN_DEVANAGARI_CHARS", "2"))

# Minimum proportion of Devanagari characters among non-whitespace text (0.0 - 1.0)
OCR_MIN_DEVANAGARI_RATIO = float(os.getenv("OCR_MIN_DEVANAGARI_RATIO", "0.50"))

# Maximum allowable ratio of non-text noise and symbol characters (0.0 - 1.0)
OCR_MAX_NOISE_RATIO = float(os.getenv("OCR_MAX_NOISE_RATIO", "0.20"))

# Maximum allowable ratio of replacement characters (?, \ufffd) (0.0 - 1.0)
OCR_MAX_REPLACEMENT_RATIO = float(os.getenv("OCR_MAX_REPLACEMENT_RATIO", "0.10"))

# Maximum consecutive identical character repetitions allowed before flagging as garbage loop
OCR_MAX_CHAR_REPETITION = int(os.getenv("OCR_MAX_CHAR_REPETITION", "4"))

# Minimum proportion of multi-character Devanagari words among all tokens (for text with >= 6 tokens)
OCR_MIN_WORD_TOKEN_RATIO = float(os.getenv("OCR_MIN_WORD_TOKEN_RATIO", "0.35"))

# Maximum allowable ratio of isolated single-character tokens (excluding dandas) (for text with >= 6 tokens)
OCR_MAX_SINGLE_CHAR_RATIO = float(os.getenv("OCR_MAX_SINGLE_CHAR_RATIO", "0.15"))

# Text completeness & visible line estimation thresholds
OCR_MIN_CHARS_PER_LINE = float(os.getenv("OCR_MIN_CHARS_PER_LINE", "8.0"))
OCR_MIN_LINE_COMPLETENESS_RATIO = float(os.getenv("OCR_MIN_LINE_COMPLETENESS_RATIO", "0.35"))
OCR_MIN_VISIBLE_LINES_FOR_CHECK = int(os.getenv("OCR_MIN_VISIBLE_LINES_FOR_CHECK", "3"))

# Corrupted / implausible token thresholds
OCR_MAX_CORRUPT_TOKEN_RATIO = float(os.getenv("OCR_MAX_CORRUPT_TOKEN_RATIO", "0.15"))
OCR_MAX_AVAGRAHA_RATIO = float(os.getenv("OCR_MAX_AVAGRAHA_RATIO", "0.05"))

# Machine Learning / NLP Models (CPU-Friendly)
TRANSLATION_MODEL_NAME = os.getenv("TRANSLATION_MODEL", "google/flan-t5-small")
LLM_MODEL_NAME = os.getenv("LLM_MODEL", "google/flan-t5-small")

# Predefined Knowledge Categories (Restricted to Three Canonical Domains)
CANONICAL_CATEGORIES = [
    "Philosophy",
    "Ayurveda",
    "Astronomy & Mathematics"
]
THREE_DOMAINS = CANONICAL_CATEGORIES

# Flask Secret Key
SECRET_KEY = os.getenv("SECRET_KEY", "sanskrypt-major-secret-key-2026")
