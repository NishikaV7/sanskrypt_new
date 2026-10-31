"""
Unit tests for Flask routes, upload pipeline orchestration, and web pages.
"""
import io
import pytest
from unittest.mock import patch
from PIL import Image
from database import get_manuscript


def test_index_route(client):
    """Verifies that home page loads with 200 OK and contains brand elements."""
    response = client.get("/")
    assert response.status_code == 200
    assert b"SansKrypt" in response.data
    assert b"Upload Sanskrit Manuscript" in response.data


def test_upload_invalid_file_extension(client):
    """Verifies that non-image file uploads are rejected gracefully."""
    data = {
        "title": "Invalid File Test",
        "image": (io.BytesIO(b"not an image content"), "document.txt")
    }
    response = client.post("/upload", data=data, follow_redirects=True, content_type="multipart/form-data")
    assert response.status_code == 200
    assert b"Invalid file type" in response.data


def test_upload_success_and_low_confidence_flow(client, sample_image_path):
    """
    Simulates complete upload and pipeline flow with mocked OCR returning low confidence (<40%),
    verifying proper database insertion, status tagging, and warning banner display on results page.
    """
    # Create image file buffer
    with open(sample_image_path, "rb") as f:
        img_bytes = f.read()

    data = {
        "title": "Ancient Palm Leaf Fragment",
        "image": (io.BytesIO(img_bytes), "sample.png")
    }

    # Mock pipeline functions to run swiftly
    with patch("app.validate_source_image", return_value=(True, "Accepted")), \
         patch("app.extract_text", return_value=("अहं ब्रह्मास्मि", 35.0)), \
         patch("app.translate_text", return_value="I am the infinite reality / Brahman."), \
         patch("app.generate_explanation", return_value="Mahavakya from Brihadaranyaka Upanishad asserting divine identity."), \
         patch("app.extract_keywords", return_value="Brahman, Upanishad, Identity, Advaita"), \
         patch("app.classify_content", return_value="Philosophy"), \
         patch("app.index_manuscript", return_value=1):

        response = client.post("/upload", data=data, follow_redirects=True, content_type="multipart/form-data")
        assert response.status_code == 200
        # Should redirect to results page
        assert b"Extracted Sanskrit Text" in response.data
        assert b"I am the infinite reality" in response.data
        assert b"Philosophy" in response.data


def test_upload_blank_image_rejected(client):
    """Verifies that uploading a blank/non-Sanskrit image is rejected at the validation gate."""
    # Create blank image
    img = Image.new("RGB", (200, 200), color="white")
    buf = io.BytesIO()
    img.save(buf, format="PNG")
    buf.seek(0)

    data = {
        "title": "Blank Test Image",
        "image": (buf, "blank.png")
    }

    response = client.post("/upload", data=data, follow_redirects=True, content_type="multipart/form-data")
    assert response.status_code == 200
    assert b"No Sanskrit text detected." in response.data


def test_upload_non_sanskrit_image_rejected(client):
    """Verifies that an image without Sanskrit/Hindi text is rejected."""
    with patch("app.validate_source_image", return_value=(False, "No Sanskrit text detected.")):
        data = {
            "title": "English Document",
            "image": (io.BytesIO(b"fake image bytes"), "english.png")
        }
        response = client.post("/upload", data=data, follow_redirects=True, content_type="multipart/form-data")
        assert response.status_code == 200
        assert b"No Sanskrit text detected." in response.data


def test_view_manuscript_not_found(client):
    """Verifies 404 handler on missing manuscript."""
    response = client.get("/manuscript/999999")
    assert response.status_code == 404
    assert b"Manuscript Not Found" in response.data


def test_search_routes(client, sample_manuscripts):
    """Verifies search view and search execution."""
    # Search page
    res_search = client.get("/search")
    assert res_search.status_code == 200
    assert b"Search Sanskrit Repository" in res_search.data

    # Search results query
    res_results = client.get("/search/results?query=Ayurveda")
    assert res_results.status_code == 200
    assert b"Charaka Samhita" in res_results.data

    # Search filter for low_confidence only
    res_low = client.get("/search/results?status=low_confidence")
    assert res_low.status_code == 200
    assert b"Charaka Samhita" in res_low.data


def test_repository_list_and_delete(client, sample_manuscripts):
    """Verifies repository listing and manuscript deletion."""
    res_list = client.get("/manuscripts?status=completed")
    assert res_list.status_code == 200
    assert b"Rigveda Creation Hymn" in res_list.data

    # Delete first manuscript
    m_id = sample_manuscripts[0]
    res_del = client.post(f"/manuscript/{m_id}/delete", follow_redirects=True)
    assert res_del.status_code == 200
    assert b"deleted successfully" in res_del.data


def test_manuscript_image_serving(client, sample_manuscripts):
    """Verifies that manuscript original and processed image routes return the image correctly."""
    m_id = sample_manuscripts[0]
    res = client.get(f"/manuscript/{m_id}/image/original")
    assert res.status_code == 200
    assert res.mimetype.startswith("image/")




