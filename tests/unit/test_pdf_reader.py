
from pypdf import PdfWriter

from app.pdf_reader import extract_text


def create_pdf(path, pages=1):
    """Create a simple PDF for testing."""

    writer = PdfWriter()

    for _ in range(pages):
        writer.add_blank_page(width=300, height=300)

    with open(path, "wb") as pdf_file:
        writer.write(pdf_file)


def test_empty_pdf_returns_empty_text(tmp_path):
    """A valid PDF with blank pages should return empty text."""

    pdf_path = tmp_path / "empty.pdf"
    create_pdf(pdf_path)

    result = extract_text(str(pdf_path))

    assert result == ""


def test_invalid_pdf_raises_value_error(tmp_path):
    """Invalid PDF content should produce a readable error."""

    pdf_path = tmp_path / "invalid.pdf"
    pdf_path.write_text("This is not a valid PDF.")

    try:
        extract_text(str(pdf_path))
        assert False, "Expected ValueError was not raised."
    except ValueError as exc:
        assert str(exc) == "Could not read this PDF."


def test_encrypted_pdf_is_rejected(tmp_path):
    """Password-protected PDFs should not be supported."""

    pdf_path = tmp_path / "encrypted.pdf"

    writer = PdfWriter()
    writer.add_blank_page(width=300, height=300)
    writer.encrypt("test-password")

    with open(pdf_path, "wb") as pdf_file:
        writer.write(pdf_file)

    try:
        extract_text(str(pdf_path))
        assert False, "Expected ValueError was not raised."
    except ValueError as exc:
        assert str(exc) == "Password-protected PDFs are not supported."


def test_page_limit_is_enforced(tmp_path, monkeypatch):
    """PDFs exceeding the configured page limit should be rejected."""

    import app.pdf_reader as pdf_reader

    pdf_path = tmp_path / "large.pdf"
    create_pdf(pdf_path, pages=2)

    monkeypatch.setattr(pdf_reader, "MAX_PAGES", 1)

    try:
        extract_text(str(pdf_path))
        assert False, "Expected ValueError was not raised."
    except ValueError as exc:
        assert "maximum limit of 1 pages" in str(exc)


def test_extracted_text_contains_page_numbers(tmp_path, monkeypatch):
    """Extracted text should include page markers."""

    import app.pdf_reader as pdf_reader

    class FakePage:
        def __init__(self, text):
            self.text = text

        def extract_text(self):
            return self.text

    class FakeReader:
        def __init__(self, path, strict=False):
            self.is_encrypted = False
            self.pages = [
                FakePage("First page content"),
                FakePage("   "),
                FakePage("Third page content"),
            ]

    monkeypatch.setattr(pdf_reader, "PdfReader", FakeReader)

    result = extract_text("fake.pdf")

    assert "[Page 1]" in result
    assert "First page content" in result
    assert "[Page 3]" in result
    assert "Third page content" in result
    assert "[Page 2]" not in result
