
from pypdf import PdfReader
from pypdf.errors import PdfReadError


MAX_PAGES = 500


def extract_text(pdf_path: str) -> str:
    try:
        reader = PdfReader(pdf_path, strict=False)

        if reader.is_encrypted:
            raise ValueError("Password-protected PDFs are not supported.")

        if len(reader.pages) > MAX_PAGES:
            raise ValueError(
                f"PDF exceeds the maximum limit of {MAX_PAGES} pages."
            )

        extracted_pages = []

        for page_number, page in enumerate(reader.pages, start=1):
            page_text = page.extract_text()

            if page_text and page_text.strip():
                extracted_pages.append(
                    f"[Page {page_number}]\n{page_text.strip()}"
                )

        return "\n\n".join(extracted_pages)

    except (PdfReadError, OSError) as exc:
        raise ValueError("Could not read this PDF.") from exc
