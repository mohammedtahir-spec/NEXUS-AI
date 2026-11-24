
SECTION_HEADINGS = {
    "professional summary",
    "executive summary",
    "summary",
    "introduction",
    "overview",
    "technical skills",
    "skills",
    "projects",
    "experience",
    "work experience",
    "professional experience",
    "education",
    "education & training",
    "training",
    "training and courses",
    "courses",
    "professional development",
    "certifications",
    "certificates",
    "languages",
    "references",
    "requirements",
    "scope",
    "objectives",
    "conclusion",
    "appendix",
}


def is_section_heading(line: str) -> bool:
    """Check whether a line looks like a section heading."""

    cleaned = line.strip().lstrip("#").strip()
    normalized = cleaned.lower().rstrip(":").strip()

    if normalized in SECTION_HEADINGS:
        return True

    if (
        len(cleaned) <= 80
        and any(char.isalpha() for char in cleaned)
        and cleaned == cleaned.upper()
        and not cleaned.startswith(("•", "-", "*"))
    ):
        return True

    return False


def get_word_safe_overlap(text: str, overlap: int) -> str:
    """Return overlap using complete words."""

    if overlap <= 0:
        return ""

    words = text.split()
    selected = []
    length = 0

    for word in reversed(words):
        added_length = len(word) + (1 if selected else 0)

        if length + added_length > overlap:
            break

        selected.insert(0, word)
        length += added_length

    return " ".join(selected)


def split_long_paragraph(
    paragraph: str,
    max_length: int,
    overlap: int = 0
) -> list[str]:
    """Split long paragraphs while keeping overlap inside the size limit."""

    if max_length <= 0:
        raise ValueError("Maximum chunk length must be positive.")

    words = paragraph.split()

    chunks = []
    current = ""

    for word in words:

        # A single word longer than the limit must be split.
        if len(word) > max_length:

            if current:
                chunks.append(current)
                current = ""

            for start in range(0, len(word), max_length):
                chunks.append(word[start:start + max_length])

            continue

        if not current:
            current = word

        elif len(current) + len(word) + 1 <= max_length:
            current += " " + word

        else:
            chunks.append(current)

            # Reserve space for the next word before adding overlap.
            overlap_text = get_word_safe_overlap(
                current,
                min(overlap, max_length - 1)
            )

            if (
                overlap_text
                and len(overlap_text) + len(word) + 1 <= max_length
            ):
                current = overlap_text + " " + word
            else:
                current = word

    if current:
        chunks.append(current)

    return chunks


def chunk_section(
    heading: str,
    paragraphs: list[str],
    chunk_size: int,
    overlap: int
) -> list[str]:
    """Create chunks while preserving section information."""

    prefix = f"{heading}\n" if heading else ""
    max_content = chunk_size - len(prefix)

    if max_content <= 0:
        raise ValueError("Section heading is too long.")

    chunks = []
    current = ""

    def save_chunk():
        if current.strip():
            chunks.append(prefix + current.strip())

    for paragraph in paragraphs:

        paragraph = paragraph.strip()

        if not paragraph:
            continue

        if len(paragraph) > max_content:

            save_chunk()
            current = ""

            parts = split_long_paragraph(
                paragraph,
                max_content,
                overlap=min(overlap, max_content - 1)
            )

            chunks.extend(prefix + part for part in parts)

            continue

        if not current:
            current = paragraph

        elif len(current) + len(paragraph) + 1 <= max_content:
            current += "\n" + paragraph

        else:
            previous = current
            save_chunk()

            available_overlap = max(
                0,
                max_content - len(paragraph) - 1
            )

            overlap_text = get_word_safe_overlap(
                previous,
                min(overlap, available_overlap)
            )

            if overlap_text:
                current = overlap_text + "\n" + paragraph
            else:
                current = paragraph

    save_chunk()

    return chunks


def chunk_text(
    text: str,
    chunk_size: int = 500,
    overlap: int = 100
) -> list[str]:
    """Split a document into section-aware, word-safe chunks."""

    if chunk_size <= 0 or overlap < 0 or overlap >= chunk_size:
        raise ValueError("Invalid chunk size or overlap.")

    lines = [
        line.strip()
        for line in text.splitlines()
        if line.strip()
    ]

    sections = []
    current_heading = ""
    current_paragraphs = []

    for line in lines:

        if is_section_heading(line):

            if current_paragraphs:
                sections.append(
                    (current_heading, current_paragraphs)
                )

            current_heading = line
            current_paragraphs = []

        else:
            current_paragraphs.append(line)

    if current_paragraphs:
        sections.append(
            (current_heading, current_paragraphs)
        )

    all_chunks = []

    for heading, paragraphs in sections:

        all_chunks.extend(
            chunk_section(
                heading,
                paragraphs,
                chunk_size,
                overlap
            )
        )

    return all_chunks
