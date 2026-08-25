import fitz  
import re




def extract_text_from_pdf(pdf_path: str) -> str:
    """Step 1: Pull raw text out of every page of the PDF."""
    doc = fitz.open(pdf_path)
    pages_text = []
    for page in doc:
        pages_text.append(page.get_text())
    doc.close()
    return "\n\n".join(pages_text)


def clean_text(raw_text: str) -> str:
    """Step 2: Fix common PDF-extraction artifacts."""

    # Fix hyphenated words split across lines: "mate-\nrial" -> "material"
    text = re.sub(r"-\n", "", raw_text)

    # Collapse single newlines (mid-sentence wraps) into spaces,
    # but preserve double newlines (real paragraph breaks).
    # Trick: temporarily protect double newlines, collapse the rest, restore.
    text = re.sub(r"\n{2,}", "<PARA>", text)
    text = re.sub(r"\n", " ", text)
    text = text.replace("<PARA>", "\n\n")

    # Collapse repeated spaces
    text = re.sub(r"[ \t]{2,}", " ", text)

    # Strip lines that repeat identically many times (likely headers/footers)
    lines = text.split("\n")
    line_counts = {}
    for line in lines:
        stripped = line.strip()
        if stripped:
            line_counts[stripped] = line_counts.get(stripped, 0) + 1

    threshold = 3  # appears on 3+ pages -> likely a header/footer
    cleaned_lines = [
        line for line in lines
        if line.strip() == "" or line_counts.get(line.strip(), 0) < threshold
    ]
    text = "\n".join(cleaned_lines)

    return text.strip()


def chunk_text(text: str, min_words: int = 40, max_words: int = 150) -> list[str]:
    """Step 3: Split cleaned text into paragraph-sized chunks."""

    # Primary split: on paragraph breaks
    raw_chunks = [c.strip() for c in text.split("\n\n") if c.strip()]

    # Merge chunks that are too short into the next chunk
    merged_chunks = []
    buffer = ""
    for chunk in raw_chunks:
        buffer = (buffer + " " + chunk).strip() if buffer else chunk
        if len(buffer.split()) >= min_words:
            merged_chunks.append(buffer)
            buffer = ""
    if buffer:
        # leftover short piece — attach to the last chunk if one exists
        if merged_chunks:
            merged_chunks[-1] += " " + buffer
        else:
            merged_chunks.append(buffer)

    # Split chunks that are too long, on sentence boundaries
    final_chunks = []
    for chunk in merged_chunks:
        words = chunk.split()
        if len(words) <= max_words:
            final_chunks.append(chunk)
        else:
            sentences = re.split(r"(?<=[.!?]) +", chunk)
            current = ""
            for sentence in sentences:
                if len((current + " " + sentence).split()) > max_words and current:
                    final_chunks.append(current.strip())
                    current = sentence
                else:
                    current = (current + " " + sentence).strip()
            if current:
                final_chunks.append(current.strip())

    return final_chunks


def ingest_pdf(pdf_path: str) -> list[str]:
    """Full Stage 1 pipeline: PDF -> list of clean text chunks."""
    raw = extract_text_from_pdf(pdf_path)
    cleaned = clean_text(raw)
    chunks = chunk_text(cleaned)
    return chunks


