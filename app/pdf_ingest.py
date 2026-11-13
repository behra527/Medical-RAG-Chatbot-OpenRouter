"""Page-faithful text extraction and section-aware, overlapping chunks."""
from __future__ import annotations
from dataclasses import dataclass
from hashlib import sha256
import re
import fitz

SECTION_RE = re.compile(r'^\s*(\d+(?:\.\d+){1,2})\s+([^\n]{4,95})\s*$')
RECOMMENDATION_RE = re.compile(r'\b\d+\.\d+\.\d+\b')
PAGE_FOOTER_RE = re.compile(r'^(?:©\s*NICE|Page\s+\d+\s+of\s+\d+|Hypertension in adults: diagnosis and management\s*\(NG136\))')


@dataclass
class Chunk:
    chunk_id: str
    page: int
    section: str
    recommendation: str | None
    text: str


def section_text(line: str) -> str | None:
    line = line.strip()
    match = SECTION_RE.match(line)
    if not match or '.....' in line or line.endswith(('.', '[2019]', '[2023]')):
        return None
    number, name = match.groups()
    # Recommendation paragraphs begin 1.2.3. Section headings only use one decimal.
    if number.count('.') != 1 or len(name) > 80:
        return None
    return f'{number} {name}'


def extract_chunks(pdf: bytes, *, words_per_chunk: int = 190, overlap: int = 40) -> list[Chunk]:
    if not pdf.startswith(b'%PDF'):
        raise ValueError('Invalid file: expected a PDF document')
    if not 0 <= overlap < words_per_chunk:
        raise ValueError('Invalid chunk overlap')
    try:
        document = fitz.open(stream=pdf, filetype='pdf')
    except Exception as exc:
        raise ValueError('Could not open PDF; it may be corrupt or password-protected') from exc
    try:
        if document.needs_pass:
            raise ValueError('Password-protected PDFs are not supported')
        result: list[Chunk] = []
        current_section = 'Unspecified section'
        for page_number, page in enumerate(document, start=1):
            lines = page.get_text('text', sort=True).splitlines()
            segments: list[tuple[str, str]] = []
            active: list[str] = []
            for raw_line in lines:
                line = ' '.join(raw_line.split())
                if not line or PAGE_FOOTER_RE.match(line):
                    continue
                if len(line) > 3000:
                    continue
                heading = section_text(line)
                if heading:
                    if active:
                        segments.append((current_section, ' '.join(active)))
                        active = []
                    current_section = heading
                else:
                    active.append(line)
            if active:
                segments.append((current_section, ' '.join(active)))
            for section, section_body in segments:
                words = section_body.split()
                if not words:
                    continue
                stride = words_per_chunk - overlap
                for start in range(0, len(words), stride):
                    piece = words[start:start + words_per_chunk]
                    if len(piece) < 12 and start != 0:
                        break
                    passage = ' '.join(piece)
                    rec = RECOMMENDATION_RE.search(passage)
                    result.append(Chunk(
                        chunk_id=sha256(f'{page_number}:{section}:{start}:{passage}'.encode()).hexdigest()[:24],
                        page=page_number,
                        section=section,
                        recommendation=rec.group(0) if rec else None,
                        text=passage,
                    ))
                    if start + words_per_chunk >= len(words):
                        break
        if not result:
            raise ValueError('PDF contains no extractable text. OCR is needed for scanned PDFs.')
        return result
    finally:
        document.close()
