"""Document routing: extractable text -> the Krun TEXT path (no vision tower); visual documents -> the vision tower.

Fast path (``route == "text"``): txt / md / html / json directly, PDF via ``pypdf`` and DOCX via ``python-docx`` when
those optional packages are installed. The extracted text becomes (part of) the decision context, so the request is
a normal text request: same tokens path, same latency class, no vision weights.

Visual path (``route == "visual"``): images of pages (scans, receipts, invoices, forms, screenshots, layout-sensitive
PDFs). A PDF page is rendered with ``pypdfium2`` when installed; images are passed through. The pages go to the SAME
vision tower as images (one perception block), tagged as modality "document".

A document goes visual when: the caller sets ``visual_required``; or extraction is impossible (no extractor / scanned);
or the extracted text is shorter than ``min_chars`` per page (typical of scans and receipts).
"""

from __future__ import annotations

import html
import io
import json
import re
from dataclasses import dataclass, field
from typing import Any

from .config import DocumentConfig

TEXT_MIMES = {"text/plain", "text/markdown", "text/csv", "application/json", "text/html"}


@dataclass
class DocumentInput:
    data: bytes | str
    mime: str
    document_type: str | None = None       # invoice, receipt, form, contract, screenshot, ...
    visual_required: bool = False          # caller knows layout matters
    pages: list[Any] | None = None         # already-rendered page images (tensors), optional


@dataclass
class DocumentRoute:
    route: str                             # "text" | "visual"
    text: str | None = None
    pages: list[Any] = field(default_factory=list)
    reason: str = ""
    meta: dict[str, Any] = field(default_factory=dict)


def _html_to_text(s: str) -> str:
    s = re.sub(r"(?is)<(script|style).*?</\1>", " ", s)
    s = re.sub(r"(?s)<[^>]+>", " ", s)
    return " ".join(html.unescape(s).split())


def extract_text(doc: DocumentInput) -> tuple[str | None, int, str]:
    """-> (text or None, page count, extractor name). Never raises for a missing optional dependency."""
    data = doc.data
    if doc.mime in TEXT_MIMES:
        s = data.decode("utf-8", "replace") if isinstance(data, bytes) else data
        if doc.mime == "text/html":
            s = _html_to_text(s)
        elif doc.mime == "application/json":
            s = json.dumps(json.loads(s), ensure_ascii=False)
        return s, 1, "builtin"
    if doc.mime == "application/pdf":
        try:
            from pypdf import PdfReader  # optional
        except ImportError:
            return None, 0, "missing:pypdf"
        r = PdfReader(io.BytesIO(data if isinstance(data, bytes) else data.encode()))
        return "\n".join((p.extract_text() or "") for p in r.pages), len(r.pages), "pypdf"
    if doc.mime in ("application/vnd.openxmlformats-officedocument.wordprocessingml.document",):
        try:
            import docx  # optional (python-docx)
        except ImportError:
            return None, 0, "missing:python-docx"
        d = docx.Document(io.BytesIO(data))
        return "\n".join(p.text for p in d.paragraphs), 1, "python-docx"
    return None, 0, "none"


def render_pdf_pages(data: bytes, max_pages: int, scale: float = 1.5) -> list[Any]:
    try:
        import pypdfium2 as pdfium  # optional
    except ImportError:
        return []
    pdf = pdfium.PdfDocument(data)
    return [pdf[i].render(scale=scale).to_pil() for i in range(min(len(pdf), max_pages))]


class DocumentRouter:
    def __init__(self, cfg: DocumentConfig):
        self.cfg = cfg

    def route(self, doc: DocumentInput) -> DocumentRoute:
        if doc.mime.startswith("image/") or doc.pages:
            if not self.cfg.visual_fallback:
                raise ValueError("visual document received but document.visual_fallback is disabled")
            return DocumentRoute("visual", pages=list(doc.pages or [doc.data]), reason="image input",
                                 meta={"document_type": doc.document_type})
        text, n_pages, how = (None, 0, "skipped") if doc.visual_required else extract_text(doc)
        enough = text is not None and len(text.strip()) >= self.cfg.min_chars * max(1, n_pages)   # per page
        if self.cfg.text_fast_path and enough and not doc.visual_required:
            return DocumentRoute("text", text=text, reason=f"extracted by {how}",
                                 meta={"pages": n_pages, "chars": len(text), "document_type": doc.document_type})
        if not self.cfg.visual_fallback:
            if text:
                return DocumentRoute("text", text=text, reason=f"short text ({how}); visual fallback disabled")
            raise ValueError(f"document has no extractable text ({how}) and visual fallback is disabled")
        pages = render_pdf_pages(doc.data, self.cfg.max_pages) if doc.mime == "application/pdf" and isinstance(doc.data, bytes) else []
        reason = "visual_required" if doc.visual_required else (f"too little text ({how})" if text is not None else f"no text ({how})")
        return DocumentRoute("visual", text=text, pages=pages, reason=reason,
                             meta={"document_type": doc.document_type, "rendered": len(pages)})
