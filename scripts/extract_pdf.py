#!/usr/bin/env python3
"""
extract_pdf.py -- PDF extraction engine for ai4reading paper discovery tool.

Extracts structured content from cached ArXiv PDFs to support the /deep-dive skill:
table of contents, high-value section detection, and raw page text.

Usage:
    python extract_pdf.py --id 2602.03837 --toc
    python extract_pdf.py --id 2602.03837 --pages "11-18,45-49"
    python extract_pdf.py --id 2602.03837 --auto
"""

import sys
import re
import json
import argparse
import urllib.request
from pathlib import Path

sys.stdout.reconfigure(encoding="utf-8")

try:
    import fitz  # PyMuPDF
except ImportError:
    print(json.dumps({"error": "PyMuPDF not installed. Run: pip install pymupdf"}))
    sys.exit(1)

# ── Constants ────────────────────────────────────────────────────────────────

PAPERS_DIR = Path(__file__).parent.parent / "papers"
ARXIV_PDF_URL = "https://arxiv.org/pdf/{id}"
USER_AGENT = "Mozilla/5.0 (compatible; ai4reading/1.0)"
MAX_SECTION_PAGES = 15

HIGH_VALUE_KEYWORDS = [
    "case study", "case studies", "experiment", "example", "counterexample",
    "result", "evaluation", "finding", "demo", "application", "illustration",
    "proof", "technique", "contribution", "ablation", "analysis",
    # Research process & collaboration
    "vibe", "neuro-symbolic", "conjecture", "derivation", "resolution",
    "bug detection", "cross-pollination", "review", "verification",
    # Algorithm & math sections (level-2 subsections in applied papers)
    "algorithm", "bound", "approximation", "complexity", "optimization",
    "information theory", "mechanism design", "cryptography", "physics",
    "graph theory", "geometry", "streaming", "machine learning",
]


# ── PDF Cache ────────────────────────────────────────────────────────────────

def clean_id(arxiv_id: str) -> str:
    """Strip version suffix: '2602.03837v2' -> '2602.03837'"""
    return re.sub(r"v\d+$", "", arxiv_id.strip())


def find_cached_pdf(arxiv_id: str) -> Path | None:
    """Look for an already-downloaded PDF in the papers/ directory."""
    base = clean_id(arxiv_id)
    PAPERS_DIR.mkdir(parents=True, exist_ok=True)
    for candidate in [
        PAPERS_DIR / f"{base}.pdf",
        PAPERS_DIR / f"{arxiv_id}.pdf",
    ]:
        if candidate.exists():
            return candidate
    return None


def download_pdf(arxiv_id: str) -> Path:
    """Download PDF from ArXiv and cache it locally."""
    base = clean_id(arxiv_id)
    dest = PAPERS_DIR / f"{base}.pdf"
    url = ARXIV_PDF_URL.format(id=arxiv_id)
    req = urllib.request.Request(url, headers={"User-Agent": USER_AGENT})
    with urllib.request.urlopen(req, timeout=60) as resp:
        data = resp.read()
    PAPERS_DIR.mkdir(parents=True, exist_ok=True)
    dest.write_bytes(data)
    return dest


def get_pdf_path(arxiv_id: str) -> Path:
    """Return local PDF path, downloading if necessary."""
    cached = find_cached_pdf(arxiv_id)
    if cached:
        return cached
    return download_pdf(arxiv_id)


# ── ToC Extraction ───────────────────────────────────────────────────────────

def extract_toc(doc: fitz.Document) -> list:
    """Extract table of contents. Falls back to font-size heuristic if no ToC."""
    raw_toc = doc.get_toc()
    if raw_toc:
        return [
            {"title": title, "level": level, "page": page}
            for level, title, page in raw_toc
        ]
    # Fallback: scan first 12 pages for large-font text (section headers)
    entries = []
    for pg_idx in range(min(12, len(doc))):
        page = doc[pg_idx]
        blocks = page.get_text("dict")["blocks"]
        for block in blocks:
            if block.get("type") != 0:
                continue
            for line in block.get("lines", []):
                for span in line.get("spans", []):
                    text = span.get("text", "").strip()
                    size = span.get("size", 0)
                    if size >= 13 and 4 < len(text) < 100:
                        entries.append({
                            "title": text,
                            "level": 1 if size >= 16 else 2,
                            "page": pg_idx + 1,
                        })
    # Deduplicate by (page, title prefix)
    seen = set()
    result = []
    for e in entries:
        key = (e["page"], e["title"][:30])
        if key not in seen:
            seen.add(key)
            result.append(e)
    return result


def is_high_value(title: str) -> tuple:
    """Check if a section title contains high-value keywords."""
    lower = title.lower()
    matched = [kw for kw in HIGH_VALUE_KEYWORDS if kw in lower]
    if matched:
        return True, ", ".join(matched)
    return False, ""


def build_high_value_sections(toc: list, total_pages: int) -> list:
    """Identify high-value sections and compute their page ranges."""
    result = []
    for i, entry in enumerate(toc):
        hv, reason = is_high_value(entry["title"])
        if not hv:
            continue
        page_start = entry["page"]
        page_end = total_pages
        for j in range(i + 1, len(toc)):
            if toc[j]["level"] <= entry["level"]:
                page_end = toc[j]["page"] - 1
                break
        page_end = max(page_start, page_end)  # guard against page_end < page_start
        page_end = min(page_end, page_start + MAX_SECTION_PAGES - 1)
        result.append({
            "title": entry["title"],
            "level": entry["level"],
            "page_start": page_start,
            "page_end": page_end,
            "reason": f"contains: {reason}",
        })
    return result


# ── Text Extraction ──────────────────────────────────────────────────────────

def parse_page_ranges(spec: str, total_pages: int) -> list:
    """Parse '11-18,45-49' into list of 1-indexed page numbers."""
    pages = []
    for part in spec.split(","):
        part = part.strip()
        if "-" in part:
            a, b = part.split("-", 1)
            pages.extend(range(int(a), int(b) + 1))
        elif part:
            pages.append(int(part))
    return [p for p in pages if 1 <= p <= total_pages]


def extract_text_pages(doc: fitz.Document, page_nums: list) -> str:
    """Extract and clean text from specified 1-indexed pages."""
    parts = []
    for pn in page_nums:
        page = doc[pn - 1]
        text = page.get_text()
        # Join soft-hyphenated line breaks, normalize whitespace
        text = re.sub(r"-\n(\w)", r"\1", text)
        text = re.sub(r"[ \t]+", " ", text)
        text = re.sub(r"\n{3,}", "\n\n", text)
        parts.append(f"--- Page {pn} ---\n{text.strip()}")
    return "\n\n".join(parts)


def get_paper_title(doc: fitz.Document) -> str:
    """Extract paper title from first page (first non-empty line)."""
    if not doc:
        return ""
    return doc[0].get_text().strip().split("\n")[0].strip()[:200]


# ── CLI Modes ────────────────────────────────────────────────────────────────

def mode_toc(doc: fitz.Document, arxiv_id: str, pdf_path: Path) -> dict:
    """Return ToC + high-value section list (no text)."""
    toc = extract_toc(doc)
    high_value = build_high_value_sections(toc, len(doc))
    return {
        "id": arxiv_id,
        "title": get_paper_title(doc),
        "total_pages": len(doc),
        "pdf_path": str(pdf_path),
        "toc": toc,
        "high_value_sections": high_value,
    }


def mode_pages(doc: fitz.Document, arxiv_id: str, pages_spec: str) -> dict:
    """Extract text from specific page ranges."""
    page_nums = parse_page_ranges(pages_spec, len(doc))
    text = extract_text_pages(doc, page_nums)
    return {
        "id": arxiv_id,
        "pages_requested": pages_spec,
        "pages_extracted": page_nums,
        "text": text,
    }


def mode_auto(doc: fitz.Document, arxiv_id: str) -> dict:
    """Auto-detect and extract all high-value sections with text."""
    toc = extract_toc(doc)
    high_value = build_high_value_sections(toc, len(doc))
    sections = []
    for sec in high_value:
        pages = list(range(sec["page_start"], sec["page_end"] + 1))
        text = extract_text_pages(doc, pages)
        sections.append({
            "title": sec["title"],
            "page_start": sec["page_start"],
            "page_end": sec["page_end"],
            "reason": sec["reason"],
            "text": text,
        })
    return {
        "id": arxiv_id,
        "title": get_paper_title(doc),
        "total_pages": len(doc),
        "high_value_sections": sections,
    }


# ── Main ─────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description="Extract structured content from ArXiv PDFs."
    )
    parser.add_argument("--id", "-i", required=True,
                        help="ArXiv ID (e.g. 2602.03837 or 2602.03837v2)")
    parser.add_argument("--toc", action="store_true",
                        help="Extract table of contents + high-value section list")
    parser.add_argument("--pages", metavar="RANGES",
                        help="Extract text from page ranges, e.g. '11-18,45-49'")
    parser.add_argument("--auto", action="store_true",
                        help="Auto-extract all high-value sections with text")
    args = parser.parse_args()

    try:
        pdf_path = get_pdf_path(args.id)
    except Exception as e:
        print(json.dumps({"error": f"Failed to get PDF: {e}"}))
        sys.exit(1)

    try:
        doc = fitz.open(str(pdf_path))
    except Exception as e:
        print(json.dumps({"error": f"Failed to open PDF: {e}"}))
        sys.exit(1)

    try:
        if args.toc:
            result = mode_toc(doc, args.id, pdf_path)
        elif args.pages:
            result = mode_pages(doc, args.id, args.pages)
        elif args.auto:
            result = mode_auto(doc, args.id)
        else:
            parser.print_help()
            sys.exit(0)
    except Exception as e:
        print(json.dumps({"error": f"Extraction failed: {e}"}))
        sys.exit(1)

    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
