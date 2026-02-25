#!/usr/bin/env python3
"""
fetch_arxiv.py — Search ArXiv for AI/ML papers via the Atom API.

Usage:
    python3 fetch_arxiv.py --query "LLM reasoning" [--category cs.LG] [--days 7] [--limit 10] [--sort-by relevance|date]

Output: JSON array of paper objects.
"""

import argparse
import json
import re
import sys
import urllib.parse
import urllib.request
from datetime import datetime, timedelta, timezone

try:
    import feedparser
    HAS_FEEDPARSER = True
except ImportError:
    HAS_FEEDPARSER = False


ARXIV_API = "https://export.arxiv.org/api/query"

VALID_CATEGORIES = [
    "cs.AI", "cs.LG", "cs.CL", "cs.CV", "cs.RO", "cs.NE",
    "cs.IR", "cs.HC", "cs.MA", "stat.ML",
]

SORT_MAP = {
    "relevance": "relevance",
    "date": "submittedDate",
    "lastUpdated": "lastUpdatedDate",
}


def parse_arxiv_id(url_or_id: str) -> str:
    """Extract ArXiv ID from a URL or raw ID string."""
    url_or_id = url_or_id.strip().rstrip("/")
    # Match patterns like abs/2312.11805 or pdf/2312.11805
    m = re.search(r"arxiv\.org/(?:abs|pdf)/([0-9]{4}\.[0-9]{4,5}(?:v\d+)?)", url_or_id)
    if m:
        return m.group(1)
    # Plain ID pattern
    m = re.match(r"^([0-9]{4}\.[0-9]{4,5}(?:v\d+)?)$", url_or_id)
    if m:
        return m.group(1)
    return url_or_id


def build_query(query: str, category: str | None) -> str:
    parts = []
    if query:
        parts.append(f"all:{query}")
    if category:
        parts.append(f"cat:{category}")
    return " AND ".join(parts) if parts else "all:AI"


def fetch_with_urllib(query_str: str, start: int, max_results: int,
                      sort_by: str, sort_order: str) -> str:
    params = urllib.parse.urlencode({
        "search_query": query_str,
        "start": start,
        "max_results": max_results,
        "sortBy": sort_by,
        "sortOrder": sort_order,
    })
    url = f"{ARXIV_API}?{params}"
    req = urllib.request.Request(url, headers={"User-Agent": "ai4reading/1.0"})
    with urllib.request.urlopen(req, timeout=30) as resp:
        return resp.read().decode("utf-8")


def parse_feed_urllib(xml: str) -> list[dict]:
    """Minimal Atom XML parser using stdlib only."""
    import xml.etree.ElementTree as ET
    NS = {
        "atom": "http://www.w3.org/2005/Atom",
        "arxiv": "http://arxiv.org/schemas/atom",
    }
    root = ET.fromstring(xml)
    papers = []
    for entry in root.findall("atom:entry", NS):
        def text(tag):
            el = entry.find(tag, NS)
            return el.text.strip() if el is not None and el.text else ""

        arxiv_id = text("atom:id").split("/abs/")[-1]
        title = re.sub(r"\s+", " ", text("atom:title"))
        abstract = re.sub(r"\s+", " ", text("atom:summary"))
        submitted = text("atom:published")
        updated = text("atom:updated")

        authors = [
            a.findtext("atom:name", "", NS).strip()
            for a in entry.findall("atom:author", NS)
        ]

        categories = [
            c.get("term", "")
            for c in entry.findall("atom:category", NS)
        ]

        papers.append({
            "id": arxiv_id,
            "title": title,
            "authors": authors,
            "abstract": abstract,
            "url": f"https://arxiv.org/abs/{arxiv_id}",
            "pdf_url": f"https://arxiv.org/pdf/{arxiv_id}",
            "submitted": submitted,
            "updated": updated,
            "categories": categories,
        })
    return papers


def parse_feed_feedparser(xml: str) -> list[dict]:
    feed = feedparser.parse(xml)
    papers = []
    for entry in feed.entries:
        arxiv_id = entry.get("id", "").split("/abs/")[-1]
        title = re.sub(r"\s+", " ", entry.get("title", ""))
        abstract = re.sub(r"\s+", " ", entry.get("summary", ""))
        submitted = entry.get("published", "")
        updated = entry.get("updated", "")
        authors = [a.get("name", "") for a in entry.get("authors", [])]
        categories = [t.get("term", "") for t in entry.get("tags", [])]
        papers.append({
            "id": arxiv_id,
            "title": title,
            "authors": authors,
            "abstract": abstract,
            "url": f"https://arxiv.org/abs/{arxiv_id}",
            "pdf_url": f"https://arxiv.org/pdf/{arxiv_id}",
            "submitted": submitted,
            "updated": updated,
            "categories": categories,
        })
    return papers


def filter_by_days(papers: list[dict], days: int) -> list[dict]:
    if days <= 0:
        return papers
    cutoff = datetime.now(timezone.utc) - timedelta(days=days)
    filtered = []
    for p in papers:
        try:
            dt = datetime.fromisoformat(p["submitted"].replace("Z", "+00:00"))
            if dt >= cutoff:
                filtered.append(p)
        except (ValueError, KeyError):
            filtered.append(p)  # Keep if can't parse date
    return filtered


def main():
    parser = argparse.ArgumentParser(description="Search ArXiv papers")
    parser.add_argument("--query", "-q", default="", help="Search query")
    parser.add_argument("--category", "-c", default=None,
                        help=f"ArXiv category (e.g. cs.LG). Options: {', '.join(VALID_CATEGORIES)}")
    parser.add_argument("--days", "-d", type=int, default=0,
                        help="Only return papers from last N days (0 = no filter)")
    parser.add_argument("--limit", "-l", type=int, default=10,
                        help="Max number of results (default: 10)")
    parser.add_argument("--sort-by", default="relevance",
                        choices=list(SORT_MAP.keys()),
                        help="Sort order: relevance|date|lastUpdated")
    parser.add_argument("--sort-order", default="descending",
                        choices=["ascending", "descending"])
    args = parser.parse_args()

    if not args.query and not args.category:
        print("Error: provide --query and/or --category", file=sys.stderr)
        sys.exit(1)

    query_str = build_query(args.query, args.category)
    sort_by = SORT_MAP.get(args.sort_by, "relevance")
    fetch_limit = min(args.limit * 3, 100)  # Fetch extra for date filtering

    try:
        xml = fetch_with_urllib(query_str, 0, fetch_limit, sort_by, args.sort_order)
    except Exception as e:
        print(json.dumps({"error": str(e)}))
        sys.exit(1)

    if HAS_FEEDPARSER:
        papers = parse_feed_feedparser(xml)
    else:
        papers = parse_feed_urllib(xml)

    if args.days > 0:
        papers = filter_by_days(papers, args.days)

    papers = papers[: args.limit]
    print(json.dumps(papers, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
