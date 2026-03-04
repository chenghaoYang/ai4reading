#!/usr/bin/env python3
"""
fetch_daily.py — Fetch today's top AI/ML papers from ArXiv and Hugging Face Daily Papers.

Usage:
    python3 fetch_daily.py [--config config.yaml] [--limit 20] [--days 1]

Output: JSON object grouped by category with ranked papers.
"""

import argparse
import json
import os
import re
import sys
import urllib.parse
import urllib.request
import xml.etree.ElementTree as ET
from datetime import datetime, timedelta, timezone
from pathlib import Path

try:
    import yaml
    HAS_YAML = True
except ImportError:
    HAS_YAML = False

ARXIV_API = "https://export.arxiv.org/api/query"
# Official Hugging Face Daily Papers API (no key needed)
HF_DAILY_API = "https://huggingface.co/api/daily_papers"

DEFAULT_CATEGORIES = ["cs.AI", "cs.LG", "cs.CL", "cs.CV"]
DEFAULT_KEYWORDS = ["large language model", "reasoning", "multimodal", "agent", "diffusion"]
DEFAULT_LIMIT = 20


def load_config(config_path: str) -> dict:
    """Load user config from YAML file."""
    if not os.path.exists(config_path):
        return {}
    if not HAS_YAML:
        print(f"Warning: PyYAML not installed, using defaults", file=sys.stderr)
        return {}
    with open(config_path, encoding="utf-8") as f:
        return yaml.safe_load(f) or {}


def fetch_arxiv_category(category: str, days: int, max_results: int) -> list[dict]:
    """Fetch recent papers from a single ArXiv category."""
    params = urllib.parse.urlencode({
        "search_query": f"cat:{category}",
        "start": 0,
        "max_results": max_results,
        "sortBy": "submittedDate",
        "sortOrder": "descending",
    })
    url = f"{ARXIV_API}?{params}"
    try:
        req = urllib.request.Request(url, headers={"User-Agent": "ai4reading/1.0"})
        with urllib.request.urlopen(req, timeout=30) as resp:
            xml = resp.read().decode("utf-8")
    except Exception as e:
        print(f"Warning: Failed to fetch {category}: {e}", file=sys.stderr)
        return []

    NS = {
        "atom": "http://www.w3.org/2005/Atom",
        "arxiv": "http://arxiv.org/schemas/atom",
    }

    cutoff = datetime.now(timezone.utc) - timedelta(days=days)
    papers = []

    try:
        root = ET.fromstring(xml)
    except ET.ParseError as e:
        print(f"Warning: XML parse error for {category}: {e}", file=sys.stderr)
        return []

    for entry in root.findall("atom:entry", NS):
        def text(tag):
            el = entry.find(tag, NS)
            return el.text.strip() if el is not None and el.text else ""

        submitted_str = text("atom:published")
        try:
            submitted = datetime.fromisoformat(submitted_str.replace("Z", "+00:00"))
            if submitted < cutoff:
                continue
        except ValueError:
            pass

        paper_id = text("atom:id").split("/abs/")[-1]
        title = re.sub(r"\s+", " ", text("atom:title"))
        abstract = re.sub(r"\s+", " ", text("atom:summary"))
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
            "id": paper_id,
            "title": title,
            "authors": authors,
            "abstract": abstract,
            "url": f"https://arxiv.org/abs/{paper_id}",
            "pdf_url": f"https://arxiv.org/pdf/{paper_id}",
            "submitted": submitted_str,
            "updated": updated,
            "categories": categories,
            "source": "arxiv",
        })

    return papers


def fetch_hf_daily_papers(date_str: str | None = None, limit: int = 50) -> list[dict]:
    """Fetch Hugging Face Daily Papers using the official API.

    Args:
        date_str: Optional date in YYYY-MM-DD format (defaults to today).
        limit: Maximum number of papers to return.
    """
    params = {"limit": limit}
    if date_str:
        params["date"] = date_str
    url = HF_DAILY_API + "?" + urllib.parse.urlencode(params)

    try:
        req = urllib.request.Request(
            url,
            headers={
                "User-Agent": "ai4reading/1.0",
                "Accept": "application/json",
            }
        )
        with urllib.request.urlopen(req, timeout=20) as resp:
            raw = json.loads(resp.read().decode("utf-8"))
    except Exception as e:
        print(f"Warning: Failed to fetch HF papers: {e}", file=sys.stderr)
        return []

    papers = []
    for item in raw:
        paper_obj = item.get("paper", item)
        arxiv_id = paper_obj.get("id", "")
        if not arxiv_id:
            continue

        authors_raw = paper_obj.get("authors", [])
        authors = [
            a.get("name", a) if isinstance(a, dict) else str(a)
            for a in authors_raw
        ]

        papers.append({
            "id": arxiv_id,
            "title": paper_obj.get("title", ""),
            "authors": authors,
            "abstract": paper_obj.get("summary", paper_obj.get("abstract", "")),
            "url": f"https://arxiv.org/abs/{arxiv_id}",
            "pdf_url": f"https://arxiv.org/pdf/{arxiv_id}",
            "submitted": (paper_obj.get("publishedAt", "") or "")[:10],
            "updated": "",
            "categories": [],
            "source": "huggingface",
            "hf_upvotes": item.get("upvotes", 0),
        })

    # Sort by upvotes descending
    papers.sort(key=lambda p: p["hf_upvotes"], reverse=True)
    return papers


def score_paper(paper: dict, keywords: list[str]) -> float:
    """Score paper relevance based on keyword matches."""
    text = (paper["title"] + " " + paper["abstract"]).lower()
    score = 0.0
    for kw in keywords:
        kw_lower = kw.lower()
        if kw_lower in text:
            # Title match is worth more
            if kw_lower in paper["title"].lower():
                score += 3.0
            else:
                score += 1.0
    # Boost HF papers (already curated/popular)
    if paper.get("source") == "huggingface":
        score += 2.0 + paper.get("hf_upvotes", 0) * 0.1
    return score


def deduplicate(papers: list[dict]) -> list[dict]:
    """Remove duplicate papers by ArXiv ID, keeping first occurrence."""
    seen = set()
    result = []
    for p in papers:
        pid = re.sub(r"v\d+$", "", p["id"])
        if pid not in seen:
            seen.add(pid)
            result.append(p)
    return result


def main():
    parser = argparse.ArgumentParser(description="Fetch daily AI/ML papers")
    parser.add_argument("--config", default="config.yaml", help="Config file path")
    parser.add_argument("--limit", type=int, default=0, help="Max papers to return (0 = use config)")
    parser.add_argument("--days", type=int, default=0, help="Days to look back (0 = use config)")
    parser.add_argument("--no-hf", action="store_true", help="Skip Hugging Face papers")
    parser.add_argument("--category", default=None, help="Only fetch specific category")
    args = parser.parse_args()

    # Load config
    config_path = args.config
    if not os.path.isabs(config_path):
        # Try relative to repo root
        script_dir = Path(__file__).parent
        repo_root = script_dir.parent
        config_path = str(repo_root / config_path)

    config = load_config(config_path)

    interests = config.get("interests", {})
    keywords = interests.get("keywords", DEFAULT_KEYWORDS)
    categories = interests.get("categories", DEFAULT_CATEGORIES)

    if args.category:
        categories = [args.category]

    daily_cfg = config.get("daily", {})
    limit = args.limit or daily_cfg.get("limit", DEFAULT_LIMIT)
    days = args.days or daily_cfg.get("days_back", 1)

    # Fetch from all sources
    all_papers = []

    # ArXiv per category
    per_cat = max(limit * 2, 50)
    for cat in categories:
        cat_papers = fetch_arxiv_category(cat, days, per_cat)
        all_papers.extend(cat_papers)

    # Hugging Face daily papers
    if not args.no_hf:
        hf_papers = fetch_hf_daily_papers()
        all_papers.extend(hf_papers)

    # Deduplicate
    all_papers = deduplicate(all_papers)

    # Score and sort by relevance
    for p in all_papers:
        p["relevance_score"] = score_paper(p, keywords)

    all_papers.sort(key=lambda p: p["relevance_score"], reverse=True)

    # Group by primary category
    grouped: dict[str, list[dict]] = {}
    for paper in all_papers:
        cats = paper.get("categories", [])
        primary = cats[0] if cats else paper.get("source", "other")
        # Normalize to known categories
        for known_cat in categories:
            if known_cat in cats:
                primary = known_cat
                break
        if paper.get("source") == "huggingface" and not cats:
            primary = "HuggingFace Daily"
        grouped.setdefault(primary, []).append(paper)

    # Apply limit per group
    per_group_limit = max(limit // max(len(grouped), 1), 5)
    for cat in grouped:
        grouped[cat] = grouped[cat][:per_group_limit]

    result = {
        "date": datetime.now(timezone.utc).strftime("%Y-%m-%d"),
        "total_fetched": len(all_papers),
        "categories": grouped,
        "top_papers": all_papers[:limit],
    }

    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
