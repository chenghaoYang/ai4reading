#!/usr/bin/env python3
"""
fetch_paper.py — Fetch detailed metadata for a single paper by ArXiv ID or URL.
Also queries Papers with Code for linked code repositories.

Usage:
    python3 fetch_paper.py --id "2312.11805"
    python3 fetch_paper.py --id "https://arxiv.org/abs/2312.11805"

Output: JSON object with paper metadata + code repos.
"""

import argparse
import json
import re
import sys
import urllib.parse
import urllib.request
import xml.etree.ElementTree as ET


ARXIV_API = "https://export.arxiv.org/api/query"
PWC_API = "https://paperswithcode.com/api/v1/papers/"
# Semantic Scholar API — free, no key required for basic fields
SS_API = "https://api.semanticscholar.org/graph/v1/paper"


def parse_arxiv_id(url_or_id: str) -> str:
    """Extract ArXiv ID from URL or raw ID string."""
    url_or_id = url_or_id.strip().rstrip("/")
    m = re.search(r"arxiv\.org/(?:abs|pdf)/([0-9]{4}\.[0-9]{4,5}(?:v\d+)?)", url_or_id)
    if m:
        return m.group(1)
    m = re.match(r"^([0-9]{4}\.[0-9]{4,5}(?:v\d+)?)$", url_or_id)
    if m:
        return m.group(1)
    return url_or_id


def fetch_arxiv_paper(arxiv_id: str) -> dict:
    """Fetch paper metadata from ArXiv API."""
    clean_id = re.sub(r"v\d+$", "", arxiv_id)  # Remove version for API query
    params = urllib.parse.urlencode({
        "id_list": clean_id,
        "max_results": 1,
    })
    url = f"{ARXIV_API}?{params}"
    req = urllib.request.Request(url, headers={"User-Agent": "ai4reading/1.0"})
    with urllib.request.urlopen(req, timeout=30) as resp:
        xml = resp.read().decode("utf-8")

    NS = {
        "atom": "http://www.w3.org/2005/Atom",
        "arxiv": "http://arxiv.org/schemas/atom",
    }
    root = ET.fromstring(xml)
    entries = root.findall("atom:entry", NS)
    if not entries:
        return {}

    entry = entries[0]

    def text(tag):
        el = entry.find(tag, NS)
        return el.text.strip() if el is not None and el.text else ""

    paper_id = text("atom:id").split("/abs/")[-1]
    title = re.sub(r"\s+", " ", text("atom:title"))
    abstract = re.sub(r"\s+", " ", text("atom:summary"))
    submitted = text("atom:published")
    updated = text("atom:updated")
    journal_ref = text("arxiv:journal_ref")
    doi = text("arxiv:doi")
    comment = text("arxiv:comment")

    authors = [
        a.findtext("atom:name", "", NS).strip()
        for a in entry.findall("atom:author", NS)
    ]

    categories = [
        c.get("term", "")
        for c in entry.findall("atom:category", NS)
    ]

    primary_cat = entry.find("arxiv:primary_category", NS)
    primary_category = primary_cat.get("term", "") if primary_cat is not None else (categories[0] if categories else "")

    return {
        "id": paper_id,
        "title": title,
        "authors": authors,
        "abstract": abstract,
        "url": f"https://arxiv.org/abs/{paper_id}",
        "pdf_url": f"https://arxiv.org/pdf/{paper_id}",
        "submitted": submitted,
        "updated": updated,
        "categories": categories,
        "primary_category": primary_category,
        "journal_ref": journal_ref,
        "doi": doi,
        "comment": comment,
    }


def fetch_pwc_repos(arxiv_id: str) -> list[dict]:
    """Fetch linked code repositories from Papers with Code."""
    clean_id = re.sub(r"v\d+$", "", arxiv_id)
    url = f"{PWC_API}?arxiv_id={urllib.parse.quote(clean_id)}"
    try:
        req = urllib.request.Request(url, headers={"User-Agent": "ai4reading/1.0"})
        with urllib.request.urlopen(req, timeout=15) as resp:
            data = json.loads(resp.read().decode("utf-8"))
        results = data.get("results", [])
        if not results:
            return []
        paper = results[0]
        repos = []
        for repo in paper.get("repositories", []):
            repos.append({
                "url": repo.get("url", ""),
                "stars": repo.get("stars", 0),
                "framework": repo.get("framework", ""),
                "is_official": repo.get("is_official", False),
            })
        # Sort by official first, then by stars
        repos.sort(key=lambda r: (not r["is_official"], -r["stars"]))
        return repos
    except Exception:
        return []


def fetch_semantic_scholar(arxiv_id: str) -> dict:
    """Fetch citation count and TLDR from Semantic Scholar."""
    clean_id = re.sub(r"v\d+$", "", arxiv_id)
    fields = "citationCount,influentialCitationCount,tldr"
    url = f"{SS_API}/arXiv:{urllib.parse.quote(clean_id)}?fields={fields}"
    try:
        req = urllib.request.Request(url, headers={
            "User-Agent": "ai4reading/1.0",
            "Accept": "application/json",
        })
        with urllib.request.urlopen(req, timeout=15) as resp:
            data = json.loads(resp.read().decode("utf-8"))
        return {
            "citation_count": data.get("citationCount", 0),
            "influential_citations": data.get("influentialCitationCount", 0),
            "semantic_tldr": (data.get("tldr") or {}).get("text", ""),
        }
    except Exception:
        return {}


def main():
    parser = argparse.ArgumentParser(description="Fetch single paper details")
    parser.add_argument("--id", "-i", required=True,
                        help="ArXiv ID (e.g. 2312.11805) or URL")
    parser.add_argument("--no-pwc", action="store_true",
                        help="Skip Papers with Code lookup")
    parser.add_argument("--no-ss", action="store_true",
                        help="Skip Semantic Scholar lookup")
    args = parser.parse_args()

    arxiv_id = parse_arxiv_id(args.id)

    try:
        paper = fetch_arxiv_paper(arxiv_id)
    except Exception as e:
        print(json.dumps({"error": f"ArXiv fetch failed: {e}"}))
        sys.exit(1)

    if not paper:
        print(json.dumps({"error": f"Paper not found: {arxiv_id}"}))
        sys.exit(1)

    if not args.no_pwc:
        paper["code_repos"] = fetch_pwc_repos(arxiv_id)
    else:
        paper["code_repos"] = []

    if not args.no_ss:
        ss_data = fetch_semantic_scholar(arxiv_id)
        paper.update(ss_data)

    print(json.dumps(paper, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
