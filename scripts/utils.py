#!/usr/bin/env python3
"""
Shared utility functions for ai4reading scripts.
"""

import os
from pathlib import Path


def ensure_dir(path) -> Path:
    """Create directory and parents if they don't exist."""
    p = Path(path)
    p.mkdir(parents=True, exist_ok=True)
    return p


def load_config(config_path=None) -> dict:
    """Load YAML config. Falls back to defaults if file missing or PyYAML not installed."""
    defaults = {
        "interests": {
            "keywords": [
                "large language model", "reasoning", "multimodal",
                "agent", "diffusion", "alignment",
            ],
            "categories": ["cs.AI", "cs.LG", "cs.CL", "cs.CV"],
        },
        "daily": {"limit": 20, "days_back": 1},
        "search": {"default_limit": 10, "default_days": 7},
    }

    if config_path is None:
        # Try to find config relative to script location
        script_dir = Path(__file__).parent
        config_path = script_dir.parent / "config.yaml"

    config_path = Path(config_path)
    if not config_path.exists():
        return defaults

    try:
        import yaml
        with open(config_path, encoding="utf-8") as f:
            user_cfg = yaml.safe_load(f) or {}
        # Deep merge
        result = dict(defaults)
        for key, val in user_cfg.items():
            if isinstance(val, dict) and key in result and isinstance(result[key], dict):
                result[key] = {**result[key], **val}
            else:
                result[key] = val
        return result
    except Exception:
        return defaults


def format_paper_brief(paper: dict, index: int = None) -> str:
    """Format a paper as a single-line entry."""
    prefix = f"{index}. " if index else "- "
    arxiv_id = paper.get("id", paper.get("arxiv_id", ""))
    title = paper.get("title", "Untitled")
    authors = paper.get("authors", [])
    date = (paper.get("submitted", paper.get("published", "")) or "")[:10]

    if len(authors) > 3:
        author_str = f"{authors[0]} et al."
    elif authors:
        author_str = ", ".join(authors)
    else:
        author_str = "Unknown"

    url = paper.get("url", paper.get("abs_url", f"https://arxiv.org/abs/{arxiv_id}"))
    return f"{prefix}**[{title}]({url})** — {author_str} ({date})"


def format_paper_full(paper: dict, index: int = None) -> str:
    """Format a paper with abstract snippet for display."""
    arxiv_id = paper.get("id", paper.get("arxiv_id", ""))
    title = paper.get("title", "Untitled")
    authors = paper.get("authors", [])
    date = (paper.get("submitted", paper.get("published", "")) or "")[:10]
    abstract = paper.get("abstract", paper.get("summary", ""))
    categories = paper.get("categories", [])
    source = paper.get("source", "arxiv")
    upvotes = paper.get("hf_upvotes", paper.get("upvotes"))
    code_repos = paper.get("code_repos", paper.get("github_repos", []))
    tldr = paper.get("semantic_tldr", "")

    prefix = f"### {index}. " if index else "### "
    url = paper.get("url", paper.get("abs_url", f"https://arxiv.org/abs/{arxiv_id}"))
    pdf_url = paper.get("pdf_url", f"https://arxiv.org/pdf/{arxiv_id}")

    if len(authors) > 5:
        author_str = ", ".join(authors[:5]) + " et al."
    elif authors:
        author_str = ", ".join(authors)
    else:
        author_str = "Unknown"

    lines = [f"{prefix}**{title}**"]
    lines.append(f"- **Authors**: {author_str}")
    if date:
        lines.append(f"- **Date**: {date}")
    if categories:
        lines.append(f"- **Categories**: {', '.join(categories[:4])}")

    links = [f"[ArXiv]({url})", f"[PDF]({pdf_url})"]
    if upvotes is not None:
        lines.append(f"- **Source**: {source} | HF Upvotes: {upvotes}")
    else:
        lines.append(f"- **Source**: {source}")
    lines.append(f"- **Links**: {' | '.join(links)}")

    if code_repos:
        repo = code_repos[0]
        repo_url = repo.get("url", "")
        stars = repo.get("stars", 0)
        official = " (official)" if repo.get("is_official") else ""
        lines.append(f"- **Code**: [{repo_url}]({repo_url}){official} ⭐{stars}")

    if tldr:
        lines.append(f"\n> **TL;DR**: {tldr}")
    elif abstract:
        snippet = abstract[:280] + ("..." if len(abstract) > 280 else "")
        lines.append(f"\n> {snippet}")

    return "\n".join(lines)
