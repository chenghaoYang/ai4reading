# ai4reading

Claude Code skills for automatically discovering and reading AI/ML frontier papers.

> **How it works**: This project adds slash commands (`/search-papers`, `/daily-papers`, etc.) directly into Claude Code. You open this folder with `claude`, and Claude uses the Python scripts to fetch live data from ArXiv and Hugging Face, then presents it in a readable format — all inside your terminal.

---

## Features

| Skill | Command | Description |
|-------|---------|-------------|
| Search Papers | `/search-papers` | Search ArXiv by keyword, topic, or category |
| Daily Digest | `/daily-papers` | Today's top AI papers from ArXiv + Hugging Face |
| Summarize | `/summarize-paper` | Structured summary of any paper from abstract |
| Deep Dive | `/deep-dive` | Extract concrete examples, case studies & key results from full PDF |
| Reading List | `/reading-list` | Track papers with notes on key results & examples |

**Data sources** (all free, no API keys required):
- [ArXiv](https://arxiv.org) — cs.AI, cs.LG, cs.CL, cs.CV, and more
- [Hugging Face Daily Papers](https://huggingface.co/papers) — community-curated highlights
- [Papers with Code](https://paperswithcode.com) — linked code repositories

---

## Setup

### 1. Prerequisites

- Python 3.10+
- [Claude Code](https://github.com/anthropics/claude-code) CLI installed

### 2. Install Python dependencies

```bash
pip install requests feedparser PyYAML beautifulsoup4
```

> **Windows note**: The `requirements.txt` contains UTF-8 characters that cause errors on Chinese Windows systems. Install packages directly as shown above.

### 3. Customize your interests

Edit `config.yaml` to match your research areas:

```yaml
interests:
  keywords:
    - "large language model"
    - "reasoning"
    - "your topic here"
  categories:
    - cs.AI
    - cs.LG
    - cs.CL
    - cs.CV

daily:
  limit: 20       # max papers in daily digest
  days_back: 1    # days to look back (use 3 for Monday catch-up)

search:
  default_limit: 10
  default_days: 7
```

### 4. Open with Claude Code

```bash
cd ai4reading
claude
```

That's it. The slash commands are now available in your Claude Code session.

> **Windows note**: If you see `UnicodeEncodeError` when running scripts directly, prefix with `PYTHONIOENCODING=utf-8`. This is not needed when using Claude Code slash commands.

---

## Usage

### `/search-papers` — Search ArXiv

Search for papers by keyword, topic, or category. Claude fetches results from ArXiv and presents them in a formatted list.

```
/search-papers <query> [--category <cat>] [--days <n>] [--limit <n>] [--sort-by relevance|date]
```

**Examples:**

```
# Search by keyword (default: last 7 days, top 10 results)
/search-papers "chain of thought reasoning"

# Narrow to a category
/search-papers "vision language model" --category cs.CV

# Only papers from the last 24 hours
/search-papers "diffusion model" --days 1 --sort-by date

# Browse a category without a keyword
/search-papers --category cs.CL --days 3 --limit 15
```

**Available categories:**

| Code | Area |
|------|------|
| `cs.AI` | Artificial Intelligence |
| `cs.LG` | Machine Learning |
| `cs.CL` | Computation and Language (NLP) |
| `cs.CV` | Computer Vision |
| `cs.RO` | Robotics |
| `cs.NE` | Neural and Evolutionary Computing |
| `stat.ML` | Statistics / Machine Learning |

**After showing results**, Claude will offer to:
- Summarize any paper in detail
- Add papers to your reading list
- Refine the search

---

### `/daily-papers` — Today's Digest

Fetch today's top AI/ML papers ranked by relevance to your interests in `config.yaml`. Pulls from both ArXiv and Hugging Face Daily Papers.

```
/daily-papers [--category <cat>] [--days <n>] [--limit <n>] [--no-hf]
```

**Examples:**

```
# Today's top papers across all your configured categories
/daily-papers

# Only NLP papers
/daily-papers --category cs.CL

# Catch up after a long weekend
/daily-papers --days 3

# More results, ArXiv only
/daily-papers --limit 30 --no-hf
```

**Output format:** Papers are grouped into "Top Picks" (highest relevance score) and by category. Each paper shows title, authors, date, source (ArXiv or Hugging Face), and abstract excerpt.

> **Tip**: ArXiv doesn't publish on weekends. Run `/daily-papers --days 3` on Mondays to catch up.

---

### `/summarize-paper` — Deep Summary

Fetch full metadata for a specific paper and generate a structured summary. Also checks [Papers with Code](https://paperswithcode.com) for linked code repositories.

```
/summarize-paper <arxiv_id_or_url>
```

**Examples:**

```
# By ArXiv ID
/summarize-paper 2312.11805

# By full URL
/summarize-paper https://arxiv.org/abs/2310.06825

# Specific version
/summarize-paper 2401.01234v2
```

**Summary sections:**
- **TL;DR** — one-sentence core contribution
- **Problem** — what gap or limitation this addresses
- **Key Contribution** — the novel idea and how it differs from prior work
- **Method** — technical approach at a high level
- **Results** — key metrics and experimental findings
- **Significance** — why this matters for the field
- **Related Work** — connected papers and directions to explore
- **Code** — linked repositories with star counts (if available)

> **Note**: Summaries are generated from the abstract and metadata, not the full PDF. Use `/deep-dive` for concrete examples.

---

### `/deep-dive` — Extract Concrete Examples

Download the full PDF and extract concrete examples, case studies, key prompts, and specific results. Automatically identifies high-value sections (experiments, case studies, counterexamples) and presents them in a structured format.

```
/deep-dive <arxiv_id_or_url>
```

**Examples:**

```
# Deep dive into a paper's case studies
/deep-dive 2602.03837

# Works with full URLs too
/deep-dive https://arxiv.org/abs/2310.06825
```

**What it surfaces (per section):**
- **核心问题** — the specific problem/question this section addresses
- **具体案例 / 步骤** — actual prompts, step-by-step process, specific numbers
- **关键发现** — bullet-point takeaways with metrics
- **可复用的技术/技巧** — techniques you can apply to your own work

> **Tip**: Run `/summarize-paper` first for a quick overview, then `/deep-dive` when you want the paper's concrete contributions in depth.

---

### `/reading-list` — Manage Your Papers

Track papers across three statuses: 📖 Unread, 📚 Reading, ✅ Done. Stored in `papers/reading_list.md` — a plain markdown file you can edit manually or sync to your notes app.

```
/reading-list <command> [arguments]
```

**Commands:**

| Command | Description |
|---------|-------------|
| `/reading-list add <arxiv_id>` | Add a paper (fetches title/authors automatically) |
| `/reading-list list` | Show all papers with status |
| `/reading-list list --unread` | Show only unread papers |
| `/reading-list list --done` | Show only completed papers |
| `/reading-list done <arxiv_id>` | Mark a paper as read |
| `/reading-list note <arxiv_id> "<text>"` | Attach a personal note |
| `/reading-list search <query>` | Search your list by title, note, or category |
| `/reading-list remove <arxiv_id>` | Remove a paper from the list |
| `/reading-list stats` | Show reading statistics |

**Examples:**

```
# Add a paper (Claude fetches the title for you)
/reading-list add 2312.11805
/reading-list add https://arxiv.org/abs/2310.06825

# Review what's in your list
/reading-list list
/reading-list list --unread

# Mark as done after reading
/reading-list done 2312.11805

# Add a note while reading
/reading-list note 2312.11805 "Great method section — revisit for implementation ideas"

# Search your list
/reading-list search "reasoning"

# View your stats
/reading-list stats
```

---

## Typical Workflow

```
# 1. Start your day — see what's new
/daily-papers

# 2. Something looks interesting — get a quick summary
/summarize-paper 2502.12345

# 2.5 Paper has rich case studies or experiments? Extract them
/deep-dive 2502.12345

# 3. Worth keeping — add to your list
/reading-list add 2502.12345

# 4. Going deep on a topic — search for more
/search-papers "test-time compute" --days 14

# 5. After reading — mark done and add notes
/reading-list done 2502.12345
/reading-list note 2502.12345 "Key insight: scaling inference beats scaling training for math"

# 6. Weekly review — check your stats
/reading-list stats
```

---

## Testing Scripts Directly

You can also run the underlying Python scripts without Claude Code:

```bash
# Search ArXiv
python3 scripts/fetch_arxiv.py --query "LLM reasoning" --limit 5

# Daily digest (reads config.yaml for your interests)
python3 scripts/fetch_daily.py

# Fetch a single paper
python3 scripts/fetch_paper.py --id "2312.11805"
```

> **Windows users**: Prefix with `PYTHONIOENCODING=utf-8` to avoid encoding errors:
> ```bash
> PYTHONIOENCODING=utf-8 python3 scripts/fetch_daily.py
> ```

---

## Directory Structure

```
ai4reading/
├── .claude/
│   └── commands/
│       ├── search-papers.md       # /search-papers skill
│       ├── daily-papers.md        # /daily-papers skill
│       ├── summarize-paper.md     # /summarize-paper skill
│       ├── deep-dive.md           # /deep-dive skill
│       └── reading-list.md        # /reading-list skill
├── scripts/
│   ├── fetch_arxiv.py             # ArXiv API wrapper
│   ├── fetch_daily.py             # Multi-source daily digest
│   ├── fetch_paper.py             # Single paper fetcher + Papers with Code
│   ├── extract_pdf.py             # PDF extraction engine for deep dives
│   └── utils.py                   # Shared utilities
├── papers/
│   ├── reading_list.md            # Your reading list (auto-created)
│   └── *.pdf                      # Cached PDFs (auto-downloaded by /deep-dive)
├── config.yaml                    # Your research interests & settings
├── requirements.txt               # Python dependencies
└── README.md
```

---

## Inspired By

- [ArxivDigest](https://github.com/AutoLLM/ArxivDigest) — LLM-personalized paper recommendations
- [GPT Paper Assistant](https://github.com/tatsu-lab/gpt_paper_assistant) — daily arXiv scanner with GitHub Actions
- [arxiv-mcp-server](https://github.com/blazickjp/arxiv-mcp-server) — MCP server for ArXiv
- [Hugging Face Daily Papers](https://huggingface.co/papers) — community paper curation
- [dair-ai/ML-Papers-of-the-Week](https://github.com/dair-ai/ML-Papers-of-the-Week) — concrete metrics & benchmarks format
- [KaleabTessera/Research-Paper-Reading-Template](https://github.com/KaleabTessera/Research-Paper-Reading-Template) — structured reading notes with examples
