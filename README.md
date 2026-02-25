# ai4reading

Claude Code skills for automatically discovering and reading AI/ML frontier papers.

## Features

| Skill | Command | Description |
|-------|---------|-------------|
| Search Papers | `/search-papers` | Search ArXiv by keyword/topic/category |
| Daily Digest | `/daily-papers` | Today's top AI papers from ArXiv + Hugging Face |
| Summarize | `/summarize-paper` | Structured summary of any paper |
| Reading List | `/reading-list` | Manage your personal paper reading list |

**Data sources** (all free, no API keys required):
- [ArXiv](https://arxiv.org) — cs.AI, cs.LG, cs.CL, cs.CV, and more
- [Hugging Face Daily Papers](https://huggingface.co/papers) — community-curated highlights
- [Papers with Code](https://paperswithcode.com) — linked code repositories

---

## Setup

### 1. Install Python dependencies

```bash
pip install -r requirements.txt
```

> Minimum requirements: Python 3.10+. The scripts use `urllib` from stdlib as fallback if optional packages are missing.

### 2. Customize your interests

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
```

### 3. Open with Claude Code

```bash
cd ai4reading
claude
```

---

## Usage

### Search for papers

```
/search-papers "chain of thought reasoning"
/search-papers "diffusion model" --category cs.CV --days 7
/search-papers --category cs.CL --days 3 --limit 15
```

### Get today's digest

```
/daily-papers
/daily-papers --category cs.AI
/daily-papers --days 3        # catch up after a weekend
```

### Summarize a specific paper

```
/summarize-paper 2312.11805
/summarize-paper https://arxiv.org/abs/2310.06825
```

The summary includes: TL;DR, Problem, Key Contribution, Method, Results, Significance, and code repo links.

### Manage your reading list

```
/reading-list add 2312.11805          # add a paper
/reading-list list                    # view all papers
/reading-list list --unread           # view unread only
/reading-list done 2312.11805         # mark as read
/reading-list note 2312.11805 "Great ideas for my project"
/reading-list search "reasoning"      # search your list
/reading-list stats                   # reading statistics
```

---

## Directory Structure

```
ai4reading/
├── .claude/
│   └── commands/
│       ├── search-papers.md       # /search-papers skill
│       ├── daily-papers.md        # /daily-papers skill
│       ├── summarize-paper.md     # /summarize-paper skill
│       └── reading-list.md        # /reading-list skill
├── scripts/
│   ├── fetch_arxiv.py             # ArXiv API wrapper
│   ├── fetch_daily.py             # Multi-source daily digest
│   └── fetch_paper.py             # Single paper fetcher
├── papers/
│   └── reading_list.md            # Your reading list
├── config.yaml                    # Your research interests
├── requirements.txt               # Python deps
└── README.md
```

---

## Testing Scripts Directly

```bash
# Test ArXiv search
python3 scripts/fetch_arxiv.py --query "LLM reasoning" --limit 5

# Test daily digest (fetches from ArXiv + HuggingFace)
python3 scripts/fetch_daily.py

# Test single paper fetch with Papers with Code
python3 scripts/fetch_paper.py --id "2312.11805"
```

---

## Inspired By

- [ArxivDigest](https://github.com/AutoLLM/ArxivDigest) — LLM-personalized paper recommendations
- [GPT Paper Assistant](https://github.com/tatsu-lab/gpt_paper_assistant) — daily arXiv scanner with GitHub Actions
- [arxiv-mcp-server](https://github.com/blazickjp/arxiv-mcp-server) — MCP server for ArXiv
- [Hugging Face Daily Papers](https://huggingface.co/papers) — community paper curation
