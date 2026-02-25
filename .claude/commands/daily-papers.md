# Daily AI Paper Digest

Fetch and display today's top AI/ML papers from ArXiv and Hugging Face Daily Papers.
Papers are filtered and ranked by your research interests defined in `config.yaml`.

## Usage
```
/daily-papers [--category <cat>] [--days <n>] [--limit <n>] [--no-hf]
```

## Examples
- `/daily-papers` — fetch today's papers across all configured categories
- `/daily-papers --category cs.CL` — only NLP/language papers
- `/daily-papers --days 3` — papers from last 3 days
- `/daily-papers --limit 30 --no-hf` — more ArXiv papers, skip Hugging Face

---

## Instructions

When this skill is invoked with `$ARGUMENTS`:

1. Parse arguments: `--category`, `--days`, `--limit`, `--no-hf`

2. Run the daily fetch script:
```bash
python3 scripts/fetch_daily.py [--category <cat>] [--days <days>] [--limit <limit>] [--no-hf]
```

3. Parse the JSON output. Present results in this structured format:

```
# Daily AI Paper Digest — <date>
Fetched <N> papers total. Showing top results ranked by relevance to your interests.

---

## 🔥 Top Picks (Most Relevant to Your Interests)

### 1. <Title>
- **ArXiv**: <url>
- **Authors**: <Author1>, <Author2> et al.
- **Date**: <YYYY-MM-DD>
- **Categories**: <cats>
- **Source**: ArXiv / 🤗 Hugging Face Daily
- **Abstract**: <first 250 chars>...

---

## By Category

### cs.AI — Artificial Intelligence (<N> papers)
1. [<Title>](<url>) — <Author1> et al. (<date>)
   > <Abstract first 150 chars>...
...

### cs.LG — Machine Learning (<N> papers)
...

### cs.CL — Computation & Language (<N> papers)
...

### cs.CV — Computer Vision (<N> papers)
...

### 🤗 Hugging Face Daily
...
```

4. After showing the digest, offer:
   - "Type `/summarize-paper <arxiv_id>` for a full summary of any paper"
   - "Type `/reading-list add <arxiv_id>` to add a paper to your reading list"
   - "Customize your interests by editing `config.yaml`"

5. If no papers are found for the last 1 day, automatically try `--days 2` and mention it.

6. Handle errors gracefully:
   - If HF fetch fails, continue with ArXiv-only results and note the issue
   - If config.yaml missing, use defaults and suggest creating it

## Notes
- The script reads `config.yaml` for your research interests
- Papers are scored by keyword matches in title (3x weight) and abstract
- Hugging Face papers get a boost as they're already curated by the community
- ArXiv weekend submissions appear Monday — use `--days 3` on Mondays
