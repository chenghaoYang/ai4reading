# Search AI Papers on ArXiv

Search for AI/ML papers on ArXiv by keyword, topic, or category.

## Usage
```
/search-papers <query> [--category <cat>] [--days <n>] [--limit <n>] [--sort-by relevance|date]
```

## Examples
- `/search-papers "chain of thought reasoning"`
- `/search-papers "vision language model" --category cs.CV --days 7`
- `/search-papers "diffusion model" --limit 5 --sort-by date`
- `/search-papers --category cs.CL --days 3 --limit 15`

## Available Categories
- `cs.AI` — Artificial Intelligence
- `cs.LG` — Machine Learning
- `cs.CL` — Computation and Language (NLP)
- `cs.CV` — Computer Vision
- `cs.RO` — Robotics
- `cs.NE` — Neural and Evolutionary Computing
- `stat.ML` — Statistics / Machine Learning

---

## Instructions

When this skill is invoked with `$ARGUMENTS`:

1. Parse the arguments to extract: `query`, `--category`, `--days` (default 0), `--limit` (default 10), `--sort-by` (default relevance)

2. Run the fetch script:
```bash
python3 scripts/fetch_arxiv.py --query "<query>" [--category <cat>] [--days <days>] [--limit <limit>] [--sort-by <sort>]
```
If only `--category` is given (no query), omit `--query`.

3. Parse the JSON output and present results in this format:

```
## Search Results: "<query>" [category: <cat>]
Found <N> papers

### 1. <Title>
- **Authors**: <Author1>, <Author2>, ...
- **Submitted**: <YYYY-MM-DD>
- **ArXiv**: <url>
- **Categories**: <cat1>, <cat2>
- **Abstract**: <first 200 chars of abstract>...

### 2. <Title>
...
```

4. After showing results, ask: "Would you like to:
   - (a) Get a full summary of any paper? (provide ArXiv ID)
   - (b) Add papers to your reading list?
   - (c) Refine the search with different keywords?"

5. If the script fails or returns empty results:
   - Check if `requirements.txt` dependencies are installed: `pip install -r requirements.txt`
   - Try a broader query or different category
   - Inform the user of the issue clearly

## Notes
- ArXiv API is free and requires no API key
- Results are sorted by relevance by default; use `--sort-by date` for newest first
- Use `--days 1` to see only papers from the past 24 hours
- Authors list is truncated at 5 names for readability; add "et al." if more
