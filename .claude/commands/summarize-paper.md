# Summarize an AI Paper

Fetch full metadata for a paper and generate a structured, easy-to-read summary.
Also checks Papers with Code for linked code repositories.

## Usage
```
/summarize-paper <arxiv_id_or_url>
```

## Examples
- `/summarize-paper 2312.11805`
- `/summarize-paper https://arxiv.org/abs/2310.06825`
- `/summarize-paper 2401.01234v2`

---

## Instructions

When this skill is invoked with `$ARGUMENTS`:

1. Extract the ArXiv ID or URL from the argument.

2. Run the paper fetch script:
```bash
python3 scripts/fetch_paper.py --id "<arxiv_id_or_url>"
```

3. Parse the JSON output. If it contains `"error"`, report the error to the user.

4. Using the paper's `title`, `abstract`, `authors`, `categories`, and `code_repos`, generate a comprehensive structured summary:

```markdown
# <Title>

**ArXiv**: <url> | **PDF**: <pdf_url>
**Authors**: <Author1>, <Author2>, ... (<total count> authors)
**Submitted**: <date> | **Categories**: <primary_category>, <other_cats>
**Journal/Conference**: <journal_ref if available, else "Not yet published">
**Code**: <repo_url (is_official marker)> ⭐<stars> | "No code available"

---

## TL;DR
<One sentence that captures the key contribution. Be concrete and specific.>

## Problem
<What specific problem or limitation does this paper address? 2-3 sentences.>

## Key Contribution
<What is the main novel idea, method, or finding? What makes this different from prior work? 3-4 sentences.>

## Method
<How does the approach work at a high level? Include key technical components, architecture choices, or algorithmic innovations. 4-5 sentences.>

## Results
<What are the main experimental results or findings? Include specific numbers/metrics if mentioned in the abstract. 3-4 sentences.>

## Key Examples
<If the paper contains case studies, experiments, or illustrative examples, list the most important ones with one sentence each. If only abstract is available, note: "Run `/deep-dive <id>` to extract concrete examples from the full paper.">

## Significance
<Why does this matter for the field? Who would benefit from reading this? 2-3 sentences.>

## Related Work to Explore
<2-3 closely related papers or research directions this builds on or competes with.>
```

5. After the summary, check if the paper likely contains rich case studies by looking for keywords in the abstract: "case study", "case studies", "experiment", "we show", "we demonstrate", "counterexample", "we prove". If found, add this line:

   > 💡 **This paper contains concrete case studies/experiments** → run `/deep-dive <arxiv_id>` to extract them with full details.

6. Then ask:
   - "Add to reading list? (`/reading-list add <id>`)"
   - "Deep dive into examples? (`/deep-dive <id>`)"
   - "Explore related papers? (`/search-papers <topic>`)"

7. If the abstract alone is insufficient for a full summary, note what information is unavailable and do your best with what's provided.

## Notes
- The summary is generated from the abstract and metadata — it does not read the full PDF
- For papers with code, official implementations are marked and shown first
- If the paper has a `journal_ref` (published at a conference), that context is valuable
- Use the `comment` field (e.g., "Accepted at NeurIPS 2024") if available
