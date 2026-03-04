# Deep Dive into a Paper's Key Sections

Extract and deeply analyze the most valuable sections of a paper's PDF — with concrete examples, experiments, prompts, and step-by-step methodology.

## Usage
```
/deep-dive <arxiv_id>
```

## Examples
- `/deep-dive 2312.11805`
- `/deep-dive 2401.01234`

---

## Instructions

When this skill is invoked with `$ARGUMENTS`:

1. **Extract the arxiv_id** from `$ARGUMENTS`.

2. **Get the Table of Contents** by running:
   ```bash
   /c/Python312/python.exe scripts/extract_pdf.py --id "<arxiv_id>" --toc
   ```
   - If the PDF is not cached locally, the script will auto-download it before parsing.
   - If the script does not exist, tell the user: "`scripts/extract_pdf.py` is required but was not found. Please ensure it exists in the `scripts/` directory."
   - If the command fails for any other reason, report the error message to the user.

3. **Parse the JSON output.** The output contains these fields:
   - `id` — the arxiv ID
   - `title` — paper title
   - `total_pages` — total number of pages in the PDF
   - `toc` — array of section objects, each with: `section`, `title`, `page_start`, `page_end`
   - `high_value_sections` — array of objects, each with: `section`, `title`, `page_start`, `page_end`, `reason`

4. **Display a compact Table of Contents.** Show every section with its page range. Mark high-value sections with ★ and include the `reason` field. Example format:

   ```
   Table of Contents — <title> (<total_pages> pages)
   ══════════════════════════════════════════════════
   1. Introduction (pp. 1–2)
   ★ 3. Method (pp. 5–9)  ← Core technical contribution
   4. Experiments (pp. 10–14)
   ★ 4.1 Case Studies (pp. 11–13)  ← Concrete worked examples
   5. Related Work (pp. 15–16)
   6. Conclusion (pp. 17–18)
   ```

5. **Auto-select or ask the user which sections to extract:**
   - If `high_value_sections` has **4 or fewer** entries: automatically proceed with all of them (no need to ask). Inform the user: "Auto-selecting all ★ high-value sections."
   - If `high_value_sections` has **more than 4** entries: display a numbered list of the high-value sections and ask:
     > "Which sections would you like to deep-dive? (enter numbers, e.g. `1,3,5` — or `all`)"
   - Wait for the user's response before proceeding.

6. **Extract the selected sections** by running:
   ```bash
   /c/Python312/python.exe scripts/extract_pdf.py --id "<arxiv_id>" --pages "<page_ranges>"
   ```
   Build `page_ranges` by taking each selected section's `page_start` and `page_end` and joining them as `start-end`, comma-separated. Example: `--pages "5-9,11-13"`. If a section is a single page, use `N-N`.

7. **Present the extracted content** using the structured template below. Produce one block per section. Fill each field from the actual extracted text — do not fabricate details.

```markdown
## ★ [Section Title] (pp. X–Y)

### 核心问题
<1-2 sentences: what specific problem or question this section addresses>

### 具体案例 / 步骤
<The key concrete example, experiment, or case study found in this section. Include:
- Actual prompts used (if any) in fenced code blocks
- Specific numbers, metrics, dataset names
- Step-by-step process if it describes a methodology
- Any tables or structured results, reproduced faithfully>

### 关键发现
<Bullet points of the main takeaways from this section, with specific numbers where available>

### 可复用的技术/技巧
<What the reader can directly apply to their own work. Be specific and actionable.>

---
```

   Repeat this block for every selected section, in reading order.

8. **After all sections are presented**, offer the following next steps:

   - "Save these insights to your reading list note? Run: `/reading-list note <arxiv_id> \"<one-line summary>\"`"
   - "Want to explore related papers? Run: `/search-papers <topic>`"

## Notes
- The section extraction reads the raw PDF text. Mathematical notation may render imperfectly; interpret it as best as possible.
- If a section's extracted text is very short (e.g., under 200 characters), note that the section may be figure-heavy and mention this to the user.
- Always cite page numbers when quoting or paraphrasing from the extracted text.
