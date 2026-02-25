# Manage Your Paper Reading List

Keep track of papers you want to read, are reading, or have finished.
The reading list is stored in `papers/reading_list.md`.

## Usage
```
/reading-list <command> [arguments]
```

## Commands
| Command | Description |
|---------|-------------|
| `/reading-list add <arxiv_id>` | Add a paper to the reading list |
| `/reading-list list` | Show all papers with status |
| `/reading-list list --unread` | Show only unread papers |
| `/reading-list list --done` | Show only completed papers |
| `/reading-list done <arxiv_id>` | Mark a paper as read |
| `/reading-list remove <arxiv_id>` | Remove a paper from the list |
| `/reading-list search <query>` | Search within the reading list |
| `/reading-list note <arxiv_id> "<note>"` | Add a note to a paper |
| `/reading-list stats` | Show reading statistics |

## Examples
- `/reading-list add 2312.11805`
- `/reading-list add https://arxiv.org/abs/2310.06825`
- `/reading-list done 2312.11805`
- `/reading-list search "reasoning"`
- `/reading-list note 2312.11805 "Great method section, revisit for implementation ideas"`

---

## Instructions

When this skill is invoked with `$ARGUMENTS`:

### Command: `add <arxiv_id>`

1. Fetch paper metadata:
```bash
python3 scripts/fetch_paper.py --id "<arxiv_id>"
```

2. Check if `papers/reading_list.md` exists; if not, create it with header:
```markdown
# Reading List

Papers I want to read, am reading, or have finished.

<!-- Format: each paper is a ## section with metadata -->

---
```

3. Check if the paper is already in the list (search for the ArXiv ID); if so, inform the user.

4. Append the paper entry to `papers/reading_list.md`:
```markdown
## <Title>
- **ArXiv**: <id> | <url>
- **Authors**: <Author1>, <Author2> et al.
- **Status**: 📖 Unread
- **Categories**: <cats>
- **Added**: <YYYY-MM-DD>
- **Notes**:

---
```

5. Confirm: "Added: **<Title>** to your reading list."

---

### Command: `list [--unread|--done|--in-progress]`

1. Read `papers/reading_list.md`
2. Parse all paper entries (each `##` section)
3. Filter by status if flag provided:
   - `--unread`: Status contains "Unread" or "📖"
   - `--done`: Status contains "Done" or "✅"
   - `--in-progress`: Status contains "Reading" or "📚"
4. Display formatted table:

```
## Your Reading List (<N> papers)

### 📖 Unread (<N>)
1. **<Title>** (ArXiv: <id>)
   Authors: <authors> | Added: <date>

### 📚 Reading (<N>)
...

### ✅ Done (<N>)
...
```

---

### Command: `done <arxiv_id>`

1. Read `papers/reading_list.md`
2. Find the paper entry containing the ArXiv ID
3. Update its status line: replace `📖 Unread` or `📚 Reading` with `✅ Done`
4. Add a completion date: append `- **Completed**: <today's date>` after Status
5. Write back to `papers/reading_list.md`
6. Confirm: "Marked **<Title>** as done! 🎉"

---

### Command: `remove <arxiv_id>`

1. Read `papers/reading_list.md`
2. Find and remove the entire `##` section for that paper (title through the `---` separator)
3. Write back to `papers/reading_list.md`
4. Confirm: "Removed **<Title>** from reading list."

---

### Command: `search <query>`

1. Read `papers/reading_list.md`
2. Search for papers where title, notes, or categories contain the query (case-insensitive)
3. Display matching papers in the same format as `list`

---

### Command: `note <arxiv_id> "<text>"`

1. Read `papers/reading_list.md`
2. Find the paper entry for that ArXiv ID
3. Update the `- **Notes**:` line to include the new text
4. Write back to `papers/reading_list.md`
5. Confirm the note was added.

---

### Command: `stats`

Read `papers/reading_list.md` and display:
```
## Reading Stats
- Total papers: <N>
- Unread: <N>
- In Progress: <N>
- Completed: <N>
- Categories: <breakdown>
- Recently added (last 7 days): <N>
```

---

## Notes
- The reading list is a plain markdown file you can edit manually too
- Status emoji: 📖 Unread, 📚 Reading, ✅ Done
- Papers are never deleted from the file when marked done — they're kept for reference
- Back up `papers/reading_list.md` to your own notes app if desired
