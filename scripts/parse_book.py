"""
parse_book.py — Parse PDF to Markdown using PyMuPDF (lightweight).

Usage:
    python parse_book.py <pdf_path> <output_dir> [--language en|ru]

Extracts text page by page, groups by chapter based on PDF ToC,
and saves each chapter as a .md file.
"""

import sys
import re
from pathlib import Path


def extract_toc(doc) -> list[dict]:
    """Extract Table of Contents from PDF."""
    raw_toc = doc.get_toc(simple=False)
    if not raw_toc:
        print("[WARN] No TOC found in PDF, will split by H1 detection")
        return []

    chapters = []
    for item in raw_toc:
        level, title, page, _ = item
        if level == 1 and title.strip():
            chapters.append({
                "title": title.strip(),
                "page": page - 1,  # 0-indexed
                "level": level,
            })

    print(f"[TOC] Found {len(chapters)} chapters in PDF TOC")
    return chapters


def clean_text(text: str) -> str:
    """Basic text cleanup."""
    # Remove excessive whitespace
    text = re.sub(r'\n{4,}', '\n\n\n', text)
    text = re.sub(r' {3,}', '  ', text)
    # Remove page numbers (standalone numbers at start/end of page)
    text = re.sub(r'\n\d{1,4}\n', '\n', text)
    # Remove running headers (common patterns)
    text = re.sub(r'\nAI Engineering.*?\n', '\n', text)
    text = re.sub(r'\n©.*?Chip Huyen.*?\n', '\n', text)
    return text.strip()


def detect_code_blocks(text: str) -> str:
    """Detect and wrap code-like content in fenced blocks."""
    lines = text.split('\n')
    result = []
    in_code = False
    code_buf = []

    for line in lines:
        # Heuristic: line is code if it has indentation and contains programming symbols
        is_code_line = bool(re.match(r'^\s{2,}', line)) and (
            '(' in line or '{' in line or '=' in line or
            'def ' in line or 'import ' in line or 'class ' in line or
            '[' in line or 'return' in line or '#' in line
        )

        if is_code_line and not in_code:
            in_code = True
            code_buf = [line]
        elif is_code_line and in_code:
            code_buf.append(line)
        elif not is_code_line and in_code:
            # End code block
            if len(code_buf) >= 2:
                result.append('```python')
                result.extend(code_buf)
                result.append('```')
            else:
                result.extend(code_buf)
            result.append(line)
            in_code = False
            code_buf = []
        else:
            result.append(line)

    if in_code and code_buf:
        result.append('```python')
        result.extend(code_buf)
        result.append('```')

    return '\n'.join(result)


def markdown_cleanup(text: str) -> str:
    """Improve markdown formatting."""
    # Fix headings — ensure space after #
    text = re.sub(r'^(#{1,6})([^ #])', r'\1 \2', text, flags=re.MULTILINE)

    # Bold markers
    text = re.sub(r'(?<!\*)\*\*(.+?)\*\*(?!\*)', r'**\1**', text)

    # Fix list markers
    text = re.sub(r'^[•·○●]\s*', '- ', text, flags=re.MULTILINE)

    # Detect tables (lines with multiple |)
    lines = text.split('\n')
    for i, line in enumerate(lines):
        if line.count('|') >= 2 and not line.strip().startswith('#'):
            # Already a table
            pass

    return text


def extract_chapters(pdf_path: Path, output_dir: Path, language: str) -> list[dict]:
    """Main extraction pipeline."""
    import fitz

    doc = fitz.open(str(pdf_path))
    total_pages = doc.page_count
    print(f"[PDF] {total_pages} pages")

    # Get TOC
    toc_entries = extract_toc(doc)

    # If no TOC, create manual split by page ranges
    if not toc_entries:
        print("[INFO] No PDF TOC — extracting all text, split manually")
        full_text_parts = []
        for i in range(total_pages):
            page = doc[i]
            text = page.get_text("text")
            full_text_parts.append(text)
        full_text = "\n\n--- PAGE BREAK ---\n\n".join(full_text_parts)

        # Save raw combined
        raw_path = output_dir / f"_raw_{language}.md"
        raw_path.write_text(full_text, encoding="utf-8")
        print(f"[Raw] Saved full text: {raw_path} ({len(full_text)} chars)")

        doc.close()
        return split_by_headings(full_text, output_dir, language)

    # Extract by chapters using TOC
    print(f"[Extract] Processing {len(toc_entries)} chapters...")
    all_chapters = []

    for idx, entry in enumerate(toc_entries):
        start_page = entry["page"]
        end_page = toc_entries[idx + 1]["page"] if idx + 1 < len(toc_entries) else total_pages

        chapter_parts = []
        for p in range(start_page, min(end_page, total_pages)):
            page = doc[p]
            text = page.get_text("text")
            chapter_parts.append(text)

        chapter_text = "\n\n".join(chapter_parts)
        chapter_text = clean_text(chapter_text)
        chapter_text = detect_code_blocks(chapter_text)
        chapter_text = markdown_cleanup(chapter_text)

        # Generate chapter ID
        chapter_id = f"{idx:02d}-{slugify(entry['title'])}"

        all_chapters.append({
            "id": chapter_id,
            "title": entry["title"],
            "content": chapter_text,
            "page_start": start_page,
            "page_end": end_page,
        })

        print(f"  [{language}] p{start_page:3d}-{end_page-1:3d}: {entry['title'][:60]}")

    doc.close()

    # Save individual chapter files
    output_dir.mkdir(parents=True, exist_ok=True)
    saved = []

    for ch in all_chapters:
        filename = f"{ch['id']}.md"
        filepath = output_dir / filename

        frontmatter = f"""---
id: "{ch['id']}"
title: "{ch['title']}"
language: "{language}"
---

# {ch['title']}

"""
        filepath.write_text(frontmatter + ch["content"], encoding="utf-8")
        saved.append({
            "id": ch["id"],
            "title": ch["title"],
            "file": f"{language}/{filename}",
        })

    print(f"\n=== Done: {len(saved)} chapters saved ===")
    return saved


def split_by_headings(full_text: str, output_dir: Path, language: str) -> list[dict]:
    """Fallback: split raw text by H1 headings."""
    lines = full_text.split('\n')
    chapters = []
    current_title = "Introduction"
    current_lines = []

    for line in lines:
        stripped = line.strip()
        # Detect chapter headings
        if (stripped.lower().startswith('chapter ') or
            stripped.lower().startswith('preface') or
            re.match(r'^(?:CHAPTER|Chapter)\s+\d+', stripped)):
            if current_lines:
                chapters.append({"title": current_title, "content": '\n'.join(current_lines)})
            current_title = stripped
            current_lines = [f"# {stripped}"]
        else:
            current_lines.append(line)

    if current_lines:
        chapters.append({"title": current_title, "content": '\n'.join(current_lines)})

    print(f"[Split] Detected {len(chapters)} chapters")

    output_dir.mkdir(parents=True, exist_ok=True)
    saved = []

    for i, ch in enumerate(chapters):
        chapter_id = f"{i:02d}-{slugify(ch['title'])}"
        filename = f"{chapter_id}.md"
        filepath = output_dir / filename

        frontmatter = f"""---
id: "{chapter_id}"
title: "{ch['title']}"
language: "{language}"
---

"""
        filepath.write_text(frontmatter + ch["content"], encoding="utf-8")
        saved.append({"id": chapter_id, "title": ch["title"], "file": f"{language}/{filename}"})

    return saved


def slugify(text: str) -> str:
    """Convert chapter title to URL-friendly slug."""
    text = text.lower()
    text = re.sub(r'[^a-z0-9\s-]', '', text)
    text = re.sub(r'\s+', '-', text)
    return text[:50]


def main():
    if len(sys.argv) < 3:
        print("Usage: python parse_book.py <pdf_path> <output_dir> [--language en|ru]")
        sys.exit(1)

    pdf_path = Path(sys.argv[1])
    output_dir = Path(sys.argv[2])
    language = "en"

    for i, arg in enumerate(sys.argv):
        if arg == "--language" and i + 1 < len(sys.argv):
            language = sys.argv[i + 1]

    if not pdf_path.exists():
        print(f"Error: PDF not found: {pdf_path}")
        sys.exit(1)

    print(f"=== Parsing: {pdf_path.name} -> {output_dir} (lang={language}) ===")

    extract_chapters(pdf_path, output_dir, language)


if __name__ == "__main__":
    main()
