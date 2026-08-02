"""
clean_chapters.py — LLM cleaning pipeline for parsed book chapters.

Reads raw parsed chapters and uses an LLM to clean them up:
- Fix heading hierarchy
- Wrap code in fenced blocks with language hints
- Fix tables
- Remove page numbers, headers, footers, OCR artifacts
- Fix broken lists, paragraphs
- Preserve formulas (LaTeX where possible)

Usage:
    python clean_chapters.py <chapters_dir> [--model deepseek-v4-pro] [--chapter 01-introduction]
"""

import sys
import os
import json
import time
from pathlib import Path
import yaml

# Add backend venv to path for any shared deps
BACKEND_DIR = Path(__file__).resolve().parent.parent / "app" / "backend"
sys.path.insert(0, str(BACKEND_DIR))

CLEANING_PROMPT = """You are cleaning a book chapter that was parsed from PDF to Markdown.

Your job: clean this chapter while preserving ALL technical content.

Rules:
1. Fix heading hierarchy — use # for chapter title, ## for sections, ### for subsections
2. Wrap ALL code in fenced code blocks with language (```python, ```bash, ```yaml, ```text if unknown)
3. Fix tables — convert to proper Markdown tables. If too complex, format as readable list.
4. REMOVE: page numbers, running headers/footers, repeated chapter titles in headers, OCR garbage
5. Fix broken paragraphs — merge split sentences, normalize line breaks
6. Preserve ALL formulas — wrap in $...$ (inline) or $$...$$ (block) LaTeX
7. Fix bullet lists and numbered lists — ensure proper indentation
8. Remove duplicated content
9. Keep ALL images references as [Image: description] or ![alt](path)
10. Keep the chapter frontmatter (--- block at top) unchanged

IMPORTANT:
- Do NOT summarize or shorten the content
- Do NOT remove technical details, examples, or explanations
- Output ONLY the cleaned Markdown, no explanations

---

CHAPTER TO CLEAN:
"""


def clean_with_llm(content: str, model: str = "deepseek-v4-pro") -> str:
    """
    Clean chapter content using an LLM API call.
    Uses the Hermes execution environment — calls external LLM API.
    """
    # This is a placeholder that will be called via Hermes during execution.
    # In production, this uses the OpenAI-compatible API.
    import urllib.request

    api_key = os.getenv("DEEPSEEK_API_KEY", "")
    if not api_key:
        print("[WARN] No DEEPSEEK_API_KEY set, skipping LLM clean")
        return content

    url = "https://api.deepseek.com/v1/chat/completions"
    headers = {
        "Content-Type": "application/json",
        "Authorization": f"Bearer {api_key}",
    }

    # Truncate if too long (DeepSeek context limit)
    max_input = 50000
    if len(content) > max_input:
        content = content[:max_input] + "\n\n[... content truncated ...]"

    body = {
        "model": model,
        "messages": [
            {"role": "system", "content": CLEANING_PROMPT},
            {"role": "user", "content": content},
        ],
        "temperature": 0.1,
        "max_tokens": 16000,
    }

    req = urllib.request.Request(
        url,
        data=json.dumps(body).encode("utf-8"),
        headers=headers,
    )

    try:
        with urllib.request.urlopen(req, timeout=300) as resp:
            result = json.loads(resp.read())
            return result["choices"][0]["message"]["content"]
    except Exception as e:
        print(f"[ERROR] LLM clean failed: {e}")
        return content


def clean_chapter(filepath: Path, model: str, dry_run: bool = False) -> bool:
    """Clean a single chapter file."""
    content = filepath.read_text(encoding="utf-8")

    # Extract frontmatter
    frontmatter = ""
    body = content
    if content.startswith("---"):
        parts = content.split("---", 2)
        if len(parts) >= 3:
            frontmatter = "---" + parts[1] + "---\n\n"
            body = parts[2]

    # Skip if already cleaned (check for marker)
    if "<!-- cleaned -->" in content:
        print(f"  [SKIP] Already cleaned: {filepath.name}")
        return True

    # Skip empty/short files
    if len(body.strip()) < 100:
        print(f"  [SKIP] Too short: {filepath.name}")
        return True

    print(f"  [CLEAN] {filepath.name} ({len(body)} chars)...")

    if dry_run:
        print(f"    [DRY RUN] Would clean {filepath.name}")
        return True

    # Clean via LLM
    cleaned_body = clean_with_llm(body, model)

    # Reassemble
    cleaned = frontmatter + "<!-- cleaned -->\n" + cleaned_body
    filepath.write_text(cleaned, encoding="utf-8")

    print(f"    Done: {len(cleaned_body)} chars -> {filepath.name}")
    return True


def main():
    if len(sys.argv) < 2:
        print("Usage: python clean_chapters.py <chapters_dir> [--model deepseek-v4-pro] [--chapter CH_ID] [--dry-run]")
        sys.exit(1)

    chapters_dir = Path(sys.argv[1])
    model = "deepseek-v4-pro"
    specific_chapter = None
    dry_run = False

    for i, arg in enumerate(sys.argv):
        if arg == "--model" and i + 1 < len(sys.argv):
            model = sys.argv[i + 1]
        elif arg == "--chapter" and i + 1 < len(sys.argv):
            specific_chapter = sys.argv[i + 1]
        elif arg == "--dry-run":
            dry_run = True

    if not chapters_dir.exists():
        print(f"Error: Directory not found: {chapters_dir}")
        sys.exit(1)

    md_files = sorted(chapters_dir.glob("*.md"))
    # Filter out raw/full files
    md_files = [f for f in md_files if not f.name.startswith("_")]

    if specific_chapter:
        md_files = [f for f in md_files if specific_chapter in f.name]

    print(f"=== Cleaning {len(md_files)} chapters (model={model}) ===")
    if dry_run:
        print("[DRY RUN — no changes will be made]")

    success = 0
    for f in md_files:
        if clean_chapter(f, model, dry_run):
            success += 1
        time.sleep(0.5)  # Rate limit

    print(f"\n=== Done: {success}/{len(md_files)} chapters ===")


if __name__ == "__main__":
    main()
