"""
extract_images.py — Extract images from PDF and map to chapters.

Usage:
    python extract_images.py <pdf_path> <output_dir> [--language en|ru]
"""
import sys
import re
from pathlib import Path
import fitz

def slugify(text):
    text = text.lower()
    text = re.sub(r'[^a-z0-9]+', '-', text)
    return text.strip('-')

def get_chapter_ranges(doc):
    """Get chapter page ranges from PDF ToC."""
    toc = doc.get_toc(simple=False)
    chapters = []
    for level, title, page, details in toc:
        if level == 1:
            chapters.append({
                "title": title.strip(),
                "start": page - 1,  # 0-based
            })
    # Add end pages
    for i, ch in enumerate(chapters):
        ch["end"] = chapters[i + 1]["start"] if i + 1 < len(chapters) else len(doc)
    return chapters

def chapter_id_from_title(title):
    """Map chapter title to stable ID."""
    mapping = {
        "preface": "00-preface",
        "introduction": "01-introduction",
        "understanding foundation": "02-understanding-foundation-models",
        "evaluation methodology": "03-evaluation-methodology",
        "evaluate ai": "04-evaluate-ai-systems",
        "prompt engineering": "05-prompt-engineering",
        "rag and agents": "06-rag-and-agents",
        "fine": "07-finetuning",
        "dataset": "08-dataset-engineering",
        "inference": "09-inference-optimization",
        "architecture": "10-ai-engineering-architecture",
    }
    t = title.lower()
    for key, val in mapping.items():
        if key in t:
            return val
    return None

def main():
    if len(sys.argv) < 3:
        print("Usage: python extract_images.py <pdf_path> <output_dir> [--language en|ru]")
        sys.exit(1)

    pdf_path = Path(sys.argv[1])
    output_dir = Path(sys.argv[2])
    language = "en"
    if "--language" in sys.argv:
        idx = sys.argv.index("--language")
        language = sys.argv[idx + 1]

    print(f"=== Extracting images from {pdf_path} ===")

    doc = fitz.open(str(pdf_path))
    chapters = get_chapter_ranges(doc)

    print(f"Found {len(chapters)} chapters via ToC:")
    for ch in chapters[:5]:
        print(f"  p{ch['start']}-{ch['end']}: {ch['title'][:50]}")
    if len(chapters) > 5:
        print(f"  ... and {len(chapters)-5} more")

    # Map page -> chapter_id
    page_to_chapter = {}
    for ch in chapters:
        ch_id = chapter_id_from_title(ch["title"])
        if ch_id:
            for p in range(ch["start"], ch["end"]):
                page_to_chapter[p] = ch_id

    # Extract images
    total = 0
    skipped = 0
    chapter_counts = {}

    for page_num in range(len(doc)):
        ch_id = page_to_chapter.get(page_num, "uncategorized")
        page = doc[page_num]
        images = page.get_images(full=True)

        for img_idx, img in enumerate(images):
            xref = img[0]
            base = doc.extract_image(xref)

            # Skip tiny images (decorative)
            if base["width"] < 100 or base["height"] < 100:
                skipped += 1
                continue

            # Create output directory
            img_dir = output_dir / ch_id
            img_dir.mkdir(parents=True, exist_ok=True)

            # Name: page_image.png
            ext = base["ext"]
            name = f"p{page_num+1:03d}_{img_idx+1:02d}.{ext}"
            path = img_dir / name

            path.write_bytes(base["image"])
            total += 1
            chapter_counts[ch_id] = chapter_counts.get(ch_id, 0) + 1

    doc.close()

    print(f"\n=== Summary ===")
    print(f"Total extracted: {total} images")
    print(f"Skipped (small): {skipped}")
    print(f"By chapter:")
    for ch_id in sorted(chapter_counts.keys()):
        print(f"  {ch_id}: {chapter_counts[ch_id]} images")
    print(f"\nOutput: {output_dir}")

if __name__ == "__main__":
    main()
