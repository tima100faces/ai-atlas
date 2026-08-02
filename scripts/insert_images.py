"""
insert_images.py — Insert extracted images into cleaned markdown chapters.

Replaces [Image: ...] markers with markdown image links.
For chapters without markers, appends an "Illustrations" section.

Usage:
    python insert_images.py <chapters_dir> <assets_dir> [--dry-run]
"""
import sys
import re
from pathlib import Path

def insert_images_for_chapter(md_path, assets_dir, chap...[truncated]