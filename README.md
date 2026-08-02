# AI Atlas

Personal AI Engineering learning platform.

**Live:** https://idealabs.co/learn  
**Owner:** Tima (tima100faces)

---

## What is this?

AI Atlas is a private, self-hosted learning system built around:

- Chip Huyen — *AI Engineering* (EN + RU)
- Anthropic Academy courses (MCP, Agents, Claude Code, etc.)
- Personal notes, concepts, and experiments

### Core principles

- **File-first**: All knowledge and progress live in plain files (Markdown + YAML). The web UI is just a beautiful viewer and interaction layer.
- **Agent-friendly**: Hermes (or any coding agent) can fully manage content and progress by editing files.
- **Markdown as primary reading format**. Original PDFs are kept as high-quality fallback.
- Clean, dark, Cursor-inspired interface.

---

## Tech Stack

| Layer        | Technology                          |
|--------------|-------------------------------------|
| Backend      | FastAPI                             |
| Frontend     | Vite + React + TypeScript + Tailwind + shadcn/ui |
| Auth         | Simple password                     |
| Content      | Markdown + YAML                     |
| PDF fallback | pdf.js                              |
| Deployment   | VPS + nginx (`idealabs.co/learn`)   |

---

## Project Structure

```text
ai-atlas/
├── content/                     # ← Source of truth (agent works here)
│   ├── book/
│   │   ├── en/                  # English chapters (markdown)
│   │   ├── ru/                  # Russian chapters (markdown)
│   │   ├── original/            # Source PDFs
│   │   ├── meta.yaml            # Table of contents + EN↔RU mapping
│   │   └── assets/
│   ├── courses/anthropic/       # Course cards + notes
│   ├── concepts/
│   ├── notes/
│   └── external/aie-book/       # Useful files from chiphuyen/aie-book
├── progress/
│   ├── book.yaml
│   ├── courses.yaml
│   └── overall.yaml
├── app/
│   ├── backend/                 # FastAPI
│   └── frontend/                # React app
├── scripts/
│   ├── parse_book.py            # PDF → raw markdown (Marker)
│   └── clean_chapters.py        # LLM cleaning pipeline
└── docs/                        # Architecture & agent plans
```

## Quick Start (for agent)

1. Read `docs/ARCHITECTURE.md`
2. Read `docs/DEVELOPMENT_PLAN.md`
3. Read `docs/FILE_CONTRACTS.md`
4. Start with Phase 0 from the development plan.

## Password

Simple password protection is enabled.
Password: `timohin2026`

## Status

🚀 **Phase 1.5 complete** — All 22 chapters (EN + RU) parsed and LLM-cleaned. Reader UI functional with language switching and progress tracking.

**Next:** Phase 2 — Dashboard & Progress System
