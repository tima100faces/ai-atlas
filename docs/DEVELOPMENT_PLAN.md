# AI Atlas — Development Plan (for Agent)

This document is the single source of truth for implementation order.

**Recommended model for coding:** DeepSeek V4 Pro (primary)  
**Alternative / polish:** Kimi 3

---

## Phase 0 — Foundation ✅

**Status:** DONE

- Directory structure
- PDFs in `content/book/original/`
- FastAPI backend (port 8001, password auth)
- Vite + React + TypeScript + Tailwind frontend
- Dark theme (Cursor-inspired)
- `run.sh` for local dev (backend :8001 + frontend :5173)

---

## Phase 1 — Book Pipeline & Reader ✅

**Status:** DONE

- `scripts/parse_book.py` — PyMuPDF-based PDF parser
- `scripts/clean_chapters.py` — LLM cleaning via DeepSeek API
- 22 cleaned chapters (EN + RU) with `<!-- cleaned -->` markers
- `meta.yaml` with ToC and EN↔RU mapping
- Reader page: sidebar ToC, language switcher, progress, notes
- `progress/book.yaml` — chapter-level progress tracking
- File-first architecture (no database)

---

## Phase 2 — Dashboard & Progress System ✅

**Status:** DONE

- Dashboard with 4 stat cards (progress %, chapters read, current chapter, language)
- "Continue Reading" button → Reader with chapter + language
- Anthropic courses block on Dashboard with inline status toggle
- Recent notes from book progress
- File-first, no DB changes

---

## Phase 3 — Anthropic Courses ✅

**Status:** DONE

- Data split: `content/courses/anthropic/index.yaml` (catalog) + `progress/courses.yaml` (user data)
- 6 courses: MCP Intro, MCP Advanced, Claude Code 101, Subagents, Agent Skills, Building Effective Agents
- Rich metadata: description, topics, related_chapters, estimated_time, lectures, quizzes
- Backend: `GET /api/content/courses/anthropic`, merge in `GET /api/progress/courses`, PUT filter
- `Courses.tsx` — full courses page with filter/sort, cards with topics, related chapters, notes, takeaways
- Sidebar navigation: Dashboard ↔ Reader ↔ Courses

---

## Phase 4 — Polish & Agent Experience

**Status:** TODO

- Full-text search
- Keyboard shortcuts
- Better agent ergonomics (clear contracts, helper scripts)
- Optional: command palette

---

## Infrastructure

- **Backend:** systemd service `ai-atlas-backend.service` (enabled, auto-restart)
- **Frontend:** Vite dev server on :5173 (run manually or via `run.sh`)
- **Prod deploys:** TBD (nginx + static build), currently dev-only

---

## Rules for the Agent

1. Always prefer editing files under `content/` and `progress/` over putting data in a database.
2. After any significant change, update this plan if needed.
3. Keep commits small and meaningful.
4. When in doubt — ask before introducing new major dependencies.
5. Port 8001 only — 8000 belongs to health-tracker.
6. Backend runs via systemd, survives reboots.
