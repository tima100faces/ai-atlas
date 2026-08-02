# AI Atlas — Development Plan (for Agent)

This document is the single source of truth for implementation order.

**Recommended model for coding:** DeepSeek V4 Pro (primary)  
**Alternative / polish:** Kimi 3

---

## Phase 0 — Foundation (do this first)

**Goal:** Working skeleton that can be deployed and opened in the browser.

### Tasks

1. Create the full directory structure (already defined in README).
2. Move the two PDF files into `content/book/original/`:
   - `AI Engineering by Chip Huyen.pdf`
   - `Хьюен Ч. - AI-инженерия - 2026.pdf`
3. Download useful files from the official repo:
   ```bash
   git clone --depth 1 https://github.com/chiphuyen/aie-book.git content/external/aie-book
   ```
   Keep at least: `ToC.md`, `chapter-summaries.md`, `resources.md`, `prompt-examples.md`, `study-notes.md`.

4. Initialize backend (`app/backend`):
   - FastAPI
   - Simple password auth (`timohin2026`)
   - Health endpoint
   - Static file serving preparation

5. Initialize frontend (`app/frontend`):
   - Vite + React + TypeScript
   - Tailwind CSS
   - shadcn/ui
   - Dark theme (Cursor-inspired)
   - Basic layout with sidebar

6. Create minimal nginx config example for `idealabs.co/learn`
7. Create `.env.example` and basic README instructions for running locally.

### Definition of done for Phase 0

- `docker compose up` or simple scripts start both backend and frontend
- Opening the site asks for password
- After login you see a dark empty shell with sidebar

---

## Phase 1 — Book Pipeline & Reader

**Goal:** Ability to read the book in Markdown (both languages) with progress tracking.

### Tasks

1. Write `scripts/parse_book.py` using Marker.
2. Write `scripts/clean_chapters.py` (LLM cleaning step).
3. Define and implement `content/book/meta.yaml` (see `FILE_CONTRACTS.md`).
4. Parse both PDFs → raw markdown → cleaned chapters in `en/` and `ru/`.
5. Build the Markdown reader page:
   - Sidebar with table of contents
   - Language switcher (EN / RU)
   - Progress indicator
   - "Mark as read" + short note
6. Add PDF.js fallback viewer.

### Definition of done

- You can open any chapter in clean Markdown
- Progress is saved to `progress/book.yaml`
- Switching language works

---

## Phase 2 — Dashboard & Progress System

**Goal:** One-screen overview of the whole learning journey.

- Main dashboard showing:
  - Book progress (overall + current chapter)
  - Anthropic courses status
  - Recent notes
  - Next recommended actions
- Robust progress reading/writing API

---

## Phase 3 — Anthropic Courses

- Create rich course cards under `content/courses/anthropic/`
- Priority courses:
  - Introduction to MCP
  - MCP: Advanced Topics
  - Claude Code 101
  - Introduction to subagents
  - Introduction to agent skills
  - Building Effective Agents (Anthropic engineering post)
- Status + personal notes for each

---

## Phase 4 — Polish & Agent Experience

- Full-text search
- Keyboard shortcuts
- Better agent ergonomics (clear contracts, helper scripts)
- Optional: command palette

---

## Rules for the Agent

1. Always prefer editing files under `content/` and `progress/` over putting data in a database.
2. After any significant change, update this plan if needed.
3. Keep commits small and meaningful.
4. When in doubt — ask before introducing new major dependencies.
