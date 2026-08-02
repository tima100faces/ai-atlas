# AI Atlas

Personal AI Engineering learning platform.

**Live (dev):** http://46.225.136.143:5173
**Planned domain:** https://idealabs.co/learn
**Owner:** Tima (tima100faces)

---

## What is this?

AI Atlas is a private, self-hosted learning system built around:

- Chip Huyen — *AI Engineering* (EN + RU, 11 chapters, LLM-cleaned)
- Anthropic Academy courses (MCP, Claude Code, Agents, Skills)
- Personal notes, takeaways, and progress tracking

### Core principles

- **File-first**: All knowledge and progress live in plain files (Markdown + YAML). The web UI is just a beautiful viewer and interaction layer.
- **Agent-friendly**: Hermes (or any coding agent) can fully manage content and progress by editing files.
- **Markdown as primary reading format**. Original PDFs are kept as high-quality fallback.
- Clean, dark, Cursor-inspired interface.

---

## Tech Stack

| Layer        | Technology                          |
|--------------|-------------------------------------|
| Backend      | FastAPI (port 8001)                 |
| Frontend     | Vite + React + TypeScript + Tailwind |
| Auth         | Simple password (cookie-based)      |
| Content      | Markdown + YAML                     |
| Deployment   | systemd (backend) + Vite dev (frontend) |

---

## Project Structure

```text
ai-atlas/
├── content/                     # ← Source of truth (agent works here)
│   ├── book/
│   │   ├── en/                  # 11 English chapters (markdown)
│   │   ├── ru/                  # 11 Russian chapters (markdown)
│   │   ├── original/            # Source PDFs
│   │   ├── assets/              # Chapter images
│   │   └── meta.yaml            # ToC + EN↔RU mapping
│   └── courses/anthropic/
│       └── index.yaml           # Course catalog (6 courses)
├── progress/
│   ├── book.yaml                # Chapter progress + notes
│   ├── courses.yaml             # Course user-data (status, notes, takeaways)
│   └── overall.yaml             # Aggregated book progress
├── app/
│   ├── backend/                 # FastAPI (venv)
│   │   └── main.py              # Auth, content, progress endpoints
│   └── frontend/                # React app (npm)
│       └── src/
│           ├── App.tsx           # Auth + routing
│           ├── Dashboard.tsx     # Main dashboard
│           ├── Reader.tsx        # Book reader
│           └── Courses.tsx       # Anthropic courses
├── scripts/
│   ├── parse_book.py            # PDF → raw markdown (PyMuPDF)
│   └── clean_chapters.py        # LLM cleaning pipeline (DeepSeek)
├── docs/
│   ├── ARCHITECTURE.md          # Architecture overview
│   ├── DEVELOPMENT_PLAN.md      # Phase plan + status
│   ├── FILE_CONTRACTS.md        # YAML schemas
│   └── AGENT_PROMPT_PHASE0.md   # Initial agent prompt
├── run.sh                       # Dev runner (backend + frontend)
└── ai-atlas-backend.service     # systemd unit
```

---

## Quick Start

### For humans

```bash
# Dev (both servers)
cd /root/ai-atlas && bash run.sh
# → Frontend: http://localhost:5173
# → Backend:  http://localhost:8001
# → Password: timohin2026
```

Backend survives reboots via systemd:
```bash
systemctl status ai-atlas-backend   # check
systemctl restart ai-atlas-backend  # restart
```

### For agents

1. Read `docs/ARCHITECTURE.md`
2. Read `docs/DEVELOPMENT_PLAN.md`
3. Read `docs/FILE_CONTRACTS.md`
4. Load skill: `skill_view("ai-atlas")`

---

## API Endpoints

All require auth (cookie `ai_atlas_auth` or header `X-AI-Atlas-Auth`):

| Method | Path | Purpose |
|--------|------|---------|
| POST | `/api/auth/login` | Login (`{"password": "timohin2026"}`) |
| GET | `/api/auth/check` | Check auth |
| POST | `/api/auth/logout` | Logout |
| GET | `/api/health` | Health (no auth) |
| GET | `/api/content/book/meta` | Book ToC |
| GET | `/api/content/book/{lang}/{chapter_id}` | Chapter markdown |
| GET | `/api/content/courses/anthropic` | Course catalog (static) |
| GET | `/api/progress/book` | Chapter progress |
| PUT | `/api/progress/book` | Update chapter progress |
| GET | `/api/progress/courses` | Courses merged (catalog + user data) |
| PUT | `/api/progress/courses` | Update course progress (filters to user fields) |
| GET | `/api/progress/overall` | Aggregated book stats |

---

## Password

`timohin2026`

---

## Status

### ✅ Phase 0 — Foundation
Directory structure, FastAPI backend, React frontend, dark theme, auth.

### ✅ Phase 1 — Book Pipeline & Reader
PDF parsing, LLM cleaning, 22 chapters (EN+RU), Reader UI with language switch and progress.

### ✅ Phase 2 — Dashboard & Progress System
Dashboard with stat cards, courses block, recent notes, "Continue Reading" button.

### ✅ Phase 3 — Anthropic Courses
Data split (catalog vs progress), 6 courses with rich metadata, Courses page with filters, notes, takeaways, related book chapters.

### 🔜 Phase 4 — Polish
Full-text search, keyboard shortcuts, command palette, nginx + HTTPS deploy.
