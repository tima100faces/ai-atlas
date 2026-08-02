# AI Atlas — Architecture

## Philosophy

**File-first + thin beautiful UI**

Everything important (book chapters, progress, notes, course status) lives in the filesystem as Markdown and YAML.  
The web application is a high-quality viewer and interaction layer on top of these files.

This design gives two big advantages:

1. Any coding agent (Hermes + DeepSeek / Kimi) can read and write the entire knowledge base without special APIs.
2. The system remains simple, debuggable, and portable.

---

## High-level components

```
┌─────────────────────────────────────────────────────┐
│                    Browser (UI)                     │
│         React + Tailwind + shadcn (Cursor-like)     │
└──────────────────────┬──────────────────────────────┘
                       │ HTTP
┌──────────────────────▼──────────────────────────────┐
│                   FastAPI Backend                   │
│  - Auth (simple password)                           │
│  - Serve markdown / progress / static               │
│  - Optional: trigger scripts                        │
└──────────────────────┬──────────────────────────────┘
                       │ reads/writes
┌──────────────────────▼──────────────────────────────┐
│                   Filesystem                        │
│  content/   +   progress/   +   original PDFs       │
└─────────────────────────────────────────────────────┘
```

---

## Content Model

### Book

- Two parallel trees: `content/book/en/` and `content/book/ru/`
- One markdown file per chapter (or major section)
- `content/book/meta.yaml` contains:
  - Ordered list of chapters
  - Mapping between EN and RU versions
  - Status, estimated pages, etc.

### Progress

All progress is stored in YAML files under `progress/`:

- `book.yaml` — per-chapter progress for both languages
- `courses.yaml` — Anthropic courses status
- `overall.yaml` — high-level dashboard summary

### Courses (Anthropic)

Because full offline mirroring of Skilljar content is not practical, we use **rich cards**:

- Title, description, link to official course
- Status (`todo` / `in_progress` / `done`)
- Personal notes
- Key takeaways (filled by user or agent)

---

## Authentication

Simple password protection (password: `timohin2026`).

Implementation options (in order of preference):
1. FastAPI dependency + session / JWT cookie
2. nginx basic auth in front of the app (simpler, good enough for personal use)

---

## Deployment

- Domain: `idealabs.co/learn`
- Reverse proxy: nginx
- Process manager: systemd or docker-compose (to be decided in Phase 0)

---

## Extensibility rules

When adding new features, prefer:

1. New files / folders under `content/` or `progress/`
2. New API endpoints that only read/write those files
3. New React pages that consume the existing API

Avoid introducing a traditional database for core learning data.
