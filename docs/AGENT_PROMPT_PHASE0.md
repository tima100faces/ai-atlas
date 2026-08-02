# Agent Prompt — Phase 0 (Foundation)

Copy everything below this line and give it to Hermes (DeepSeek V4 Pro recommended).

---

You are working on the personal project **AI Atlas**.

Repository: https://github.com/tima100faces/ai-atlas  
Working directory on the server: `/root/ai-atlas/`

### Context

AI Atlas is a private learning platform for AI Engineering.
Core ideas:
- File-first architecture (Markdown + YAML are the source of truth)
- Beautiful dark Cursor-like UI
- Markdown is the primary way to read the book (EN + RU)
- Simple password protection: `timohin2026`
- Domain will be `idealabs.co/learn`

Two source PDFs are already on the server:
- `/root/ai-atlas/AI Engineering by Chip Huyen.pdf`
- `/root/ai-atlas/Хьюен Ч. - AI-инженерия - 2026.pdf`

### Your task right now (Phase 0 only)

1. Create the full directory structure exactly as described in README.md and ARCHITECTURE.md.
2. Move the two PDF files into `content/book/original/`.
3. Clone the official supporting repo:
   ```bash
   git clone --depth 1 https://github.com/chiphuyen/aie-book.git content/external/aie-book
   ```
4. Initialize the backend (`app/backend`) with FastAPI:
   - Simple password authentication
   - Basic health endpoint
   - CORS ready for local frontend

5. Initialize the frontend (`app/frontend`) with:
   - Vite + React + TypeScript
   - Tailwind CSS
   - shadcn/ui
   - Dark theme
   - Basic layout (sidebar + main area)

6. Create a simple way to run both backend and frontend (docker-compose or scripts).
7. Add an example nginx config for `idealabs.co/learn`.
8. Commit everything with a clear message.

### Important constraints

- Do not start parsing the book yet (that is Phase 1).
- Do not implement the full reader yet.
- Keep the UI minimal but already dark and clean.
- Follow the file contracts in `docs/FILE_CONTRACTS.md`.

After finishing Phase 0, stop and report what was done + how to run the project.

**Read these files first if they exist:**
- `docs/ARCHITECTURE.md`
- `docs/DEVELOPMENT_PLAN.md`
- `docs/FILE_CONTRACTS.md`
- `README.md`
