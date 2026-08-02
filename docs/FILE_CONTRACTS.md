# AI Atlas — File Contracts

These contracts must be respected by both the application and any agent.

---

## 1. `content/book/meta.yaml`

```yaml
title: "AI Engineering"
author: "Chip Huyen"
versions:
  en:
    source_pdf: "original/AI Engineering by Chip Huyen.pdf"
    language: "en"
  ru:
    source_pdf: "original/Хьюен Ч. - AI-инженерия - 2026.pdf"
    language: "ru"

chapters:
  - id: "01-introduction"
    title:
      en: "Introduction to Building AI Applications with Foundation Models"
      ru: "Введение в создание AI-приложений на фундаментных моделях"
    file:
      en: "en/01-introduction.md"
      ru: "ru/01-introduction.md"
    order: 1

  - id: "02-understanding-foundation-models"
    title:
      en: "Understanding Foundation Models"
      ru: "Понимание фундаментных моделей"
    file:
      en: "en/02-understanding-foundation-models.md"
      ru: "ru/02-understanding-foundation-models.md"
    order: 2
```

---

## 2. `progress/book.yaml`

```yaml
updated_at: "2026-08-02T12:00:00+03:00"

overall:
  percent: 12
  current_chapter_id: "02-understanding-foundation-models"
  current_language: "ru"

chapters:
  "01-introduction":
    status: "done"          # todo | in_progress | done
    language_progress:
      en:
        status: "done"
        last_page: null
        notes: "Хорошая карта стека"
      ru:
        status: "done"
        notes: ""
    finished_at: "2026-08-01"

  "02-understanding-foundation-models":
    status: "in_progress"
    language_progress:
      en:
        status: "todo"
      ru:
        status: "in_progress"
        last_position: "section-3"
        notes: "Нужно ещё раз прочитать про sampling"
```

---

## 3. `progress/courses.yaml`

```yaml
updated_at: "2026-08-02T12:00:00+03:00"

courses:
  - id: "mcp-intro"
    title: "Introduction to Model Context Protocol"
    url: "https://anthropic.skilljar.com/introduction-to-model-context-protocol"
    status: "todo"          # todo | in_progress | done
    priority: 1
    notes: ""
    takeaways: []

  - id: "mcp-advanced"
    title: "Model Context Protocol: Advanced Topics"
    url: "https://anthropic.skilljar.com/model-context-protocol-advanced-topics"
    status: "todo"
    priority: 2
    notes: ""
    takeaways: []
```

---

## 4. Chapter Markdown files

Each chapter file should start with optional frontmatter:

```markdown
---
id: "01-introduction"
title: "Introduction to Building AI Applications with Foundation Models"
language: "en"
---

# Introduction to Building AI Applications with Foundation Models

...
```

---

## Rules

1. IDs must be stable (never rename chapter IDs lightly).
2. Agents should update `updated_at` when changing progress files.
3. The application should treat missing progress entries as `todo`.

---

## 5. `content/courses/anthropic/index.yaml` (NEW — Phase 3)

Static course catalog. User progress lives separately in `progress/courses.yaml`.

```yaml
courses:
  - id: "mcp-intro"
    title: "Introduction to Model Context Protocol"
    url: "https://anthropic.skilljar.com/introduction-to-model-context-protocol"
    type: "course"              # course | article
    priority: 1
    estimated_time: "1h"
    lectures: 16
    quizzes: 1
    description: >
      Learn to build modular AI applications using MCP...
    topics:
      - "MCP architecture & client-server model"
      - "Building MCP servers with Python SDK"
    related_chapters:
      - "06-rag-and-agents"
```

---

## 6. `progress/courses.yaml` (UPDATED — Phase 3)

Now contains ONLY user-progress fields. Backend merges with catalog on GET.

```yaml
updated_at: "2026-08-02T18:00:00+03:00"

courses:
  - id: "mcp-intro"
    status: "todo"
    notes: ""
    takeaways: []
    started_at: null
    completed_at: null
```

**Merge contract:** `GET /api/progress/courses` reads both files and returns merged dicts.
`PUT /api/progress/courses` extracts only user-progress fields before writing.
`GET /api/content/courses/anthropic` returns the raw catalog.

---

## Rules (continued)

4. `courses.yaml` must NOT contain static metadata (title, url, description, topics, etc.) — those live in the catalog.
