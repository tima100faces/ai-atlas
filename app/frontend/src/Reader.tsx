import { useState, useEffect, useCallback } from 'react'

interface Chapter {
  id: string
  title: { en: string; ru: string }
  file: { en: string; ru: string }
  order: number
}

interface ChapterProgress {
  status: 'todo' | 'in_progress' | 'done'
  language_progress: {
    en: { status: string; notes: string }
    ru: { status: string; notes: string }
  }
}

interface Meta {
  title: string
  author: string
  chapters: Chapter[]
}

type Language = 'en' | 'ru'

interface ReaderProps {
  onNavigate: (view: string) => void
}

export default function Reader({ onNavigate }: ReaderProps) {
  const [meta, setMeta] = useState<Meta | null>(null)
  const [progress, setProgress] = useState<Record<string, ChapterProgress>>({})
  const [language, setLanguage] = useState<Language>('ru')
  const [activeChapter, setActiveChapter] = useState<Chapter | null>(null)
  const [content, setContent] = useState<string>('')
  const [note, setNote] = useState('')
  const [loading, setLoading] = useState(false)
  const [sidebarOpen, setSidebarOpen] = useState(true)

  // Load book meta
  useEffect(() => {
    fetch('/api/content/book/meta')
      .then(r => r.json())
      .then(data => {
        if (data.chapters) {
          setMeta(data)
          // Start with chapter from progress or first chapter
          const savedChapterId = data.current_chapter_id
          const firstChapter = savedChapterId
            ? data.chapters.find((c: Chapter) => c.id === savedChapterId)
            : data.chapters[0]
          if (firstChapter) setActiveChapter(firstChapter)
        }
      })
      .catch(() => {})
  }, [])

  // Load progress
  useEffect(() => {
    fetch('/api/progress/book')
      .then(r => r.json())
      .then(data => {
        if (data.chapters) setProgress(data.chapters)
      })
      .catch(() => {})
  }, [])

  // Load chapter content
  const loadChapter = useCallback(async (chapter: Chapter) => {
    setLoading(true)
    setNote('')
    setActiveChapter(chapter)
    try {
      const filename = (chapter.file[language]?.split('/').pop() || chapter.file[language] || '').replace('.md', '')
      const res = await fetch(`/api/content/book/${language}/${filename}`)
      const data = await res.json()
      if (data.content) {
        setContent(data.content)
      }
      // Load saved note for this chapter+language
      const chProgress = progress[chapter.id]
      if (chProgress?.language_progress?.[language]?.notes) {
        setNote(chProgress.language_progress[language].notes || '')
      }
    } catch {
      setContent('*Chapter not available yet*')
    }
    setLoading(false)
  }, [language, progress])

  useEffect(() => {
    if (activeChapter) loadChapter(activeChapter)
  }, [language, activeChapter, loadChapter])

  // Mark chapter as read
  const markAsRead = async (status: 'done' | 'in_progress') => {
    if (!activeChapter) return

    const updated = { ...progress }
    if (!updated[activeChapter.id]) {
      updated[activeChapter.id] = {
        status: 'todo',
        language_progress: {
          en: { status: 'todo', notes: '' },
          ru: { status: 'todo', notes: '' },
        },
      }
    }
    updated[activeChapter.id].language_progress[language] = {
      status,
      notes: note,
    }
    updated[activeChapter.id].status = status

    const now = new Date().toISOString()
    const body = {
      updated_at: now,
      overall: {
        current_chapter_id: activeChapter.id,
        current_language: language,
      },
      chapters: updated,
    }

    await fetch('/api/progress/book', {
      method: 'PUT',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify(body),
    })
    setProgress(updated)
  }

  // Simple markdown-to-HTML renderer (for inline display)
  const renderMarkdown = (md: string) => {
    // Basic rendering: code blocks, headers, bold, italic, links
    let html = md
      // Code blocks
      .replace(/```(\w*)\n([\s\S]*?)```/g, '<pre class="code-block"><code>$2</code></pre>')
      // Inline code
      .replace(/`([^`]+)`/g, '<code class="inline-code">$1</code>')
      // Headers
      .replace(/^#### (.+)$/gm, '<h4>$1</h4>')
      .replace(/^### (.+)$/gm, '<h3>$1</h3>')
      .replace(/^## (.+)$/gm, '<h2>$1</h2>')
      .replace(/^# (.+)$/gm, '<h1>$1</h1>')
      // Bold + italic
      .replace(/\*\*\*(.+?)\*\*\*/g, '<strong><em>$1</em></strong>')
      .replace(/\*\*(.+?)\*\*/g, '<strong>$1</strong>')
      .replace(/\*(.+?)\*/g, '<em>$1</em>')
      // Links
      .replace(/\[([^\]]+)\]\(([^)]+)\)/g, '<a href="$2" target="_blank">$1</a>')
      // Images
      .replace(/!\[([^\]]*)\]\(([^)]+)\)/g, '<span class="image-ref">[Image: $1]</span>')
      // Tables (simple)
      .replace(/^\|(.+)\|$/gm, (match: string) => {
        const cells = match.split('|').filter(c => c.trim())
        if (match.includes('---')) return ''
        return '<tr>' + cells.map(c => `<td>${c.trim()}</td>`).join('') + '</tr>'
      })
      // Paragraphs
      .replace(/\n\n/g, '</p><p>')
      // Single newlines as <br>
      .replace(/\n/g, '<br>')

    return `<p>${html}</p>`
  }

  const chProgress = activeChapter ? progress[activeChapter.id] : null
  const langProgress = chProgress?.language_progress?.[language]
  const isRead = langProgress?.status === 'done'

  return (
    <div style={{ display: 'flex', minHeight: '100vh', background: 'var(--bg-primary)' }}>
      {/* Sidebar */}
      {sidebarOpen && (
        <aside style={{
          width: 280, minWidth: 280, height: '100vh',
          background: 'var(--bg-sidebar)', borderRight: '1px solid var(--border)',
          display: 'flex', flexDirection: 'column',
          position: 'sticky', top: 0,
        }}>
          {/* Header */}
          <div style={{ padding: '16px 20px', borderBottom: '1px solid var(--border)' }}>
            <div
              style={{ fontSize: 16, fontWeight: 700, color: 'var(--accent)', cursor: 'pointer' }}
              onClick={() => onNavigate('dashboard')}
            >
              AI Atlas
            </div>
            <div style={{ fontSize: 11, color: 'var(--text-secondary)', marginTop: 2 }}>
              {meta?.title || 'AI Engineering'}
            </div>
          </div>

          {/* Language switcher */}
          <div style={{ padding: '8px 20px', borderBottom: '1px solid var(--border)', display: 'flex', gap: 8 }}>
            <LangButton label="EN" active={language === 'en'} onClick={() => setLanguage('en')} />
            <LangButton label="RU" active={language === 'ru'} onClick={() => setLanguage('ru')} />
          </div>

          {/* Chapters */}
          <div style={{ flex: 1, overflowY: 'auto', padding: '8px 0' }}>
            {meta?.chapters?.map(ch => {
              const p = progress[ch.id]
              const lp = p?.language_progress?.[language]
              const isActive = activeChapter?.id === ch.id
              const statusIcon = lp?.status === 'done' ? '✓' : lp?.status === 'in_progress' ? '○' : '·'

              return (
                <div
                  key={ch.id}
                  onClick={() => setActiveChapter(ch)}
                  style={{
                    padding: '8px 20px', cursor: 'pointer',
                    fontSize: 13, lineHeight: 1.4,
                    color: isActive ? 'var(--text-primary)' : 'var(--text-secondary)',
                    background: isActive ? 'var(--bg-secondary)' : 'transparent',
                    borderLeft: isActive ? '2px solid var(--accent)' : '2px solid transparent',
                    transition: 'all 0.1s',
                    display: 'flex', gap: 8, alignItems: 'flex-start',
                  }}
                >
                  <span style={{
                    fontSize: 11, color: lp?.status === 'done' ? '#4ade80' : 'var(--text-secondary)',
                    minWidth: 16, textAlign: 'center', marginTop: 1,
                  }}>
                    {statusIcon}
                  </span>
                  <span>{ch.title[language] || ch.title.en}</span>
                </div>
              )
            })}
          </div>

          {/* Footer */}
          <div style={{ padding: '12px 20px', borderTop: '1px solid var(--border)', fontSize: 12, color: 'var(--text-secondary)' }}>
            <div style={{ cursor: 'pointer', marginBottom: 4 }} onClick={() => onNavigate('dashboard')}>
              ← Dashboard
            </div>
          </div>
        </aside>
      )}

      {/* Toggle sidebar button */}
      <div
        onClick={() => setSidebarOpen(!sidebarOpen)}
        style={{
          position: 'fixed', left: sidebarOpen ? 280 : 0, top: 16, zIndex: 10,
          width: 28, height: 28, cursor: 'pointer',
          display: 'flex', alignItems: 'center', justifyContent: 'center',
          background: 'var(--bg-secondary)', borderRadius: '0 6px 6px 0',
          border: '1px solid var(--border)', borderLeft: 'none',
          color: 'var(--text-secondary)', fontSize: 14,
        }}
      >
        {sidebarOpen ? '◂' : '▸'}
      </div>

      {/* Main content area */}
      <div style={{ flex: 1, padding: '40px 60px', maxWidth: 860, margin: '0 auto' }}>
        {loading ? (
          <div style={{ color: 'var(--text-secondary)', textAlign: 'center', marginTop: 100 }}>
            Loading...
          </div>
        ) : (
          <>
            {/* Chapter title */}
            <div style={{ marginBottom: 32 }}>
              <div style={{ fontSize: 12, color: 'var(--text-secondary)', marginBottom: 4 }}>
                Chapter {activeChapter?.order}
              </div>
              <h1 style={{ fontSize: 28, fontWeight: 700, color: 'var(--text-primary)', lineHeight: 1.3 }}>
                {activeChapter?.title[language] || activeChapter?.title.en}
              </h1>
            </div>

            {/* Rendered content */}
            <div
              className="chapter-content"
              dangerouslySetInnerHTML={{ __html: renderMarkdown(content) }}
              style={{
                fontSize: 15, lineHeight: 1.8, color: 'var(--text-primary)',
              }}
            />

            {/* Progress controls */}
            <div style={{
              marginTop: 48, paddingTop: 24, borderTop: '2px solid var(--border)',
              display: 'flex', alignItems: 'center', gap: 12, flexWrap: 'wrap',
            }}>
              <button
                onClick={() => markAsRead('done')}
                disabled={isRead}
                style={{
                  padding: '10px 24px', borderRadius: 8, border: 'none',
                  background: isRead ? '#1a3a1a' : 'var(--accent)',
                  color: isRead ? '#4ade80' : '#fff',
                  cursor: isRead ? 'default' : 'pointer',
                  fontSize: 14, fontWeight: 500,
                }}
              >
                {isRead ? '✓ Read' : 'Mark as Read'}
              </button>

              {!isRead && (
                <button
                  onClick={() => markAsRead('in_progress')}
                  style={{
                    padding: '10px 24px', borderRadius: 8,
                    border: '1px solid var(--border)', background: 'transparent',
                    color: 'var(--text-secondary)', cursor: 'pointer',
                    fontSize: 14,
                  }}
                >
                  In Progress
                </button>
              )}

              <input
                type="text"
                placeholder="Short note..."
                value={note}
                onChange={e => setNote(e.target.value)}
                onBlur={() => { if (note && langProgress?.status) markAsRead(langProgress.status as 'done' | 'in_progress') }}
                style={{
                  flex: 1, minWidth: 200, padding: '10px 14px',
                  background: 'var(--bg-secondary)', border: '1px solid var(--border)',
                  borderRadius: 8, color: 'var(--text-primary)', fontSize: 13,
                  outline: 'none',
                }}
              />
            </div>

            {/* PDF fallback */}
            <div style={{ marginTop: 16, fontSize: 12, color: 'var(--text-secondary)' }}>
              <a href="#" onClick={e => { e.preventDefault() }} style={{ color: 'var(--accent)' }}>
                Open original PDF →
              </a>
            </div>
          </>
        )}
      </div>

      {/* Chapter content styling */}
      <style>{`
        .chapter-content h1 { font-size: 24px; font-weight: 700; margin: 32px 0 16px; color: var(--text-primary); }
        .chapter-content h2 { font-size: 20px; font-weight: 600; margin: 28px 0 12px; color: var(--text-primary); }
        .chapter-content h3 { font-size: 17px; font-weight: 600; margin: 20px 0 10px; color: var(--text-primary); }
        .chapter-content h4 { font-size: 15px; font-weight: 600; margin: 16px 0 8px; color: var(--text-secondary); }
        .chapter-content p { margin: 10px 0; line-height: 1.8; }
        .chapter-content .code-block {
          background: #0a0a0a; border: 1px solid var(--border); border-radius: 8px;
          padding: 16px 20px; margin: 16px 0; overflow-x: auto;
          font-family: 'JetBrains Mono', 'Fira Code', monospace; font-size: 13px; line-height: 1.6;
        }
        .chapter-content .inline-code {
          background: var(--bg-secondary); padding: 2px 6px; border-radius: 4px;
          font-family: 'JetBrains Mono', 'Fira Code', monospace; font-size: 13px;
        }
        .chapter-content a { color: var(--accent); text-decoration: underline; }
        .chapter-content ul, .chapter-content ol { padding-left: 24px; margin: 10px 0; }
        .chapter-content li { margin: 4px 0; }
        .chapter-content table { border-collapse: collapse; margin: 16px 0; width: 100%; font-size: 13px; }
        .chapter-content td, .chapter-content th {
          border: 1px solid var(--border); padding: 8px 12px; text-align: left;
        }
        .chapter-content th { background: var(--bg-secondary); font-weight: 600; }
        .chapter-content blockquote {
          border-left: 3px solid var(--accent); padding: 4px 16px; margin: 12px 0;
          color: var(--text-secondary); font-style: italic;
        }
        .chapter-content .image-ref {
          display: block; padding: 20px; margin: 12px 0;
          background: var(--bg-secondary); border: 1px dashed var(--border);
          border-radius: 8px; text-align: center; color: var(--text-secondary); font-size: 13px;
        }
      `}</style>
    </div>
  )
}

function LangButton({ label, active, onClick }: { label: string; active: boolean; onClick: () => void }) {
  return (
    <button
      onClick={onClick}
      style={{
        padding: '4px 12px', borderRadius: 6, border: 'none',
        background: active ? 'var(--accent)' : 'transparent',
        color: active ? '#fff' : 'var(--text-secondary)',
        cursor: 'pointer', fontSize: 12, fontWeight: 600,
        transition: 'all 0.1s',
      }}
    >
      {label}
    </button>
  )
}
