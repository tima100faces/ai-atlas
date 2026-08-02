import { useState, useEffect } from 'react'

interface Progress {
  book_percent: number
  current_chapter_id: string | null
  current_language: string | null
  total_chapters: number
  done_chapters: number
}

interface Course {
  id: string
  title: string
  url: string
  status: 'todo' | 'in_progress' | 'done'
  priority: number
  notes: string
  takeaways: string[]
}

interface ChapterProgress {
  status: string
  language_progress: {
    en: { status: string; notes: string }
    ru: { status: string; notes: string }
  }
}

interface Note {
  chapterId: string
  language: string
  text: string
}

interface ChapterMeta {
  id: string
  title: { en: string; ru: string }
  order: number
}

interface Meta {
  title: string
  author: string
  chapters: ChapterMeta[]
}

interface DashboardProps {
  onNavigate: (view: string, params?: Record<string, string>) => void
}

export default function Dashboard({ onNavigate }: DashboardProps) {
  const [progress, setProgress] = useState<Progress | null>(null)
  const [courses, setCourses] = useState<Course[]>([])
  const [notes, setNotes] = useState<Note[]>([])
  const [meta, setMeta] = useState<Meta | null>(null)

  useEffect(() => {
    fetch('/api/progress/overall')
      .then(r => r.json())
      .then(data => {
        if (!data.detail) setProgress(data)
      })
      .catch(() => {})

    fetch('/api/progress/courses')
      .then(r => r.json())
      .then(data => {
        if (data.courses) setCourses(data.courses)
      })
      .catch(() => {})

    fetch('/api/progress/book')
      .then(r => r.json())
      .then(data => {
        if (data.chapters) {
          const allNotes: Note[] = []
          for (const [chId, chData] of Object.entries(data.chapters)) {
            const cp = chData as ChapterProgress
            for (const lang of ['en', 'ru'] as const) {
              const lp = cp.language_progress?.[lang]
              if (lp?.notes) {
                allNotes.push({ chapterId: chId, language: lang, text: lp.notes })
              }
            }
          }
          setNotes(allNotes.slice(-5).reverse())
        }
      })
      .catch(() => {})

    fetch('/api/content/book/meta')
      .then(r => r.json())
      .then(data => {
        if (data.chapters) setMeta(data)
      })
      .catch(() => {})
  }, [])

  const updateCourseStatus = async (courseId: string, newStatus: 'todo' | 'in_progress' | 'done') => {
    const updated = courses.map(c =>
      c.id === courseId ? { ...c, status: newStatus } : c
    )
    setCourses(updated)

    await fetch('/api/progress/courses', {
      method: 'PUT',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({
        updated_at: new Date().toISOString(),
        courses: updated,
      }),
    })
  }

  const getChapterTitle = (chId: string, lang: string): string => {
    if (!meta?.chapters) return chId
    const ch = meta.chapters.find(c => c.id === chId)
    const title = ch?.title as Record<string, string> | undefined
    return title?.[lang] || title?.en || chId
  }

  const statusColor = (s: string) => {
    switch (s) {
      case 'done': return '#4ade80'
      case 'in_progress': return '#fbbf24'
      default: return 'var(--text-secondary)'
    }
  }

  const doneChapters = progress?.done_chapters || 0
  const totalChapters = progress?.total_chapters || meta?.chapters?.length || 11
  const currentChap = progress?.current_chapter_id
  const currentLang = progress?.current_language || 'en'

  return (
    <div style={{ display: 'flex', minHeight: '100vh', background: 'var(--bg-primary)' }}>
      {/* Sidebar */}
      <aside style={{
        width: 260, minWidth: 260, height: '100vh',
        background: 'var(--bg-sidebar)', borderRight: '1px solid var(--border)',
        padding: 20, display: 'flex', flexDirection: 'column', gap: 8,
        position: 'sticky', top: 0,
      }}>
        <h2 style={{ fontSize: 18, fontWeight: 600, color: 'var(--accent)', marginBottom: 20 }}>
          AI Atlas
        </h2>

        <SidebarSection title="Book">
          <SidebarItem label="Reader" onClick={() => onNavigate('reader')} />
          <SidebarItem label="Dashboard" active />
        </SidebarSection>

        <SidebarSection title="Courses">
          <SidebarItem label="Anthropic" onClick={() => onNavigate('courses')} />
        </SidebarSection>

        <SidebarSection title="Notes">
          <SidebarItem label="All Notes" />
        </SidebarSection>
      </aside>

      {/* Main */}
      <div style={{
        flex: 1, padding: '32px 40px',
        maxWidth: 1100, width: '100%',
      }}>
        <h1 style={{ fontSize: 28, fontWeight: 700, marginBottom: 4 }}>
          AI Atlas
        </h1>
        <p style={{ color: 'var(--text-secondary)', fontSize: 14, marginBottom: 32 }}>
          {meta?.title || 'AI Engineering'} by {meta?.author || 'Chip Huyen'}
        </p>

        {/* Stat cards */}
        <div style={{
          display: 'grid',
          gridTemplateColumns: 'repeat(auto-fit, minmax(180px, 1fr))',
          gap: 16, marginBottom: 28,
        }}>
          <StatCard label="Book Progress" value={`${progress?.book_percent || 0}%`} />
          <StatCard label="Chapters Read" value={`${doneChapters} / ${totalChapters}`} />
          <StatCard
            label="Current Chapter"
            value={currentChap ? getChapterTitle(currentChap, currentLang) : 'Not started'}
          />
          <StatCard label="Language" value={currentLang.toUpperCase()} />
        </div>

        {/* Quick actions */}
        <div style={{ display: 'flex', gap: 12, marginBottom: 40 }}>
          <button
            onClick={() => onNavigate('reader', {
              chapterId: currentChap || '',
              language: currentLang,
            })}
            style={{
              padding: '12px 24px', borderRadius: 8, border: 'none',
              background: 'var(--accent)', color: '#fff',
              cursor: 'pointer', fontSize: 14, fontWeight: 500,
            }}
          >
            Continue Reading →
          </button>
          <button
            onClick={() => onNavigate('reader')}
            style={{
              padding: '12px 24px', borderRadius: 8,
              border: '1px solid var(--border)', background: 'transparent',
              color: 'var(--text-secondary)', cursor: 'pointer',
              fontSize: 14,
            }}
          >
            Open Reader
          </button>
        </div>

        {/* Two-column: Courses + Notes */}
        <div style={{
          display: 'grid',
          gridTemplateColumns: 'minmax(0, 1fr) minmax(0, 1fr)',
          gap: 32,
        }}>
          {/* Courses */}
          <div>
            <h2 style={{ fontSize: 18, fontWeight: 600, marginBottom: 16, color: 'var(--text-primary)' }}>
              Anthropic Courses
            </h2>
            <div style={{ display: 'flex', flexDirection: 'column', gap: 8 }}>
              {courses
                .sort((a, b) => a.priority - b.priority)
                .map(course => (
                  <div key={course.id} style={{
                    background: 'var(--bg-secondary)', borderRadius: 10,
                    border: '1px solid var(--border)', padding: '14px 16px',
                  }}>
                    <div style={{
                      fontSize: 14, fontWeight: 500, marginBottom: 8,
                      color: 'var(--text-primary)',
                    }}>
                      {course.title}
                    </div>
                    <div style={{ display: 'flex', gap: 12, alignItems: 'center' }}>
                      <select
                        value={course.status}
                        onChange={e => updateCourseStatus(course.id, e.target.value as any)}
                        style={{
                          padding: '3px 8px', borderRadius: 6,
                          background: 'var(--bg-primary)', border: '1px solid var(--border)',
                          color: statusColor(course.status), fontSize: 12,
                          cursor: 'pointer', outline: 'none',
                        }}
                      >
                        <option value="todo">Todo</option>
                        <option value="in_progress">In Progress</option>
                        <option value="done">Done</option>
                      </select>
                      <a
                        href={course.url}
                        target="_blank"
                        rel="noopener noreferrer"
                        style={{
                          fontSize: 12, color: 'var(--accent)',
                          textDecoration: 'none',
                        }}
                      >
                        Open ↗
                      </a>
                    </div>
                  </div>
                ))}
            </div>
          </div>

          {/* Recent Notes */}
          <div>
            <h2 style={{ fontSize: 18, fontWeight: 600, marginBottom: 16, color: 'var(--text-primary)' }}>
              Recent Notes
            </h2>
            {notes.length === 0 ? (
              <div style={{
                background: 'var(--bg-secondary)', borderRadius: 10,
                border: '1px solid var(--border)', padding: '20px 16px',
              }}>
                <p style={{ color: 'var(--text-secondary)', fontSize: 14, lineHeight: 1.6 }}>
                  No notes yet. Add notes to chapters as you read — they'll appear here.
                </p>
              </div>
            ) : (
              <div style={{ display: 'flex', flexDirection: 'column', gap: 8 }}>
                {notes.map((note, i) => (
                  <div key={i} style={{
                    background: 'var(--bg-secondary)', borderRadius: 10,
                    border: '1px solid var(--border)', padding: '12px 16px',
                    cursor: 'pointer',
                  }}
                  onClick={() => onNavigate('reader', {
                    chapterId: note.chapterId,
                    language: note.language,
                  })}>
                    <div style={{
                      fontSize: 11, color: 'var(--text-secondary)', marginBottom: 4,
                      display: 'flex', gap: 8,
                    }}>
                      <span>{getChapterTitle(note.chapterId, note.language)}</span>
                      <span style={{ color: 'var(--accent)' }}>{note.language.toUpperCase()}</span>
                    </div>
                    <div style={{ fontSize: 13, lineHeight: 1.5, color: 'var(--text-primary)' }}>
                      {note.text}
                    </div>
                  </div>
                ))}
              </div>
            )}
          </div>
        </div>
      </div>
    </div>
  )
}

function SidebarSection({ title, children }: { title: string; children: React.ReactNode }) {
  return (
    <div style={{ marginBottom: 8 }}>
      <div style={{
        fontSize: 11, fontWeight: 600, color: 'var(--text-secondary)',
        textTransform: 'uppercase', letterSpacing: 1, padding: '8px 8px 4px',
      }}>
        {title}
      </div>
      {children}
    </div>
  )
}

function SidebarItem({ label, active, onClick }: { label: string; active?: boolean; onClick?: () => void }) {
  return (
    <div
      onClick={onClick}
      style={{
        padding: '6px 8px', borderRadius: 6, fontSize: 13,
        color: active ? 'var(--text-primary)' : 'var(--text-secondary)',
        background: active ? 'var(--bg-secondary)' : 'transparent',
        cursor: onClick ? 'pointer' : 'default',
        transition: 'background 0.15s',
      }}
    >
      {label}
    </div>
  )
}

function StatCard({ label, value }: { label: string; value: string }) {
  return (
    <div style={{
      background: 'var(--bg-secondary)', borderRadius: 12, padding: 20,
      border: '1px solid var(--border)',
    }}>
      <div style={{ fontSize: 12, color: 'var(--text-secondary)', marginBottom: 6 }}>
        {label}
      </div>
      <div style={{ fontSize: 22, fontWeight: 600, color: 'var(--text-primary)' }}>
        {value}
      </div>
    </div>
  )
}
