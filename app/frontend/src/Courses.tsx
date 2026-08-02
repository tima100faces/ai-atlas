import { useState, useEffect, useCallback } from 'react'

interface Course {
  id: string
  title: string
  url: string
  type: 'course' | 'article'
  status: 'todo' | 'in_progress' | 'done'
  priority: number
  estimated_time: string
  lectures: number
  quizzes: number
  description: string
  topics: string[]
  related_chapters: string[]
  notes: string
  takeaways: string[]
  started_at: string | null
  completed_at: string | null
}

interface ChapterMeta {
  id: string
  title: { en: string; ru: string }
  order: number
}

type FilterStatus = 'all' | 'todo' | 'in_progress' | 'done'
type SortMode = 'priority' | 'status'

interface CoursesProps {
  onNavigate: (view: string, params?: Record<string, string>) => void
}

export default function Courses({ onNavigate }: CoursesProps) {
  const [courses, setCourses] = useState<Course[]>([])
  const [meta, setMeta] = useState<ChapterMeta[]>([])
  const [filter, setFilter] = useState<FilterStatus>('all')
  const [sort, setSort] = useState<SortMode>('priority')
  const [expanded, setExpanded] = useState<Set<string>>(new Set())
  const [editingNotes, setEditingNotes] = useState<Record<string, string>>({})
  const [newTakeaway, setNewTakeaway] = useState<Record<string, string>>({})

  // Fetch courses + book meta
  useEffect(() => {
    fetch('/api/progress/courses')
      .then(r => r.json())
      .then(data => {
        if (data.courses) setCourses(data.courses)
      })
      .catch(() => {})

    fetch('/api/content/book/meta')
      .then(r => r.json())
      .then(data => {
        if (data.chapters) setMeta(data.chapters)
      })
      .catch(() => {})
  }, [])

  // Save entire courses list (backend filters user-progress fields)
  const save = useCallback(async (updated: Course[]) => {
    setCourses(updated)
    await fetch('/api/progress/courses', {
      method: 'PUT',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({
        updated_at: new Date().toISOString(),
        courses: updated,
      }),
    })
  }, [])

  const updateStatus = (courseId: string, status: 'todo' | 'in_progress' | 'done') => {
    const now = new Date().toISOString().split('T')[0]
    const updated = courses.map(c => {
      if (c.id !== courseId) return c
      return {
        ...c,
        status,
        started_at: status === 'in_progress' && !c.started_at ? now : c.started_at,
        completed_at: status === 'done' ? now : c.completed_at,
      }
    })
    save(updated)
  }

  const saveNotes = (courseId: string) => {
    const notes = editingNotes[courseId]
    if (notes === undefined) return
    const updated = courses.map(c =>
      c.id === courseId ? { ...c, notes } : c
    )
    save(updated)
  }

  const addTakeaway = (courseId: string) => {
    const text = newTakeaway[courseId]?.trim()
    if (!text) return
    const updated = courses.map(c =>
      c.id === courseId ? { ...c, takeaways: [...c.takeaways, text] } : c
    )
    setNewTakeaway(prev => ({ ...prev, [courseId]: '' }))
    save(updated)
  }

  const removeTakeaway = (courseId: string, idx: number) => {
    const updated = courses.map(c =>
      c.id === courseId
        ? { ...c, takeaways: c.takeaways.filter((_, i) => i !== idx) }
        : c
    )
    save(updated)
  }

  const toggleExpand = (id: string) => {
    setExpanded(prev => {
      const next = new Set(prev)
      if (next.has(id)) next.delete(id)
      else next.add(id)
      return next
    })
  }

  const getChapterTitle = (chId: string): string => {
    const ch = meta.find(c => c.id === chId)
    return ch?.title?.ru || ch?.title?.en || chId
  }

  const statusColor = (s: string): string => {
    switch (s) {
      case 'done': return '#4ade80'
      case 'in_progress': return '#fbbf24'
      default: return 'var(--text-secondary)'
    }
  }

  // Filter + sort
  const filtered = courses
    .filter(c => filter === 'all' || c.status === filter)
    .sort((a, b) => {
      if (sort === 'priority') return a.priority - b.priority
      const order: Record<string, number> = { in_progress: 0, todo: 1, done: 2 }
      return (order[a.status] ?? 1) - (order[b.status] ?? 1)
    })

  const stats = {
    total: courses.length,
    done: courses.filter(c => c.status === 'done').length,
    inProgress: courses.filter(c => c.status === 'in_progress').length,
  }

  return (
    <div style={{ display: 'flex', minHeight: '100vh', background: 'var(--bg-primary)' }}>
      {/* Sidebar */}
      <aside style={{
        width: 260, minWidth: 260, height: '100vh',
        background: 'var(--bg-sidebar)', borderRight: '1px solid var(--border)',
        padding: 20, display: 'flex', flexDirection: 'column', gap: 8,
        position: 'sticky', top: 0,
      }}>
        <div
          style={{ fontSize: 18, fontWeight: 600, color: 'var(--accent)', marginBottom: 20, cursor: 'pointer' }}
          onClick={() => onNavigate('dashboard')}
        >
          AI Atlas
        </div>

        <SidebarSection title="Book">
          <SidebarItem label="Reader" onClick={() => onNavigate('reader')} />
          <SidebarItem label="Dashboard" onClick={() => onNavigate('dashboard')} />
        </SidebarSection>

        <SidebarSection title="Courses">
          <SidebarItem label="Anthropic" active />
        </SidebarSection>
      </aside>

      {/* Main */}
      <div style={{ flex: 1, padding: '32px 40px', maxWidth: 900, width: '100%' }}>
        {/* Header */}
        <div style={{ marginBottom: 28 }}>
          <h1 style={{ fontSize: 28, fontWeight: 700, marginBottom: 4 }}>
            Anthropic Courses
          </h1>
          <div style={{ display: 'flex', gap: 20, color: 'var(--text-secondary)', fontSize: 13 }}>
            <span>{stats.total} courses</span>
            <span style={{ color: '#4ade80' }}>{stats.done} done</span>
            <span style={{ color: '#fbbf24' }}>{stats.inProgress} in progress</span>
          </div>
        </div>

        {/* Filters */}
        <div style={{
          display: 'flex', gap: 8, marginBottom: 24,
          flexWrap: 'wrap', alignItems: 'center',
        }}>
          {(['all', 'todo', 'in_progress', 'done'] as FilterStatus[]).map(f => (
            <button
              key={f}
              onClick={() => setFilter(f)}
              style={{
                padding: '6px 14px', borderRadius: 6, border: '1px solid var(--border)',
                background: filter === f ? 'var(--accent)' : 'transparent',
                color: filter === f ? '#fff' : 'var(--text-secondary)',
                cursor: 'pointer', fontSize: 12, fontWeight: 500,
                textTransform: 'capitalize',
              }}
            >
              {f === 'all' ? 'All' : f.replace('_', ' ')}
            </button>
          ))}
          <div style={{ flex: 1 }} />
          <select
            value={sort}
            onChange={e => setSort(e.target.value as SortMode)}
            style={{
              padding: '5px 10px', borderRadius: 6,
              background: 'var(--bg-secondary)', border: '1px solid var(--border)',
              color: 'var(--text-primary)', fontSize: 12, outline: 'none',
            }}
          >
            <option value="priority">By Priority</option>
            <option value="status">By Status</option>
          </select>
        </div>

        {/* Course cards */}
        <div style={{ display: 'flex', flexDirection: 'column', gap: 12 }}>
          {filtered.map(course => {
            const isExpanded = expanded.has(course.id)
            const noteText = editingNotes[course.id] ?? course.notes

            return (
              <div key={course.id} style={{
                background: 'var(--bg-secondary)', borderRadius: 12,
                border: '1px solid var(--border)',
                overflow: 'hidden',
              }}>
                {/* Card header */}
                <div style={{
                  padding: '16px 20px', display: 'flex', alignItems: 'flex-start',
                  gap: 14,
                }}>
                  {/* Priority badge */}
                  <div style={{
                    width: 28, height: 28, borderRadius: 8,
                    background: 'var(--bg-primary)', border: '1px solid var(--border)',
                    display: 'flex', alignItems: 'center', justifyContent: 'center',
                    fontSize: 13, fontWeight: 600, color: 'var(--accent)',
                    flexShrink: 0,
                  }}>
                    {course.priority}
                  </div>

                  <div style={{ flex: 1, minWidth: 0 }}>
                    <div style={{ display: 'flex', alignItems: 'baseline', gap: 8, marginBottom: 4 }}>
                      <span style={{ fontSize: 15, fontWeight: 600, color: 'var(--text-primary)' }}>
                        {course.title}
                      </span>
                      {course.type === 'article' && (
                        <span style={{
                          padding: '1px 6px', borderRadius: 4, fontSize: 10,
                          background: 'var(--bg-primary)', border: '1px solid var(--border)',
                          color: 'var(--text-secondary)',
                        }}>
                          article
                        </span>
                      )}
                    </div>

                    <div style={{ display: 'flex', gap: 14, flexWrap: 'wrap', fontSize: 12, color: 'var(--text-secondary)' }}>
                      {course.estimated_time && <span>⏱ {course.estimated_time}</span>}
                      {course.lectures > 0 && <span>{course.lectures} lectures</span>}
                      {course.quizzes > 0 && <span>{course.quizzes} quizzes</span>}
                    </div>

                    {/* Topics (always visible) */}
                    {course.topics.length > 0 && (
                      <div style={{ display: 'flex', flexWrap: 'wrap', gap: 4, marginTop: 8 }}>
                        {course.topics.slice(0, isExpanded ? course.topics.length : 4).map((t, i) => (
                          <span key={i} style={{
                            padding: '2px 8px', borderRadius: 4,
                            background: 'var(--bg-primary)', border: '1px solid var(--border)',
                            fontSize: 11, color: 'var(--text-secondary)',
                          }}>
                            {t}
                          </span>
                        ))}
                        {!isExpanded && course.topics.length > 4 && (
                          <span style={{ fontSize: 11, color: 'var(--accent)', cursor: 'pointer', padding: '2px 4px' }}
                            onClick={() => toggleExpand(course.id)}>
                            +{course.topics.length - 4} more
                          </span>
                        )}
                      </div>
                    )}
                  </div>

                  {/* Status selector */}
                  <select
                    value={course.status}
                    onChange={e => updateStatus(course.id, e.target.value as any)}
                    style={{
                      padding: '4px 10px', borderRadius: 6,
                      background: 'var(--bg-primary)', border: '1px solid var(--border)',
                      color: statusColor(course.status), fontSize: 12,
                      cursor: 'pointer', outline: 'none', flexShrink: 0,
                    }}
                  >
                    <option value="todo">Todo</option>
                    <option value="in_progress">In Progress</option>
                    <option value="done">Done</option>
                  </select>
                </div>

                {/* Expandable section */}
                {isExpanded && (
                  <div style={{ padding: '0 20px 20px', borderTop: '1px solid var(--border)' }}>
                    {/* Description */}
                    <div style={{ marginTop: 14, fontSize: 13, lineHeight: 1.7, color: 'var(--text-secondary)' }}>
                      {course.description.trim()}
                    </div>

                    {/* Related chapters */}
                    {course.related_chapters.length > 0 && (
                      <div style={{ marginTop: 14 }}>
                        <div style={{ fontSize: 11, fontWeight: 600, color: 'var(--text-secondary)', marginBottom: 6, textTransform: 'uppercase', letterSpacing: 0.5 }}>
                          Related Chapters
                        </div>
                        <div style={{ display: 'flex', flexWrap: 'wrap', gap: 6 }}>
                          {course.related_chapters.map(chId => (
                            <span
                              key={chId}
                              onClick={() => onNavigate('reader', { chapterId: chId, language: 'ru' })}
                              style={{
                                padding: '3px 10px', borderRadius: 6,
                                background: 'var(--bg-primary)', border: '1px solid var(--accent)',
                                fontSize: 12, color: 'var(--accent)', cursor: 'pointer',
                                transition: 'background 0.15s',
                              }}
                            >
                              {getChapterTitle(chId)}
                            </span>
                          ))}
                        </div>
                      </div>
                    )}

                    {/* Personal notes */}
                    <div style={{ marginTop: 14 }}>
                      <div style={{ fontSize: 11, fontWeight: 600, color: 'var(--text-secondary)', marginBottom: 6, textTransform: 'uppercase', letterSpacing: 0.5 }}>
                        Notes
                      </div>
                      <textarea
                        value={noteText}
                        onChange={e => setEditingNotes(prev => ({ ...prev, [course.id]: e.target.value }))}
                        onBlur={() => saveNotes(course.id)}
                        placeholder="Your notes on this course..."
                        rows={3}
                        style={{
                          width: '100%', padding: '10px 14px',
                          background: 'var(--bg-primary)', border: '1px solid var(--border)',
                          borderRadius: 8, color: 'var(--text-primary)', fontSize: 13,
                          resize: 'vertical', outline: 'none', lineHeight: 1.5,
                          fontFamily: 'inherit',
                        }}
                      />
                    </div>

                    {/* Takeaways */}
                    <div style={{ marginTop: 14 }}>
                      <div style={{ fontSize: 11, fontWeight: 600, color: 'var(--text-secondary)', marginBottom: 6, textTransform: 'uppercase', letterSpacing: 0.5 }}>
                        Takeaways
                      </div>
                      {course.takeaways.length > 0 && (
                        <div style={{ display: 'flex', flexDirection: 'column', gap: 6, marginBottom: 8 }}>
                          {course.takeaways.map((t, i) => (
                            <div key={i} style={{
                              display: 'flex', alignItems: 'flex-start', gap: 8,
                              padding: '6px 10px', borderRadius: 6,
                              background: 'var(--bg-primary)', border: '1px solid var(--border)',
                            }}>
                              <span style={{ fontSize: 13, lineHeight: 1.5, flex: 1, color: 'var(--text-primary)' }}>{t}</span>
                              <button
                                onClick={() => removeTakeaway(course.id, i)}
                                style={{
                                  background: 'none', border: 'none',
                                  color: 'var(--text-secondary)', cursor: 'pointer',
                                  fontSize: 14, padding: '0 2px', lineHeight: 1,
                                }}
                              >
                                ×
                              </button>
                            </div>
                          ))}
                        </div>
                      )}
                      <div style={{ display: 'flex', gap: 8 }}>
                        <input
                          type="text"
                          value={newTakeaway[course.id] || ''}
                          onChange={e => setNewTakeaway(prev => ({ ...prev, [course.id]: e.target.value }))}
                          onKeyDown={e => { if (e.key === 'Enter') addTakeaway(course.id) }}
                          placeholder="Add a key takeaway..."
                          style={{
                            flex: 1, padding: '7px 12px',
                            background: 'var(--bg-primary)', border: '1px solid var(--border)',
                            borderRadius: 6, color: 'var(--text-primary)', fontSize: 13,
                            outline: 'none',
                          }}
                        />
                        <button
                          onClick={() => addTakeaway(course.id)}
                          style={{
                            padding: '7px 14px', borderRadius: 6, border: 'none',
                            background: 'var(--accent)', color: '#fff',
                            cursor: 'pointer', fontSize: 12, fontWeight: 500,
                          }}
                        >
                          Add
                        </button>
                      </div>
                    </div>

                    {/* Dates */}
                    {(course.started_at || course.completed_at) && (
                      <div style={{ marginTop: 12, fontSize: 11, color: 'var(--text-secondary)', display: 'flex', gap: 16 }}>
                        {course.started_at && <span>Started: {course.started_at}</span>}
                        {course.completed_at && <span>Completed: {course.completed_at}</span>}
                      </div>
                    )}
                  </div>
                )}

                {/* Card footer */}
                <div style={{
                  padding: '10px 20px', borderTop: '1px solid var(--border)',
                  display: 'flex', gap: 16, fontSize: 12,
                }}>
                  <span
                    onClick={() => toggleExpand(course.id)}
                    style={{ color: 'var(--accent)', cursor: 'pointer' }}
                  >
                    {isExpanded ? '▲ Less' : '▼ More'}
                  </span>
                  <a
                    href={course.url}
                    target="_blank"
                    rel="noopener noreferrer"
                    style={{ color: 'var(--text-secondary)', textDecoration: 'none' }}
                  >
                    Open on Anthropic ↗
                  </a>
                </div>
              </div>
            )
          })}
        </div>

        {filtered.length === 0 && (
          <div style={{ textAlign: 'center', color: 'var(--text-secondary)', marginTop: 60, fontSize: 14 }}>
            No courses match this filter.
          </div>
        )}
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
