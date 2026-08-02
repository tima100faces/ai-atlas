import { useState, useEffect } from 'react'
import Reader from './Reader'

interface Progress {
  book_percent: number
  current_chapter_id: string | null
  current_language: string | null
}

function LoginScreen({ onLogin }: { onLogin: () => void }) {
  const [password, setPassword] = useState('')
  const [error, setError] = useState('')

  const handleSubmit = async (e: React.FormEvent) => {
    e.preventDefault()
    try {
      const res = await fetch('/api/auth/login', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ password }),
      })
      if (res.ok) {
        onLogin()
      } else {
        setError('Wrong password')
      }
    } catch {
      setError('Connection error')
    }
  }

  return (
    <div style={{
      display: 'flex', alignItems: 'center', justifyContent: 'center',
      minHeight: '100vh', background: 'var(--bg-primary)'
    }}>
      <form onSubmit={handleSubmit} style={{
        display: 'flex', flexDirection: 'column', gap: 16,
        width: 320, padding: 32
      }}>
        <h1 style={{ fontSize: 24, fontWeight: 600, color: 'var(--text-primary)', textAlign: 'center' }}>
          AI Atlas
        </h1>
        <input
          type="password"
          placeholder="Password"
          value={password}
          onChange={e => setPassword(e.target.value)}
          autoFocus
          style={{
            padding: '10px 14px',
            background: 'var(--bg-secondary)',
            border: '1px solid var(--border)',
            borderRadius: 8,
            color: 'var(--text-primary)',
            fontSize: 14,
            outline: 'none',
          }}
        />
        {error && <p style={{ color: '#ff6b6b', fontSize: 13 }}>{error}</p>}
        <button type="submit" style={{
          padding: '10px 0',
          background: 'var(--accent)',
          color: '#fff',
          border: 'none',
          borderRadius: 8,
          fontSize: 14,
          cursor: 'pointer',
          fontWeight: 500,
        }}>
          Enter
        </button>
      </form>
    </div>
  )
}

function Dashboard({ onNavigate }: { onNavigate: (view: string) => void }) {
  const [progress, setProgress] = useState<Progress | null>(null)

  useEffect(() => {
    fetch('/api/progress/overall')
      .then(r => r.json())
      .then(data => {
        if (data.detail) return
        setProgress(data)
      })
      .catch(() => {})
  }, [])

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

        <Section title="Book">
          <NavItem label="Reader" active={false} onClick={() => onNavigate('reader')} />
          <NavItem label="Progress" />
        </Section>

        <Section title="Courses">
          <NavItem label="Anthropic" />
        </Section>

        <Section title="Notes">
          <NavItem label="Concepts" />
          <NavItem label="All Notes" />
        </Section>

        <div style={{ marginTop: 'auto' }}>
          <NavItem label="Dashboard" active />
        </div>
      </aside>

      {/* Main */}
      <div style={{ flex: 1, padding: 40 }}>
        <h1 style={{ fontSize: 28, fontWeight: 700, marginBottom: 8 }}>
          Welcome to AI Atlas
        </h1>
        <p style={{ color: 'var(--text-secondary)', fontSize: 14, marginBottom: 32 }}>
          Your personal AI Engineering learning platform
        </p>

        {progress && (
          <div style={{
            display: 'grid', gridTemplateColumns: 'repeat(auto-fit, minmax(200px, 1fr))',
            gap: 16, marginBottom: 32,
          }}>
            <StatCard label="Book Progress" value={`${progress.book_percent}%`} />
            <StatCard
              label="Current Chapter"
              value={progress.current_chapter_id || 'Not started'}
            />
            <StatCard label="Language" value={progress.current_language?.toUpperCase() || '—'} />
          </div>
        )}

        {/* Quick action to reader */}
        <button
          onClick={() => onNavigate('reader')}
          style={{
            padding: '12px 24px', borderRadius: 8, border: 'none',
            background: 'var(--accent)', color: '#fff',
            cursor: 'pointer', fontSize: 14, fontWeight: 500,
          }}
        >
          Open Reader →
        </button>
      </div>
    </div>
  )
}

function Section({ title, children }: { title: string; children: React.ReactNode }) {
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

function NavItem({ label, active, onClick }: { label: string; active?: boolean; onClick?: () => void }) {
  return (
    <div
      onClick={onClick}
      style={{
        padding: '6px 8px', borderRadius: 6, fontSize: 13,
        color: active ? 'var(--text-primary)' : 'var(--text-secondary)',
        background: active ? 'var(--bg-secondary)' : 'transparent',
        cursor: 'pointer',
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

export default function App() {
  const [authed, setAuthed] = useState(false)
  const [view, setView] = useState('dashboard')

  useEffect(() => {
    fetch('/api/auth/check')
      .then(r => r.json())
      .then(data => {
        if (data.authenticated) setAuthed(true)
      })
      .catch(() => {})
  }, [])

  if (!authed) {
    return <LoginScreen onLogin={() => setAuthed(true)} />
  }

  if (view === 'reader') {
    return <Reader onNavigate={setView} />
  }

  return <Dashboard onNavigate={setView} />
}
