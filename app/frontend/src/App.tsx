import { useState, useEffect } from 'react'
import Dashboard from './Dashboard'
import Reader from './Reader'

export default function App() {
  const [authed, setAuthed] = useState(false)
  const [view, setView] = useState('dashboard')
  const [readerParams, setReaderParams] = useState<Record<string, string>>({})

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
    return (
      <Reader
        onNavigate={(v) => { setView(v); setReaderParams({}) }}
        initialChapterId={readerParams.chapterId}
        initialLanguage={readerParams.language}
      />
    )
  }

  return (
    <Dashboard
      onNavigate={(v, params) => {
        setView(v)
        setReaderParams(params || {})
      }}
    />
  )
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
