import { useState, useRef, useEffect } from 'react'
import Header from './Header'
import MessageList from './MessageList'
import InputBar from './InputBar'
import WelcomeScreen from './WelcomeScreen'
import './Chat.css'

const API_BASE = 'http://localhost:8000'

export default function Chat() {
  const [messages, setMessages]   = useState([])
  const [loading, setLoading]     = useState(false)
  const [error, setError]         = useState(null)
  const bottomRef                 = useRef(null)

  // Auto-scroll to latest message
  useEffect(() => {
    bottomRef.current?.scrollIntoView({ behavior: 'smooth' })
  }, [messages, loading])

  // Build history array for the API (last 6 turns)
  const buildHistory = () => {
    return messages.slice(-6).map(m => ({
      role:    m.role,
      content: m.content,
    }))
  }

  const sendMessage = async (query) => {
    if (!query.trim() || loading) return

    setError(null)

    // Add user message immediately
    const userMsg = { id: Date.now(), role: 'user', content: query }
    setMessages(prev => [...prev, userMsg])
    setLoading(true)

    try {
      const res = await fetch(`${API_BASE}/api/chat`, {
        method:  'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({
          query,
          history: buildHistory(),
          top_k: 5,
        }),
      })

      if (!res.ok) {
        throw new Error(`Server error: ${res.status}`)
      }

      const data = await res.json()

      const aiMsg = {
        id:      Date.now() + 1,
        role:    'assistant',
        content: data.answer,
        sources: data.sources || [],
      }

      setMessages(prev => [...prev, aiMsg])

    } catch (err) {
      setError(err.message.includes('fetch')
        ? 'Cannot connect to backend. Is the server running on port 8000?'
        : err.message
      )
      // Remove the user message if request failed
      setMessages(prev => prev.slice(0, -1))
    } finally {
      setLoading(false)
    }
  }

  const clearChat = () => {
    setMessages([])
    setError(null)
  }

  return (
    <div className="chat-shell">
      <Header onClear={clearChat} hasMessages={messages.length > 0} />

      <main className="chat-body">
        {messages.length === 0 && !loading ? (
          <WelcomeScreen onSuggestion={sendMessage} />
        ) : (
          <MessageList
            messages={messages}
            loading={loading}
          />
        )}
        <div ref={bottomRef} />
      </main>

      {error && (
        <div className="error-banner">
          <span className="error-icon">⚠</span>
          {error}
          <button onClick={() => setError(null)} className="error-close">✕</button>
        </div>
      )}

      <InputBar onSend={sendMessage} disabled={loading} />
    </div>
  )
}