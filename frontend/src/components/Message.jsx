import { useState } from 'react'
import Sources from './Sources'
import './Message.css'

// Minimal markdown-like renderer for bold, italics, numbered lists, bullets
function renderContent(text) {
  if (!text) return null

  const lines = text.split('\n')
  const elements = []
  let i = 0

  while (i < lines.length) {
    const line = lines[i]

    // Skip empty lines (add spacing via CSS gap)
    if (!line.trim()) {
      elements.push(<div key={i} className="msg-spacer" />)
      i++
      continue
    }

    // Numbered list item: "1. ..." or "1) ..."
    if (/^\d+[\.\)]\s/.test(line)) {
      const items = []
      while (i < lines.length && /^\d+[\.\)]\s/.test(lines[i])) {
        items.push(
          <li key={i}>{renderInline(lines[i].replace(/^\d+[\.\)]\s/, ''))}</li>
        )
        i++
      }
      elements.push(<ol key={`ol-${i}`} className="msg-ol">{items}</ol>)
      continue
    }

    // Bullet list item: "* ..." or "- ..." or "• ..."
    if (/^[\*\-•]\s/.test(line)) {
      const items = []
      while (i < lines.length && /^[\*\-•]\s/.test(lines[i])) {
        items.push(
          <li key={i}>{renderInline(lines[i].replace(/^[\*\-•]\s/, ''))}</li>
        )
        i++
      }
      elements.push(<ul key={`ul-${i}`} className="msg-ul">{items}</ul>)
      continue
    }

    // Normal paragraph
    elements.push(
      <p key={i} className="msg-para">{renderInline(line)}</p>
    )
    i++
  }

  return elements
}

// Render inline formatting: **bold**, *italic*, `code`
function renderInline(text) {
  // Split on bold (**text**), italic (*text*), code (`text`)
  const parts = text.split(/(\*\*[^*]+\*\*|\*[^*]+\*|`[^`]+`)/g)
  return parts.map((part, i) => {
    if (part.startsWith('**') && part.endsWith('**')) {
      return <strong key={i}>{part.slice(2, -2)}</strong>
    }
    if (part.startsWith('*') && part.endsWith('*')) {
      return <em key={i}>{part.slice(1, -1)}</em>
    }
    if (part.startsWith('`') && part.endsWith('`')) {
      return <code key={i} className="msg-code">{part.slice(1, -1)}</code>
    }
    return part
  })
}

export default function Message({ message, isLast }) {
  const [showSources, setShowSources] = useState(false)
  const isUser = message.role === 'user'
  const hasSources = message.sources && message.sources.length > 0

  return (
    <div
      className={`message-wrap ${isUser ? 'user' : 'ai'} ${isLast ? 'is-last' : ''}`}
    >
      {/* Avatar */}
      <div className={`msg-avatar ${isUser ? 'avatar-user' : 'avatar-ai'}`}>
        {isUser ? 'U' : '⚖'}
      </div>

      {/* Bubble */}
      <div className="msg-bubble-col">
        <div className={`msg-bubble ${isUser ? 'bubble-user' : 'bubble-ai'}`}>
          <div className="msg-content">
            {renderContent(message.content)}
          </div>
        </div>

        {/* Sources toggle */}
        {!isUser && hasSources && (
          <div className="msg-footer">
            <button
              className="sources-toggle"
              onClick={() => setShowSources(v => !v)}
            >
              <span className="sources-icon">§</span>
              {showSources ? 'Hide' : 'Show'} sources
              <span className="sources-count">{message.sources.length}</span>
              <span className="sources-chevron">{showSources ? '▲' : '▼'}</span>
            </button>

            {showSources && <Sources sources={message.sources} />}
          </div>
        )}
      </div>
    </div>
  )
}