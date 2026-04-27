import { useState, useRef, useEffect } from 'react'
import './InputBar.css'

export default function InputBar({ onSend, disabled }) {
  const [value, setValue]   = useState('')
  const textareaRef         = useRef(null)

  // Auto-resize textarea
  useEffect(() => {
    const ta = textareaRef.current
    if (!ta) return
    ta.style.height = 'auto'
    ta.style.height = Math.min(ta.scrollHeight, 160) + 'px'
  }, [value])

  const handleSend = () => {
    const trimmed = value.trim()
    if (!trimmed || disabled) return
    onSend(trimmed)
    setValue('')
    // Reset height
    if (textareaRef.current) {
      textareaRef.current.style.height = 'auto'
    }
  }

  const handleKey = (e) => {
    // Send on Enter (not Shift+Enter)
    if (e.key === 'Enter' && !e.shiftKey) {
      e.preventDefault()
      handleSend()
    }
  }

  const charCount = value.length
  const nearLimit = charCount > 800

  return (
    <div className="inputbar-shell">
      <div className="inputbar-wrap">
        <div className={`inputbar ${disabled ? 'inputbar-disabled' : ''}`}>

          {/* Textarea */}
          <textarea
            ref={textareaRef}
            className="inputbar-textarea"
            placeholder="Ask about the Constitution of India…"
            value={value}
            onChange={e => setValue(e.target.value)}
            onKeyDown={handleKey}
            disabled={disabled}
            rows={1}
            maxLength={1000}
          />

          {/* Right side actions */}
          <div className="inputbar-actions">
            {nearLimit && (
              <span className={`char-count ${charCount > 950 ? 'count-warn' : ''}`}>
                {charCount}/1000
              </span>
            )}

            <button
              className={`send-btn ${value.trim() && !disabled ? 'send-active' : ''}`}
              onClick={handleSend}
              disabled={!value.trim() || disabled}
              title="Send (Enter)"
            >
              {disabled ? (
                <span className="send-spinner" />
              ) : (
                <svg width="16" height="16" viewBox="0 0 16 16" fill="none">
                  <path
                    d="M2 8h12M8 2l6 6-6 6"
                    stroke="currentColor"
                    strokeWidth="1.8"
                    strokeLinecap="round"
                    strokeLinejoin="round"
                  />
                </svg>
              )}
            </button>
          </div>
        </div>

        <p className="inputbar-hint">
          Enter to send · Shift+Enter for new line · Answers cite Constitutional articles only
        </p>
      </div>
    </div>
  )
}