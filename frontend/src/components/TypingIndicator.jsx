import './TypingIndicator.css'

export default function TypingIndicator() {
  return (
    <div className="typing-wrap">
      <div className="typing-avatar">⚖</div>
      <div className="typing-bubble">
        <span className="typing-label">Consulting the Constitution</span>
        <div className="typing-dots">
          <span /><span /><span />
        </div>
      </div>
    </div>
  )
}