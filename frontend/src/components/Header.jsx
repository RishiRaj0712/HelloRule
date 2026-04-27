import './Header.css'

export default function Header({ onClear, hasMessages }) {
  return (
    <header className="header">
      <div className="header-left">
        <div className="header-emblem">
          <span className="emblem-icon">⚖</span>
        </div>
        <div className="header-title-group">
          <h1 className="header-title">LawBook India</h1>
          <p className="header-subtitle">Constitution of India · RAG-Powered Q&A</p>
        </div>
      </div>

      <div className="header-right">
        <div className="status-badge">
          <span className="status-dot" />
          <span className="status-text">Gemini 2.5 Flash</span>
        </div>

        {hasMessages && (
          <button className="clear-btn" onClick={onClear} title="Clear conversation">
            <span>Clear</span>
          </button>
        )}
      </div>
    </header>
  )
}