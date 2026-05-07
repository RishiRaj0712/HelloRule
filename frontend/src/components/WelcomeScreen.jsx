import './WelcomeScreen.css'

const SUGGESTIONS = [
  // Constitution
  "What are the Fundamental Rights guaranteed by the Constitution?",
  "What does Article 21 say about the right to life?",
  // BNS (criminal law)
  "What is the punishment for murder under BNS?",
  "What is the BNS equivalent of IPC Section 420?",
  "What are the new offences added in BNS that weren't in IPC?",
  "What does BNS say about cybercrime and organised crime?",
]

export default function WelcomeScreen({ onSuggestion }) {
  return (
    <div className="welcome">
      <div className="welcome-hero">
        <div className="welcome-seal">⚖</div>
        <h2 className="welcome-heading">
          Ask Indian Law
        </h2>
        <p className="welcome-desc">
          Ask any question about the Constitution of India or the
          Bharatiya Nyaya Sanhita (BNS) 2023. Get accurate, cited answers
          drawn directly from 395 Articles, 12 Schedules, 106 Amendments,
          and 358 BNS Sections.
        </p>
        <div className="welcome-divider">
          <span className="divider-line" />
          <span className="divider-text">Suggested questions</span>
          <span className="divider-line" />
        </div>
      </div>

      <div className="suggestions-grid">
        {SUGGESTIONS.map((s, i) => (
          <button
            key={i}
            className="suggestion-card"
            onClick={() => onSuggestion(s)}
            style={{ animationDelay: `${i * 0.07}s` }}
          >
            <span className="suggestion-arrow">→</span>
            <span className="suggestion-text">{s}</span>
          </button>
        ))}
      </div>

      <p className="welcome-disclaimer">
        Answers are grounded in the Constitution of India and the Bharatiya Nyaya Sanhita (BNS) 2023 only. For BNSS, BSA, or other specific Acts, consult a legal expert.
      </p>
    </div>
  )
}