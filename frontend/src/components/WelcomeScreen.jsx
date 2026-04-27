import './WelcomeScreen.css'

const SUGGESTIONS = [
  "What are the Fundamental Rights guaranteed by the Constitution?",
  "What does Article 21 say about the right to life?",
  "What are the Directive Principles of State Policy?",
  "How can the Constitution be amended?",
  "What is the Preamble of the Indian Constitution?",
  "What powers does the President have during an emergency?",
]

export default function WelcomeScreen({ onSuggestion }) {
  return (
    <div className="welcome">
      <div className="welcome-hero">
        <div className="welcome-seal">⚖</div>
        <h2 className="welcome-heading">
          Ask the Constitution
        </h2>
        <p className="welcome-desc">
          Ask any question about the Constitution of India. Get accurate,
          cited answers drawn directly from all 395 Articles, 12 Schedules,
          and 106 Amendments.
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
        Answers are grounded in constitutional text only. For IPC, CrPC, or specific Acts, consult a legal expert.
      </p>
    </div>
  )
}