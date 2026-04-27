import './Sources.css'

export default function Sources({ sources }) {
  if (!sources || sources.length === 0) return null

  // Deduplicate by article number
  const seen = new Set()
  const unique = sources.filter(s => {
    const key = s.article || s.title
    if (seen.has(key)) return false
    seen.add(key)
    return true
  })

  return (
    <div className="sources-panel">
      <div className="sources-header">
        <span className="sources-label">Constitutional Sources</span>
      </div>
      <div className="sources-list">
        {unique.map((src, i) => (
          <SourceChip key={i} source={src} />
        ))}
      </div>
    </div>
  )
}

function SourceChip({ source }) {
  const isActive  = source.status === 'active'
  const typeLabel = source.type === 'amendment' ? 'AMDT'
                  : source.type === 'schedule'  ? 'SCH'
                  : source.type === 'preamble'  ? 'PRE'
                  : 'ART'

  const score = source.score ? Math.round(source.score * 100) : null

  return (
    <div className={`source-chip ${!isActive ? 'chip-inactive' : ''}`}>
      <span className="chip-type">{typeLabel}</span>

      <div className="chip-body">
        <span className="chip-article">
          {source.type === 'article' && source.article
            ? `Article ${source.article}`
            : source.title?.replace('Constitution', 'Const.') || 'Unknown'}
        </span>
        {source.part && (
          <span className="chip-part">Part {source.part}</span>
        )}
      </div>

      {!isActive && (
        <span className="chip-status-badge">Repealed</span>
      )}

      {score && (
        <span className="chip-score">{score}%</span>
      )}
    </div>
  )
}