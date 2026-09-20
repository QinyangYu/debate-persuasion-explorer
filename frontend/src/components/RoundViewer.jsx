import { useState } from 'react'

export default function RoundViewer({ rounds }) {
  const [activeRound, setActiveRound] = useState(1)
  const selected = rounds.find((round) => round.round === activeRound) || rounds[0]

  return (
    <section className="round-viewer panel">
      <div className="round-header">
        <div>
          <span className="step-label">04</span>
          <h2>Read the debate arguments</h2>
          <p>The same text can become the input to a later Transformer model.</p>
        </div>
        <div className="round-tabs">
          {rounds.map((round) => (
            <button className={round.round === selected?.round ? 'active' : ''} key={round.round} onClick={() => setActiveRound(round.round)}>
              Round {round.round}
            </button>
          ))}
        </div>
      </div>
      {selected ? (
        <div className="arguments-grid">
          {['PRO', 'CON'].map((side) => {
            const argument = selected.arguments.find((item) => item.side === side)
            return (
              <article className={`argument ${side.toLowerCase()}`} key={side}>
                <header><span>{side}</span><small>{(argument?.text || '').split(/\s+/).filter(Boolean).length} words</small></header>
                <div className="argument-text">{argument?.text || 'No argument recorded for this side.'}</div>
              </article>
            )
          })}
        </div>
      ) : <p>No rounds were recorded.</p>}
    </section>
  )
}
