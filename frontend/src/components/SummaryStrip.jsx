const transitions = [
  ['PRO → PRO', 'PRO -> PRO'],
  ['PRO → CON', 'PRO -> CON'],
  ['CON → PRO', 'CON -> PRO'],
  ['CON → CON', 'CON -> CON'],
]

export default function SummaryStrip({ summary }) {
  return (
    <section className="summary-strip panel">
      <div className="summary-primary">
        <span>Observed switch rate</span>
        <strong>{(summary.switch_rate * 100).toFixed(1)}%</strong>
        <small>{summary.switchers} of {summary.valid_transitions} valid transitions</small>
      </div>
      <div className="transition-cards">
        {transitions.map(([label, key]) => (
          <div className={`transition-stat ${key.includes('PRO -> CON') || key.includes('CON -> PRO') ? 'switch' : ''}`} key={key}>
            <span>{label}</span>
            <strong>{summary.transitions[key]}</strong>
          </div>
        ))}
      </div>
      <div className="validity-note">
        <span className="info-icon">i</span>
        Strict rate excludes ties, missing selections, and ambiguous records.
      </div>
    </section>
  )
}
