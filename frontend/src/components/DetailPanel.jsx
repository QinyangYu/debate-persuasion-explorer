import { useEffect, useState } from 'react'

const labels = ['Needs review', 'Notable switch', 'Stable stance', 'Exclude later']

export default function DetailPanel({ voter, profile, annotation, onSave, backendOnline, prediction, predictionLoading, predictionError }) {
  const [label, setLabel] = useState('Needs review')
  const [note, setNote] = useState('')
  const [message, setMessage] = useState('')

  useEffect(() => {
    setLabel(annotation?.label || (voter?.switched ? 'Notable switch' : 'Stable stance'))
    setNote(annotation?.note || '')
    setMessage('')
  }, [voter?.username])

  if (!voter) {
    return (
      <aside className="detail-panel panel empty-detail">
        <div className="empty-orbit"><span>+</span></div>
        <h2>Select a voter</h2>
        <p>Click any outer node to inspect its recorded transition and add a research label.</p>
      </aside>
    )
  }

  async function submit(event) {
    event.preventDefault()
    const result = await onSave(voter.username, label, note.trim())
    setMessage(result.stored === 'database' ? 'Saved to SQLite' : 'Saved in this browser')
  }

  return (
    <aside className="detail-panel panel">
      <div className="panel-heading compact">
        <div>
          <span className="step-label">03</span>
          <h2>Voter detail</h2>
        </div>
        <span className={`change-badge ${voter.switched ? 'yes' : ''}`}>{voter.switched ? 'Changed' : 'Stable'}</span>
      </div>
      <div className="voter-identity">
        <div className="avatar">{voter.username.slice(0, 2).toUpperCase()}</div>
        <div><strong>{voter.username}</strong><span>{voter.vote_time || 'Vote time unavailable'}</span></div>
      </div>
      <div className="transition-flow">
        <div className={voter.before.toLowerCase()}><span>Before</span><strong>{voter.before}</strong></div>
        <b>→</b>
        <div className={voter.after.toLowerCase()}><span>After</span><strong>{voter.after}</strong></div>
      </div>
      {profile && (
        <div className="profile-grid">
          <div title="Political ideology self-reported on the user's Debate.org profile"><span>Self-reported politics</span><b>{profile.political_ideology || 'Not available'}</b></div>
          <div title="Debate.org's Elo-style participant rating"><span>Debate.org Elo</span><b>{profile.elo_ranking || 'Not available'}</b></div>
          <div title="Number of debates in which this user was a debater"><span>As participant</span><b>{profile.number_of_all_debates}</b></div>
          <div title="Number of debates on which this user cast a vote"><span>Votes cast</span><b>{profile.number_of_voted_debates}</b></div>
        </div>
      )}
      <section className="prediction-card">
        <div className="prediction-title"><h3>MLP prediction</h3></div>
        {predictionLoading && <p className="prediction-state">Running inference…</p>}
        {predictionError && <p className="prediction-state error">{predictionError}</p>}
        {prediction && !predictionLoading && (
          <>
            <div className="prediction-primary">
              <span>Predicted result</span>
              <strong>{prediction.predicted_class} · {prediction.predicted_transition}</strong>
            </div>
            <div className="probability-row"><span>Switch probability</span><b>{(prediction.switch_probability * 100).toFixed(1)}%</b></div>
            <div className="probability-track"><i style={{ width: `${prediction.switch_probability * 100}%` }} /></div>
            <div className="probability-classes">
              {Object.entries(prediction.probabilities).map(([name, probability]) => <span key={name}><i>{name.replaceAll(' ', '')}</i><b>{(probability * 100).toFixed(1)}%</b></span>)}
            </div>
            <p className={`prediction-check ${prediction.predicted_switched === voter.switched ? 'correct' : 'incorrect'}`}>
              {prediction.predicted_switched === voter.switched ? '✓ Matches the recorded Switch/Stay outcome' : '✕ Does not match the recorded Switch/Stay outcome'}
            </p>
            <details className="model-inputs">
              <summary>Model inputs used for this prediction</summary>
              <ul>
                <li><b>Initial stance:</b> {prediction.before}</li>
                <li><b>Debate:</b> {prediction.input_summary.category}; {prediction.input_summary.rounds} rounds</li>
                <li><b>Arguments:</b> {prediction.input_summary.pro_argument_words} PRO words and {prediction.input_summary.con_argument_words} CON words, plus separate hashed text vectors</li>
                <li><b>User profile:</b> politics {prediction.input_summary.political_ideology || 'unavailable'}; religion {prediction.input_summary.religious_ideology || 'unavailable'}; Elo {prediction.input_summary.elo_ranking || 'unavailable'}</li>
                <li><b>User activity:</b> {prediction.input_summary.debates_as_participant ?? 'unavailable'} debates; {prediction.input_summary.votes_cast ?? 'unavailable'} votes</li>
              </ul>
              <small>The recorded after stance and post-debate judging fields are excluded.</small>
            </details>
          </>
        )}
        {!prediction && !predictionLoading && !predictionError && <p className="prediction-state">Start the backend and train the model to enable inference.</p>}
      </section>
      <form className="annotation-form" onSubmit={submit}>
        <div className="annotation-title">
          <div><h3>Research annotation</h3></div>
          <small title="Where this optional manual label and note will be stored">{backendOnline ? 'SQLite' : 'Local'}</small>
        </div>
        <p className="annotation-help">Optional manual label and note for later research review.</p>
        <label>
          <span>Label</span>
          <select value={label} onChange={(event) => setLabel(event.target.value)}>
            {labels.map((item) => <option key={item}>{item}</option>)}
          </select>
        </label>
        <label>
          <span>Note</span>
          <textarea value={note} onChange={(event) => setNote(event.target.value)} placeholder="Why is this transition noteworthy?" rows="3" />
        </label>
        <button type="submit">Save annotation</button>
        {message && <p className="save-message">✓ {message}</p>}
      </form>
    </aside>
  )
}
