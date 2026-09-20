import { useEffect, useState } from 'react'

const labels = ['Needs review', 'Notable switch', 'Stable stance', 'Exclude later']

export default function DetailPanel({ voter, profile, annotation, onSave, backendOnline }) {
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
          <div><span>Political view</span><b>{profile.political_ideology || 'Not available'}</b></div>
          <div><span>Elo ranking</span><b>{profile.elo_ranking || 'Not available'}</b></div>
          <div><span>Debates entered</span><b>{profile.number_of_all_debates}</b></div>
          <div><span>Debates voted</span><b>{profile.number_of_voted_debates}</b></div>
        </div>
      )}
      <form className="annotation-form" onSubmit={submit}>
        <div className="annotation-title">
          <div><span className="step-label">+10</span><h3>Research annotation</h3></div>
          <small>{backendOnline ? 'SQLite' : 'Local'}</small>
        </div>
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
