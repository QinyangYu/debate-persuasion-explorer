import { useEffect, useMemo, useState } from 'react'
import DebateGraph from './components/DebateGraph'
import DebateList from './components/DebateList'
import DetailPanel from './components/DetailPanel'
import RoundViewer from './components/RoundViewer'
import SummaryStrip from './components/SummaryStrip'

const API = '/api'

function sortIndex(items, sortBy) {
  return [...items].sort((a, b) => {
    if (sortBy === 'switchers') return b.switchers - a.switchers
    if (sortBy === 'switchRate') return b.switch_rate - a.switch_rate
    if (sortBy === 'date') return (b.start_date || '').localeCompare(a.start_date || '')
    return b.valid_transitions - a.valid_transitions
  })
}

export default function App() {
  const [index, setIndex] = useState([])
  const [activeId, setActiveId] = useState(null)
  const [debate, setDebate] = useState(null)
  const [selectedVoter, setSelectedVoter] = useState(null)
  const [mode, setMode] = useState('before')
  const [query, setQuery] = useState('')
  const [category, setCategory] = useState('All categories')
  const [sortBy, setSortBy] = useState('valid')
  const [onlySwitchers, setOnlySwitchers] = useState(false)
  const [nodeLimit, setNodeLimit] = useState(60)
  const [annotations, setAnnotations] = useState({})
  const [backendOnline, setBackendOnline] = useState(false)
  const [loading, setLoading] = useState(true)
  const [error, setError] = useState('')

  useEffect(() => {
    fetch('/data/debates_index.json')
      .then((response) => {
        if (!response.ok) throw new Error('Could not load the debate index.')
        return response.json()
      })
      .then((data) => {
        setIndex(data)
        setActiveId(data[0]?.id || null)
      })
      .catch((reason) => setError(reason.message))
      .finally(() => setLoading(false))
  }, [])

  useEffect(() => {
    fetch(`${API}/health`)
      .then((response) => setBackendOnline(response.ok))
      .catch(() => setBackendOnline(false))
  }, [])

  useEffect(() => {
    if (!activeId) return
    setDebate(null)
    setSelectedVoter(null)
    fetch(`/data/debates/${activeId}.json`)
      .then((response) => {
        if (!response.ok) throw new Error('Could not load the selected debate.')
        return response.json()
      })
      .then(setDebate)
      .catch((reason) => setError(reason.message))

    const local = JSON.parse(localStorage.getItem(`dpe:${activeId}`) || '{}')
    setAnnotations(local)
    fetch(`${API}/annotations?debate_id=${encodeURIComponent(activeId)}`)
      .then((response) => {
        if (!response.ok) throw new Error()
        return response.json()
      })
      .then((rows) => {
        const fromServer = Object.fromEntries(rows.map((row) => [row.username, row]))
        setAnnotations((current) => ({ ...current, ...fromServer }))
        setBackendOnline(true)
      })
      .catch(() => setBackendOnline(false))
  }, [activeId])

  const categories = useMemo(
    () => ['All categories', ...new Set(index.map((item) => item.category).sort())],
    [index],
  )

  const filteredIndex = useMemo(() => {
    const normalized = query.trim().toLowerCase()
    const filtered = index.filter((item) => {
      const inCategory = category === 'All categories' || item.category === category
      const searchable = `${item.title} ${item.participants.map((p) => p.username).join(' ')}`.toLowerCase()
      return inCategory && searchable.includes(normalized)
    })
    return sortIndex(filtered, sortBy)
  }, [index, query, category, sortBy])

  async function saveAnnotation(username, label, note) {
    const next = { ...annotations, [username]: { username, label, note } }
    setAnnotations(next)
    localStorage.setItem(`dpe:${activeId}`, JSON.stringify(next))
    if (!backendOnline) return { stored: 'browser' }
    try {
      const response = await fetch(`${API}/annotations/${encodeURIComponent(activeId)}/${encodeURIComponent(username)}`, {
        method: 'PUT',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ label, note }),
      })
      if (!response.ok) throw new Error()
      const saved = await response.json()
      setAnnotations((current) => ({ ...current, [username]: saved }))
      return { stored: 'database' }
    } catch {
      setBackendOnline(false)
      return { stored: 'browser' }
    }
  }

  if (loading) return <div className="page-state">Loading debate sample…</div>
  if (error && !index.length) return <div className="page-state error">{error}</div>

  return (
    <div className="app-shell">
      <header className="topbar">
        <div className="brand-mark" aria-hidden="true">D</div>
        <div className="brand-copy">
          <div className="eyebrow">DS 7400 · Homework 1</div>
          <h1>Debate Persuasion Explorer</h1>
        </div>
        <div className="research-question">
          <span>Research lens</span>
          How do audience positions differ before and after a debate?
        </div>
        <div className={`status-pill ${backendOnline ? 'online' : ''}`} title="Annotations always save locally; start the backend for SQLite persistence.">
          <span className="status-dot" />
          {backendOnline ? 'SQLite connected' : 'Browser storage'}
        </div>
      </header>

      <main className="dashboard">
        <aside className="debate-sidebar panel">
          <div className="panel-heading">
            <div>
              <span className="step-label">01</span>
              <h2>Choose a debate</h2>
            </div>
            <span className="count-badge" title="Curated browser sample, not the full raw dataset">{filteredIndex.length} shown</span>
          </div>
          <label className="field-label" htmlFor="debate-search">Search title or debater</label>
          <div className="search-wrap">
            <span aria-hidden="true">⌕</span>
            <input id="debate-search" value={query} onChange={(event) => setQuery(event.target.value)} placeholder="e.g. climate, religion…" />
          </div>
          <div className="filter-grid">
            <label>
              <span>Category</span>
              <select value={category} onChange={(event) => setCategory(event.target.value)}>
                {categories.map((name) => <option key={name}>{name}</option>)}
              </select>
            </label>
            <label>
              <span>Sort by</span>
              <select value={sortBy} onChange={(event) => setSortBy(event.target.value)}>
                <option value="valid">Valid voters</option>
                <option value="switchers">Switchers</option>
                <option value="switchRate">Switch rate</option>
                <option value="date">Date</option>
              </select>
            </label>
          </div>
          <p className="sample-note">
            Curated browser sample · the full audit covers 78,376 debates.
          </p>
          <DebateList items={filteredIndex} activeId={activeId} onSelect={setActiveId} />
        </aside>

        <section className="workspace">
          {debate ? (
            <>
              <section className="debate-hero panel">
                <div className="title-block">
                  <div className="eyebrow">{debate.category} · {debate.start_date || 'Date unavailable'}</div>
                  <h2>{debate.title}</h2>
                  <div className="participant-line">
                    {debate.participants.map((participant) => (
                      <span className={participant.position === 'PRO' ? 'pro-text' : 'con-text'} key={`${participant.username}-${participant.position}`}>
                        <b>{participant.position}</b> {participant.username}
                      </span>
                    ))}
                  </div>
                </div>
                <div className="source-stack">
                  <a
                    className="source-link"
                    href={`/data/debates/${debate.id}.json`}
                    target="_blank"
                    rel="noreferrer"
                    title="Open the processed dataset record used by this viewer"
                  >
                    Dataset record ↗
                  </a>
                  <a
                    className="dataset-citation"
                    href="https://doi.org/10.6084/m9.figshare.28326899"
                    target="_blank"
                    rel="noreferrer"
                    title="Figshare page from which the project data files were downloaded"
                  >
                    Downloaded from Figshare · record 28326899
                  </a>
                  <a
                    className="original-citation"
                    href="https://aclanthology.org/P19-1057/"
                    target="_blank"
                    rel="noreferrer"
                    title="Original DDO corpus paper"
                  >
                    Original DDO data: Durmus &amp; Cardie
                  </a>
                  <small>Collected from Debate.org · 2007–2017</small>
                </div>
              </section>

              <SummaryStrip summary={debate.summary} />

              <section className="explorer-grid">
                <div className="graph-card panel">
                  <div className="graph-toolbar">
                    <div>
                      <span className="step-label">02</span>
                      <h2>Explore audience stance</h2>
                    </div>
                    <div className="segmented" aria-label="Stance timing">
                      <button className={mode === 'before' ? 'active' : ''} onClick={() => setMode('before')}>Before</button>
                      <button className={mode === 'after' ? 'active' : ''} onClick={() => setMode('after')}>After</button>
                    </div>
                  </div>
                  <div className="graph-controls">
                    <label className="checkbox-control">
                      <input type="checkbox" checked={onlySwitchers} onChange={(event) => setOnlySwitchers(event.target.checked)} />
                      Only switchers
                    </label>
                    <label className="range-control">
                      Voters shown <b>{nodeLimit}</b>
                      <input type="range" min="12" max="120" step="6" value={nodeLimit} onChange={(event) => setNodeLimit(Number(event.target.value))} />
                    </label>
                    <div className="legend" aria-label="Node color legend">
                      <span><i className="legend-dot pro" />PRO</span>
                      <span><i className="legend-dot con" />CON</span>
                      <span><i className="legend-dot other" />Tie / unknown</span>
                    </div>
                  </div>
                  <div className="graph-caption">
                    <span><b>Selected line</b> this voter voted on the debate</span>
                    <span><b>Colored dashed lines</b> PRO/CON debater role</span>
                    <span><b>Two voter rings</b> spacing only</span>
                    <span><b>Gold outline</b> stance changed</span>
                  </div>
                  <DebateGraph
                    debate={debate}
                    mode={mode}
                    selectedVoter={selectedVoter}
                    onSelectVoter={setSelectedVoter}
                    onlySwitchers={onlySwitchers}
                    nodeLimit={nodeLimit}
                    annotations={annotations}
                  />
                </div>

                <DetailPanel
                  voter={selectedVoter}
                  profile={selectedVoter ? debate.profiles[selectedVoter.username] : null}
                  annotation={selectedVoter ? annotations[selectedVoter.username] : null}
                  onSave={saveAnnotation}
                  backendOnline={backendOnline}
                />
              </section>

              <RoundViewer key={debate.id} rounds={debate.rounds} participants={debate.participants} />
            </>
          ) : <div className="panel page-state">Loading selected debate…</div>}
        </section>
      </main>

      <footer>
        <span>Access: Figshare record 28326899 · Original DDO: Durmus &amp; Cardie</span>
        <span>66,297 valid before/after transitions audited from DDO</span>
      </footer>
    </div>
  )
}
