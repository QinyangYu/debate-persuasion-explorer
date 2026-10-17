import { useEffect, useMemo, useRef, useState } from 'react'

const defaultConfig = {
  learning_rate: 0.001,
  epochs: 150,
  hidden_units: 64,
  hidden_layers: 1,
  activation: 'tanh',
  batch_size: 128,
}

function pct(value) {
  return Number.isFinite(value) ? `${(value * 100).toFixed(1)}%` : '—'
}

function MetricCard({ label, value, note }) {
  return (
    <div className="metric-card">
      <span>{label}</span>
      <strong>{value}</strong>
      <small>{note}</small>
    </div>
  )
}

function LossChart({ history }) {
  const geometry = useMemo(() => {
    if (!history.length) return null
    const rows = history.map((item) => ({
      train: item.train.loss,
      validation: item.validation.loss,
    }))
    const values = rows.flatMap((item) => [item.train, item.validation])
    const low = Math.min(...values)
    const high = Math.max(...values)
    const span = Math.max(high - low, 0.001)
    const points = (key) => rows.map((item, index) => {
      const x = 42 + (index / Math.max(rows.length - 1, 1)) * 526
      const y = 224 - ((item[key] - low) / span) * 184
      return `${x.toFixed(1)},${y.toFixed(1)}`
    }).join(' ')
    return { train: points('train'), validation: points('validation'), low, high }
  }, [history])

  if (!geometry) {
    return <div className="chart-empty">Start training to stream loss curves here.</div>
  }
  return (
    <div className="loss-chart-wrap">
      <div className="chart-legend"><span className="train-line">Train loss</span><span className="val-line">Validation loss</span></div>
      <svg className="loss-chart" viewBox="0 0 600 250" role="img" aria-label="Training and validation loss by epoch">
        <line x1="42" y1="224" x2="568" y2="224" />
        <line x1="42" y1="40" x2="42" y2="224" />
        <text x="8" y="44">{geometry.high.toFixed(2)}</text>
        <text x="8" y="226">{geometry.low.toFixed(2)}</text>
        <text x="542" y="244">epoch</text>
        <polyline className="train" points={geometry.train} />
        <polyline className="validation" points={geometry.validation} />
      </svg>
    </div>
  )
}

export default function TrainingLab({ backendOnline, onModelUpdated }) {
  const [config, setConfig] = useState(defaultConfig)
  const [status, setStatus] = useState({ state: 'idle', history: [], epoch: 0 })
  const [metrics, setMetrics] = useState(null)
  const [error, setError] = useState('')
  const completedRun = useRef(null)

  useEffect(() => {
    if (!backendOnline) return
    Promise.all([
      fetch('/api/model').then((response) => response.json()),
      fetch('/api/training/status').then((response) => response.json()),
    ]).then(([model, training]) => {
      if (model.ready) setMetrics(model.metrics)
      setStatus(training)
    }).catch(() => setError('Could not read model status.'))
  }, [backendOnline])

  useEffect(() => {
    if (!backendOnline || status.state !== 'running') return undefined
    const timer = window.setInterval(() => {
      fetch('/api/training/status')
        .then((response) => response.json())
        .then((next) => {
          setStatus(next)
          if (next.state === 'completed') {
            setMetrics(next.metrics)
            if (next.promoted && completedRun.current !== next.run_id) {
              completedRun.current = next.run_id
              onModelUpdated()
            }
          }
          if (next.state === 'failed') setError(next.error || 'Training failed.')
        })
        .catch(() => setError('Lost connection to the training monitor.'))
    }, 700)
    return () => window.clearInterval(timer)
  }, [backendOnline, status.state, onModelUpdated])

  function update(name, value) {
    setConfig((current) => ({ ...current, [name]: value }))
  }

  async function startTraining(event) {
    event.preventDefault()
    setError('')
    const response = await fetch('/api/training/start', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify(config),
    })
    const body = await response.json()
    if (!response.ok) {
      setError(body.detail || 'Training could not start.')
      return
    }
    setStatus(body)
  }

  const history = status.history?.length ? status.history : metrics?.history || []
  const result = status.state === 'completed' ? status.metrics : metrics
  const progress = status.state === 'running' ? (status.epoch / status.epochs) * 100 : status.state === 'completed' ? 100 : 0

  return (
    <section className="training-lab panel">
      <div className="training-heading">
        <div>
          <span className="step-label">04</span>
          <div><h2>Neural network training lab</h2><p>Train the PyTorch MLP with interactive training and live monitor.</p></div>
        </div>
      </div>
      <div className="dataset-split-banner">
        <b>Dataset boundary</b>
        <span><strong>{result?.split?.train_debates?.toLocaleString() || '—'}</strong> background train debates</span>
        <span><strong>{result?.split?.validation_debates?.toLocaleString() || '—'}</strong> background validation debates</span>
        <span className="test-only"><strong>51</strong> browser debates · test only</span>
      </div>
      <div className="training-layout">
        <form className="hyperparameter-form" onSubmit={startTraining}>
          <h3>Hyperparameters</h3>
          <label><span>Learning rate</span><input type="number" min="0.00001" max="0.05" step="0.00001" value={config.learning_rate} onChange={(event) => update('learning_rate', Number(event.target.value))} /></label>
          <label><span>Epochs</span><input type="number" min="5" max="200" value={config.epochs} onChange={(event) => update('epochs', Number(event.target.value))} /></label>
          <label><span>Hidden units</span><select value={config.hidden_units} onChange={(event) => update('hidden_units', Number(event.target.value))}>{[16, 32, 64, 128, 256].map((value) => <option key={value}>{value}</option>)}</select></label>
          <label><span>Hidden layers</span><select value={config.hidden_layers} onChange={(event) => update('hidden_layers', Number(event.target.value))}>{[1, 2, 3, 4].map((value) => <option key={value}>{value}</option>)}</select></label>
          <label><span>Activation</span><select value={config.activation} onChange={(event) => update('activation', event.target.value)}><option value="relu">ReLU</option><option value="tanh">Tanh</option><option value="gelu">GELU</option></select></label>
          <label><span>Batch size</span><select value={config.batch_size} onChange={(event) => update('batch_size', Number(event.target.value))}>{[32, 64, 128, 256].map((value) => <option key={value}>{value}</option>)}</select></label>
          <button disabled={!backendOnline || status.state === 'running'}>{status.state === 'running' ? 'Training…' : 'Start a new training run'}</button>
          <p>Training and model selection use only non-showcase background debates. The 51 browser debates never enter fitting or the validation quality gate.</p>
        </form>
        <div className="training-monitor">
          <div className="monitor-status">
            <div><span className={`run-light ${status.state}`} /> <b>{status.state === 'running' ? `Epoch ${status.epoch} of ${status.epochs}` : status.state === 'completed' ? 'Training complete' : 'Ready'}</b></div>
            <span>{status.config ? `${status.config.hidden_layers} × ${status.config.hidden_units} ${status.config.activation.toUpperCase()}` : 'Default trained model'}</span>
          </div>
          <div className="progress-track"><i style={{ width: `${progress}%` }} /></div>
          <LossChart history={history} />
          {status.state === 'completed' && status.message && <p className={`promotion-message ${status.promoted ? 'promoted' : 'preserved'}`}>{status.message}</p>}
          {error && <p className="training-error">{error}</p>}
        </div>
        <div className="metric-stack">
          <h3>Test performance · 51 debates</h3>
          <MetricCard label="Test accuracy" value={pct(result?.test?.accuracy)} note={`${result?.test?.examples || 0} voter–debate pairs`} />
          <MetricCard label="Balanced accuracy" value={pct(result?.test?.balanced_accuracy)} note="Average recall for Stay and Switch" />
          <MetricCard label="Switch F1" value={pct(result?.test?.switch_f1)} note="Precision/recall for the rare class" />
          <MetricCard label="Switch PR-AUC" value={pct(result?.test?.switch_pr_auc)} note="Threshold-independent ranking quality" />
          <MetricCard label="Logistic baseline" value={pct(result?.baselines?.logistic_regression?.test?.balanced_accuracy)} note="Shallow benchmark balanced accuracy" />
          {result && <p className="benchmark-result">MLP balanced error: {pct(1 - result.test.balanced_accuracy)} · logistic baseline: {pct(1 - result.baselines.logistic_regression.test.balanced_accuracy)}</p>}
        </div>
      </div>
    </section>
  )
}

