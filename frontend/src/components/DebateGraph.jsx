import { useMemo, useRef, useState } from 'react'

const WIDTH = 900
const HEIGHT = 590

function stanceClass(stance) {
  if (stance === 'PRO') return 'pro'
  if (stance === 'CON') return 'con'
  return 'other'
}

export default function DebateGraph({ debate, mode, selectedVoter, onSelectVoter, onlySwitchers, nodeLimit, annotations }) {
  const svgRef = useRef(null)
  const dragRef = useRef(null)
  const [view, setView] = useState({ x: 0, y: 0, w: WIDTH, h: HEIGHT })
  const [hovered, setHovered] = useState(null)

  const voters = useMemo(() => {
    const sorted = [...debate.voters].sort((a, b) => Number(b.switched) - Number(a.switched) || a.username.localeCompare(b.username))
    return sorted.filter((voter) => !onlySwitchers || voter.switched).slice(0, nodeLimit)
  }, [debate, onlySwitchers, nodeLimit])

  const positioned = useMemo(() => voters.map((voter, index) => {
    const count = voters.length
    const ring = index < Math.min(count, 40) ? 0 : 1
    const ringStart = ring === 0 ? 0 : 40
    const ringCount = ring === 0 ? Math.min(count, 40) : count - 40
    const angle = ((index - ringStart) / Math.max(ringCount, 1)) * Math.PI * 2 - Math.PI / 2
    const radiusX = ring === 0 ? 280 : 385
    const radiusY = ring === 0 ? 185 : 250
    return {
      ...voter,
      x: WIDTH / 2 + Math.cos(angle) * radiusX,
      y: HEIGHT / 2 + Math.sin(angle) * radiusY,
    }
  }), [voters])

  function onWheel(event) {
    event.preventDefault()
    const factor = event.deltaY > 0 ? 1.12 : 0.88
    const nextW = Math.min(WIDTH * 1.8, Math.max(WIDTH * 0.45, view.w * factor))
    const nextH = nextW * HEIGHT / WIDTH
    setView({
      x: view.x + (view.w - nextW) / 2,
      y: view.y + (view.h - nextH) / 2,
      w: nextW,
      h: nextH,
    })
  }

  function onPointerDown(event) {
    if (event.target.closest('.voter-node')) return
    svgRef.current.setPointerCapture(event.pointerId)
    dragRef.current = { clientX: event.clientX, clientY: event.clientY, view }
  }

  function onPointerMove(event) {
    if (!dragRef.current) return
    const rect = svgRef.current.getBoundingClientRect()
    const dx = (event.clientX - dragRef.current.clientX) * view.w / rect.width
    const dy = (event.clientY - dragRef.current.clientY) * view.h / rect.height
    setView({ ...dragRef.current.view, x: dragRef.current.view.x - dx, y: dragRef.current.view.y - dy })
  }

  function resetView() {
    setView({ x: 0, y: 0, w: WIDTH, h: HEIGHT })
  }

  const pro = debate.participants.find((p) => p.position === 'PRO')
  const con = debate.participants.find((p) => p.position === 'CON')

  return (
    <div className="graph-stage">
      <svg
        ref={svgRef}
        viewBox={`${view.x} ${view.y} ${view.w} ${view.h}`}
        onWheel={onWheel}
        onPointerDown={onPointerDown}
        onPointerMove={onPointerMove}
        onPointerUp={() => { dragRef.current = null }}
        onPointerLeave={() => { dragRef.current = null }}
        aria-label={`${mode} debate stance network`}
      >
        <defs>
          <radialGradient id="debateCore">
            <stop offset="0%" stopColor="#244e63" />
            <stop offset="100%" stopColor="#102f40" />
          </radialGradient>
          <filter id="nodeShadow" x="-50%" y="-50%" width="200%" height="200%">
            <feDropShadow dx="0" dy="3" stdDeviation="4" floodOpacity="0.18" />
          </filter>
        </defs>
        <g className="edges">
          {positioned.map((voter) => <line key={voter.username} x1={WIDTH / 2} y1={HEIGHT / 2} x2={voter.x} y2={voter.y} />)}
          <line className="participant-edge pro" x1={WIDTH / 2} y1={HEIGHT / 2} x2={205} y2={HEIGHT / 2} />
          <line className="participant-edge con" x1={WIDTH / 2} y1={HEIGHT / 2} x2={695} y2={HEIGHT / 2} />
        </g>

        <g className="participant-node" transform={`translate(205 ${HEIGHT / 2})`}>
          <circle r="47" className="pro" />
          <text className="role" y="-7">PRO</text>
          <text className="name" y="13">{pro?.username?.slice(0, 18)}</text>
        </g>
        <g className="participant-node" transform={`translate(695 ${HEIGHT / 2})`}>
          <circle r="47" className="con" />
          <text className="role" y="-7">CON</text>
          <text className="name" y="13">{con?.username?.slice(0, 18)}</text>
        </g>
        <g className="debate-node" transform={`translate(${WIDTH / 2} ${HEIGHT / 2})`}>
          <circle r="72" fill="url(#debateCore)" filter="url(#nodeShadow)" />
          <text y="-10">SELECTED</text>
          <text className="main" y="12">DEBATE</text>
          <text className="sub" y="34">{voters.length} voters shown</text>
        </g>

        {positioned.map((voter) => {
          const selected = selectedVoter?.username === voter.username
          const annotated = annotations[voter.username]
          return (
            <g
              className={`voter-node ${stanceClass(voter[mode])} ${selected ? 'selected' : ''} ${voter.switched ? 'switched' : ''}`}
              transform={`translate(${voter.x} ${voter.y})`}
              key={voter.username}
              onClick={(event) => { event.stopPropagation(); onSelectVoter(voter) }}
              onMouseEnter={() => setHovered(voter)}
              onMouseLeave={() => setHovered(null)}
              tabIndex="0"
              role="button"
              onKeyDown={(event) => { if (event.key === 'Enter') onSelectVoter(voter) }}
              aria-label={`${voter.username}: ${voter[mode]}`}
            >
              <circle r="10" />
              {voter.switched && <circle className="switch-ring" r="15" />}
              {annotated && <path className="annotation-flag" d="M 8 -13 L 19 -18 L 17 -7 Z" />}
            </g>
          )
        })}

        {hovered && (
          <g className="graph-tooltip" transform={`translate(${Math.min(hovered.x + 18, 720)} ${Math.max(hovered.y - 44, 12)})`}>
            <rect width="166" height="58" rx="8" />
            <text className="tooltip-name" x="12" y="21">{hovered.username.slice(0, 22)}</text>
            <text x="12" y="42">{hovered.before} → {hovered.after}{hovered.switched ? ' · switched' : ''}</text>
          </g>
        )}
      </svg>
      <div className="graph-help">Scroll to zoom · drag to pan · click a voter</div>
      <button className="reset-view" onClick={resetView}>Reset view</button>
    </div>
  )
}
