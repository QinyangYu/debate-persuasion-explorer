export default function DebateList({ items, activeId, onSelect }) {
  if (!items.length) return <div className="empty-list">No debates match these filters.</div>
  return (
    <div className="debate-list">
      {items.map((item) => (
        <button key={item.id} className={`debate-row ${item.id === activeId ? 'active' : ''}`} onClick={() => onSelect(item.id)}>
          <span className="debate-row-category">{item.category}</span>
          <strong>{item.title}</strong>
          <span className="debate-row-meta">
            <span>{item.valid_transitions} valid voters</span>
            <span>{item.switchers} switched</span>
          </span>
        </button>
      ))}
    </div>
  )
}
