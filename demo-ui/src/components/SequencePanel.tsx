import type { ReactElement } from 'react'
import type { SequenceSnapshot, ZoneStatus } from '../contracts'

const STATUS_LABEL: Record<ZoneStatus, string> = {
  current: '현재',
  next: '다음',
  completed: '완료',
  pending: '대기',
}

export function SequencePanel({ snapshot }: { snapshot: SequenceSnapshot }): ReactElement {
  return (
    <section aria-label="painting-sequence" className="sequence-panel">
      <header className="sequence-panel__header">
        <h2 className="sequence-panel__title">{snapshot.label}</h2>
        <p className="sequence-panel__level-id">{snapshot.level_id}</p>
      </header>
      <ol className="sequence-panel__list">
        {snapshot.zones.map((zone) => (
          <li
            key={zone.zone_id}
            data-testid={`zone-${zone.zone_id}`}
            data-status={zone.status}
            className="sequence-panel__zone"
          >
            <span className="sequence-panel__zone-id">{zone.zone_id}</span>
            <span className="sequence-panel__zone-label">{zone.label}</span>
            <span className="sequence-panel__zone-badge">{STATUS_LABEL[zone.status]}</span>
          </li>
        ))}
      </ol>
    </section>
  )
}
