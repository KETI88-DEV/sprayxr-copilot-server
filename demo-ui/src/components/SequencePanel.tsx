import type { ReactElement } from 'react'
import type { SequenceSnapshot, ZoneStatus } from '../contracts'
import type { Phase, SimSession } from '../simContracts'

const STATUS_LABEL: Record<ZoneStatus, string> = {
  current: '현재',
  next: '다음',
  completed: '완료',
  pending: '대기',
}

const PHASE_LABEL: Record<Phase, string> = {
  start: '세션 시작',
  runtime: '런타임',
  session_end: '세션 끝',
}

type Props = {
  readonly snapshot: SequenceSnapshot
  readonly session: SimSession | null
}

export function SequencePanel({ snapshot, session }: Props): ReactElement {
  const completed = snapshot.zones.filter((zone) => zone.status === 'completed').length

  return (
    <section aria-label="painting-sequence" className="sequence">
      <header className="sequence__header">
        <p className="panel__eyebrow">도장 순서</p>
        <h3 className="sequence__title">{snapshot.label}</h3>
        <dl className="sequence__id">
          <dt className="sequence__id-label">레벨 ID</dt>
          <dd className="sequence__id-value" title="도장 순서 조회에 사용하는 식별자">
            {snapshot.level_id}
          </dd>
        </dl>
        <p className="sequence__session">
          <span className="sequence__session-phase" data-testid="sim-phase">
            {session === null ? '대기' : PHASE_LABEL[session.phase]}
          </span>
          {session !== null && (
            <span className="sequence__session-id" data-testid="sim-session-id">
              {session.session_id.slice(0, 8)}
            </span>
          )}
        </p>
        {session !== null && session.unresolved_zone_id !== null && (
          <p className="sequence__unresolved" data-testid="sim-unresolved">
            미매핑 구역 {session.unresolved_zone_id}
          </p>
        )}
        <p className="sequence__progress">
          구역 {completed} / {snapshot.zones.length} 완료
        </p>
      </header>

      <ol className="sequence__list">
        {snapshot.zones.map((zone, index) => (
          <li
            key={zone.zone_id}
            data-testid={`zone-${zone.zone_id}`}
            data-status={zone.status}
            className="sequence__zone"
          >
            <span className="sequence__zone-index">{String(index + 1).padStart(2, '0')}</span>
            <span className="sequence__zone-id">{zone.zone_id}</span>
            <span className="sequence__zone-label">{zone.label}</span>
            <span className="sequence__zone-badge">{STATUS_LABEL[zone.status]}</span>
          </li>
        ))}
      </ol>
    </section>
  )
}
