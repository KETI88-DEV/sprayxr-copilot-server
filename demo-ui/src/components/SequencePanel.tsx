import type { ReactElement } from 'react'
import type { SessionSequenceView } from '../contracts'
import type { Phase, SimSession } from '../simContracts'

const STATUS_LABEL = { current: '분사 중', completed: '지나감' } as const

const PHASE_LABEL: Record<Phase, string> = {
  start: '세션 시작',
  runtime: '런타임',
  session_end: '세션 끝',
}

type Props = {
  readonly view: SessionSequenceView
  readonly session: SimSession | null
}

export function SequencePanel({ view, session }: Props): ReactElement {
  const summary =
    view.sequence_count === 0
      ? '아직 받은 분사가 없습니다'
      : `구역 ${view.zones.length}곳에서 분사 ${view.sequence_count}회` +
        (view.unassigned_count > 0 ? `, 미명중 ${view.unassigned_count}회` : '')

  return (
    <section aria-label="painting-sequence" className="sequence">
      <header className="sequence__header">
        <p className="panel__eyebrow">도장 순서</p>
        <h3 className="sequence__title">지나간 구역</h3>
        <dl className="sequence__id">
          <dt className="sequence__id-label">레벨</dt>
          <dd className="sequence__id-value" title="시퀀스의 scenario_id. 구역 번호는 이 레벨 안에서만 유일하다">
            {view.level_id ?? '—'}
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
        {/* live region 으로 만들지 않는다 — 모니터 패널에 이미 하나 있고, 둘이면 서로 경쟁한다. */}
        <p className="sequence__progress" data-testid="sequence-summary">
          {summary}
        </p>
      </header>

      {view.zones.length === 0 ? (
        <p className="sequence__empty">시뮬레이터가 시퀀스를 보내면 지나간 구역이 여기 쌓입니다.</p>
      ) : (
        <ol className="sequence__list">
          {view.zones.map((zone, index) => (
            <li
              key={zone.zone_id}
              data-testid={`zone-${zone.zone_id}`}
              data-status={zone.status}
              className="sequence__zone"
            >
              <span className="sequence__zone-index">{String(index + 1).padStart(2, '0')}</span>
              <span className="sequence__zone-id">{zone.zone_id}번</span>
              <span className="sequence__zone-label">분사 {zone.sequence_count}회</span>
              <span className="sequence__zone-badge">{STATUS_LABEL[zone.status]}</span>
            </li>
          ))}
        </ol>
      )}
    </section>
  )
}
