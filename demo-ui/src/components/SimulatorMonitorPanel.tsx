import { useMemo, type ReactElement } from 'react'
import type { MonitorEvent, Phase } from '../simContracts'
import '../simMonitor.css'

const PHASE_LABEL: Record<Phase, string> = {
  start: '세션 시작',
  runtime: '런타임',
  session_end: '세션 끝',
}

const PHASE_ORDER: readonly Phase[] = ['start', 'runtime', 'session_end']

type Props = {
  readonly events: readonly MonitorEvent[]
  readonly onSendDummy: () => void
}

/**
 * Read-only timeline of simulator interface I/O, grouped by session then phase.
 *
 * The component takes events as props instead of subscribing itself, so the
 * rendering stays deterministic under test and the stream lives in one hook.
 */
export function SimulatorMonitorPanel({ events, onSendDummy }: Props): ReactElement {
  const bySession = useMemo(() => {
    const grouped = new Map<string, MonitorEvent[]>()
    for (const event of events) {
      const existing = grouped.get(event.session_id) ?? []
      existing.push(event)
      grouped.set(event.session_id, existing)
    }
    return [...grouped.entries()]
  }, [events])

  return (
    <section className="sim-monitor" aria-label="시뮬레이터 인터페이스 모니터">
      <div className="sim-monitor__bar">
        <h3 className="sim-monitor__title">시뮬레이터 인터페이스 모니터</h3>
        <button type="button" className="sim-monitor__send" onClick={onSendDummy}>
          더미 세션 송신
        </button>
      </div>

      {bySession.length === 0 && <p className="sim-monitor__empty">아직 수신된 I/O가 없습니다.</p>}

      <div className="sim-monitor__scroll">
        {bySession.map(([sessionId, sessionEvents]) => (
          <div className="sim-monitor__session" key={sessionId}>
            <p className="sim-monitor__session-id">세션 {sessionId}</p>
            {PHASE_ORDER.map((phase) => {
              const phaseEvents = sessionEvents.filter((event) => event.phase === phase)
              if (phaseEvents.length === 0) {
                return null
              }
              return (
                <div className="sim-monitor__phase" data-phase={phase} key={phase}>
                  <p className="sim-monitor__phase-label">{PHASE_LABEL[phase]}</p>
                  <ol className="sim-monitor__rows">
                    {phaseEvents.map((event) => (
                      <li className="sim-monitor__row" data-direction={event.direction} key={event.seq}>
                        <span className="sim-monitor__dir">{event.direction}</span>
                        <code className="sim-monitor__payload">{JSON.stringify(event.payload)}</code>
                      </li>
                    ))}
                  </ol>
                </div>
              )
            })}
          </div>
        ))}
      </div>
    </section>
  )
}
