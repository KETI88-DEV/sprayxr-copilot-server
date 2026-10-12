import type { ReactElement } from 'react'
import type { SessionSequenceView } from '../contracts'
import type { SimSession } from '../simContracts'
import { EMPTY_READOUT, formatCount, formatDurationMs, type QueryMetrics } from '../telemetry'
import { SequencePanel } from './SequencePanel'

export type RunState = 'idle' | 'running' | 'done' | 'error'

const RUN_STATE_LABEL: Record<RunState, string> = {
  idle: '대기 중',
  running: '실행 중',
  done: '완료',
  error: '오류',
}

type Props = {
  readonly online: boolean
  readonly runState: RunState
  readonly stageLabel: string | null
  readonly elapsedSec: number | null
  readonly metrics: QueryMetrics | null
  readonly sequenceView: SessionSequenceView
  readonly simSession: SimSession | null
  readonly onReconnect: () => void
}

function Readout({
  label,
  value,
  tone,
}: {
  readonly label: string
  readonly value: string
  readonly tone?: 'neutral' | 'alert' | 'verify'
}): ReactElement {
  return (
    <div className="readout" data-tone={tone ?? 'neutral'}>
      <p className="readout__label">{label}</p>
      <p className="readout__value">{value}</p>
    </div>
  )
}

export function MonitorPanel({
  online,
  runState,
  stageLabel,
  elapsedSec,
  metrics,
  sequenceView,
  simSession,
  onReconnect,
}: Props): ReactElement {
  const running = runState === 'running'
  const activity = running ? (stageLabel ?? RUN_STATE_LABEL.running) : RUN_STATE_LABEL[runState]
  const degradedTone = metrics?.degraded == null ? 'neutral' : metrics.degraded ? 'alert' : 'verify'
  const degradedValue = metrics?.degraded == null ? EMPTY_READOUT : metrics.degraded ? '예' : '아니오'
  const retrievalValue = metrics?.retrievalUsed == null ? EMPTY_READOUT : metrics.retrievalUsed ? '사용' : '안 함'

  return (
    <aside className="monitor" aria-label="monitoring">
      <section className="panel">
        <p className="panel__eyebrow">세션</p>
        <div className="monitor__link">
          <span className="monitor__link-state" data-online={online} data-testid="connection-state">
            {online ? '온라인' : '오프라인'}
          </span>
          {!online && (
            <button type="button" className="monitor__retry" onClick={onReconnect}>
              재연결
            </button>
          )}
        </div>
        <p className="monitor__mode-note">질의는 POST /api/v1/copilot/query 로 보냅니다</p>
      </section>

      <section className="panel">
        <p className="panel__eyebrow">실행 상태</p>
        <p className="monitor__activity" data-state={runState} role="status" aria-live="polite">
          {activity}
        </p>
        <p className="monitor__elapsed">
          {running ? `${elapsedSec ?? 0}초 경과` : metrics === null ? '요청 없음' : '마지막 턴 종료'}
        </p>
        <div className="monitor__meter" data-running={running} aria-hidden="true">
          <span className="monitor__meter-run" />
        </div>
      </section>

      <section className="panel">
        <p className="panel__eyebrow">응답 계측</p>
        <div className="monitor__readouts">
          {/* 질의 API 는 스트리밍이 아니라 첫 토큰 시간이 없다. 왕복 − 서버 처리 = 네트워크·대기. */}
          <Readout label="왕복 시간" value={formatDurationMs(metrics?.roundTripMs ?? null)} />
          <Readout label="서버 처리" value={formatDurationMs(metrics?.serverMs ?? null)} />
          <Readout label="인용" value={formatCount(metrics?.citationCount ?? null, '건')} />
          <Readout label="문서 검색" value={retrievalValue} />
          <Readout label="축약 응답" value={degradedValue} tone={degradedTone} />
          <Readout label="상태" value={metrics?.contextStatus ?? EMPTY_READOUT} />
        </div>
      </section>

      <SequencePanel view={sequenceView} session={simSession} />
    </aside>
  )
}
