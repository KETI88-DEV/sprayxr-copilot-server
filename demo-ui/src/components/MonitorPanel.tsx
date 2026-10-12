import type { ReactElement } from 'react'
import type { SessionSequenceView } from '../contracts'
import type { SimSession } from '../simContracts'
import { EMPTY_READOUT, formatCount, formatDurationMs, type TurnMetrics } from '../telemetry'
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
  readonly mode: 'demo' | 'live'
  readonly runState: RunState
  readonly stageLabel: string | null
  readonly elapsedSec: number | null
  readonly metrics: TurnMetrics | null
  readonly sequenceView: SessionSequenceView
  readonly simSession: SimSession | null
  readonly onModeChange: (mode: 'demo' | 'live') => void
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
  mode,
  runState,
  stageLabel,
  elapsedSec,
  metrics,
  sequenceView,
  simSession,
  onModeChange,
  onReconnect,
}: Props): ReactElement {
  const running = runState === 'running'
  const activity = running ? (stageLabel ?? RUN_STATE_LABEL.running) : RUN_STATE_LABEL[runState]
  const deadlineTone = metrics === null ? 'neutral' : metrics.deadlineHit ? 'alert' : 'verify'
  const deadlineValue = metrics === null ? EMPTY_READOUT : metrics.deadlineHit ? '초과' : '준수'

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
        <div className="monitor__modes" role="group" aria-label="실행 모드">
          {(['demo', 'live'] as const).map((option) => (
            <button
              key={option}
              type="button"
              className="monitor__mode"
              data-active={mode === option}
              aria-pressed={mode === option}
              onClick={() => onModeChange(option)}
            >
              {option}
            </button>
          ))}
        </div>
        <p className="monitor__mode-note">
          {mode === 'demo' ? '고정 시나리오로 응답합니다' : '실제 파이프라인과 MCP 도구를 사용합니다'}
        </p>
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
          <Readout label="첫 응답" value={formatDurationMs(metrics?.ttftMs ?? null)} />
          <Readout label="총 소요" value={formatDurationMs(metrics?.totalMs ?? null)} />
          <Readout label="종합 생성" value={formatDurationMs(metrics?.synthesisMs ?? null)} />
          <Readout label="도구 호출" value={formatCount(metrics?.toolCallCount ?? null, '회')} />
          <Readout label="참조 청크" value={formatCount(metrics?.ragChunkCount ?? null, '건')} />
          <Readout label="응답 시한" value={deadlineValue} tone={deadlineTone} />
        </div>
      </section>

      <SequencePanel view={sequenceView} session={simSession} />
    </aside>
  )
}
