import { useEffect, useRef, type ReactElement } from 'react'
import { describeStage } from '../stageLabels'
import { formatClock, formatCount, formatDurationMs, type TurnMetrics } from '../telemetry'

export type ToolLogEntry =
  | {
      readonly kind: 'stage'
      readonly at: number
      readonly stage: string
      readonly chunks: number | null
      readonly elapsedSec: number | null
    }
  | {
      readonly kind: 'tool_result'
      readonly at: number
      readonly name: string
      readonly arguments: Record<string, unknown>
      readonly result: Record<string, unknown>
    }
  | { readonly kind: 'error'; readonly at: number; readonly message: string; readonly retryable: boolean }
  | { readonly kind: 'complete'; readonly at: number; readonly metrics: TurnMetrics }

/**
 * In-flight progress. The backend repeats this every few seconds, so it is held
 * as a single value and rendered as one row that mutates in place — it is never
 * appended to `entries`.
 */
export type LiveProgress = {
  readonly stage: string
  readonly chunks: number | null
  readonly elapsedSec: number | null
}

type Props = {
  readonly entries: readonly ToolLogEntry[]
  readonly live: LiveProgress | null
  readonly open: boolean
  readonly onToggle: () => void
}

function stageDetail(chunks: number | null, elapsedSec: number | null): string | null {
  const parts: string[] = []
  if (chunks !== null) {
    parts.push(`청크 ${chunks}건`)
  }
  if (elapsedSec !== null) {
    parts.push(`${elapsedSec}초 경과`)
  }
  return parts.length === 0 ? null : parts.join(' · ')
}

function completeDetail(metrics: TurnMetrics): string {
  return [
    `첫 응답 ${formatDurationMs(metrics.ttftMs)}`,
    `총 ${formatDurationMs(metrics.totalMs)}`,
    `도구 ${formatCount(metrics.toolCallCount, '회')}`,
    `청크 ${formatCount(metrics.ragChunkCount, '건')}`,
  ].join(' · ')
}

function EntryRow({ entry }: { readonly entry: ToolLogEntry }): ReactElement {
  const clock = formatClock(entry.at)

  if (entry.kind === 'stage') {
    const description = describeStage(entry.stage)
    const detail = stageDetail(entry.chunks, entry.elapsedSec)
    return (
      <li className="log__row" data-kind="stage" data-known={description.known}>
        <span className="log__clock">{clock}</span>
        <span className="log__label">{description.label}</span>
        {detail !== null && <span className="log__detail">{detail}</span>}
      </li>
    )
  }

  if (entry.kind === 'tool_result') {
    return (
      <li className="log__row" data-kind="tool_result">
        <span className="log__clock">{clock}</span>
        <details className="log__tool">
          <summary className="log__tool-summary">
            <span className="log__label">도구 호출</span>
            <span className="log__tool-name">{entry.name}</span>
          </summary>
          <div className="log__payload">
            <p className="log__payload-label">요청</p>
            <pre className="log__code">{JSON.stringify(entry.arguments, null, 2)}</pre>
            <p className="log__payload-label">응답</p>
            <pre className="log__code">{JSON.stringify(entry.result, null, 2)}</pre>
          </div>
        </details>
      </li>
    )
  }

  if (entry.kind === 'complete') {
    return (
      <li className="log__row" data-kind="complete">
        <span className="log__clock">{clock}</span>
        <span className="log__label">응답 완료</span>
        <span className="log__detail">{completeDetail(entry.metrics)}</span>
        {entry.metrics.deadlineHit && <span className="log__flag">시한 초과</span>}
      </li>
    )
  }

  return (
    <li className="log__row" data-kind="error">
      <span className="log__clock">{clock}</span>
      <span className="log__label">오류</span>
      <span className="log__detail">{entry.message}</span>
      {entry.retryable && <span className="log__flag" data-tone="retry">재시도 가능</span>}
    </li>
  )
}

export function ToolLogPanel({ entries, live, open, onToggle }: Props): ReactElement {
  const listRef = useRef<HTMLOListElement | null>(null)

  useEffect(() => {
    const element = listRef.current
    if (element === null) {
      return
    }
    element.scrollTop = Math.max(0, element.scrollHeight - element.clientHeight)
  }, [entries.length, live])

  const liveDescription = live === null ? null : describeStage(live.stage)
  const liveDetail = live === null ? null : stageDetail(live.chunks, live.elapsedSec)

  return (
    <section className="log" data-open={open} aria-label="activity-log">
      <button type="button" className="log__toggle" aria-expanded={open} onClick={onToggle}>
        <span className="log__toggle-chevron" aria-hidden="true">
          <svg viewBox="0 0 12 12" width="12" height="12" focusable="false">
            <path d="M2.5 4.5 6 8l3.5-3.5" fill="none" stroke="currentColor" strokeWidth="1.4" strokeLinecap="round" />
          </svg>
        </span>
        <span className="log__toggle-label">활동 로그</span>
        <span className="log__toggle-count">{entries.length}</span>
        {live !== null && <span className="log__toggle-live">진행 중</span>}
      </button>

      {open && (
        <ol className="log__list" ref={listRef}>
          {entries.map((entry, index) => (
            <EntryRow entry={entry} key={index} />
          ))}

          {liveDescription !== null && (
            <li className="log__row" data-kind="live" aria-live="polite">
              <span className="log__clock">진행</span>
              <span className="log__label">{liveDescription.label}</span>
              {liveDetail !== null && <span className="log__detail">{liveDetail}</span>}
              <span className="log__pulse" aria-hidden="true" />
            </li>
          )}
        </ol>
      )}
    </section>
  )
}
