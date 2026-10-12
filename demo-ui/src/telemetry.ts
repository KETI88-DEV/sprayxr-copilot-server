/**
 * Turn telemetry carried by the terminal `complete` event, plus the pure
 * formatters the monitoring panel and the activity log share.
 */

export type TurnMetrics = {
  readonly ttftMs: number | null
  readonly totalMs: number | null
  readonly synthesisMs: number | null
  readonly toolCallCount: number
  readonly ragChunkCount: number
  readonly deadlineHit: boolean
}

/**
 * v2.5 질의 한 번의 계측. 스트리밍이 아니므로 첫 토큰 시간은 없다 — 화면에서 잰 왕복과
 * 서버가 적은 처리 시간(latency_ms)만 있다. 둘의 차이가 네트워크·대기다.
 */
export type QueryMetrics = {
  readonly roundTripMs: number
  readonly serverMs: number | null
  readonly citationCount: number
  readonly retrievalUsed: boolean | null
  readonly degraded: boolean | null
  readonly contextStatus: string | null
}

export const EMPTY_READOUT = '—'

const MS_PER_SECOND = 1000
const MS_PER_MINUTE = 60 * MS_PER_SECOND

function isMeasurable(value: number | null): value is number {
  return value !== null && Number.isFinite(value) && value >= 0
}

/** `4200` -> `4.2 초`, `900` -> `900 ms`, `null` -> `—`. */
export function formatDurationMs(value: number | null): string {
  if (!isMeasurable(value)) {
    return EMPTY_READOUT
  }

  if (value < MS_PER_SECOND) {
    return `${Math.round(value)} ms`
  }

  if (value < MS_PER_MINUTE) {
    return `${(value / MS_PER_SECOND).toFixed(1)} 초`
  }

  const minutes = Math.floor(value / MS_PER_MINUTE)
  const seconds = Math.round((value % MS_PER_MINUTE) / MS_PER_SECOND)
  return seconds === 0 ? `${minutes} 분` : `${minutes} 분 ${seconds} 초`
}

/** `151` -> `151 초`, `null` -> `—`. */
export function formatSeconds(value: number | null): string {
  return isMeasurable(value) ? `${Math.round(value)} 초` : EMPTY_READOUT
}

export function formatCount(value: number | null, unit: string): string {
  return isMeasurable(value) ? `${Math.round(value)}${unit}` : EMPTY_READOUT
}

/** Local wall clock, zero padded, no locale dependency. */
export function formatClock(epochMs: number): string {
  const at = new Date(epochMs)
  const pad = (part: number): string => String(part).padStart(2, '0')
  return `${pad(at.getHours())}:${pad(at.getMinutes())}:${pad(at.getSeconds())}`
}
