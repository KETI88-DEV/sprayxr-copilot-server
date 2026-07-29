import { useCallback, useEffect, useMemo, useRef, useState, type ReactElement } from 'react'
import { ApiError, fetchHealth, fetchSequenceSnapshot, streamChat } from './api'
import { ChatPanel, type ChatMessage, type ChatProgress } from './components/ChatPanel'
import { MonitorPanel, type RunState } from './components/MonitorPanel'
import { SimulatorMonitorPanel } from './components/SimulatorMonitorPanel'
import { TitleBar } from './components/TitleBar'
import { ToolLogPanel, type LiveProgress, type ToolLogEntry } from './components/ToolLogPanel'
import type { SequenceSnapshot, SequenceUpdate } from './contracts'
import { applySequenceUpdate } from './sequenceState'
import {
  fetchSimSession,
  sendRuntimeEvent,
  sendSessionEnd,
  sendSessionStart,
} from './simMonitorClient'
import type { SimSession } from './simContracts'
import type { SimLifecycle } from './components/SimulatorMonitorPanel'
import { describeStage } from './stageLabels'
import type { TurnMetrics } from './telemetry'
import { useMonitorStream } from './useMonitorStream'
import './styles.css'

const LEVEL_ID = 'TR-L01'

const INITIAL_SNAPSHOT: SequenceSnapshot = {
  level_id: LEVEL_ID,
  label: 'Training Level 01',
  zones: [
    { zone_id: 'A-1', label: '상부', status: 'completed' },
    { zone_id: 'A-2', label: '중간', status: 'next' },
    { zone_id: 'A-3', label: '하부', status: 'pending' },
  ],
  current_zone_id: 'A-1',
  next_zone_ids: ['A-2', 'A-3'],
  completed_zone_ids: ['A-1'],
}

function updateAssistantMessage(messages: readonly ChatMessage[], index: number, delta: string): ChatMessage[] {
  const next = [...messages]
  const message = next[index]
  if (message === undefined || message.role !== 'assistant') {
    return next
  }

  next[index] = { ...message, text: `${message.text}${delta}` }
  return next
}

function getStreamErrorMessage(error: unknown): string {
  if (error instanceof Error) {
    return error.message
  }
  return '알 수 없는 오류'
}

export function App(): ReactElement {
  const [snapshot, setSnapshot] = useState<SequenceSnapshot>(INITIAL_SNAPSHOT)
  const [messages, setMessages] = useState<readonly ChatMessage[]>([])
  const [busy, setBusy] = useState<boolean>(false)
  const [runState, setRunState] = useState<RunState>('idle')
  const [live, setLive] = useState<LiveProgress | null>(null)
  const [metrics, setMetrics] = useState<TurnMetrics | null>(null)
  const [logEntries, setLogEntries] = useState<readonly ToolLogEntry[]>([])
  const [logOpen, setLogOpen] = useState<boolean>(true)
  const [online, setOnline] = useState<boolean>(false)
  const [mode, setMode] = useState<'demo' | 'live'>('demo')
  const [sessionId, setSessionId] = useState<string>(() => crypto.randomUUID())
  const [turnStartedAt, setTurnStartedAt] = useState<number | null>(null)
  const [elapsedSec, setElapsedSec] = useState<number>(0)
  const [simLifecycle, setSimLifecycle] = useState<SimLifecycle>('idle')
  const [simSessionId, setSimSessionId] = useState<string | null>(null)
  const [simSession, setSimSession] = useState<SimSession | null>(null)
  const simRuntimeCountRef = useRef<number>(0)
  const abortControllerRef = useRef<AbortController | null>(null)
  /** Milestone stages already logged for the current turn, so nothing repeats. */
  const loggedStagesRef = useRef<Set<string>>(new Set())
  const monitorEvents = useMonitorStream()

  // A queued request can wait a minute before its first token; a locally ticked
  // counter is what tells the operator the console is alive between backend
  // progress events.
  useEffect(() => {
    if (!busy || turnStartedAt === null) {
      return
    }

    setElapsedSec(Math.max(0, Math.round((Date.now() - turnStartedAt) / 1000)))
    const timer = window.setInterval(() => {
      setElapsedSec(Math.max(0, Math.round((Date.now() - turnStartedAt) / 1000)))
    }, 1000)

    return () => window.clearInterval(timer)
  }, [busy, turnStartedAt])

  const appendLog = useCallback((entry: ToolLogEntry): void => {
    setLogEntries((previous) => [...previous, entry])
  }, [])

  const loadOverview = useCallback(async (): Promise<void> => {
    const [healthResult, snapshotResult] = await Promise.allSettled([
      fetchHealth(),
      fetchSequenceSnapshot(LEVEL_ID),
    ])

    setOnline(healthResult.status === 'fulfilled')

    if (snapshotResult.status === 'fulfilled') {
      setSnapshot(snapshotResult.value)
      return
    }

    const reason = snapshotResult.reason
    appendLog({
      kind: 'error',
      at: Date.now(),
      message: reason instanceof Error ? reason.message : '스냅샷 불러오기 실패',
      retryable: true,
    })
  }, [appendLog])

  useEffect(() => {
    void loadOverview()
  }, [loadOverview])

  // 서버가 세션 상태의 단일 소유자다. 모니터 이벤트는 재조회 신호로만 쓴다.
  useEffect(() => {
    if (monitorEvents.length === 0) {
      return
    }

    let cancelled = false
    void (async () => {
      const [snapshotResult, sessionResult] = await Promise.allSettled([
        fetchSequenceSnapshot(LEVEL_ID),
        fetchSimSession(),
      ])
      if (cancelled) {
        return
      }
      if (snapshotResult.status === 'fulfilled') {
        setSnapshot(snapshotResult.value)
      }
      if (sessionResult.status === 'fulfilled') {
        setSimSession(sessionResult.value)
      }
      if (snapshotResult.status === 'rejected' || sessionResult.status === 'rejected') {
        appendLog({
          kind: 'error',
          at: Date.now(),
          message: '시뮬레이터 상태 갱신 실패',
          retryable: true,
        })
      }
    })()

    return () => {
      cancelled = true
    }
  }, [monitorEvents.length, appendLog])

  const onSubmit = useCallback(
    async (message: string): Promise<void> => {
      const nextAssistantIndex = messages.length + 1
      setMessages((previous) => [
        ...previous,
        { role: 'user', text: message },
        { role: 'assistant', text: '' },
      ])
      loggedStagesRef.current = new Set()
      setBusy(true)
      setRunState('running')
      setLive(null)
      setMetrics(null)
      setElapsedSec(0)
      setTurnStartedAt(Date.now())

      const controller = new AbortController()
      abortControllerRef.current = controller

      try {
        for await (const event of streamChat(
          { message, session_id: sessionId, mode, level_id: snapshot.level_id },
          controller.signal,
        )) {
          switch (event.event) {
            case 'session':
              setSessionId(event.data.session_id)
              setMode(event.data.mode)
              break
            case 'status': {
              const stage = event.data.stage
              const chunks = event.data.chunks ?? null
              const elapsed = event.data.elapsed_sec ?? null

              // Repeating progress only mutates the single live row.
              setLive({ stage, chunks, elapsedSec: elapsed })

              const description = describeStage(stage)
              if (description.kind === 'milestone' && !loggedStagesRef.current.has(stage)) {
                loggedStagesRef.current.add(stage)
                appendLog({ kind: 'stage', at: Date.now(), stage, chunks, elapsedSec: elapsed })
              }
              break
            }
            case 'message_delta':
              setMessages((previous) => updateAssistantMessage(previous, nextAssistantIndex, event.data.text))
              break
            case 'tool_result':
              appendLog({
                kind: 'tool_result',
                at: Date.now(),
                name: event.data.name,
                arguments: event.data.arguments,
                result: event.data.result,
              })
              break
            case 'sequence_update': {
              const nextUpdate: SequenceUpdate = event.data
              setSnapshot((previous) => ({
                ...previous,
                level_id: nextUpdate.level_id,
                current_zone_id: nextUpdate.current_zone_id,
                next_zone_ids: nextUpdate.next_zone_ids,
                completed_zone_ids: nextUpdate.completed_zone_ids,
                zones: applySequenceUpdate(previous.zones, nextUpdate),
              }))
              break
            }
            case 'complete': {
              const turnMetrics: TurnMetrics = {
                ttftMs: event.data.ttft_ms ?? null,
                totalMs: event.data.total_ms ?? null,
                synthesisMs: event.data.synthesis_ms ?? null,
                toolCallCount: event.data.tool_call_count,
                ragChunkCount: event.data.rag_chunk_count,
                deadlineHit: event.data.deadline_hit ?? false,
              }
              setMetrics(turnMetrics)
              appendLog({ kind: 'complete', at: Date.now(), metrics: turnMetrics })
              setLive(null)
              setRunState('done')
              setBusy(false)
              break
            }
            case 'error':
              appendLog({
                kind: 'error',
                at: Date.now(),
                message: event.data.message,
                retryable: event.data.retryable,
              })
              setLive(null)
              setRunState('error')
              setBusy(false)
              break
            default: {
              const _exhaustive: never = event
              return _exhaustive
            }
          }
        }
      } catch (error: unknown) {
        if (controller.signal.aborted) {
          setRunState('idle')
          return
        }
        appendLog({
          kind: 'error',
          at: Date.now(),
          message: error instanceof ApiError ? error.message : getStreamErrorMessage(error),
          retryable: true,
        })
        setRunState('error')
      } finally {
        abortControllerRef.current = null
        setBusy(false)
        setLive(null)
        // A stream that ends without a terminal event must not leave the
        // console claiming it is still running.
        setRunState((previous) => (previous === 'running' ? 'idle' : previous))
      }
    },
    [appendLog, messages.length, mode, sessionId, snapshot.level_id],
  )

  const onStop = useCallback(() => {
    abortControllerRef.current?.abort()
  }, [])

  const onReset = useCallback(() => {
    abortControllerRef.current?.abort()
    loggedStagesRef.current = new Set()
    setMessages([])
    setLogEntries([])
    setLive(null)
    setMetrics(null)
    setRunState('idle')
    setTurnStartedAt(null)
    setElapsedSec(0)
    setSessionId(crypto.randomUUID())
  }, [])

  const onReconnect = useCallback(() => {
    void loadOverview()
  }, [loadOverview])

  const onToggleLog = useCallback(() => {
    setLogOpen((previous) => !previous)
  }, [])

  const onSimStart = useCallback(() => {
    const id = crypto.randomUUID()
    simRuntimeCountRef.current = 0
    setSimSessionId(id)
    setSimLifecycle('started')
    void sendSessionStart(id, snapshot.level_id)
  }, [snapshot.level_id])

  const onSimRuntime = useCallback(() => {
    if (simSessionId === null) {
      return
    }
    const index = snapshot.zones.findIndex((zone) => zone.zone_id === snapshot.current_zone_id)
    const target = snapshot.zones[index + 1] ?? snapshot.zones[index]
    if (target === undefined) {
      return
    }
    const count = simRuntimeCountRef.current
    simRuntimeCountRef.current = count + 1
    void sendRuntimeEvent(simSessionId, `ev-${count}`, target.zone_id, count === 0)
  }, [simSessionId, snapshot])

  const onSimEnd = useCallback(() => {
    if (simSessionId === null) {
      return
    }
    setSimLifecycle('idle')
    void sendSessionEnd(simSessionId)
  }, [simSessionId])

  const stageLabel = live === null ? null : describeStage(live.stage).label

  const chatProgress = useMemo<ChatProgress | null>(
    () => (busy ? { label: stageLabel ?? '요청 전송 중', elapsedSec } : null),
    [busy, elapsedSec, stageLabel],
  )

  return (
    <div className="app-shell">
      <TitleBar
        logOpen={logOpen}
        busy={busy}
        onToggleLog={onToggleLog}
        onReset={onReset}
        onReconnect={onReconnect}
      />
      <div className="workspace">
        <MonitorPanel
          online={online}
          mode={mode}
          runState={runState}
          stageLabel={stageLabel}
          elapsedSec={busy ? elapsedSec : null}
          metrics={metrics}
          snapshot={snapshot}
          simSession={simSession}
          onModeChange={setMode}
          onReconnect={onReconnect}
        />
        <ChatPanel
          messages={messages}
          busy={busy}
          streaming={busy}
          progress={chatProgress}
          onSubmit={onSubmit}
          onStop={onStop}
          onReset={onReset}
        />
      </div>
      <SimulatorMonitorPanel
        events={monitorEvents}
        lifecycle={simLifecycle}
        onStart={onSimStart}
        onRuntime={onSimRuntime}
        onEnd={onSimEnd}
      />
      <ToolLogPanel entries={logEntries} live={live} open={logOpen} onToggle={onToggleLog} />
    </div>
  )
}
