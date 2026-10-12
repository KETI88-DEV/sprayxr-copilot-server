import { useCallback, useEffect, useMemo, useRef, useState, type ReactElement } from 'react'
import { ApiError, fetchHealth, fetchSequenceSnapshot, queryCopilot } from './api'
import { ChatPanel, type ChatMessage, type ChatProgress } from './components/ChatPanel'
import { MonitorPanel, type RunState } from './components/MonitorPanel'
import { SimulatorMonitorPanel } from './components/SimulatorMonitorPanel'
import { TitleBar } from './components/TitleBar'
import { ToolLogPanel, type LiveProgress, type ToolLogEntry } from './components/ToolLogPanel'
import { DEFAULT_QUERY_OPTIONS, TASK_PRESETS, sequenceViewFromPayloads } from './contracts'
import type { CopilotTaskType, QueryOptions } from './contracts'
import type { SequenceSnapshot } from './contracts'
import {
  fetchSimSession,
  sendRuntimeEvent,
  sendSessionEnd,
  sendSessionStart,
} from './simMonitorClient'
import type { SimSession } from './simContracts'
import type { SimLifecycle } from './components/SimulatorMonitorPanel'
import { describeStage } from './stageLabels'
import type { QueryMetrics } from './telemetry'
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
  const [queryOptions, setQueryOptions] = useState<QueryOptions>(DEFAULT_QUERY_OPTIONS)
  const [runState, setRunState] = useState<RunState>('idle')
  const [live, setLive] = useState<LiveProgress | null>(null)
  const [metrics, setMetrics] = useState<QueryMetrics | null>(null)
  const [logEntries, setLogEntries] = useState<readonly ToolLogEntry[]>([])
  const [logOpen, setLogOpen] = useState<boolean>(true)
  const [online, setOnline] = useState<boolean>(false)
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
  // 시퀀스 본문은 이미 /api/monitor 로 흐른다(telemetry 라우터가 recorder 를 받는다).
  const sequenceView = useMemo(
    () => sequenceViewFromPayloads(monitorEvents.filter((e) => e.direction === 'input').map((e) => e.payload)),
    [monitorEvents],
  )

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

  // 채팅은 v2.5 질의 API 를 시험하는 자리다. 자유 질문은 knowledge_search, 기능 버튼은 각자의 task_type 으로
  // POST /api/v1/copilot/query 를 부른다. 대상 세션은 패널이 보고 있는 최근 시퀀스의 세션이다.
  const runQuery = useCallback(
    async (taskType: CopilotTaskType, prompt: string, shown: string): Promise<void> => {
      if (busy) {
        return
      }
      const assistantIndex = messages.length + 1
      setMessages((previous) => [...previous, { role: 'user', text: shown }, { role: 'assistant', text: '' }])
      const controller = new AbortController()
      abortControllerRef.current = controller
      setBusy(true)
      setRunState('running')
      setMetrics(null)
      try {
        const context = queryOptions.context.trim()
        const { text, metrics: measured } = await queryCopilot(
          {
            user_id: sequenceView.user_id ?? 'demo-ui',
            ...(sequenceView.session_id === null ? {} : { session_id: sequenceView.session_id }),
            ...(sequenceView.level_id === null ? {} : { scenario_id: sequenceView.level_id }),
            request_id: crypto.randomUUID(),
            task_type: taskType,
            prompt,
            use_rag: queryOptions.use_rag,
            ...(context === '' ? {} : { context }),
          },
          controller.signal,
        )
        setMessages((previous) => updateAssistantMessage(previous, assistantIndex, text))
        setMetrics(measured)
        setRunState('done')
      } catch (error: unknown) {
        const stopped = controller.signal.aborted
        setMessages((previous) =>
          updateAssistantMessage(
            previous,
            assistantIndex,
            stopped ? '중단했습니다.' : `요청에 실패했습니다: ${getStreamErrorMessage(error)}`,
          ),
        )
        setRunState(stopped ? 'idle' : 'error')
      } finally {
        setBusy(false)
      }
    },
    [busy, messages.length, queryOptions, sequenceView.level_id, sequenceView.session_id, sequenceView.user_id],
  )

  const onPreset = useCallback(
    (taskType: CopilotTaskType): void => {
      const preset = TASK_PRESETS.find((candidate) => candidate.task_type === taskType)
      if (preset !== undefined) {
        void runQuery(taskType, preset.prompt, `[${preset.label}] ${preset.prompt}`)
      }
    },
    [runQuery],
  )

  const onSubmit = useCallback(
    (message: string): void => {
      void runQuery(queryOptions.task_type, message, message)
    },
    [queryOptions.task_type, runQuery],
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
          runState={runState}
          stageLabel={stageLabel}
          elapsedSec={busy ? elapsedSec : null}
          metrics={metrics}
          sequenceView={sequenceView}
          simSession={simSession}
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
          onPreset={onPreset}
          queryOptions={queryOptions}
          onQueryOptionsChange={setQueryOptions}
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
