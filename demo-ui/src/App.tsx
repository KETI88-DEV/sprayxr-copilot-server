import { useCallback, useEffect, useRef, useState, type ReactElement } from 'react'
import { ApiError, fetchHealth, fetchSequenceSnapshot, streamChat } from './api'
import { ChatPanel, type ChatMessage } from './components/ChatPanel'
import { SequencePanel } from './components/SequencePanel'
import { StatusBar } from './components/StatusBar'
import { ToolLogPanel, type ToolLogEntry } from './components/ToolLogPanel'
import type { SequenceSnapshot, SequenceUpdate } from './contracts'
import { applySequenceUpdate } from './sequenceState'
import './styles.css'

const LEVEL_ID = 'SP0035-L01'

const INITIAL_SNAPSHOT: SequenceSnapshot = {
  level_id: LEVEL_ID,
  label: 'SP0035 Level 01',
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
  const [stage, setStage] = useState<string>('idle')
  const [logEntries, setLogEntries] = useState<readonly ToolLogEntry[]>([])
  const [online, setOnline] = useState<boolean>(false)
  const [mode, setMode] = useState<'demo' | 'live'>('demo')
  const [sessionId, setSessionId] = useState<string>(() => crypto.randomUUID())
  const abortControllerRef = useRef<AbortController | null>(null)

  const loadOverview = useCallback(async (): Promise<void> => {
    const [healthResult, snapshotResult] = await Promise.allSettled([
      fetchHealth(),
      fetchSequenceSnapshot(LEVEL_ID),
    ])

    if (healthResult.status === 'fulfilled') {
      setOnline(true)
    } else {
      setOnline(false)
    }

    if (snapshotResult.status === 'fulfilled') {
      setSnapshot(snapshotResult.value)
    } else if (snapshotResult.reason instanceof Error) {
      setLogEntries((previous) => [...previous, { kind: 'error', message: snapshotResult.reason.message }])
    } else {
      setLogEntries((previous) => [...previous, { kind: 'error', message: '스냅샷 불러오기 실패' }])
    }
  }, [])

  useEffect(() => {
    void loadOverview()
  }, [loadOverview])

  const onSubmit = useCallback(
    async (message: string): Promise<void> => {
      const nextAssistantIndex = messages.length + 1
      setMessages((previous) => [
        ...previous,
        { role: 'user', text: message },
        { role: 'assistant', text: '' },
      ])
      setBusy(true)

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
            case 'status':
              setStage(event.data.stage)
              setLogEntries((previous) => [
                ...previous,
                {
                  kind: 'status',
                  stage: event.data.stage,
                  ...(event.data.chunks === undefined ? {} : { chunks: event.data.chunks }),
                  ...(event.data.elapsed_sec === undefined ? {} : { elapsedSec: event.data.elapsed_sec }),
                },
              ])
              break
            case 'message_delta':
              setMessages((previous) => updateAssistantMessage(previous, nextAssistantIndex, event.data.text))
              break
            case 'tool_result':
              setLogEntries((previous) => [
                ...previous,
                {
                  kind: 'tool_result',
                  name: event.data.name,
                  arguments: event.data.arguments,
                  result: event.data.result,
                },
              ])
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
            case 'complete':
              setBusy(false)
              break
            case 'error':
              setLogEntries((previous) => [...previous, { kind: 'error', message: event.data.message }])
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
          setBusy(false)
          return
        }
        const messageText = error instanceof ApiError ? error.message : getStreamErrorMessage(error)
        setLogEntries((previous) => [...previous, { kind: 'error', message: messageText }])
      } finally {
        abortControllerRef.current = null
        setBusy(false)
      }
    },
    [messages.length, mode, sessionId, snapshot.level_id],
  )

  const onStop = useCallback(() => {
    abortControllerRef.current?.abort()
  }, [])

  const onReset = useCallback(() => {
    abortControllerRef.current?.abort()
    setMessages([])
    setLogEntries([])
    setStage('idle')
    setSessionId(crypto.randomUUID())
  }, [])

  const onReconnect = useCallback(() => {
    void loadOverview()
  }, [loadOverview])

  return (
    <div className="app-shell">
      <StatusBar
        online={online}
        levelId={snapshot.level_id}
        mode={mode}
        stage={stage}
        onModeChange={setMode}
        onReconnect={onReconnect}
      />
      <SequencePanel snapshot={snapshot} />
      <ChatPanel messages={messages} busy={busy} streaming={busy} onSubmit={onSubmit} onStop={onStop} onReset={onReset} />
      <ToolLogPanel entries={logEntries} />
    </div>
  )
}
