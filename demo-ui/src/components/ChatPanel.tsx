import {
  useCallback,
  useEffect,
  useMemo,
  useRef,
  useState,
  type FormEvent,
  type KeyboardEvent,
  type ReactElement,
} from 'react'
import { TASK_PRESETS } from '../contracts'
import type { CopilotTaskType } from '../contracts'
import { AnswerBody } from './AnswerBody'

export type ChatMessage = { readonly role: 'user' | 'assistant'; readonly text: string }

export type ChatProgress = {
  readonly label: string
  readonly elapsedSec: number | null
}

type Props = {
  readonly messages: readonly ChatMessage[]
  readonly busy: boolean
  readonly streaming?: boolean
  readonly progress?: ChatProgress | null
  readonly onSubmit: (message: string) => void
  readonly onStop: () => void
  readonly onReset: () => void
  /** 기능 4개의 기본 질문을 보낸다. 없으면 버튼 줄을 그리지 않는다. */
  readonly onPreset?: (taskType: CopilotTaskType) => void
}

const ROLE_LABEL: Record<ChatMessage['role'], string> = {
  user: '작업자 질문',
  assistant: '코파일럿 응답',
}

/** Pixels of slack still counted as "the operator is reading the newest turn". */
const BOTTOM_SLACK = 24

function ordinalLabel(ordinal: number): string {
  return String(ordinal).padStart(2, '0')
}

function WaitingBlock({ progress }: { readonly progress: ChatProgress | null }): ReactElement {
  const elapsed = progress?.elapsedSec ?? null

  return (
    <div className="waiting" role="status" aria-live="polite">
      <p className="waiting__stage">{progress?.label ?? '요청 전송 중'}</p>
      <p className="waiting__note">
        {elapsed === null ? '응답 대기 중' : `${elapsed}초 경과`} · 첫 문장까지 30초 이상 걸릴 수 있습니다
      </p>
      <div className="waiting__meter" aria-hidden="true">
        <span className="waiting__meter-run" />
      </div>
    </div>
  )
}

export function ChatPanel({
  messages,
  busy,
  streaming = false,
  progress = null,
  onSubmit,
  onStop,
  onReset,
  onPreset,
}: Props): ReactElement {
  const [draft, setDraft] = useState<string>('')
  const [pinnedToLatest, setPinnedToLatest] = useState<boolean>(true)
  const transcriptRef = useRef<HTMLOListElement | null>(null)
  const pinnedRef = useRef<boolean>(true)

  const lastMessage = messages[messages.length - 1]
  const lastAssistantTextLength = lastMessage?.role === 'assistant' ? lastMessage.text.length : 0

  const turns = useMemo(() => {
    let ordinal = 0
    return messages.map((message) => {
      if (message.role === 'user') {
        ordinal += 1
      }
      return { message, ordinal }
    })
  }, [messages])

  const scrollToLatest = useCallback((): void => {
    const element = transcriptRef.current
    if (element === null) {
      return
    }
    element.scrollTop = Math.max(0, element.scrollHeight - element.clientHeight)
    pinnedRef.current = true
    setPinnedToLatest(true)
  }, [])

  // Only follow the stream while the operator is already at the bottom, so
  // scrolling back through earlier turns is not yanked away mid-answer.
  useEffect(() => {
    const element = transcriptRef.current
    if (element === null || !pinnedRef.current) {
      return
    }
    element.scrollTop = Math.max(0, element.scrollHeight - element.clientHeight)
  }, [lastAssistantTextLength, messages.length])

  function handleScroll(): void {
    const element = transcriptRef.current
    if (element === null) {
      return
    }
    const distanceFromBottom = element.scrollHeight - element.clientHeight - element.scrollTop
    const atBottom = distanceFromBottom <= BOTTOM_SLACK
    pinnedRef.current = atBottom
    setPinnedToLatest(atBottom)
  }

  function submitDraft(): void {
    const trimmed = draft.trim()
    if (trimmed.length === 0 || busy) {
      return
    }
    onSubmit(trimmed)
    setDraft('')
    scrollToLatest()
  }

  function handleSubmit(event: FormEvent<HTMLFormElement>): void {
    event.preventDefault()
    submitDraft()
  }

  function handleKeyDown(event: KeyboardEvent<HTMLTextAreaElement>): void {
    if (event.key !== 'Enter' || event.shiftKey) {
      return
    }

    event.preventDefault()
    submitDraft()
  }

  return (
    <section className="chat" aria-label="chat">
      <div className="chat__stage">
        <ol ref={transcriptRef} className="chat__transcript" onScroll={handleScroll}>
          {turns.map(({ message, ordinal }, index) => {
            const isLast = index === messages.length - 1
            const isStreamingTurn = streaming && isLast && message.role === 'assistant'
            const awaitingFirstToken = isStreamingTurn && message.text.length === 0

            return (
              <li
                key={index}
                className="turn"
                data-role={message.role}
                data-streaming={isStreamingTurn ? 'true' : undefined}
              >
                <p className="turn__meta">
                  <span className="turn__ordinal">문답 {ordinalLabel(ordinal)}</span>
                  <span className="turn__role">{ROLE_LABEL[message.role]}</span>
                </p>
                {message.role === 'user' ? (
                  <p className="turn__question">{message.text}</p>
                ) : awaitingFirstToken ? (
                  <WaitingBlock progress={progress} />
                ) : (
                  <AnswerBody text={message.text} streaming={isStreamingTurn} />
                )}
              </li>
            )
          })}
        </ol>

        {messages.length === 0 && (
          <div className="chat__empty">
            <p className="chat__empty-title">도장 훈련 코파일럿</p>
            <p className="chat__empty-body">
              도막 두께 판정, 결함 원인, 다음 도장 구역처럼 현장에서 바로 확인해야 하는 내용을 물어보세요. 답변은 근거
              문서와 함께 제시됩니다.
            </p>
          </div>
        )}

        {!pinnedToLatest && messages.length > 0 && (
          <button type="button" className="chat__jump" onClick={scrollToLatest}>
            최신 응답 보기
          </button>
        )}
      </div>

      {onPreset !== undefined && (
        <div className="chat__presets" role="group" aria-label="기능별 기본 질문">
          {TASK_PRESETS.map((preset) => (
            <button
              key={preset.task_type}
              type="button"
              className="chat__action"
              data-testid={`preset-${preset.task_type}`}
              title={preset.prompt}
              disabled={busy}
              onClick={() => {
                onPreset(preset.task_type)
                scrollToLatest()
              }}
            >
              {preset.label}
            </button>
          ))}
        </div>
      )}

      <form className="chat__composer" onSubmit={handleSubmit}>
        <textarea
          className="chat__input"
          placeholder="도장 관련 질문을 입력하세요"
          value={draft}
          onChange={(event) => setDraft(event.target.value)}
          onKeyDown={handleKeyDown}
          rows={3}
        />
        <div className="chat__composer-bar">
          <p className="chat__hint">Enter 전송 · Shift+Enter 줄바꿈</p>
          <div className="chat__actions">
            <button type="submit" className="chat__action" data-variant="primary" disabled={busy}>
              전송
            </button>
            <button type="button" className="chat__action" onClick={onStop} disabled={!busy}>
              중단
            </button>
            <button type="button" className="chat__action" onClick={onReset} disabled={busy}>
              초기화
            </button>
          </div>
        </div>
      </form>
    </section>
  )
}
