import { useEffect, useRef, useState, type FormEvent, type KeyboardEvent, type ReactElement } from 'react'

export type ChatMessage = { readonly role: 'user' | 'assistant'; readonly text: string }

type Props = {
  readonly messages: readonly ChatMessage[]
  readonly busy: boolean
  readonly streaming?: boolean
  readonly onSubmit: (message: string) => void
  readonly onStop: () => void
  readonly onReset: () => void
}

export function ChatPanel({ messages, busy, streaming = false, onSubmit, onStop, onReset }: Props): ReactElement {
  const [draft, setDraft] = useState<string>('')
  const messagesRef = useRef<HTMLOListElement | null>(null)

  const lastMessage = messages[messages.length - 1]
  const lastAssistantTextLength = lastMessage?.role === 'assistant' ? lastMessage.text.length : 0

  useEffect(() => {
    const element = messagesRef.current
    if (element === null) {
      return
    }
    element.scrollTop = Math.max(0, element.scrollHeight - element.clientHeight)
  }, [lastAssistantTextLength, messages.length])

  function submitDraft(): void {
    const trimmed = draft.trim()
    if (trimmed.length === 0 || busy) {
      return
    }
    onSubmit(trimmed)
    setDraft('')
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
    <section className="chat-panel" aria-label="chat">
      <ol ref={messagesRef} className="chat-panel__messages">
        {messages.map((message, index) => (
          <li
            key={index}
            className="chat-panel__message"
            data-role={message.role}
            data-streaming={streaming && index === messages.length - 1 && message.role === 'assistant' ? 'true' : undefined}
          >
            {message.text}
          </li>
        ))}
      </ol>
      <form className="chat-panel__composer" onSubmit={handleSubmit}>
        <textarea
          className="chat-panel__input"
          placeholder="도장 관련 질문을 입력하세요"
          value={draft}
          onChange={(event) => setDraft(event.target.value)}
          onKeyDown={handleKeyDown}
          rows={3}
        />
        <div className="chat-panel__actions">
          <button type="submit" disabled={busy}>
            전송
          </button>
          <button type="button" onClick={onStop} disabled={!busy}>
            중단
          </button>
          <button type="button" onClick={onReset} disabled={busy}>
            초기화
          </button>
        </div>
      </form>
    </section>
  )
}
