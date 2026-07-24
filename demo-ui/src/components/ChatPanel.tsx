import { useState, type FormEvent, type ReactElement } from 'react'

export type ChatMessage = { readonly role: 'user' | 'assistant'; readonly text: string }

type Props = {
  readonly messages: readonly ChatMessage[]
  readonly busy: boolean
  readonly onSubmit: (message: string) => void
  readonly onStop: () => void
  readonly onReset: () => void
}

export function ChatPanel({ messages, busy, onSubmit, onStop, onReset }: Props): ReactElement {
  const [draft, setDraft] = useState<string>('')

  function handleSubmit(event: FormEvent<HTMLFormElement>): void {
    event.preventDefault()
    const trimmed = draft.trim()
    if (trimmed.length === 0 || busy) {
      return
    }
    onSubmit(trimmed)
    setDraft('')
  }

  return (
    <section className="chat-panel" aria-label="chat">
      <ol className="chat-panel__messages">
        {messages.map((message, index) => (
          <li key={index} className="chat-panel__message" data-role={message.role}>
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
