import { useState, type ReactElement } from 'react'

export type ToolLogEntry =
  | {
      readonly kind: 'status'
      readonly stage: string
      readonly chunks?: number
      readonly elapsedSec?: number
    }
  | {
      readonly kind: 'tool_result'
      readonly name: string
      readonly arguments: Record<string, unknown>
      readonly result: Record<string, unknown>
    }
  | { readonly kind: 'error'; readonly message: string }

export function ToolLogPanel({ entries }: { readonly entries: readonly ToolLogEntry[] }): ReactElement {
  const [open, setOpen] = useState<boolean>(true)

  return (
    <section className="tool-log" data-open={open}>
      <button
        type="button"
        className="tool-log__toggle"
        aria-expanded={open}
        onClick={() => setOpen((prev) => !prev)}
      >
        도구 로그 ({entries.length})
      </button>
      {open && (
        <ol className="tool-log__list">
          {entries.map((entry, index) => (
            <li key={index} className="tool-log__entry" data-kind={entry.kind}>
              {entry.kind === 'status' && (
                <span>
                  RAG stage={entry.stage}
                  {entry.chunks === undefined ? '' : ` chunks=${entry.chunks}`}
                  {entry.elapsedSec === undefined ? '' : ` elapsed=${entry.elapsedSec}s`}
                </span>
              )}
              {entry.kind === 'tool_result' && (
                <details>
                  <summary>{entry.name}</summary>
                  <pre>{JSON.stringify(entry.result, null, 2)}</pre>
                </details>
              )}
              {entry.kind === 'error' && <span className="tool-log__error">오류: {entry.message}</span>}
            </li>
          ))}
        </ol>
      )}
    </section>
  )
}
