import { act, render, screen, waitFor, within } from '@testing-library/react'
import userEvent from '@testing-library/user-event'
import { afterEach, beforeEach, describe, expect, it, vi } from 'vitest'
import { App } from '../src/App'

function sseStream(frames: string[]): Response {
  const body = new ReadableStream<Uint8Array>({
    start(controller) {
      const encoder = new TextEncoder()
      for (const frame of frames) {
        controller.enqueue(encoder.encode(frame))
      }
      controller.close()
    },
  })
  return new Response(body, {
    status: 200,
    headers: { 'Content-Type': 'text/event-stream' },
  })
}

type ControlledStream = {
  readonly response: Response
  readonly push: (frame: string) => Promise<void>
  readonly close: () => Promise<void>
}

/** An SSE response the test drives frame by frame, so mid-stream state is observable. */
function controlledSseStream(): ControlledStream {
  const encoder = new TextEncoder()
  let sink: ReadableStreamDefaultController<Uint8Array> | null = null

  const body = new ReadableStream<Uint8Array>({
    start(controller) {
      sink = controller
    },
  })

  return {
    response: new Response(body, { status: 200, headers: { 'Content-Type': 'text/event-stream' } }),
    push: async (frame: string): Promise<void> => {
      await act(async () => {
        sink?.enqueue(encoder.encode(frame))
        await Promise.resolve()
      })
    },
    close: async (): Promise<void> => {
      await act(async () => {
        sink?.close()
        await Promise.resolve()
      })
    },
  }
}

const SNAPSHOT_JSON = {
  level_id: 'TR-L01',
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

const SESSION_FRAME = 'event: session\ndata: {"event":"session","data":{"session_id":"s-1","mode":"demo"}}\n\n'

function progressFrame(elapsedSec: number): string {
  return `event: status\ndata: {"event":"status","data":{"stage":"generating","elapsed_sec":${elapsedSec}}}\n\n`
}

const SSE_FRAMES = [
  SESSION_FRAME,
  'event: status\ndata: {"event":"status","data":{"stage":"rag","chunks":1}}\n\n',
  'event: status\ndata: {"event":"status","data":{"stage":"deadline_forced_synthesis","elapsed_sec":151}}\n\n',
  'event: sequence_update\ndata: {"event":"sequence_update","data":{"type":"update_sequence","level_id":"TR-L01","current_zone_id":"A-1","next_zone_ids":["A-2","A-3"],"completed_zone_ids":["A-1"]}}\n\n',
  'event: message_delta\ndata: {"event":"message_delta","data":{"text":"다음 도장 구역: A-2"}}\n\n',
  'event: complete\ndata: {"event":"complete","data":{"tool_call_count":0,"rag_chunk_count":1,"ttft_ms":4200,"synthesis_ms":15000,"total_ms":61000,"deadline_hit":true}}\n\n',
]

async function ask(question: string): Promise<void> {
  await userEvent.type(screen.getByPlaceholderText('도장 관련 질문을 입력하세요'), question)
  await userEvent.click(screen.getByRole('button', { name: '전송' }))
}

function activityLog(): HTMLElement {
  return screen.getByRole('region', { name: 'activity-log' })
}

describe('App', () => {
  let fetchSpy: ReturnType<typeof vi.spyOn>
  let nextChatResponse: () => Response
  const chatBodies: string[] = []

  beforeEach(() => {
    chatBodies.length = 0
    nextChatResponse = () => sseStream(SSE_FRAMES)
    fetchSpy = vi.spyOn(globalThis, 'fetch').mockImplementation(async (input, init) => {
      const url = typeof input === 'string' ? input : (input as Request).url
      if (url.endsWith('/api/health')) {
        return new Response(JSON.stringify({ status: 'ok', servers: 1, timestamp: '2026-01-01T00:00:00Z' }), {
          status: 200, headers: { 'Content-Type': 'application/json' },
        })
      }
      if (url.includes('/api/sequence/')) {
        return new Response(JSON.stringify(SNAPSHOT_JSON), { status: 200, headers: { 'Content-Type': 'application/json' } })
      }
      if (url.endsWith('/api/chat')) {
        chatBodies.push(typeof init?.body === 'string' ? init.body : '')
        return nextChatResponse()
      }
      // The simulator monitor subscribes on mount, so this branch keeps the chat tests from throwing.
      if (url.includes('/api/monitor/stream')) {
        return new Response(
          new ReadableStream<Uint8Array>({
            start(controller) {
              controller.close()
            },
          }),
          { status: 200, headers: { 'Content-Type': 'text/event-stream' } },
        )
      }
      throw new Error(`unexpected fetch: ${url}`)
    })
  })

  afterEach(() => {
    fetchSpy.mockRestore()
  })

  it('shows level, sends chat, updates sequence panel', async () => {
    render(<App />)
    await waitFor(() => {
      expect(screen.getByText('Training Level 01')).toBeInTheDocument()
    })

    await ask('다음에 어떤 구역을 칠해야 해?')

    await waitFor(() => {
      expect(screen.getByText(/다음 도장 구역: A-2/)).toBeInTheDocument()
    })
    await waitFor(() => {
      expect(screen.getByTestId('zone-A-2')).toHaveAttribute('data-status', 'next')
    })

    // The deadline stage reaches the operator as a Korean label, with the raw
    // backend vocabulary and the "RAG" prefix gone.
    const log = activityLog()
    expect(within(log).getByText('시간 초과, 응답 정리 중')).toBeInTheDocument()
    expect(within(log).getByText('151초 경과')).toBeInTheDocument()
    expect(within(log).getByText('문서 검색 완료')).toBeInTheDocument()
    expect(screen.queryByText(/^RAG stage=/)).not.toBeInTheDocument()
  })

  it('surfaces completion telemetry in the monitoring panel', async () => {
    render(<App />)
    await waitFor(() => {
      expect(screen.getByText('Training Level 01')).toBeInTheDocument()
    })

    await ask('도막 두께 판정해줘')

    const monitor = screen.getByRole('complementary', { name: 'monitoring' })
    await waitFor(() => {
      expect(within(monitor).getByText('4.2 초')).toBeInTheDocument()
    })
    expect(within(monitor).getByText('1 분 1 초')).toBeInTheDocument()
    expect(within(monitor).getByText('15.0 초')).toBeInTheDocument()
    expect(within(monitor).getByText('0회')).toBeInTheDocument()
    expect(within(monitor).getByText('1건')).toBeInTheDocument()
    expect(within(monitor).getByText('초과')).toBeInTheDocument()
    expect(within(monitor).getByRole('status')).toHaveTextContent('완료')
  })

  it('collapses repeated progress events into one live row instead of appending', async () => {
    const stream = controlledSseStream()
    nextChatResponse = () => stream.response

    render(<App />)
    await waitFor(() => {
      expect(screen.getByText('Training Level 01')).toBeInTheDocument()
    })

    await ask('지금 도막 상태 어때?')

    await stream.push(SESSION_FRAME)
    await stream.push(progressFrame(3))
    await stream.push(progressFrame(6))
    await stream.push(progressFrame(9))

    const log = activityLog()
    await waitFor(() => {
      expect(within(log).getByText('9초 경과')).toBeInTheDocument()
    })
    // Three progress events, one row: the row mutates in place.
    expect(within(log).getAllByRole('listitem')).toHaveLength(1)
    expect(within(log).getAllByText('응답 생성 중')).toHaveLength(1)
    expect(within(log).queryByText('3초 경과')).not.toBeInTheDocument()
    expect(within(log).queryByText('6초 경과')).not.toBeInTheDocument()

    await stream.close()

    // The live row is transient: once the turn ends it leaves nothing behind.
    await waitFor(() => {
      expect(within(log).queryByText('응답 생성 중')).not.toBeInTheDocument()
    })
  })

  it('switches to live mode and sends live chat requests', async () => {
    render(<App />)

    await waitFor(() => {
      expect(screen.getByText('Training Level 01')).toBeInTheDocument()
    })

    await userEvent.click(screen.getByRole('button', { name: /live/i }))
    await ask('다음에 어떤 구역을 칠해야 해?')

    await waitFor(() => {
      expect(chatBodies.at(-1)).toContain('"mode":"live"')
    })
  })
})
