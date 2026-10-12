import { render, screen, waitFor, within } from '@testing-library/react'
import userEvent from '@testing-library/user-event'
import { afterEach, beforeEach, describe, expect, it, vi } from 'vitest'
import { App } from '../src/App'

// 채팅은 v2.5 질의 API(POST /api/v1/copilot/query)를 시험하는 자리다. /api/chat 은 부르지 않는다.

const SNAPSHOT_JSON = {
  level_id: 'TR-L01',
  label: 'Training Level 01',
  zones: [{ zone_id: '1', label: '1번', status: 'current' }],
  current_zone_id: '1',
  next_zone_ids: [],
  completed_zone_ids: [],
}

function json(body: unknown, status = 200): Response {
  return new Response(JSON.stringify(body), { status, headers: { 'Content-Type': 'application/json' } })
}

const ANSWER = {
  status: 'SUCCESS',
  code: 200,
  message: 'Copilot query completed',
  data: {
    answer: '저도막 구역은 체류 시간을 늘려 재도장한다.',
    citations: ['DOC-doc_0036-C0098'],
    latency_ms: 31187,
    retrieval_used: true,
    degraded: false,
    context_status: 'available',
  },
}

async function ask(question: string): Promise<void> {
  await userEvent.type(screen.getByPlaceholderText('도장 관련 질문을 입력하세요'), question)
  await userEvent.click(screen.getByRole('button', { name: '전송' }))
}

describe('App', () => {
  let fetchSpy: ReturnType<typeof vi.spyOn>
  const queries: Record<string, unknown>[] = []
  let chatCalls = 0

  beforeEach(() => {
    queries.length = 0
    chatCalls = 0
    fetchSpy = vi.spyOn(globalThis, 'fetch').mockImplementation(async (input, init) => {
      const url = typeof input === 'string' ? input : (input as Request).url
      if (url.endsWith('/api/health')) {
        return json({ status: 'ok', servers: 1, timestamp: '2026-01-01T00:00:00Z' })
      }
      if (url.includes('/api/sequence/')) {
        return json(SNAPSHOT_JSON)
      }
      if (url.endsWith('/api/v1/copilot/query')) {
        queries.push(JSON.parse(typeof init?.body === 'string' ? init.body : '{}') as Record<string, unknown>)
        return json(ANSWER)
      }
      if (url.endsWith('/api/chat')) {
        chatCalls += 1
        return json({}, 500)
      }
      // The simulator monitor subscribes on mount.
      if (url.includes('/api/monitor/stream')) {
        return new Response(new ReadableStream<Uint8Array>({ start: (c) => c.close() }), {
          status: 200,
          headers: { 'Content-Type': 'text/event-stream' },
        })
      }
      throw new Error(`unexpected fetch: ${url}`)
    })
  })

  afterEach(() => {
    fetchSpy.mockRestore()
  })

  it('sends a typed question to the v2.5 query API as knowledge_search, not to /api/chat', async () => {
    render(<App />)
    await waitFor(() => expect(screen.getByText('지나간 구역')).toBeInTheDocument())

    await ask('저도막 구역은 어떻게 해야해?')

    await waitFor(() => expect(screen.getByText(/체류 시간을 늘려 재도장한다/)).toBeInTheDocument())
    expect(chatCalls).toBe(0)
    expect(queries).toHaveLength(1)
    expect(queries[0]).toMatchObject({ task_type: 'knowledge_search', prompt: '저도막 구역은 어떻게 해야해?', use_rag: 'Y' })
    expect(typeof queries[0]?.['request_id']).toBe('string')
    // 인용은 본문과 따로 와서 칩으로 보인다.
    expect(screen.getByText('DOC-doc_0036-C0098')).toBeInTheDocument()
  })

  it('fills the measurement panel from the query response', async () => {
    render(<App />)
    await waitFor(() => expect(screen.getByText('지나간 구역')).toBeInTheDocument())

    await ask('저도막 구역은 어떻게 해야해?')

    const monitor = screen.getByRole('complementary', { name: 'monitoring' })
    // 서버가 적은 latency_ms 31187 → 31.2 초. 화면 왕복은 모의 응답이라 거의 0 이다.
    await waitFor(() => expect(within(monitor).getByText('31.2 초')).toBeInTheDocument())
    expect(within(monitor).getByText('1건')).toBeInTheDocument()
    expect(within(monitor).getByText('사용')).toBeInTheDocument()
    expect(within(monitor).getByText('아니오')).toBeInTheDocument()
    expect(within(monitor).getByText('available')).toBeInTheDocument()
  })

  it('carries the chosen task_type, use_rag and context in the request', async () => {
    render(<App />)
    await waitFor(() => expect(screen.getByText('지나간 구역')).toBeInTheDocument())

    await userEvent.selectOptions(screen.getByTestId('option-task-type'), 'pose_recomm')
    await userEvent.click(screen.getByTestId('option-use-rag'))
    await userEvent.type(screen.getByTestId('option-context'), 'SEQ-012')
    await ask('이 자세 어때?')

    await waitFor(() => expect(queries).toHaveLength(1))
    expect(queries[0]).toMatchObject({ task_type: 'pose_recomm', use_rag: 'N', context: 'SEQ-012' })
  })

  it('omits context when it is left empty', async () => {
    render(<App />)
    await waitFor(() => expect(screen.getByText('지나간 구역')).toBeInTheDocument())

    await ask('질문')

    await waitFor(() => expect(queries).toHaveLength(1))
    expect(queries[0]).not.toHaveProperty('context')
  })

  it('shows the server reason when the query fails', async () => {
    fetchSpy.mockImplementation(async (input: RequestInfo | URL) => {
      const url = typeof input === 'string' ? input : (input as Request).url
      if (url.endsWith('/api/v1/copilot/query')) {
        return json({ status: 'ERROR', code: 503, message: 'Copilot pipeline is not configured', data: null }, 503)
      }
      if (url.includes('/api/sequence/')) return json(SNAPSHOT_JSON)
      if (url.endsWith('/api/health')) return json({ status: 'ok', servers: 1, timestamp: '2026-01-01T00:00:00Z' })
      return new Response(new ReadableStream<Uint8Array>({ start: (c) => c.close() }), {
        status: 200,
        headers: { 'Content-Type': 'text/event-stream' },
      })
    })
    render(<App />)
    await waitFor(() => expect(screen.getByText('지나간 구역')).toBeInTheDocument())

    await ask('질문')

    const transcript = screen.getByRole('region', { name: 'chat' })
    await waitFor(() => expect(within(transcript).getByText(/Copilot pipeline is not configured/)).toBeInTheDocument())
  })
})
