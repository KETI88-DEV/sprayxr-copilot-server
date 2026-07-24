import { render, screen, waitFor } from '@testing-library/react'
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

const SNAPSHOT_JSON = {
  level_id: 'SP0035-L01',
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

const SSE_FRAMES = [
  'event: session\ndata: {"event":"session","data":{"session_id":"s-1","mode":"demo"}}\n\n',
  'event: status\ndata: {"event":"status","data":{"stage":"rag","chunks":1}}\n\n',
  'event: sequence_update\ndata: {"event":"sequence_update","data":{"type":"update_sequence","level_id":"SP0035-L01","current_zone_id":"A-1","next_zone_ids":["A-2","A-3"],"completed_zone_ids":["A-1"]}}\n\n',
  'event: message_delta\ndata: {"event":"message_delta","data":{"text":"다음 도장 구역: A-2"}}\n\n',
  'event: complete\ndata: {"event":"complete","data":{"tool_call_count":0,"rag_chunk_count":1}}\n\n',
]

describe('App', () => {
  let fetchSpy: ReturnType<typeof vi.spyOn>

  beforeEach(() => {
    fetchSpy = vi.spyOn(globalThis, 'fetch').mockImplementation(async (input) => {
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
        return sseStream(SSE_FRAMES)
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
      expect(screen.getByText('SP0035 Level 01')).toBeInTheDocument()
    })

    const composer = screen.getByPlaceholderText('도장 관련 질문을 입력하세요')
    await userEvent.type(composer, '다음에 어떤 구역을 칠해야 해?')
    await userEvent.click(screen.getByRole('button', { name: '전송' }))

    await waitFor(() => {
      expect(screen.getByText(/다음 도장 구역: A-2/)).toBeInTheDocument()
    })
    await waitFor(() => {
      expect(screen.getByTestId('zone-A-2')).toHaveAttribute('data-status', 'next')
    })
  })
})
