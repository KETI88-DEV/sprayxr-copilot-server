import { afterEach, describe, expect, it, vi } from 'vitest'
import { sendDummySession, streamMonitor } from '../src/simMonitorClient'
import type { MonitorEvent } from '../src/simContracts'

afterEach(() => {
  vi.unstubAllGlobals()
})

function sseResponse(frames: readonly string[]): Response {
  const body = new ReadableStream<Uint8Array>({
    start(controller) {
      const encoder = new TextEncoder()
      for (const frame of frames) {
        controller.enqueue(encoder.encode(frame))
      }
      controller.close()
    },
  })
  return new Response(body, { status: 200, headers: { 'Content-Type': 'text/event-stream' } })
}

describe('sendDummySession', () => {
  it('issues start, one event, and complete POSTs', async () => {
    const calls: string[] = []
    vi.stubGlobal(
      'fetch',
      vi.fn(async (url: string) => {
        calls.push(url)
        return new Response(JSON.stringify({ ok: true }), { status: 200 })
      }),
    )

    await sendDummySession('s1')

    expect(calls.some((url) => url.endsWith('/api/sim/sessions/start'))).toBe(true)
    expect(calls.some((url) => url.includes('/api/sim/sessions/s1/events'))).toBe(true)
    expect(calls.some((url) => url.endsWith('/api/sim/sessions/s1/complete'))).toBe(true)
  })

  it('raises when the interface rejects a payload', async () => {
    vi.stubGlobal('fetch', vi.fn(async () => new Response('{}', { status: 422 })))

    await expect(sendDummySession('s1')).rejects.toThrow('sim request failed: 422')
  })
})

describe('streamMonitor', () => {
  it('skips the gap marker and yields parsed monitor events', async () => {
    vi.stubGlobal(
      'fetch',
      vi.fn(async () =>
        sseResponse([
          'event: gap\ndata: {"dropped_before":3}\n\n',
          'event: monitor\ndata: {"seq":3,"session_id":"s1","phase":"start","direction":"input","timestamp":"t","payload":{"level_id":"SP0035-L01"}}\n\n',
          'event: monitor\ndata: {"seq":4,"session_id":"s1","phase":"start","direction":"output","timestamp":"t","payload":{"flow_source":"rule"}}\n\n',
        ]),
      ),
    )

    const received: MonitorEvent[] = []
    for await (const event of streamMonitor(0)) {
      received.push(event)
    }

    expect(received.map((event) => event.seq)).toEqual([3, 4])
    expect(received[1]?.direction).toBe('output')
  })

  it('requests the stream from the caller-supplied cursor', async () => {
    const calls: string[] = []
    vi.stubGlobal(
      'fetch',
      vi.fn(async (url: string) => {
        calls.push(url)
        return sseResponse([])
      }),
    )

    for await (const _event of streamMonitor(12)) {
      throw new Error('no events expected')
    }

    expect(calls[0]).toContain('/api/monitor/stream?cursor=12')
  })
})
