import { afterEach, describe, expect, it, vi } from 'vitest'
import {
  fetchSimSession,
  sendRuntimeEvent,
  sendSessionEnd,
  sendSessionStart,
  streamMonitor,
} from '../src/simMonitorClient'
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

describe('3단계 송신기', () => {
  it('세션 시작은 start 엔드포인트에 레벨과 세션 ID를 보낸다', async () => {
    const fetchMock = vi.fn(async () => new Response('{}', { status: 200 }))
    vi.stubGlobal('fetch', fetchMock)

    await sendSessionStart('s-1', 'TR-L01')

    const [url, init] = fetchMock.mock.calls[0] as unknown as [string, RequestInit]
    expect(url).toContain('/api/sim/sessions/start')
    expect(JSON.parse(String(init.body))).toMatchObject({
      session_id: 's-1',
      level_id: 'TR-L01',
    })
  })

  it('런타임은 zone_id를 싣고 이상 여부에 따라 파라미터를 바꾼다', async () => {
    const fetchMock = vi.fn(async () => new Response('{}', { status: 200 }))
    vi.stubGlobal('fetch', fetchMock)

    await sendRuntimeEvent('s-1', 'ev-1', 'A-2', true)

    const [url, init] = fetchMock.mock.calls[0] as unknown as [string, RequestInit]
    expect(url).toContain('/api/sim/sessions/s-1/events')
    const body = JSON.parse(String(init.body)) as Record<string, unknown>
    expect(body).toMatchObject({ event_id: 'ev-1', zone_id: 'A-2' })
    expect(body.telemetry).toMatchObject({ gun_metrics: { standoff_mm: 350 } })
  })

  it('정상 런타임은 대역 안 파라미터를 보낸다', async () => {
    const fetchMock = vi.fn(async () => new Response('{}', { status: 200 }))
    vi.stubGlobal('fetch', fetchMock)

    await sendRuntimeEvent('s-1', 'ev-2', 'A-3', false)

    const [, init] = fetchMock.mock.calls[0] as unknown as [string, RequestInit]
    const body = JSON.parse(String(init.body)) as Record<string, unknown>
    expect(body.telemetry).toMatchObject({ gun_metrics: { standoff_mm: 300 } })
  })

  it('세션 끝은 complete 엔드포인트를 호출한다', async () => {
    const fetchMock = vi.fn(async () => new Response('{}', { status: 200 }))
    vi.stubGlobal('fetch', fetchMock)

    await sendSessionEnd('s-1')

    const [url] = fetchMock.mock.calls[0] as unknown as [string]
    expect(url).toContain('/api/sim/sessions/s-1/complete')
  })

  it('인터페이스가 거부하면 예외를 던진다', async () => {
    vi.stubGlobal('fetch', vi.fn(async () => new Response('{}', { status: 422 })))

    await expect(sendSessionStart('s-1', 'TR-L01')).rejects.toThrow('sim request failed: 422')
  })
})

describe('fetchSimSession', () => {
  it('session이 null이면 null을 돌려준다', async () => {
    vi.stubGlobal(
      'fetch',
      vi.fn(async () => new Response(JSON.stringify({ session: null }), { status: 200 })),
    )

    await expect(fetchSimSession()).resolves.toBeNull()
  })

  it('세션이 있으면 검증된 값을 돌려준다', async () => {
    const session = {
      session_id: 's-1',
      level_id: 'TR-L01',
      phase: 'runtime',
      recommended_flow: ['A-1', 'A-2', 'A-3'],
      current_zone_id: 'A-2',
      completed_zone_ids: ['A-1'],
      last_event_id: 'ev-1',
      anomaly_count: 1,
      unresolved_zone_id: null,
      updated_at: 't',
    }
    vi.stubGlobal(
      'fetch',
      vi.fn(async () => new Response(JSON.stringify({ session }), { status: 200 })),
    )

    const parsed = await fetchSimSession()
    expect(parsed?.current_zone_id).toBe('A-2')
  })

  it('실패 응답이면 예외를 던진다', async () => {
    vi.stubGlobal('fetch', vi.fn(async () => new Response('{}', { status: 500 })))

    await expect(fetchSimSession()).rejects.toThrow('sim session request failed: 500')
  })
})

describe('streamMonitor', () => {
  it('skips the gap marker and yields parsed monitor events', async () => {
    vi.stubGlobal(
      'fetch',
      vi.fn(async () =>
        sseResponse([
          'event: gap\ndata: {"dropped_before":3}\n\n',
          'event: monitor\ndata: {"seq":3,"session_id":"s1","phase":"start","direction":"input","timestamp":"t","payload":{"level_id":"TR-L01"}}\n\n',
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
