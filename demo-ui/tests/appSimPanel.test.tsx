import { render, screen, waitFor, within } from '@testing-library/react'
import { afterEach, describe, expect, it, vi } from 'vitest'
import { App } from '../src/App'

// The factory is hoisted above imports, so the fixtures live inside it.
vi.mock('../src/simMonitorClient', () => ({
  streamMonitor: async function* () {
    yield {
      seq: 1,
      session_id: 'qa-1',
      phase: 'start',
      direction: 'input',
      timestamp: 't1',
      payload: { level_id: 'TR-L01' },
    }
    yield {
      seq: 2,
      session_id: 'qa-1',
      phase: 'start',
      direction: 'output',
      timestamp: 't2',
      payload: { flow_source: 'rule' },
    }
  },
  sendSessionStart: vi.fn(async () => undefined),
  sendRuntimeEvent: vi.fn(async () => undefined),
  sendSessionEnd: vi.fn(async () => undefined),
  fetchSimSession: vi.fn(async () => null),
}))

afterEach(() => {
  vi.restoreAllMocks()
})

describe('App simulator monitor panel', () => {
  it('renders streamed simulator I/O next to the existing console', async () => {
    vi.spyOn(globalThis, 'fetch').mockImplementation(async () => new Response('{}', { status: 500 }))

    render(<App />)

    const panel = await screen.findByLabelText('시뮬레이터 인터페이스 모니터')
    expect(await within(panel).findByRole('button', { name: '세션 시작' })).toBeInTheDocument()
    expect(within(panel).getByText('세션 시작', { selector: '.sim-monitor__phase-label' })).toBeInTheDocument()
    expect(await within(panel).findByText('input')).toBeInTheDocument()
    expect(await within(panel).findByText('output')).toBeInTheDocument()
    expect(within(panel).getByText('세션 qa-1')).toBeInTheDocument()
    // The existing console stays intact.
    expect(screen.getByPlaceholderText('도장 관련 질문을 입력하세요')).toBeInTheDocument()
  })
})

describe('App 시뮬레이터 3단계 버튼', () => {
  const SNAPSHOT = {
    level_id: 'TR-L01',
    label: 'Training Level 01',
    zones: [
      { zone_id: 'A-1', label: '상부', status: 'current' },
      { zone_id: 'A-2', label: '중간', status: 'pending' },
      { zone_id: 'A-3', label: '하부', status: 'pending' },
    ],
    current_zone_id: 'A-1',
    next_zone_ids: ['A-2', 'A-3'],
    completed_zone_ids: [],
  }

  function stubFetch(): void {
    vi.spyOn(globalThis, 'fetch').mockImplementation(async (input: RequestInfo | URL) => {
      const url = String(input)
      if (url.includes('/api/sequence/')) {
        return new Response(JSON.stringify(SNAPSHOT), { status: 200 })
      }
      return new Response(JSON.stringify({ status: 'ok', servers: 1, timestamp: 't' }), {
        status: 200,
      })
    })
  }

  it('세션 시작 전에는 런타임과 세션 끝이 비활성이다', async () => {
    stubFetch()
    render(<App />)

    const panel = await screen.findByLabelText('시뮬레이터 인터페이스 모니터')
    expect(within(panel).getByRole('button', { name: '세션 시작' })).toBeEnabled()
    expect(within(panel).getByRole('button', { name: '런타임' })).toBeDisabled()
  })

  it('세션 시작을 누르면 레벨과 함께 송신하고 런타임이 열린다', async () => {
    stubFetch()
    const client = await import('../src/simMonitorClient')
    render(<App />)

    const panel = await screen.findByLabelText('시뮬레이터 인터페이스 모니터')
    within(panel).getByRole('button', { name: '세션 시작' }).click()

    await waitFor(() => {
      expect(client.sendSessionStart).toHaveBeenCalledTimes(1)
    })
    const [, levelId] = vi.mocked(client.sendSessionStart).mock.calls[0] as [string, string]
    expect(levelId).toBe('TR-L01')
    await waitFor(() => {
      expect(within(panel).getByRole('button', { name: '런타임' })).toBeEnabled()
    })
  })

  it('런타임 클릭은 현재 구역의 다음 구역을 보낸다', async () => {
    stubFetch()
    const client = await import('../src/simMonitorClient')
    render(<App />)

    const panel = await screen.findByLabelText('시뮬레이터 인터페이스 모니터')
    within(panel).getByRole('button', { name: '세션 시작' }).click()
    await waitFor(() => {
      expect(within(panel).getByRole('button', { name: '런타임' })).toBeEnabled()
    })
    within(panel).getByRole('button', { name: '런타임' }).click()

    await waitFor(() => {
      expect(client.sendRuntimeEvent).toHaveBeenCalledTimes(1)
    })
    const call = vi.mocked(client.sendRuntimeEvent).mock.calls[0] as [
      string,
      string,
      string,
      boolean,
    ]
    expect(call[2]).toBe('A-2')
    expect(call[3]).toBe(true)
  })

  it('세션 끝을 누르면 대기 상태로 돌아간다', async () => {
    stubFetch()
    const client = await import('../src/simMonitorClient')
    render(<App />)

    const panel = await screen.findByLabelText('시뮬레이터 인터페이스 모니터')
    within(panel).getByRole('button', { name: '세션 시작' }).click()
    await waitFor(() => {
      expect(within(panel).getByRole('button', { name: '세션 끝' })).toBeEnabled()
    })
    within(panel).getByRole('button', { name: '세션 끝' }).click()

    await waitFor(() => {
      expect(client.sendSessionEnd).toHaveBeenCalledTimes(1)
    })
    await waitFor(() => {
      expect(within(panel).getByRole('button', { name: '세션 시작' })).toBeEnabled()
    })
  })

  it('모니터 이벤트가 오면 순서 스냅샷과 세션 상태를 다시 조회한다', async () => {
    stubFetch()
    const client = await import('../src/simMonitorClient')
    render(<App />)

    await waitFor(() => {
      expect(client.fetchSimSession).toHaveBeenCalled()
    })
  })
})
