import { render, screen, within } from '@testing-library/react'
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
      payload: { level_id: 'SP0035-L01' },
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
  sendDummySession: vi.fn(),
}))

afterEach(() => {
  vi.restoreAllMocks()
})

describe('App simulator monitor panel', () => {
  it('renders streamed simulator I/O next to the existing console', async () => {
    vi.spyOn(globalThis, 'fetch').mockImplementation(async () => new Response('{}', { status: 500 }))

    render(<App />)

    const panel = await screen.findByLabelText('시뮬레이터 인터페이스 모니터')
    expect(await within(panel).findByText('세션 시작')).toBeInTheDocument()
    expect(await within(panel).findByText('input')).toBeInTheDocument()
    expect(await within(panel).findByText('output')).toBeInTheDocument()
    expect(within(panel).getByText('세션 qa-1')).toBeInTheDocument()
    // The existing console stays intact.
    expect(screen.getByPlaceholderText('도장 관련 질문을 입력하세요')).toBeInTheDocument()
  })
})
