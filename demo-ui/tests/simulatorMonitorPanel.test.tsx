import { fireEvent, render, screen, within } from '@testing-library/react'
import { describe, expect, it, vi } from 'vitest'
import { SimulatorMonitorPanel } from '../src/components/SimulatorMonitorPanel'
import type { MonitorEvent } from '../src/simContracts'

const EVENTS: MonitorEvent[] = [
  {
    seq: 1,
    session_id: 's1',
    phase: 'start',
    direction: 'input',
    timestamp: 't1',
    payload: { level_id: 'SP0035-L01' },
  },
  {
    seq: 2,
    session_id: 's1',
    phase: 'start',
    direction: 'output',
    timestamp: 't2',
    payload: { recommended_flow: ['A-1'] },
  },
  {
    seq: 3,
    session_id: 's1',
    phase: 'runtime',
    direction: 'input',
    timestamp: 't3',
    payload: { event_id: 'e1' },
  },
]

describe('SimulatorMonitorPanel', () => {
  it('renders phase-grouped input/output rows', () => {
    render(<SimulatorMonitorPanel events={EVENTS} onSendDummy={() => {}} />)

    expect(screen.getByText('세션 시작')).toBeInTheDocument()
    expect(screen.getByText('런타임')).toBeInTheDocument()
    expect(screen.queryByText('세션 끝')).not.toBeInTheDocument()
    expect(screen.getAllByText('input')).toHaveLength(2)
    expect(screen.getAllByText('output')).toHaveLength(1)
    expect(screen.getByText(/SP0035-L01/)).toBeInTheDocument()
  })

  it('keeps sessions separate and orders phases start -> runtime -> session_end', () => {
    const events: MonitorEvent[] = [
      ...EVENTS,
      {
        seq: 4,
        session_id: 's2',
        phase: 'session_end',
        direction: 'output',
        timestamp: 't4',
        payload: { summary: 'done' },
      },
    ]

    render(<SimulatorMonitorPanel events={events} onSendDummy={() => {}} />)

    expect(screen.getByText('세션 s1')).toBeInTheDocument()
    const secondSession = screen.getByText('세션 s2').parentElement
    expect(secondSession).not.toBeNull()
    expect(within(secondSession as HTMLElement).getByText('세션 끝')).toBeInTheDocument()
  })

  it('shows an empty state before any I/O arrives', () => {
    render(<SimulatorMonitorPanel events={[]} onSendDummy={() => {}} />)

    expect(screen.getByText('아직 수신된 I/O가 없습니다.')).toBeInTheDocument()
  })

  it('fires onSendDummy when the button is clicked', () => {
    const onSend = vi.fn()
    render(<SimulatorMonitorPanel events={[]} onSendDummy={onSend} />)

    fireEvent.click(screen.getByRole('button', { name: '더미 세션 송신' }))

    expect(onSend).toHaveBeenCalledOnce()
  })
})
