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
    render(
      <SimulatorMonitorPanel
        events={EVENTS}
        lifecycle="idle"
        onStart={() => {}}
        onRuntime={() => {}}
        onEnd={() => {}}
      />,
    )

    // 단계 라벨은 타임라인 안에서만 확인한다. 상단 바에 같은 이름의 버튼이 있다.
    const timeline = document.querySelector('.sim-monitor__scroll')
    expect(timeline).not.toBeNull()
    const inTimeline = within(timeline as HTMLElement)
    expect(inTimeline.getByText('세션 시작')).toBeInTheDocument()
    expect(inTimeline.getByText('런타임')).toBeInTheDocument()
    expect(inTimeline.queryByText('세션 끝')).not.toBeInTheDocument()
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

    render(
      <SimulatorMonitorPanel
        events={events}
        lifecycle="idle"
        onStart={() => {}}
        onRuntime={() => {}}
        onEnd={() => {}}
      />,
    )

    expect(screen.getByText('세션 s1')).toBeInTheDocument()
    const secondSession = screen.getByText('세션 s2').parentElement
    expect(secondSession).not.toBeNull()
    expect(within(secondSession as HTMLElement).getByText('세션 끝')).toBeInTheDocument()
  })

  it('shows an empty state before any I/O arrives', () => {
    render(
      <SimulatorMonitorPanel
        events={[]}
        lifecycle="idle"
        onStart={() => {}}
        onRuntime={() => {}}
        onEnd={() => {}}
      />,
    )

    expect(screen.getByText('아직 수신된 I/O가 없습니다.')).toBeInTheDocument()
  })

  it('fires onStart when the start button is clicked', () => {
    const onSend = vi.fn()
    render(
      <SimulatorMonitorPanel
        events={[]}
        lifecycle="idle"
        onStart={onSend}
        onRuntime={() => {}}
        onEnd={() => {}}
      />,
    )

    fireEvent.click(screen.getByRole('button', { name: '세션 시작' }))

    expect(onSend).toHaveBeenCalledOnce()
  })
})

describe('SimulatorMonitorPanel 3단계 버튼', () => {
  function renderPanel(lifecycle: 'idle' | 'started') {
    const onStart = vi.fn()
    const onRuntime = vi.fn()
    const onEnd = vi.fn()
    render(
      <SimulatorMonitorPanel
        events={[]}
        lifecycle={lifecycle}
        onStart={onStart}
        onRuntime={onRuntime}
        onEnd={onEnd}
      />,
    )
    return { onStart, onRuntime, onEnd }
  }

  it('대기 상태에서는 런타임과 세션 끝이 비활성이다', () => {
    renderPanel('idle')
    expect(screen.getByRole('button', { name: '세션 시작' })).toBeEnabled()
    expect(screen.getByRole('button', { name: '런타임' })).toBeDisabled()
    expect(screen.getByRole('button', { name: '세션 끝' })).toBeDisabled()
  })

  it('시작 후에는 세 버튼의 활성 상태가 뒤바뀐다', () => {
    renderPanel('started')
    expect(screen.getByRole('button', { name: '세션 시작' })).toBeDisabled()
    expect(screen.getByRole('button', { name: '런타임' })).toBeEnabled()
    expect(screen.getByRole('button', { name: '세션 끝' })).toBeEnabled()
  })

  it('각 버튼이 대응하는 핸들러를 부른다', () => {
    const { onStart, onRuntime, onEnd } = renderPanel('started')
    fireEvent.click(screen.getByRole('button', { name: '런타임' }))
    fireEvent.click(screen.getByRole('button', { name: '세션 끝' }))
    expect(onRuntime).toHaveBeenCalledTimes(1)
    expect(onEnd).toHaveBeenCalledTimes(1)
    expect(onStart).not.toHaveBeenCalled()
  })

  it('대기 상태에서 세션 시작만 동작한다', () => {
    const { onStart, onRuntime } = renderPanel('idle')
    fireEvent.click(screen.getByRole('button', { name: '세션 시작' }))
    fireEvent.click(screen.getByRole('button', { name: '런타임' }))
    expect(onStart).toHaveBeenCalledTimes(1)
    expect(onRuntime).not.toHaveBeenCalled()
  })
})
