import { render, screen } from '@testing-library/react'
import { describe, expect, it } from 'vitest'
import { SequencePanel } from '../src/components/SequencePanel'
import type { SequenceSnapshot } from '../src/contracts'

const snapshot: SequenceSnapshot = {
  level_id: 'TR-L01',
  label: 'Training Level 01',
  zones: [
    { zone_id: 'A-1', label: '상부', status: 'completed' },
    { zone_id: 'A-2', label: '중간', status: 'current' },
    { zone_id: 'A-3', label: '하부', status: 'next' },
  ],
  current_zone_id: 'A-2',
  next_zone_ids: ['A-3'],
  completed_zone_ids: ['A-1'],
}

describe('SequencePanel', () => {
  it('renders level label and every zone with a status badge', () => {
    render(<SequencePanel snapshot={snapshot} session={null} />)
    expect(screen.getByText('Training Level 01')).toBeInTheDocument()
    expect(screen.getByTestId('zone-A-1')).toHaveAttribute('data-status', 'completed')
    expect(screen.getByTestId('zone-A-2')).toHaveAttribute('data-status', 'current')
    expect(screen.getByTestId('zone-A-3')).toHaveAttribute('data-status', 'next')
    expect(screen.getByText('상부')).toBeInTheDocument()
    expect(screen.getByText('중간')).toBeInTheDocument()
    expect(screen.getByText('하부')).toBeInTheDocument()
  })
})

describe('SequencePanel 시뮬레이터 세션 배지', () => {
  const session = {
    session_id: 'abcdef12-0000-0000-0000-000000000000',
    level_id: 'TR-L01',
    phase: 'runtime' as const,
    recommended_flow: ['A-1', 'A-2', 'A-3'],
    current_zone_id: 'A-2',
    completed_zone_ids: ['A-1'],
    last_event_id: 'ev-1',
    anomaly_count: 1,
    unresolved_zone_id: null,
    updated_at: '2026-07-28T00:00:00+00:00',
  }

  it('세션이 없으면 대기로 표시한다', () => {
    render(<SequencePanel snapshot={snapshot} session={null} />)
    expect(screen.getByTestId('sim-phase')).toHaveTextContent('대기')
    expect(screen.queryByTestId('sim-session-id')).toBeNull()
  })

  it('세션 단계와 짧은 세션 ID를 표시한다', () => {
    render(<SequencePanel snapshot={snapshot} session={session} />)
    expect(screen.getByTestId('sim-phase')).toHaveTextContent('런타임')
    expect(screen.getByTestId('sim-session-id')).toHaveTextContent('abcdef12')
  })

  it('세션 끝 단계를 표시한다', () => {
    render(<SequencePanel snapshot={snapshot} session={{ ...session, phase: 'session_end' }} />)
    expect(screen.getByTestId('sim-phase')).toHaveTextContent('세션 끝')
  })

  it('미매핑 구역을 알린다', () => {
    render(
      <SequencePanel snapshot={snapshot} session={{ ...session, unresolved_zone_id: 'B-3' }} />,
    )
    expect(screen.getByTestId('sim-unresolved')).toHaveTextContent('B-3')
  })

  it('미매핑 구역이 없으면 표시하지 않는다', () => {
    render(<SequencePanel snapshot={snapshot} session={session} />)
    expect(screen.queryByTestId('sim-unresolved')).toBeNull()
  })
})
