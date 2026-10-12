import { render, screen } from '@testing-library/react'
import { describe, expect, it } from 'vitest'
import { SequencePanel } from '../src/components/SequencePanel'
import { sequenceViewFromPayloads } from '../src/contracts'

// 시뮬레이터가 보내는 시퀀스 본문 모양(벤더 세션 샘플). 구역은 레벨 안의 번호다.
const body = (zone: string, seq: string) => ({ scenario_id: '3', zone_id: zone, sequence_id: seq })
const view = sequenceViewFromPayloads([
  body('1', 'SEQ-001'),
  body('1', 'SEQ-002'),
  body('2', 'SEQ-003'),
  body('-1', 'SEQ-004'),
])

describe('SequencePanel', () => {
  it('레벨과 지나간 구역을 분사 횟수와 함께 보여준다', () => {
    render(<SequencePanel view={view} session={null} />)
    expect(screen.getByText('3')).toBeInTheDocument()
    expect(screen.getByTestId('zone-1')).toHaveAttribute('data-status', 'completed')
    expect(screen.getByTestId('zone-2')).toHaveAttribute('data-status', 'current')
    expect(screen.getByTestId('zone-1')).toHaveTextContent('분사 2회')
  })

  it('미명중 분사는 구역으로 세지 않고 요약에만 적는다', () => {
    render(<SequencePanel view={view} session={null} />)
    expect(screen.queryByTestId('zone--1')).toBeNull()
    // 모니터 패널에 live region 이 이미 있다. 여기까지 live 로 만들면 서로 경쟁한다.
    const summary = screen.getByTestId('sequence-summary')
    expect(summary).not.toHaveAttribute('role')
    expect(summary).toHaveTextContent('구역 2곳에서 분사 4회, 미명중 1회')
  })

  it('아직 시퀀스가 없으면 빈 상태를 안내한다', () => {
    render(<SequencePanel view={sequenceViewFromPayloads([])} session={null} />)
    expect(screen.getByTestId('sequence-summary')).toHaveTextContent('아직 받은 분사가 없습니다')
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
    render(<SequencePanel view={view} session={null} />)
    expect(screen.getByTestId('sim-phase')).toHaveTextContent('대기')
    expect(screen.queryByTestId('sim-session-id')).toBeNull()
  })

  it('세션 단계와 짧은 세션 ID를 표시한다', () => {
    render(<SequencePanel view={view} session={session} />)
    expect(screen.getByTestId('sim-phase')).toHaveTextContent('런타임')
    expect(screen.getByTestId('sim-session-id')).toHaveTextContent('abcdef12')
  })

  it('세션 끝 단계를 표시한다', () => {
    render(<SequencePanel view={view} session={{ ...session, phase: 'session_end' }} />)
    expect(screen.getByTestId('sim-phase')).toHaveTextContent('세션 끝')
  })

  it('미매핑 구역을 알린다', () => {
    render(
      <SequencePanel view={view} session={{ ...session, unresolved_zone_id: 'B-3' }} />,
    )
    expect(screen.getByTestId('sim-unresolved')).toHaveTextContent('B-3')
  })

  it('미매핑 구역이 없으면 표시하지 않는다', () => {
    render(<SequencePanel view={view} session={session} />)
    expect(screen.queryByTestId('sim-unresolved')).toBeNull()
  })
})
