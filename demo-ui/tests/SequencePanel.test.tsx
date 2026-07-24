import { render, screen } from '@testing-library/react'
import { describe, expect, it } from 'vitest'
import { SequencePanel } from '../src/components/SequencePanel'
import type { SequenceSnapshot } from '../src/contracts'

const snapshot: SequenceSnapshot = {
  level_id: 'SP0035-L01',
  label: 'SP0035 Level 01',
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
    render(<SequencePanel snapshot={snapshot} />)
    expect(screen.getByText('SP0035 Level 01')).toBeInTheDocument()
    expect(screen.getByTestId('zone-A-1')).toHaveAttribute('data-status', 'completed')
    expect(screen.getByTestId('zone-A-2')).toHaveAttribute('data-status', 'current')
    expect(screen.getByTestId('zone-A-3')).toHaveAttribute('data-status', 'next')
    expect(screen.getByText('상부')).toBeInTheDocument()
    expect(screen.getByText('중간')).toBeInTheDocument()
    expect(screen.getByText('하부')).toBeInTheDocument()
  })
})
