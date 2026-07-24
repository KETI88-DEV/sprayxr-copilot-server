import { describe, expect, it } from 'vitest'
import { sequenceSnapshotSchema, sequenceUpdateSchema } from '../src/contracts'

describe('sequence contracts', () => {
  it('rejects duplicate next zones in update', () => {
    const parsed = sequenceUpdateSchema.safeParse({
      type: 'update_sequence',
      level_id: 'SP0035-L01',
      current_zone_id: 'A-1',
      next_zone_ids: ['A-2', 'A-2'],
      completed_zone_ids: ['A-1'],
    })
    expect(parsed.success).toBe(false)
  })

  it('parses a snapshot', () => {
    const parsed = sequenceSnapshotSchema.parse({
      level_id: 'SP0035-L01',
      label: 'Level 01',
      zones: [{ zone_id: 'A-1', label: '상부', status: 'completed' }],
      current_zone_id: 'A-1',
      next_zone_ids: [],
      completed_zone_ids: ['A-1'],
    })
    expect(parsed.current_zone_id).toBe('A-1')
  })
})
