import { describe, expect, it } from 'vitest'
import { applySequenceUpdate } from '../src/sequenceState'
import type { SequenceZone, SequenceUpdate } from '../src/contracts'

const zones: SequenceZone[] = [
  { zone_id: 'A-1', label: '상부', status: 'pending' },
  { zone_id: 'A-2', label: '중간', status: 'pending' },
  { zone_id: 'A-3', label: '하부', status: 'pending' },
]

describe('applySequenceUpdate', () => {
  it('maps completed, next, and pending statuses', () => {
    const update: SequenceUpdate = {
      type: 'update_sequence',
      level_id: 'SP0035-L01',
      current_zone_id: 'A-1',
      next_zone_ids: ['A-2'],
      completed_zone_ids: ['A-1'],
    }
    const result = applySequenceUpdate(zones, update)
    expect(result.map(zone => zone.status)).toEqual(['completed', 'next', 'pending'])
  })

  it('gives completed precedence over current', () => {
    const update: SequenceUpdate = {
      type: 'update_sequence',
      level_id: 'SP0035-L01',
      current_zone_id: 'A-1',
      next_zone_ids: [],
      completed_zone_ids: ['A-1'],
    }
    const result = applySequenceUpdate(zones, update)
    expect(result[0]?.status).toBe('completed')
  })

  it('leaves unknown zones untouched', () => {
    const update: SequenceUpdate = {
      type: 'update_sequence',
      level_id: 'SP0035-L01',
      current_zone_id: 'Z-9',
      next_zone_ids: ['Z-8'],
      completed_zone_ids: [],
    }
    const result = applySequenceUpdate(zones, update)
    expect(result.map(zone => zone.status)).toEqual(['pending', 'pending', 'pending'])
  })

  it('does not mutate the input', () => {
    const snapshot: SequenceZone[] = zones.map(zone => ({ ...zone }))
    applySequenceUpdate(snapshot, {
      type: 'update_sequence',
      level_id: 'SP0035-L01',
      current_zone_id: 'A-1',
      next_zone_ids: [],
      completed_zone_ids: [],
    })
    expect(snapshot).toEqual(zones)
  })
})
