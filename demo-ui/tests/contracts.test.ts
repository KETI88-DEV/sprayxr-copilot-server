import { describe, expect, it } from 'vitest'
import { sequenceSnapshotSchema, sequenceUpdateSchema, streamEventSchema } from '../src/contracts'

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

describe('stream event timing contracts', () => {
  it('accepts complete timings and budget metadata', () => {
    const parsed = streamEventSchema.safeParse({
      event: 'complete',
      data: {
        tool_call_count: 1,
        rag_chunk_count: 2,
        ttft_ms: 4200,
        synthesis_ms: 15_000,
        total_ms: 61_000,
        deadline_hit: false,
      },
    })

    expect(parsed.success).toBe(true)
  })

  it('accepts null ttft when generation ends before the first token', () => {
    const parsed = streamEventSchema.safeParse({
      event: 'complete',
      data: {
        tool_call_count: 0,
        rag_chunk_count: 0,
        ttft_ms: null,
        synthesis_ms: 0,
        total_ms: 900,
        deadline_hit: false,
      },
    })

    expect(parsed.success).toBe(true)
  })

  it('accepts deadline status without a rag chunk count', () => {
    const parsed = streamEventSchema.safeParse({
      event: 'status',
      data: {
        stage: 'deadline_forced_synthesis',
        elapsed_sec: 151,
      },
    })

    expect(parsed.success).toBe(true)
  })

  it('accepts timed errors emitted before the first token', () => {
    const parsed = streamEventSchema.safeParse({
      event: 'error',
      data: {
        message: 'streaming pipeline failed',
        retryable: true,
        ttft_ms: null,
        total_ms: 42,
      },
    })

    expect(parsed.success).toBe(true)
  })
})
