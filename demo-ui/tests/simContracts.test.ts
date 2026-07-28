import { describe, expect, it } from 'vitest'
import { monitorEventSchema, monitorSnapshotSchema } from '../src/simContracts'

describe('simContracts', () => {
  it('parses a valid monitor event', () => {
    const event = monitorEventSchema.parse({
      seq: 1,
      session_id: 's1',
      phase: 'start',
      direction: 'input',
      timestamp: 't',
      payload: { a: 1 },
    })

    expect(event.phase).toBe('start')
    expect(event.direction).toBe('input')
  })

  it('rejects an unknown phase', () => {
    expect(() =>
      monitorEventSchema.parse({
        seq: 1,
        session_id: 's1',
        phase: 'bogus',
        direction: 'input',
        timestamp: 't',
        payload: {},
      }),
    ).toThrow()
  })

  it('rejects an unexpected field so backend drift is caught at the boundary', () => {
    expect(() =>
      monitorEventSchema.parse({
        seq: 1,
        session_id: 's1',
        phase: 'runtime',
        direction: 'output',
        timestamp: 't',
        payload: {},
        unexpected: true,
      }),
    ).toThrow()
  })

  it('parses a snapshot with nullable dropped_before', () => {
    const snapshot = monitorSnapshotSchema.parse({ events: [], next_cursor: 0, dropped_before: null })

    expect(snapshot.next_cursor).toBe(0)
    expect(snapshot.dropped_before).toBeNull()
  })

  it('parses a snapshot that reports a gap', () => {
    const snapshot = monitorSnapshotSchema.parse({
      events: [
        {
          seq: 3,
          session_id: 's1',
          phase: 'session_end',
          direction: 'output',
          timestamp: 't',
          payload: { risk_flags: [] },
        },
      ],
      next_cursor: 3,
      dropped_before: 3,
    })

    expect(snapshot.dropped_before).toBe(3)
    expect(snapshot.events[0]?.phase).toBe('session_end')
  })
})
