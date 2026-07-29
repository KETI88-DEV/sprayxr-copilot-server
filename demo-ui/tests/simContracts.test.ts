import { describe, expect, it } from 'vitest'
import {
  monitorEventSchema,
  monitorSnapshotSchema,
  simSessionResponseSchema,
  simSessionSchema,
} from '../src/simContracts'

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

describe('simSessionSchema', () => {
  const valid = {
    session_id: 's-1',
    level_id: 'TR-L01',
    phase: 'runtime',
    recommended_flow: ['A-1', 'A-2', 'A-3'],
    current_zone_id: 'A-2',
    completed_zone_ids: ['A-1'],
    last_event_id: 'ev-1',
    anomaly_count: 1,
    unresolved_zone_id: null,
    updated_at: '2026-07-28T00:00:00+00:00',
  }

  it('백엔드 SimSessionView 봉투를 그대로 받는다', () => {
    expect(simSessionSchema.parse(valid).current_zone_id).toBe('A-2')
  })

  it('알 수 없는 키를 거부한다', () => {
    expect(() => simSessionSchema.parse({ ...valid, extra: 1 })).toThrow()
  })

  it('세션 없음을 null로 표현한다', () => {
    expect(simSessionResponseSchema.parse({ session: null }).session).toBeNull()
  })

  it('세션이 있으면 중첩 스키마로 검증한다', () => {
    const parsed = simSessionResponseSchema.parse({ session: valid })
    expect(parsed.session?.completed_zone_ids).toEqual(['A-1'])
  })
})
