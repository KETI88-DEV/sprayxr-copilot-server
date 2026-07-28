import { z } from 'zod'

export const phaseSchema = z.enum(['start', 'runtime', 'session_end'])
export type Phase = z.infer<typeof phaseSchema>

export const directionSchema = z.enum(['input', 'output'])
export type Direction = z.infer<typeof directionSchema>

export const monitorEventSchema = z
  .object({
    seq: z.number().int().nonnegative(),
    session_id: z.string(),
    phase: phaseSchema,
    direction: directionSchema,
    timestamp: z.string(),
    payload: z.record(z.string(), z.unknown()),
  })
  .strict()
export type MonitorEvent = z.infer<typeof monitorEventSchema>

export const monitorSnapshotSchema = z
  .object({
    events: z.array(monitorEventSchema),
    next_cursor: z.number().int().nonnegative(),
    dropped_before: z.number().int().nonnegative().nullable().default(null),
  })
  .strict()
export type MonitorSnapshot = z.infer<typeof monitorSnapshotSchema>

export const simSessionSchema = z
  .object({
    session_id: z.string(),
    level_id: z.string(),
    phase: phaseSchema,
    recommended_flow: z.array(z.string()),
    current_zone_id: z.string(),
    completed_zone_ids: z.array(z.string()),
    last_event_id: z.string().nullable(),
    anomaly_count: z.number().int().nonnegative(),
    unresolved_zone_id: z.string().nullable(),
    updated_at: z.string(),
  })
  .strict()
export type SimSession = z.infer<typeof simSessionSchema>

export const simSessionResponseSchema = z
  .object({ session: simSessionSchema.nullable() })
  .strict()
export type SimSessionResponse = z.infer<typeof simSessionResponseSchema>
