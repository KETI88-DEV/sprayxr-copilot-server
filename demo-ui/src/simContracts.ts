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
