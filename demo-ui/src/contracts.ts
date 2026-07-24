import { z } from 'zod'

export const zoneStatusSchema = z.enum(['current', 'next', 'completed', 'pending'])
export type ZoneStatus = z.infer<typeof zoneStatusSchema>

export const sequenceZoneSchema = z
  .object({
    zone_id: z.string().min(1),
    label: z.string().min(1),
    status: zoneStatusSchema.default('pending'),
  })
  .strict()
export type SequenceZone = z.infer<typeof sequenceZoneSchema>

export const sequenceSnapshotSchema = z
  .object({
    level_id: z.string().min(1),
    label: z.string().min(1),
    zones: z.array(sequenceZoneSchema),
    current_zone_id: z.string().min(1),
    next_zone_ids: z.array(z.string()),
    completed_zone_ids: z.array(z.string()),
  })
  .strict()
  .superRefine((snapshot, ctx) => {
    const zoneIds = new Set(snapshot.zones.map((zone) => zone.zone_id))

    if (!zoneIds.has(snapshot.current_zone_id)) {
      ctx.addIssue({
        code: z.ZodIssueCode.custom,
        message: 'current_zone_id must reference an existing zone',
        path: ['current_zone_id'],
      })
    }

    snapshot.next_zone_ids.forEach((zoneId, index) => {
      if (!zoneIds.has(zoneId)) {
        ctx.addIssue({
          code: z.ZodIssueCode.custom,
          message: 'next_zone_ids must reference existing zones',
          path: ['next_zone_ids', index],
        })
      }
    })

    snapshot.completed_zone_ids.forEach((zoneId, index) => {
      if (!zoneIds.has(zoneId)) {
        ctx.addIssue({
          code: z.ZodIssueCode.custom,
          message: 'completed_zone_ids must reference existing zones',
          path: ['completed_zone_ids', index],
        })
      }
    })
  })
export type SequenceSnapshot = z.infer<typeof sequenceSnapshotSchema>

export const sequenceUpdateSchema = z
  .object({
    type: z.literal('update_sequence'),
    level_id: z.string().min(1),
    current_zone_id: z.string().min(1),
    next_zone_ids: z.array(z.string()),
    completed_zone_ids: z.array(z.string()),
  })
  .strict()
  .superRefine((update, ctx) => {
    const seen = new Set<string>()

    update.next_zone_ids.forEach((zoneId, index) => {
      if (seen.has(zoneId)) {
        ctx.addIssue({
          code: z.ZodIssueCode.custom,
          message: 'next_zone_ids must be unique',
          path: ['next_zone_ids', index],
        })
      }
      seen.add(zoneId)
    })
  })
export type SequenceUpdate = z.infer<typeof sequenceUpdateSchema>

export const chatRequestSchema = z
  .object({
    message: z.string().min(1).max(4000),
    session_id: z.string().min(1),
    mode: z.enum(['demo', 'live']).default('demo'),
    level_id: z.string().min(1).default('SP0035-L01'),
  })
  .strict()
export type ChatRequest = z.infer<typeof chatRequestSchema>

const streamSessionSchema = z
  .object({
    event: z.literal('session'),
    data: z
      .object({
        session_id: z.string(),
        mode: z.enum(['demo', 'live']),
      })
      .strict(),
  })
  .strict()

const streamStatusSchema = z
  .object({
    event: z.literal('status'),
    data: z
      .object({
        stage: z.string(),
        chunks: z.number().int().nonnegative(),
      })
      .strict(),
  })
  .strict()

const streamMessageDeltaSchema = z
  .object({
    event: z.literal('message_delta'),
    data: z
      .object({
        text: z.string(),
      })
      .strict(),
  })
  .strict()

const streamToolResultSchema = z
  .object({
    event: z.literal('tool_result'),
    data: z
      .object({
        name: z.string(),
        arguments: z.record(z.string(), z.unknown()),
        result: z.record(z.string(), z.unknown()),
      })
      .strict(),
  })
  .strict()

const streamSequenceUpdateSchema = z
  .object({
    event: z.literal('sequence_update'),
    data: sequenceUpdateSchema,
  })
  .strict()

const streamCompleteSchema = z
  .object({
    event: z.literal('complete'),
    data: z
      .object({
        tool_call_count: z.number().int().nonnegative(),
        rag_chunk_count: z.number().int().nonnegative(),
      })
      .strict(),
  })
  .strict()

const streamErrorSchema = z
  .object({
    event: z.literal('error'),
    data: z
      .object({
        message: z.string(),
        retryable: z.boolean(),
      })
      .strict(),
  })
  .strict()

export const streamEventSchema = z.discriminatedUnion('event', [
  streamSessionSchema,
  streamStatusSchema,
  streamMessageDeltaSchema,
  streamToolResultSchema,
  streamSequenceUpdateSchema,
  streamCompleteSchema,
  streamErrorSchema,
])
export type StreamEvent = z.infer<typeof streamEventSchema>
