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
    level_id: z.string().min(1).default('TR-L01'),
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
        chunks: z.number().int().nonnegative().optional(),
        elapsed_sec: z.number().int().nonnegative().optional(),
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
        ttft_ms: z.number().int().nonnegative().nullable().optional(),
        synthesis_ms: z.number().int().nonnegative().optional(),
        total_ms: z.number().int().nonnegative().optional(),
        deadline_hit: z.boolean().optional(),
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
        ttft_ms: z.number().int().nonnegative().nullable().optional(),
        total_ms: z.number().int().nonnegative().optional(),
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

// --- v2.5 세션 시퀀스 뷰 -----------------------------------------------------
//
// 입력은 시뮬레이터가 보내는 시퀀스 본문(`POST /api/v1/copilot/analyze-sequence`)이다.
// 벤더 세션 샘플(docs/examples, 2026-10-08)과 같은 모양이며, 패널이 쓰는 세 필드만 읽는다.
// 구역은 레벨마다 1부터 매긴 번호이고 레벨 안에서만 유일하다 — 레벨(`scenario_id`)과 함께 본다.

/** 도막에 명중하지 않은 분사. 구역이 아니라 따로 센다(벤더 샘플 notes). */
export const UNASSIGNED_ZONE_ID = '-1'

export const sequenceBodySchema = z.object({
  scenario_id: z.string().min(1),
  zone_id: z.string().min(1),
  sequence_id: z.string().min(1),
})

export type ZoneProgress = {
  readonly zone_id: string
  readonly sequence_count: number
  readonly status: 'current' | 'completed'
}

export type SessionSequenceView = {
  readonly level_id: string | null
  readonly zones: readonly ZoneProgress[]
  readonly sequence_count: number
  readonly unassigned_count: number
}

export const EMPTY_SEQUENCE_VIEW: SessionSequenceView = {
  level_id: null,
  zones: [],
  sequence_count: 0,
  unassigned_count: 0,
}

/** 레벨에 구역 목록(카탈로그)이 없어 아직 안 간 구역은 알 수 없다 — 지나간 구역만 처음 간 순서로 낸다. */
export function sequenceViewFromPayloads(payloads: readonly unknown[]): SessionSequenceView {
  const counts = new Map<string, number>()
  let levelId: string | null = null
  let total = 0
  let unassigned = 0
  let lastZoneId: string | null = null

  for (const payload of payloads) {
    const parsed = sequenceBodySchema.safeParse(payload)
    if (!parsed.success) continue
    total += 1
    levelId = parsed.data.scenario_id
    if (parsed.data.zone_id === UNASSIGNED_ZONE_ID) {
      unassigned += 1
      continue
    }
    counts.set(parsed.data.zone_id, (counts.get(parsed.data.zone_id) ?? 0) + 1)
    lastZoneId = parsed.data.zone_id
  }

  return {
    level_id: levelId,
    zones: [...counts].map(([zoneId, count]) => ({
      zone_id: zoneId,
      sequence_count: count,
      status: zoneId === lastZoneId ? 'current' : 'completed',
    })),
    sequence_count: total,
    unassigned_count: unassigned,
  }
}
