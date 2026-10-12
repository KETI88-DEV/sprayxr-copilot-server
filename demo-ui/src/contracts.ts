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
  session_id: z.string().min(1),
  user_id: z.string().min(1),
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
  readonly session_id: string | null
  readonly user_id: string | null
  readonly level_id: string | null
  readonly zones: readonly ZoneProgress[]
  readonly sequence_count: number
  readonly unassigned_count: number
}

export const EMPTY_SEQUENCE_VIEW: SessionSequenceView = {
  session_id: null,
  user_id: null,
  level_id: null,
  zones: [],
  sequence_count: 0,
  unassigned_count: 0,
}

/**
 * 레벨에 구역 목록(카탈로그)이 없어 아직 안 간 구역은 알 수 없다 — 지나간 구역만 처음 간 순서로 낸다.
 * 모니터에는 지난 세션도 남아 있으므로 가장 최근 세션의 시퀀스만 센다. 안 그러면 데모를 다시 돌릴 때마다 횟수가 쌓인다.
 */
export function sequenceViewFromPayloads(payloads: readonly unknown[]): SessionSequenceView {
  const counts = new Map<string, number>()
  let levelId: string | null = null
  let total = 0
  let unassigned = 0
  let lastZoneId: string | null = null

  const bodies = payloads.flatMap((payload) => {
    const parsed = sequenceBodySchema.safeParse(payload)
    return parsed.success ? [parsed.data] : []
  })
  const latestSession = bodies.at(-1)?.session_id
  // 같은 sequence_id 를 다시 받으면 같은 시퀀스다(서버도 duplicate 로 본다). 한 번만 센다.
  const unique = new Map(bodies.filter((b) => b.session_id === latestSession).map((b) => [b.sequence_id, b]))

  for (const body of unique.values()) {
    total += 1
    levelId = body.scenario_id
    if (body.zone_id === UNASSIGNED_ZONE_ID) {
      unassigned += 1
      continue
    }
    counts.set(body.zone_id, (counts.get(body.zone_id) ?? 0) + 1)
    lastZoneId = body.zone_id
  }

  const latest = bodies.at(-1)
  return {
    session_id: latest?.session_id ?? null,
    user_id: latest?.user_id ?? null,
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

// --- v2.5 질의 기능 4개 ----------------------------------------------------------
//
// `POST /api/v1/copilot/query` 의 task_type 마다 기본 질문 하나. 채팅(`/api/chat`)으로 보내면
// task_type 이 없어 pose_recomm 이 채점기를 타지 않으므로 질의 API 를 직접 부른다.

export type CopilotTaskType = 'knowledge_search' | 'pose_recomm' | 'trajectory_recomm' | 'error_analysis'

export const TASK_PRESETS: readonly { readonly task_type: CopilotTaskType; readonly label: string; readonly prompt: string }[] = [
  { task_type: 'knowledge_search', label: '지식 검색', prompt: '에폭시 도료의 재도장 간격은?' },
  { task_type: 'pose_recomm', label: '자세 추천', prompt: '방금 분사한 자세를 평가해 줘' },
  { task_type: 'trajectory_recomm', label: '궤적 추천', prompt: '다음 분사 궤적을 추천해 줘' },
  { task_type: 'error_analysis', label: '오류 분석', prompt: '방금 시퀀스의 오류를 분석해 줘' },
]

export const copilotQueryResponseSchema = z.object({
  message: z.string(),
  data: z.object({ answer: z.string().nullable().optional() }).passthrough().nullable(),
})
