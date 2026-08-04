/**
 * Backend `status` stage vocabulary translated for the operator.
 *
 * The stream emits a progress `status` every few seconds while the model is
 * working. Those repeats are noise in a log, so every stage also carries a
 * `kind` telling the UI whether it belongs in the single in-place live row
 * (`progress`) or deserves a permanent log row (`milestone`).
 */

export type StageKind = 'progress' | 'milestone'

export type StageDescription = {
  /** Raw stage value exactly as received from the backend. */
  readonly stage: string
  /** Operator-facing Korean label, or the raw value when the stage is unknown. */
  readonly label: string
  readonly kind: StageKind
  readonly known: boolean
}

const UNKNOWN_STAGE_LABEL = '알 수 없는 단계'

const STAGE_TABLE = new Map<string, { readonly label: string; readonly kind: StageKind }>([
  ['initializing', { label: '모델 초기화', kind: 'progress' }],
  ['generating', { label: '응답 생성 중', kind: 'progress' }],
  ['rag', { label: '문서 검색 완료', kind: 'milestone' }],
  ['deadline_forced_synthesis', { label: '시간 초과, 응답 정리 중', kind: 'milestone' }],
])

/**
 * Never throws. An unrecognised stage degrades to its raw value and is treated
 * as a milestone so that information the backend adds later still surfaces,
 * while the known high-frequency progress stages stay out of the log.
 */
export function describeStage(stage: string): StageDescription {
  const known = STAGE_TABLE.get(stage)
  if (known !== undefined) {
    return { stage, label: known.label, kind: known.kind, known: true }
  }

  const trimmed = stage.trim()
  return {
    stage,
    label: trimmed.length === 0 ? UNKNOWN_STAGE_LABEL : trimmed,
    kind: 'milestone',
    known: false,
  }
}
