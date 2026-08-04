import { describe, expect, it } from 'vitest'
import { describeStage } from '../src/stageLabels'

describe('describeStage', () => {
  it('translates every backend stage the demo API emits', () => {
    expect(describeStage('initializing').label).toBe('모델 초기화')
    expect(describeStage('generating').label).toBe('응답 생성 중')
    expect(describeStage('rag').label).toBe('문서 검색 완료')
    expect(describeStage('deadline_forced_synthesis').label).toBe('시간 초과, 응답 정리 중')
  })

  it('keeps repeating progress stages out of the permanent log', () => {
    expect(describeStage('initializing').kind).toBe('progress')
    expect(describeStage('generating').kind).toBe('progress')
    expect(describeStage('rag').kind).toBe('milestone')
    expect(describeStage('deadline_forced_synthesis').kind).toBe('milestone')
  })

  it('degrades an unknown stage to its raw value instead of crashing', () => {
    const description = describeStage('tool_dispatch')
    expect(description.known).toBe(false)
    expect(description.label).toBe('tool_dispatch')
    expect(description.kind).toBe('milestone')
  })

  it('substitutes a readable label when the stage is blank', () => {
    expect(describeStage('   ').label).toBe('알 수 없는 단계')
    expect(describeStage('').stage).toBe('')
  })
})
