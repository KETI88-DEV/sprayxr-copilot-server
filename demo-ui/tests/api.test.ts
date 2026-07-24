import { describe, expect, it } from 'vitest'
import { createSseParser } from '../src/api'

describe('createSseParser', () => {
  it('parses an event split across chunks', () => {
    const parser = createSseParser()
    const first = parser.push('event: message_delta\ndata: {"event":"message_')
    expect(first).toEqual([])
    const rest = parser.push('delta","data":{"text":"안녕"}}\n\n')
    expect(rest).toHaveLength(1)
    expect(rest[0]?.event).toBe('message_delta')
    expect((rest[0]?.data as { text: string }).text).toBe('안녕')
  })

  it('parses multiple events in one chunk', () => {
    const parser = createSseParser()
    const frames = parser.push(
      'event: session\ndata: {"event":"session","data":{"session_id":"s","mode":"demo"}}\n\n' +
      'event: complete\ndata: {"event":"complete","data":{"tool_call_count":0,"rag_chunk_count":0}}\n\n'
    )
    expect(frames.map(f => f.event)).toEqual(['session', 'complete'])
  })

  it('drops frames with invalid JSON', () => {
    const parser = createSseParser()
    expect(parser.push('event: session\ndata: not-json\n\n')).toEqual([])
  })
})
