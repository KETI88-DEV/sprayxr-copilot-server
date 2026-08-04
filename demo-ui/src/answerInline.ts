import type { AnswerLine, HeadingBlock, HeadingLevel, InlineSegment } from './answerTypes'

const STRONG_SOURCE = String.raw`\*\*([^*\n]+?)\*\*`
/** ATX heading, optionally closed: `### 제목 ###`. Deeper than four is prose. */
const HEADING_SOURCE = String.raw`^[ \t]*(#{1,4})[ \t]+(.*?)[ \t]*#*[ \t]*$`
const HEADING_LEVELS: readonly HeadingLevel[] = [1, 2, 3, 4]
const KOREAN_POSTPOSITION_BOUNDARY = /(?<=[0-9A-Za-z%°℃µμ])(?=(?:으로|에서|은|는|이|가|을|를|로|와|과|에|의|도|만)(?=[^가-힣]|$))/g

function bindKoreanPostpositions(text: string): string {
  return text.replace(KOREAN_POSTPOSITION_BOUNDARY, '\u2060')
}

export function parseInline(text: string): AnswerLine {
  const pattern = new RegExp(STRONG_SOURCE, 'g')
  const segments: InlineSegment[] = []
  let cursor = 0

  let match = pattern.exec(text)
  while (match !== null) {
    const emphasised = match[1] ?? ''
    if (match.index > cursor) {
      segments.push({ kind: 'text', text: bindKoreanPostpositions(text.slice(cursor, match.index)) })
    }
    if (emphasised.trim().length > 0) {
      segments.push({ kind: 'strong', text: bindKoreanPostpositions(emphasised) })
    }
    cursor = match.index + match[0].length
    match = pattern.exec(text)
  }

  if (cursor < text.length) {
    segments.push({ kind: 'text', text: bindKoreanPostpositions(text.slice(cursor)) })
  }

  return segments.length === 0 ? [{ kind: 'text', text: bindKoreanPostpositions(text) }] : segments
}

export function parseHeading(line: string): HeadingBlock | null {
  const match = new RegExp(HEADING_SOURCE).exec(line)
  if (match === null) {
    return null
  }

  const content = (match[2] ?? '').trim()
  // `###` on its own is a half-arrived delta, not a heading yet.
  if (content.length === 0) {
    return null
  }

  const depth = (match[1] ?? '#').length
  return { kind: 'heading', level: HEADING_LEVELS[depth - 1] ?? 4, line: parseInline(content) }
}
