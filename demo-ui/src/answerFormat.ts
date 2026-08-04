/**
 * Pure renderer model for a coating co-pilot answer.
 *
 * The model emits several things at once and the raw string is unreadable:
 *   1. light Markdown  — `**강조**`, `- 항목`, `### 제목`, `| 표 |`, hard line breaks
 *   2. bracketed Korean sections — `[분석] [판정] [원인] [조치] [근거]`
 *   3. chunk citation ids — `DOC-doc_0039-C0239`, frequently repeated verbatim
 *
 * `formatAnswer` turns that into a small document tree and lifts every citation
 * into one de-duplicated list, so the view layer stays declarative. No parser
 * dependency: the grammar here is deliberately smaller than Markdown.
 *
 * The function is total — any string, including a partial streaming buffer,
 * produces a document.
 */

import { parseHeading, parseInline } from './answerInline'
import { parseTable } from './answerTable'
import type { AnswerBlock, AnswerDocument, AnswerLine, BodyBlock } from './answerTypes'

export type {
  AnswerBlock,
  AnswerDocument,
  AnswerLine,
  BodyBlock,
  BulletsBlock,
  ColumnAlign,
  HeadingBlock,
  HeadingLevel,
  InlineSegment,
  ParagraphBlock,
  SectionBlock,
  TableBlock,
} from './answerTypes'

/** `DOC-<doc id>-C<chunk sequence>` as produced by `chunk_docs.py`. */
const CITATION_SOURCE = String.raw`DOC-[A-Za-z0-9_.]+-C\d+`
/** A line that opens a labelled block: `[판정] 합격`. Korean labels only. */
const SECTION_SOURCE = String.raw`^[ \t]*[[［]([가-힣]{1,10})[\]］][ \t]*(.*)$`
const BULLET_SOURCE = String.raw`^[ \t]*[-*•‣][ \t]+(.*)$`
type RawGroup = { readonly label: string | null; readonly lines: string[] }

function citationPattern(): RegExp {
  return new RegExp(CITATION_SOURCE, 'g')
}

function collectCitations(raw: string): string[] {
  const pattern = citationPattern()
  const seen = new Set<string>()
  const ordered: string[] = []

  let match = pattern.exec(raw)
  while (match !== null) {
    const id = match[0]
    if (!seen.has(id)) {
      seen.add(id)
      ordered.push(id)
    }
    match = pattern.exec(raw)
  }

  return ordered
}

/** Repairs the punctuation a removed citation id leaves behind. */
function tidyLine(line: string): string {
  return line
    .replace(/[[(（【][ \t]*(?:[,;·、/][ \t]*)*[\])）】]/g, '')
    .replace(/[ \t]*[,;·、][ \t]*(?=[,;·、])/g, '')
    .replace(/[ \t]*[,;·、][ \t]*$/, '')
    .replace(/[ \t]{2,}/g, ' ')
    .replace(/[ \t]+([.,;:!?])/g, '$1')
    .trimEnd()
}

/** True when nothing but punctuation or a dangling `근거:` style label remains. */
function isCitationResidue(text: string): boolean {
  const trimmed = text.trim()
  if (trimmed.length === 0) {
    return true
  }
  if (/^[\p{P}\p{S}\s]+$/u.test(trimmed)) {
    return true
  }
  return /^[^\s:：]{1,12}[ \t]*[:：]$/u.test(trimmed)
}

function stripCitationLines(lines: readonly string[]): string[] {
  const kept: string[] = []

  for (const line of lines) {
    if (!citationPattern().test(line)) {
      kept.push(line)
      continue
    }

    const cleaned = tidyLine(line.replace(citationPattern(), ''))
    if (isCitationResidue(cleaned)) {
      continue
    }
    kept.push(cleaned)
  }

  return kept
}

function groupSections(lines: readonly string[]): RawGroup[] {
  const pattern = new RegExp(SECTION_SOURCE)
  const groups: RawGroup[] = [{ label: null, lines: [] }]

  for (const line of lines) {
    const match = pattern.exec(line)
    if (match === null) {
      groups[groups.length - 1]?.lines.push(line)
      continue
    }

    const group: RawGroup = { label: match[1] ?? '', lines: [] }
    const trailing = (match[2] ?? '').trim()
    if (trailing.length > 0) {
      group.lines.push(trailing)
    }
    groups.push(group)
  }

  return groups
}

function parseBody(lines: readonly string[]): BodyBlock[] {
  const bulletPattern = new RegExp(BULLET_SOURCE)
  const blocks: BodyBlock[] = []
  let paragraph: AnswerLine[] = []
  let bullets: AnswerLine[] = []

  const flushParagraph = (): void => {
    if (paragraph.length > 0) {
      blocks.push({ kind: 'paragraph', lines: paragraph })
      paragraph = []
    }
  }

  const flushBullets = (): void => {
    if (bullets.length > 0) {
      blocks.push({ kind: 'bullets', items: bullets })
      bullets = []
    }
  }

  let index = 0
  while (index < lines.length) {
    const line = lines[index] ?? ''
    const start = index
    index += 1

    if (line.trim().length === 0) {
      flushParagraph()
      flushBullets()
      continue
    }

    const heading = parseHeading(line)
    if (heading !== null) {
      flushParagraph()
      flushBullets()
      blocks.push(heading)
      continue
    }

    const table = parseTable(lines, start)
    if (table !== null) {
      flushParagraph()
      flushBullets()
      blocks.push(table.block)
      index = table.next
      continue
    }

    const bulletMatch = bulletPattern.exec(line)
    if (bulletMatch !== null) {
      flushParagraph()
      const content = (bulletMatch[1] ?? '').trim()
      if (content.length > 0) {
        bullets.push(parseInline(content))
      }
      continue
    }

    flushBullets()
    paragraph.push(parseInline(line.trim()))
  }

  flushParagraph()
  flushBullets()
  return blocks
}

export function formatAnswer(raw: string): AnswerDocument {
  const citations = collectCitations(raw)
  const lines = stripCitationLines(raw.split(/\r?\n/))
  const blocks: AnswerBlock[] = []

  for (const group of groupSections(lines)) {
    const body = parseBody(group.lines)
    if (group.label === null) {
      blocks.push(...body)
      continue
    }
    // A section whose only content was citations is dropped: the citation row
    // already carries that information, and an empty `[근거]` header is noise.
    if (body.length === 0) {
      continue
    }
    blocks.push({ kind: 'section', label: group.label, body })
  }

  return { blocks, citations }
}
