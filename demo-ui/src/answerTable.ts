import { parseInline } from './answerInline'
import type { AnswerLine, ColumnAlign, TableBlock } from './answerTypes'

/** A table alignment cell: `---`, `:---`, `---:`, `:---:`. */
const ALIGNMENT_CELL = /^:?-+:?$/

/** Splits on unescaped pipes and drops the optional outer ones. */
function splitTableRow(line: string): string[] {
  const trimmed = line.trim()
  const cells: string[] = []
  let current = ''
  let index = 0

  while (index < trimmed.length) {
    const char = trimmed[index] ?? ''
    if (char === '\\' && trimmed[index + 1] === '|') {
      current += '|'
      index += 2
      continue
    }
    if (char === '|') {
      cells.push(current)
      current = ''
      index += 1
      continue
    }
    current += char
    index += 1
  }
  cells.push(current)

  if (cells.length > 1 && trimmed.startsWith('|')) {
    cells.shift()
  }
  if (cells.length > 1 && trimmed.endsWith('|')) {
    cells.pop()
  }
  return cells.map((cell) => cell.trim())
}

/**
 * Requiring a pipe-delimited alignment row keeps ordinary prose and thematic
 * breaks from being mistaken for a table.
 */
function isAlignmentRow(line: string): boolean {
  if (!line.includes('|')) {
    return false
  }
  const cells = splitTableRow(line)
  return cells.length > 0 && cells.every((cell) => ALIGNMENT_CELL.test(cell))
}

function readAlignment(cell: string): ColumnAlign {
  const opensLeft = cell.startsWith(':')
  const closesRight = cell.endsWith(':')
  if (opensLeft && closesRight) {
    return 'center'
  }
  if (closesRight) {
    return 'right'
  }
  return opensLeft ? 'left' : 'default'
}

export function parseTable(
  lines: readonly string[],
  start: number,
): { readonly block: TableBlock; readonly next: number } | null {
  const headerLine = lines[start]
  const alignmentLine = lines[start + 1]
  if (headerLine === undefined || alignmentLine === undefined) {
    return null
  }
  if (!headerLine.includes('|') || !isAlignmentRow(alignmentLine)) {
    return null
  }

  const headerCells = splitTableRow(headerLine)
  const alignmentCells = splitTableRow(alignmentLine)
  if (headerCells.length !== alignmentCells.length) {
    return null
  }

  const bodyRows: string[][] = []
  let cursor = start + 2
  while (cursor < lines.length) {
    const row = lines[cursor] ?? ''
    if (row.trim().length === 0 || !row.includes('|') || isAlignmentRow(row)) {
      break
    }
    const successor = lines[cursor + 1]
    if (successor !== undefined && isAlignmentRow(successor)) {
      break
    }

    const cells = splitTableRow(row)
    cursor += 1
    if (cells.every((cell) => cell.length === 0)) {
      continue
    }
    bodyRows.push(cells)
  }

  const columnCount = Math.max(headerCells.length, ...bodyRows.map((row) => row.length))
  const toRow = (cells: readonly string[]): AnswerLine[] =>
    Array.from({ length: columnCount }, (_unused, column) => parseInline(cells[column] ?? ''))
  const alignments = Array.from({ length: columnCount }, (_unused, column) =>
    readAlignment(alignmentCells[column] ?? ''),
  )

  return {
    block: { kind: 'table', header: toRow(headerCells), rows: bodyRows.map(toRow), alignments },
    next: cursor,
  }
}
