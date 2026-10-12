import { describe, expect, it } from 'vitest'
import {
  formatAnswer,
  type AnswerBlock,
  type AnswerLine,
  type HeadingBlock,
  type SectionBlock,
  type TableBlock,
} from '../src/answerFormat'

function plainText(line: AnswerLine | undefined): string {
  return (line ?? []).map((segment) => segment.text).join('')
}

function blockAt(blocks: readonly AnswerBlock[], index: number): AnswerBlock {
  const block = blocks[index]
  if (block === undefined) {
    throw new Error(`missing block ${index}`)
  }
  return block
}

function paragraphAt(blocks: readonly AnswerBlock[], index: number): readonly AnswerLine[] {
  const block = blockAt(blocks, index)
  if (block.kind !== 'paragraph') {
    throw new Error(`block ${index} is ${block.kind}, expected paragraph`)
  }
  return block.lines
}

function bulletsAt(blocks: readonly AnswerBlock[], index: number): readonly AnswerLine[] {
  const block = blockAt(blocks, index)
  if (block.kind !== 'bullets') {
    throw new Error(`block ${index} is ${block.kind}, expected bullets`)
  }
  return block.items
}

function sectionAt(blocks: readonly AnswerBlock[], index: number): SectionBlock {
  const block = blockAt(blocks, index)
  if (block.kind !== 'section') {
    throw new Error(`block ${index} is ${block.kind}, expected section`)
  }
  return block
}

function headingAt(blocks: readonly AnswerBlock[], index: number): HeadingBlock {
  const block = blockAt(blocks, index)
  if (block.kind !== 'heading') {
    throw new Error(`block ${index} is ${block.kind}, expected heading`)
  }
  return block
}

function tableAt(blocks: readonly AnswerBlock[], index: number): TableBlock {
  const block = blockAt(blocks, index)
  if (block.kind !== 'table') {
    throw new Error(`block ${index} is ${block.kind}, expected table`)
  }
  return block
}

function tableText(table: TableBlock): { readonly header: string[]; readonly rows: string[][] } {
  return {
    header: table.header.map(plainText),
    rows: table.rows.map((row) => row.map(plainText)),
  }
}

describe('formatAnswer', () => {
  it('returns an empty document for blank input', () => {
    expect(formatAnswer('')).toEqual({ blocks: [], citations: [] })
    expect(formatAnswer('   \n\n  ')).toEqual({ blocks: [], citations: [] })
  })

  it('keeps a plain line as a single paragraph line', () => {
    const lines = paragraphAt(formatAnswer('다음 도장 구역: A-2').blocks, 0)
    expect(lines).toHaveLength(1)
    expect(plainText(lines[0])).toBe('다음 도장 구역: A-2')
  })

  it('splits bold runs into strong segments and leaves the rest as text', () => {
    const lines = paragraphAt(formatAnswer('도막 두께는 **215 µm**로 측정되었다').blocks, 0)
    expect(lines[0]).toEqual([
      { kind: 'text', text: '도막 두께는 ' },
      { kind: 'strong', text: '215 µm' },
      { kind: 'text', text: '로 측정되었다' },
    ])
  })

  it('leaves an unterminated bold marker as literal text while streaming', () => {
    const lines = paragraphAt(formatAnswer('도막 두께는 **215').blocks, 0)
    expect(lines[0]).toEqual([{ kind: 'text', text: '도막 두께는 **215' }])
  })

  it('binds a mixed-script measurement to its Korean postposition', () => {
    const lines = paragraphAt(formatAnswer('nominal 대비 -2.2%로 양호합니다').blocks, 0)
    expect(plainText(lines[0])).toBe('nominal 대비 -2.2%\u2060로 양호합니다')
  })

  it('groups consecutive dash and asterisk lines into one bullet block', () => {
    const document = formatAnswer('점검 항목\n- 상부 구역 재도장\n* 하부 구역 건조 확인\n\n종료')
    expect(document.blocks).toHaveLength(3)
    expect(bulletsAt(document.blocks, 1).map(plainText)).toEqual(['상부 구역 재도장', '하부 구역 건조 확인'])
    expect(paragraphAt(document.blocks, 2).map(plainText)).toEqual(['종료'])
  })

  it('promotes bracketed Korean labels to sections that own the following lines', () => {
    const document = formatAnswer(
      ['[분석] 평균 도막 두께 215 µm', '표준편차 18 µm', '[판정] 합격', '[조치]', '- 재검사 불필요'].join('\n'),
    )
    expect(document.blocks).toHaveLength(3)

    const analysis = sectionAt(document.blocks, 0)
    expect(analysis.label).toBe('분석')
    expect(paragraphAt(analysis.body, 0).map(plainText)).toEqual(['평균 도막 두께 215 µm', '표준편차 18 µm'])

    const verdict = sectionAt(document.blocks, 1)
    expect(verdict.label).toBe('판정')
    expect(paragraphAt(verdict.body, 0).map(plainText)).toEqual(['합격'])

    const action = sectionAt(document.blocks, 2)
    expect(action.label).toBe('조치')
    expect(bulletsAt(action.body, 0).map(plainText)).toEqual(['재검사 불필요'])
  })

  it('does not treat a numeric marker as a section label', () => {
    const lines = paragraphAt(formatAnswer('[1] 첫 번째 근거').blocks, 0)
    expect(plainText(lines[0])).toBe('[1] 첫 번째 근거')
  })

  it('drops a multi-word evidence label left behind by a lifted citation', () => {
    // 2026-10-12 데모: `[근거] 참조 문서 ID: DOC-…` 에서 ID 를 칩으로 옮긴 뒤 `참조 문서 ID:` 만 남았다.
    const document = formatAnswer(['[답변] 저도막은 체류 시간을 늘린다.', '[근거] 참조 문서 ID: DOC-doc_0036-C0098'].join('\n'))

    expect(document.citations).toEqual(['DOC-doc_0036-C0098'])
    expect(document.blocks).toHaveLength(1)
    expect(sectionAt(document.blocks, 0).label).toBe('답변')
  })

  it('keeps a real sentence that only mentions a citation', () => {
    const document = formatAnswer('[답변] 체류 시간은 문서 규정을 따른다: DOC-doc_0036-C0098 참고하여 30초 이상 유지한다.')

    expect(sectionAt(document.blocks, 0).label).toBe('답변')
    expect(paragraphAt(sectionAt(document.blocks, 0).body, 0).map(plainText).join(' ')).toContain('30초')
  })

  it('lifts repeated citation ids into one de-duplicated list', () => {
    const document = formatAnswer(
      ['[판정] 합격 (DOC-doc_0039-C0239)', '[근거] DOC-doc_0039-C0239, DOC-doc_0041-C0101, DOC-doc_0039-C0239'].join(
        '\n',
      ),
    )

    expect(document.citations).toEqual(['DOC-doc_0039-C0239', 'DOC-doc_0041-C0101'])
    // The `[근거]` section held nothing but citations, so only `[판정]` survives.
    expect(document.blocks).toHaveLength(1)

    const verdict = sectionAt(document.blocks, 0)
    expect(verdict.label).toBe('판정')
    expect(paragraphAt(verdict.body, 0).map(plainText)).toEqual(['합격'])
  })

  it('drops a bare citation line but keeps prose that surrounds an inline citation', () => {
    const document = formatAnswer(
      ['SSPC-PA2 규정[DOC-doc_0002-C0007]에 따른다', 'DOC-doc_0002-C0007', '근거: DOC-doc_0002-C0007'].join('\n'),
    )

    expect(document.citations).toEqual(['DOC-doc_0002-C0007'])
    expect(document.blocks).toHaveLength(1)
    expect(paragraphAt(document.blocks, 0).map(plainText)).toEqual(['SSPC-PA2 규정에 따른다'])
  })

  it('never throws on adversarial input', () => {
    for (const raw of [
      '**',
      '[]',
      '[근거]',
      '- ',
      '***강조***',
      'DOC--C',
      '[분석]\n\n\n',
      '\r\n\r\n',
      '#',
      '#####  깊은 제목',
      '|',
      '||',
      '|---|',
      '| a |\n|---|\n|',
      '|\n|---|\n| 값 |',
    ]) {
      expect(() => formatAnswer(raw)).not.toThrow()
    }
  })
})

describe('formatAnswer — ATX headings', () => {
  it('reads one through four hashes as heading levels', () => {
    const document = formatAnswer(['# 개요', '## DFT 합부 판정 결과', '### 측정 방법', '#### 비고'].join('\n'))
    expect(document.blocks).toHaveLength(4)
    expect(headingAt(document.blocks, 0)).toMatchObject({ level: 1 })
    expect(headingAt(document.blocks, 1)).toMatchObject({ level: 2 })
    expect(headingAt(document.blocks, 2)).toMatchObject({ level: 3 })
    expect(headingAt(document.blocks, 3)).toMatchObject({ level: 4 })
    expect(plainText(headingAt(document.blocks, 1).line)).toBe('DFT 합부 판정 결과')
  })

  it('resolves inline bold inside a heading and strips closing hashes', () => {
    const heading = headingAt(formatAnswer('### 판정: **PASS** ###').blocks, 0)
    expect(heading.line).toEqual([
      { kind: 'text', text: '판정: ' },
      { kind: 'strong', text: 'PASS' },
    ])
  })

  it('leaves a bare hash run, a hash without a space, and five hashes as prose', () => {
    expect(paragraphAt(formatAnswer('###').blocks, 0).map(plainText)).toEqual(['###'])
    expect(paragraphAt(formatAnswer('#제목').blocks, 0).map(plainText)).toEqual(['#제목'])
    expect(paragraphAt(formatAnswer('##### 너무 깊음').blocks, 0).map(plainText)).toEqual(['##### 너무 깊음'])
  })

  it('opens a heading inside a bracketed section without closing the section', () => {
    const section = sectionAt(formatAnswer('[분석]\n### 분포\n평균 81.8 um').blocks, 0)
    expect(section.label).toBe('분석')
    expect(headingAt(section.body, 0).level).toBe(3)
    expect(paragraphAt(section.body, 1).map(plainText)).toEqual(['평균 81.8 um'])
  })
})

describe('formatAnswer — pipe tables', () => {
  it('builds a header and body from a well formed table', () => {
    const document = formatAnswer(
      ['| 항목 | 값 |', '|---|---|', '| 평균 DFT | 97.8 um |', '| 규격 이탈 | 0건 |'].join('\n'),
    )
    expect(document.blocks).toHaveLength(1)
    expect(tableText(tableAt(document.blocks, 0))).toEqual({
      header: ['항목', '값'],
      rows: [
        ['평균 DFT', '97.8 um'],
        ['규격 이탈', '0건'],
      ],
    })
  })

  it('reads column alignment markers', () => {
    const table = tableAt(formatAnswer(['| 순서 | 구역 | 두께 |', '|:---|:---:|---:|', '| 1 | A-1 | 97 |'].join('\n')).blocks, 0)
    expect(table.alignments).toEqual(['left', 'center', 'right'])
  })

  it('defaults alignment when the separator carries no markers', () => {
    const table = tableAt(formatAnswer('| a | b |\n| --- | --- |\n| 1 | 2 |').blocks, 0)
    expect(table.alignments).toEqual(['default', 'default'])
  })

  it('resolves inline bold inside a cell', () => {
    const table = tableAt(formatAnswer('| 순서 | 구역 |\n|---|---|\n| **2** | **A-2** |').blocks, 0)
    expect(table.rows[0]?.[0]).toEqual([{ kind: 'strong', text: '2' }])
    expect(table.rows[0]?.[1]).toEqual([{ kind: 'strong', text: 'A-2' }])
  })

  it('keeps a partially streamed table as prose until the separator row lands', () => {
    const headerOnly = formatAnswer('### 분포 분석\n| 항목 | 값 |')
    expect(headerOnly.blocks).toHaveLength(2)
    expect(paragraphAt(headerOnly.blocks, 1).map(plainText)).toEqual(['| 항목 | 값 |'])

    const partialSeparator = formatAnswer('| 항목 | 값 |\n|---')
    expect(partialSeparator.blocks).toHaveLength(1)
    expect(paragraphAt(partialSeparator.blocks, 0).map(plainText)).toEqual(['| 항목 | 값 |', '|---'])
  })

  it('accepts a table that has no body row yet and grows without breaking', () => {
    const opening = tableAt(formatAnswer('| 항목 | 값 |\n|---|---|').blocks, 0)
    expect(tableText(opening)).toEqual({ header: ['항목', '값'], rows: [] })

    const truncated = tableAt(formatAnswer('| 항목 | 값 |\n|---|---|\n|').blocks, 0)
    expect(tableText(truncated)).toEqual({ header: ['항목', '값'], rows: [] })

    const arriving = tableAt(formatAnswer('| 항목 | 값 |\n|---|---|\n| 평균 DFT').blocks, 0)
    expect(tableText(arriving)).toEqual({ header: ['항목', '값'], rows: [['평균 DFT', '']] })
  })

  it('pads a short ragged row and widens the table for a long one', () => {
    const table = tableAt(
      formatAnswer(['| 순서 | 구역 | 상태 |', '|---|---|---|', '| 1 | A-1 |', '| 2 | A-2 | 다음 | 비고 |'].join('\n'))
        .blocks,
      0,
    )
    expect(tableText(table)).toEqual({
      header: ['순서', '구역', '상태', ''],
      rows: [
        ['1', 'A-1', '', ''],
        ['2', 'A-2', '다음', '비고'],
      ],
    })
    expect(table.alignments).toEqual(['default', 'default', 'default', 'default'])
  })

  it('never mistakes a pipe in prose or a thematic break for a table', () => {
    const prose = formatAnswer('측정값은 90% | 나머지는 10% 규칙을 따른다')
    expect(paragraphAt(prose.blocks, 0).map(plainText)).toEqual(['측정값은 90% | 나머지는 10% 규칙을 따른다'])

    const thematicBreak = formatAnswer('90% 규칙 | 10% 규칙\n---\n다음 문단')
    expect(thematicBreak.blocks.every((block) => block.kind !== 'table')).toBe(true)
  })

  it('starts a new table when a second header follows immediately', () => {
    const document = formatAnswer(
      ['| 순서 | 구역 |', '|---|---|', '| 1 | A-1 |', '| 순서 | 블록 |', '|---|---|', '| 현재 | BLK-A |'].join('\n'),
    )
    expect(document.blocks).toHaveLength(2)
    expect(tableText(tableAt(document.blocks, 0))).toEqual({ header: ['순서', '구역'], rows: [['1', 'A-1']] })
    expect(tableText(tableAt(document.blocks, 1))).toEqual({ header: ['순서', '블록'], rows: [['현재', 'BLK-A']] })
  })

  it('honours escaped pipes inside a cell', () => {
    const table = tableAt(formatAnswer('| 기호 | 설명 |\n|---|---|\n| \\| | 세로줄 |').blocks, 0)
    expect(tableText(table).rows).toEqual([['|', '세로줄']])
  })

  it('lifts a citation out of a table row and drops a row that held nothing else', () => {
    const document = formatAnswer(
      ['| 항목 | 근거 |', '|---|---|', '| 90% 규칙 | DOC-doc_0039-C0239 |', '| DOC-doc_0041-C0101 | |'].join('\n'),
    )
    expect(document.citations).toEqual(['DOC-doc_0039-C0239', 'DOC-doc_0041-C0101'])
    expect(tableText(tableAt(document.blocks, 0)).rows).toEqual([['90% 규칙', '']])
  })
})
