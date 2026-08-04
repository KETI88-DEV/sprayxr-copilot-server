export type InlineSegment =
  | { readonly kind: 'text'; readonly text: string }
  | { readonly kind: 'strong'; readonly text: string }

export type AnswerLine = readonly InlineSegment[]

export type HeadingLevel = 1 | 2 | 3 | 4

export type ColumnAlign = 'default' | 'left' | 'center' | 'right'

export type ParagraphBlock = { readonly kind: 'paragraph'; readonly lines: readonly AnswerLine[] }

export type BulletsBlock = { readonly kind: 'bullets'; readonly items: readonly AnswerLine[] }

export type HeadingBlock = { readonly kind: 'heading'; readonly level: HeadingLevel; readonly line: AnswerLine }

export type TableBlock = {
  readonly kind: 'table'
  readonly header: readonly AnswerLine[]
  readonly rows: readonly (readonly AnswerLine[])[]
  readonly alignments: readonly ColumnAlign[]
}

export type BodyBlock = ParagraphBlock | BulletsBlock | HeadingBlock | TableBlock

export type SectionBlock = {
  readonly kind: 'section'
  readonly label: string
  readonly body: readonly BodyBlock[]
}

export type AnswerBlock = BodyBlock | SectionBlock

export type AnswerDocument = {
  readonly blocks: readonly AnswerBlock[]
  /** Unique citation ids in first-seen order. */
  readonly citations: readonly string[]
}
