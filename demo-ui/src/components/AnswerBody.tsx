import { Fragment, useMemo, type ReactElement, type ReactNode } from 'react'
import {
  formatAnswer,
  type AnswerLine,
  type BodyBlock,
  type HeadingBlock,
  type TableBlock,
} from '../answerFormat'

/**
 * Text segments render as bare text nodes on purpose: keeping them as direct
 * children of the paragraph keeps the answer selectable, copyable, and
 * addressable by plain text queries.
 */
function renderLine(line: AnswerLine): ReactNode[] {
  return line.map((segment, index) =>
    segment.kind === 'strong' ? (
      <strong className="answer__strong" key={index}>
        {segment.text}
      </strong>
    ) : (
      <Fragment key={index}>{segment.text}</Fragment>
    ),
  )
}

function HeadingView({ block }: { readonly block: HeadingBlock }): ReactElement {
  const content = renderLine(block.line)
  return block.level <= 2 ? (
    <h5 className="answer__heading" data-level={block.level}>
      {content}
    </h5>
  ) : (
    <h6 className="answer__heading" data-level={block.level}>
      {content}
    </h6>
  )
}

function TableView({ block }: { readonly block: TableBlock }): ReactElement {
  return (
    <div className="answer__table-wrap">
      <table className="answer__table">
        <thead>
          <tr>
            {block.header.map((cell, column) => (
              <th key={column} scope="col" data-align={block.alignments[column] ?? 'default'}>
                {renderLine(cell)}
              </th>
            ))}
          </tr>
        </thead>
        <tbody>
          {block.rows.map((row, rowIndex) => (
            <tr key={rowIndex}>
              {row.map((cell, column) => (
                <td key={column} data-align={block.alignments[column] ?? 'default'}>
                  {renderLine(cell)}
                </td>
              ))}
            </tr>
          ))}
        </tbody>
      </table>
    </div>
  )
}

function BodyBlockView({ block }: { readonly block: BodyBlock }): ReactElement {
  switch (block.kind) {
    case 'heading':
      return <HeadingView block={block} />
    case 'table':
      return <TableView block={block} />
    case 'bullets':
      return (
        <ul className="answer__bullets">
          {block.items.map((item, index) => (
            <li className="answer__bullet" key={index}>
              {renderLine(item)}
            </li>
          ))}
        </ul>
      )
    case 'paragraph':
      return (
        <div className="answer__paragraph">
          {block.lines.map((line, index) => (
            <p className="answer__line" key={index}>
              {renderLine(line)}
            </p>
          ))}
        </div>
      )
  }
}

type Props = {
  readonly text: string
  readonly streaming?: boolean
}

export function AnswerBody({ text, streaming = false }: Props): ReactElement {
  const answer = useMemo(() => formatAnswer(text), [text])

  return (
    <div className="answer">
      {answer.blocks.map((block, index) => {
        if (block.kind === 'section') {
          return (
            <section className="answer__section" data-label={block.label} key={index}>
              <h4 className="answer__section-label">{block.label}</h4>
              <div className="answer__section-body">
                {block.body.map((child, childIndex) => (
                  <BodyBlockView block={child} key={childIndex} />
                ))}
              </div>
            </section>
          )
        }
        return <BodyBlockView block={block} key={index} />
      })}

      {answer.citations.length > 0 && (
        <footer className="answer__citations">
          <p className="answer__citations-label">참조 문서 {answer.citations.length}건</p>
          <ul className="answer__citation-list">
            {answer.citations.map((citation) => (
              <li className="answer__citation" key={citation}>
                {citation}
              </li>
            ))}
          </ul>
        </footer>
      )}

      {streaming && <span className="answer__caret" aria-hidden="true" />}
    </div>
  )
}
