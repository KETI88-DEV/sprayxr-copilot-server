import type { ReactElement } from 'react'

type Props = {
  readonly online: boolean
  readonly levelId: string
  readonly mode: 'demo' | 'live'
  readonly stage: string
  readonly onModeChange: (mode: 'demo' | 'live') => void
  readonly onReconnect: () => void
}

export function StatusBar({ online, levelId, mode, stage, onModeChange, onReconnect }: Props): ReactElement {
  return (
    <header className="status-bar" data-online={online}>
      <div className="status-bar__left">
        <span className="status-bar__title">Ship Painting AI Co-Pilot</span>
        <div className="status-bar__mode-selector" role="group" aria-label="chat mode">
          {(['demo', 'live'] as const).map((option) => (
            <button
              key={option}
              type="button"
              className="status-bar__mode-option"
              data-active={mode === option}
              aria-pressed={mode === option}
              onClick={() => onModeChange(option)}
            >
              {option}
            </button>
          ))}
        </div>
      </div>
      <div className="status-bar__right">
        <span className="status-bar__level">{levelId}</span>
        <span className="status-bar__stage">{stage}</span>
        <span className="status-bar__health" data-online={online} data-testid="connection-state">
          {online ? '온라인' : '오프라인'}
        </span>
        {!online && (
          <button type="button" className="status-bar__retry" onClick={onReconnect}>
            재연결
          </button>
        )}
      </div>
    </header>
  )
}
