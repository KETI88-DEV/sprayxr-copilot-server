import { useEffect, useRef, useState, type ReactElement } from 'react'

type Props = {
  readonly logOpen: boolean
  readonly busy: boolean
  readonly onToggleLog: () => void
  readonly onReset: () => void
  readonly onReconnect: () => void
}

type DesktopRuntime = { readonly platform: string; readonly electronVersion: string }

/**
 * Reads the only thing preload exposes. Absent in a browser tab, so every
 * access is narrowed rather than asserted.
 */
function readDesktopRuntime(): DesktopRuntime | null {
  const scope: object = globalThis
  if (!('desktop' in scope)) {
    return null
  }

  const bridge: unknown = scope.desktop
  if (typeof bridge !== 'object' || bridge === null) {
    return null
  }
  if (!('platform' in bridge) || !('electronVersion' in bridge)) {
    return null
  }

  const { platform, electronVersion } = bridge
  if (typeof platform !== 'string' || typeof electronVersion !== 'string') {
    return null
  }
  return { platform, electronVersion }
}

function MenuGlyph(): ReactElement {
  return (
    <svg viewBox="0 0 16 16" width="16" height="16" aria-hidden="true" focusable="false">
      <path
        d="M2.5 4h11M2.5 8h11M2.5 12h11"
        fill="none"
        stroke="currentColor"
        strokeWidth="1.4"
        strokeLinecap="round"
      />
    </svg>
  )
}

export function TitleBar({ logOpen, busy, onToggleLog, onReset, onReconnect }: Props): ReactElement {
  const [open, setOpen] = useState<boolean>(false)
  const [runtime] = useState<DesktopRuntime | null>(() => readDesktopRuntime())
  const menuRef = useRef<HTMLDivElement | null>(null)

  useEffect(() => {
    if (!open) {
      return
    }

    function handlePointerDown(event: MouseEvent): void {
      const root = menuRef.current
      if (root !== null && event.target instanceof Node && !root.contains(event.target)) {
        setOpen(false)
      }
    }

    function handleKeyDown(event: globalThis.KeyboardEvent): void {
      if (event.key === 'Escape') {
        setOpen(false)
      }
    }

    window.addEventListener('mousedown', handlePointerDown)
    window.addEventListener('keydown', handleKeyDown)
    return () => {
      window.removeEventListener('mousedown', handlePointerDown)
      window.removeEventListener('keydown', handleKeyDown)
    }
  }, [open])

  function runAndClose(action: () => void): void {
    setOpen(false)
    action()
  }

  return (
    <header className="titlebar" data-platform={runtime?.platform ?? 'web'}>
      <div className="titlebar__inner">
        <div className="titlebar__menu" ref={menuRef}>
          <button
            type="button"
            className="titlebar__menu-button"
            aria-haspopup="menu"
            aria-expanded={open}
            aria-label="메뉴"
            onClick={() => setOpen((previous) => !previous)}
          >
            <MenuGlyph />
          </button>

          {open && (
            <div className="titlebar__dropdown" role="menu">
              <button
                type="button"
                role="menuitem"
                className="titlebar__item"
                disabled={busy}
                onClick={() => runAndClose(onReset)}
              >
                대화 초기화
              </button>
              <button
                type="button"
                role="menuitem"
                className="titlebar__item"
                onClick={() => runAndClose(onReconnect)}
              >
                연결 다시 확인
              </button>
              <button
                type="button"
                role="menuitem"
                className="titlebar__item"
                onClick={() => runAndClose(onToggleLog)}
              >
                {logOpen ? '활동 로그 숨기기' : '활동 로그 표시'}
              </button>
              <p className="titlebar__runtime">
                {runtime === null ? '웹 브라우저' : `${runtime.platform} · Electron ${runtime.electronVersion}`}
              </p>
            </div>
          )}
        </div>

        <span className="titlebar__name">선박 도장 AI 코파일럿</span>
        <span className="titlebar__divider" aria-hidden="true" />
        <span className="titlebar__kicker">TRAINING CONSOLE</span>
      </div>
    </header>
  )
}
