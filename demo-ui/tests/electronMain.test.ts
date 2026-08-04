import { describe, expect, it, vi } from 'vitest'
import { browserWindowOptions } from '../electron/main'

const state = vi.hoisted(() => ({
  createdWindowOptions: [] as unknown[],
  applicationMenus: [] as unknown[],
}))

vi.mock('electron', () => {
  class MockBrowserWindow {
    static getAllWindows(): unknown[] {
      return []
    }

    webContents = {
      setWindowOpenHandler: () => undefined,
      on: () => undefined,
      getURL: () => '',
    }

    constructor(options: unknown) {
      state.createdWindowOptions.push(options)
    }

    once(_event: string, handler: () => void): void {
      handler()
    }

    show(): void {}

    loadFile(): Promise<void> {
      return Promise.resolve()
    }

    loadURL(): Promise<void> {
      return Promise.resolve()
    }
  }

  return {
    app: { whenReady: () => Promise.resolve(), on: () => undefined, quit: () => undefined },
    BrowserWindow: MockBrowserWindow,
    Menu: {
      setApplicationMenu: (menu: unknown) => {
        state.applicationMenus.push(menu)
      },
    },
    shell: { openExternal: () => Promise.resolve() },
    contextBridge: { exposeInMainWorld: () => undefined },
  }
})

describe('browserWindowOptions', () => {
  it('isolates and sandboxes the renderer with the given preload', () => {
    const options = browserWindowOptions('/tmp/preload.js')
    expect(options.webPreferences).toMatchObject({
      contextIsolation: true,
      nodeIntegration: false,
      sandbox: true,
      preload: '/tmp/preload.js',
    })
  })

  it('sets sensible default window bounds and background', () => {
    const options = browserWindowOptions('/tmp/preload.js')
    expect(options.width).toBeGreaterThanOrEqual(1024)
    expect(options.height).toBeGreaterThanOrEqual(700)
    expect(options.backgroundColor).toMatch(/^#[0-9a-fA-F]{6}$/)
  })

  it('hides the native title bar and themes the window control overlay', () => {
    const options = browserWindowOptions('/tmp/preload.js')
    expect(options.titleBarStyle).toBe('hidden')
    expect(options.titleBarOverlay).toMatchObject({
      color: options.backgroundColor,
      symbolColor: '#e6ebee',
      height: 44,
    })
  })
})

describe('electron main entry', () => {
  it('opens a window when loaded as the entry module regardless of argv shape', async () => {
    process.argv = ['/path/to/electron', 'dist-electron/main.js']

    await vi.waitFor(() => {
      expect(state.createdWindowOptions.length).toBeGreaterThan(0)
    })
  })

  it('removes the default application menu so only the in-app menu remains', async () => {
    await vi.waitFor(() => {
      expect(state.applicationMenus).toContain(null)
    })
  })
})
