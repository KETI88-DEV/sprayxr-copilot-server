import '@testing-library/jest-dom/vitest'
import { vi } from 'vitest'

vi.stubEnv('VITEST', '1')

vi.mock('electron', () => ({
  app: { whenReady: () => Promise.resolve(), on: () => undefined, quit: () => undefined },
  BrowserWindow: class {
    static getAllWindows(): unknown[] {
      return []
    }

    webContents = {
      setWindowOpenHandler: () => undefined,
      on: () => undefined,
      getURL: () => '',
    }

    once() {}

    show() {}

    loadFile() {
      return Promise.resolve()
    }

    loadURL() {
      return Promise.resolve()
    }
  },
  Menu: { setApplicationMenu: () => undefined },
  shell: { openExternal: () => Promise.resolve() },
  contextBridge: { exposeInMainWorld: () => undefined },
}))
