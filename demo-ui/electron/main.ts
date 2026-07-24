import { app, BrowserWindow, type BrowserWindowConstructorOptions } from 'electron'
import * as path from 'node:path'
import { fileURLToPath } from 'node:url'

const DEFAULT_WIDTH = 1280
const DEFAULT_HEIGHT = 800
const BACKGROUND = '#0f1216'

export function browserWindowOptions(preloadPath: string): BrowserWindowConstructorOptions {
  return {
    width: DEFAULT_WIDTH,
    height: DEFAULT_HEIGHT,
    backgroundColor: BACKGROUND,
    show: false,
    webPreferences: {
      preload: preloadPath,
      contextIsolation: true,
      nodeIntegration: false,
      sandbox: true,
    },
  }
}

function attachSecurityHandlers(window: BrowserWindow, allowedOrigin: string | null): void {
  window.webContents.setWindowOpenHandler(() => ({ action: 'deny' }))

  window.webContents.on('will-navigate', (event, targetUrl) => {
    if (allowedOrigin === null) {
      event.preventDefault()
      return
    }

    if (!targetUrl.startsWith(allowedOrigin)) {
      event.preventDefault()
    }
  })
}

async function createWindow(): Promise<void> {
  const currentDir = path.dirname(fileURLToPath(import.meta.url))
  const preloadPath = path.join(currentDir, 'preload.js')
  const window = new BrowserWindow(browserWindowOptions(preloadPath))

  const devUrl = process.env.VITE_DEV_SERVER_URL ?? ''
  if (devUrl.length > 0) {
    attachSecurityHandlers(window, new URL(devUrl).origin)
    await window.loadURL(devUrl)
  } else {
    const indexPath = path.join(currentDir, '..', 'dist', 'index.html')
    attachSecurityHandlers(window, null)
    await window.loadFile(indexPath)
  }

  window.once('ready-to-show', () => window.show())
}

function isRunAsMain(): boolean {
  const currentFile = fileURLToPath(import.meta.url)
  return process.argv[1] === currentFile
}

if (isRunAsMain()) {
  void app.whenReady().then(() => {
    void createWindow()
    app.on('activate', () => {
      if (BrowserWindow.getAllWindows().length === 0) {
        void createWindow()
      }
    })
  })

  app.on('window-all-closed', () => {
    if (process.platform !== 'darwin') {
      app.quit()
    }
  })
}
