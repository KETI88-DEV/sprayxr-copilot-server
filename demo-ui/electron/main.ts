import { app, BrowserWindow, Menu, type BrowserWindowConstructorOptions } from 'electron'
import * as path from 'node:path'
import { fileURLToPath } from 'node:url'

const DEFAULT_WIDTH = 1280
const DEFAULT_HEIGHT = 800
const MIN_WIDTH = 960
const MIN_HEIGHT = 640

/** Must stay in step with `--hull-900` / `--ink-100` in `src/styles.css`. */
const BACKGROUND = '#0e1215'
const SYMBOL = '#e6ebee'
/** Must stay in step with the `.titlebar` height in `src/styles.css`. */
const TITLE_BAR_HEIGHT = 44

export function browserWindowOptions(preloadPath: string): BrowserWindowConstructorOptions {
  return {
    width: DEFAULT_WIDTH,
    height: DEFAULT_HEIGHT,
    minWidth: MIN_WIDTH,
    minHeight: MIN_HEIGHT,
    backgroundColor: BACKGROUND,
    show: false,
    // One themed row instead of a light native title bar stacked on a dark app
    // header: the renderer draws the bar and the OS draws its controls into it.
    titleBarStyle: 'hidden',
    titleBarOverlay: { color: BACKGROUND, symbolColor: SYMBOL, height: TITLE_BAR_HEIGHT },
    trafficLightPosition: { x: 14, y: 14 },
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

void app.whenReady().then(() => {
  // The default File/Edit/View/Window menu is replaced by the in-app menu.
  Menu.setApplicationMenu(null)
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
