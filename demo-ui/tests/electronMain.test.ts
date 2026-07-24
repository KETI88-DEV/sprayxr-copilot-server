import { describe, expect, it } from 'vitest'
import { browserWindowOptions } from '../electron/main'

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
})
