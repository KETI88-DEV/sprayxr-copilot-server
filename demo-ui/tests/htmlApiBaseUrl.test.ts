import { readFileSync } from 'node:fs'
import { fileURLToPath } from 'node:url'
import * as path from 'node:path'
import * as vm from 'node:vm'
import { describe, expect, it } from 'vitest'

const PLACEHOLDER = '%VITE_API_BASE_URL%'

function bootstrapScript(): string {
  const htmlPath = path.join(path.dirname(fileURLToPath(import.meta.url)), '..', 'index.html')
  const html = readFileSync(htmlPath, 'utf8')
  const match = /<script>([\s\S]*?)<\/script>/.exec(html)
  if (match?.[1] === undefined) {
    throw new Error('index.html is missing the API base URL bootstrap script')
  }
  return match[1]
}

function evaluateWith(replacement: string): unknown {
  const source = bootstrapScript().split(PLACEHOLDER).join(replacement)
  const window: Record<string, unknown> = {}
  vm.runInNewContext(source, { window })
  return window.__DEMO_API_BASE_URL__
}

describe('index.html API base URL bootstrap', () => {
  it('keeps the build-time injected base URL', () => {
    expect(evaluateWith('http://192.168.0.10:8120')).toBe('http://192.168.0.10:8120')
  })

  it('keeps a hand-edited base URL in a deployed bundle', () => {
    expect(evaluateWith('https://demo.example.com')).toBe('https://demo.example.com')
  })

  it('falls back to undefined when the placeholder is not replaced', () => {
    expect(evaluateWith(PLACEHOLDER)).toBeUndefined()
  })
})
