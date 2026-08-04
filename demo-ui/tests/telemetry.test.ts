import { describe, expect, it } from 'vitest'
import { EMPTY_READOUT, formatClock, formatCount, formatDurationMs, formatSeconds } from '../src/telemetry'

describe('formatDurationMs', () => {
  it('keeps sub-second timings in milliseconds', () => {
    expect(formatDurationMs(0)).toBe('0 ms')
    expect(formatDurationMs(940)).toBe('940 ms')
  })

  it('switches to one decimal second between one second and one minute', () => {
    expect(formatDurationMs(4200)).toBe('4.2 초')
    expect(formatDurationMs(59_900)).toBe('59.9 초')
  })

  it('splits long turns into minutes and seconds', () => {
    expect(formatDurationMs(61_000)).toBe('1 분 1 초')
    expect(formatDurationMs(120_000)).toBe('2 분')
  })

  it('renders an em dash for missing or impossible values', () => {
    expect(formatDurationMs(null)).toBe(EMPTY_READOUT)
    expect(formatDurationMs(-1)).toBe(EMPTY_READOUT)
    expect(formatDurationMs(Number.NaN)).toBe(EMPTY_READOUT)
  })
})

describe('formatSeconds and formatCount', () => {
  it('appends the unit for measurable values', () => {
    expect(formatSeconds(151)).toBe('151 초')
    expect(formatCount(0, '회')).toBe('0회')
    expect(formatCount(8, '건')).toBe('8건')
  })

  it('renders an em dash before the first complete event', () => {
    expect(formatSeconds(null)).toBe(EMPTY_READOUT)
    expect(formatCount(null, '회')).toBe(EMPTY_READOUT)
  })
})

describe('formatClock', () => {
  it('zero pads local wall clock time without a locale dependency', () => {
    const at = new Date(2026, 6, 27, 9, 5, 3).getTime()
    expect(formatClock(at)).toBe('09:05:03')
  })
})
