import { API_BASE_URL, createSseParser } from './api'
import {
  monitorEventSchema,
  simSessionResponseSchema,
  type MonitorEvent,
  type SimSession,
} from './simContracts'

const OUT_OF_BAND = { speed_m_per_s: 0.8, standoff_mm: 350.0, angle_deg: 55.0 }
const IN_BAND = { speed_m_per_s: 0.8, standoff_mm: 300.0, angle_deg: 72.0 }

async function postJson(path: string, body: unknown): Promise<void> {
  const response = await fetch(`${API_BASE_URL}${path}`, {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify(body),
  })
  if (!response.ok) {
    throw new Error(`sim request failed: ${response.status}`)
  }
}

/**
 * Subscribes to the read-only monitor stream.
 *
 * The server replays everything after `cursor` before switching to live events,
 * so a `gap` frame is the only signal that the ring buffer dropped older ones.
 * Non-monitor frames are skipped rather than parsed as events.
 */
export async function* streamMonitor(cursor = 0, signal?: AbortSignal): AsyncGenerator<MonitorEvent> {
  const init: RequestInit = signal === undefined ? {} : { signal }
  const response = await fetch(`${API_BASE_URL}/api/monitor/stream?cursor=${cursor}`, init)
  if (!response.ok) {
    throw new Error(`monitor stream failed: ${response.status}`)
  }
  if (response.body === null) {
    throw new Error('monitor stream missing body')
  }

  const reader = response.body.getReader()
  const decoder = new TextDecoder()
  const parser = createSseParser()

  try {
    while (true) {
      const { done, value } = await reader.read()
      if (done) {
        break
      }

      for (const frame of parser.push(decoder.decode(value, { stream: true }))) {
        if (frame.event === 'monitor') {
          yield monitorEventSchema.parse(frame.data)
        }
      }
    }
  } finally {
    reader.releaseLock()
  }
}

export async function sendSessionStart(sessionId: string, levelId: string): Promise<void> {
  await postJson('/api/sim/sessions/start', {
    session_id: sessionId,
    level_id: levelId,
    worker: { trainee_id: 't-dummy', skill_level: 'beginner', handedness: 'right' },
    spec: { nominal_um: 100.0 },
  })
}

export async function sendRuntimeEvent(
  sessionId: string,
  eventId: string,
  zoneId: string,
  anomaly: boolean,
): Promise<void> {
  await postJson(`/api/sim/sessions/${encodeURIComponent(sessionId)}/events`, {
    event_id: eventId,
    zone_id: zoneId,
    telemetry: { gun_metrics: anomaly ? OUT_OF_BAND : IN_BAND },
  })
}

export async function fetchSimSession(signal?: AbortSignal): Promise<SimSession | null> {
  const init: RequestInit = signal === undefined ? {} : { signal }
  const response = await fetch(`${API_BASE_URL}/api/sim/session`, init)
  if (!response.ok) {
    throw new Error(`sim session request failed: ${response.status}`)
  }
  const body: unknown = await response.json()
  return simSessionResponseSchema.parse(body).session
}

export async function sendSessionEnd(sessionId: string): Promise<void> {
  await postJson(`/api/sim/sessions/${encodeURIComponent(sessionId)}/complete`, {
    session_id: sessionId,
    scenario_id: 'dummy',
    mode: 'live',
    zone_id: 'A-2',
    zone_type: 'panel',
    metrics: {
      avg_dft: 100.0,
      dft_values: [95.0, 102.0, 88.0],
      avg_speed: 0.8,
      speed_variance: 0.05,
      avg_distance: 0.3,
      distance_variance: 0.05,
      avg_angle: 70.0,
      angle_consistency: 0.9,
      sequence_correct: true,
      under_thickness_count: 1,
    },
  })
}
