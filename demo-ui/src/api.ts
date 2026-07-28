import { z } from 'zod'
import { chatRequestSchema, sequenceSnapshotSchema, streamEventSchema } from './contracts'
import type { ChatRequest, SequenceSnapshot, StreamEvent } from './contracts'

export const API_BASE_URL =
  typeof window !== 'undefined' && window.__DEMO_API_BASE_URL__ !== undefined
    ? window.__DEMO_API_BASE_URL__
    : import.meta.env.VITE_API_BASE_URL ?? ''

export class ApiError extends Error {
  constructor(message: string, public status?: number) {
    super(message)
    this.name = 'ApiError'
  }
}

export type SseFrame = { readonly event: string; readonly data: unknown }

type SseParser = { push(chunk: string): SseFrame[] }

function isRecord(value: unknown): value is Record<string, unknown> {
  return typeof value === 'object' && value !== null
}

function parseFrame(rawFrame: string): SseFrame | null {
  const lines = rawFrame.split('\n')
  const eventLine = lines.find((line) => line.startsWith('event: '))
  const dataLine = lines.find((line) => line.startsWith('data: '))

  if (eventLine === undefined || dataLine === undefined) {
    return null
  }

  const event = eventLine.slice('event: '.length).trim()
  if (event.length === 0) {
    return null
  }

  try {
    const parsedJson: unknown = JSON.parse(dataLine.slice('data: '.length))
    if (isRecord(parsedJson) && 'data' in parsedJson) {
      return { event, data: parsedJson.data }
    }
    return { event, data: parsedJson }
  } catch {
    return null
  }
}

export function createSseParser(): SseParser {
  let buffer = ''

  return {
    push(chunk: string): SseFrame[] {
      buffer += chunk
      const frames: SseFrame[] = []

      while (true) {
        const separatorIndex = buffer.indexOf('\n\n')
        if (separatorIndex === -1) {
          break
        }

        const rawFrame = buffer.slice(0, separatorIndex)
        buffer = buffer.slice(separatorIndex + 2)

        const parsed = parseFrame(rawFrame)
        if (parsed !== null) {
          frames.push(parsed)
        }
      }

      return frames
    },
  }
}

const healthSchema = z
  .object({
    status: z.string(),
    servers: z.number().int().nonnegative(),
    timestamp: z.string(),
  })
  .strict()

async function readJsonResponse(response: Response): Promise<unknown> {
  const json: unknown = await response.json()
  return json
}

async function fetchJson(path: string, signal?: AbortSignal): Promise<unknown> {
  try {
    const init: RequestInit = signal === undefined ? {} : { signal }
    const response = await fetch(`${API_BASE_URL}${path}`, init)
    if (!response.ok) {
      throw new ApiError(`request failed: ${response.status}`, response.status)
    }
    return await readJsonResponse(response)
  } catch (error: unknown) {
    if (error instanceof ApiError) {
      throw error
    }
    if (error instanceof Error) {
      throw new ApiError(error.message)
    }
    throw new ApiError('request failed')
  }
}

export async function fetchHealth(signal?: AbortSignal): Promise<{
  readonly status: string
  readonly servers: number
  readonly timestamp: string
}> {
  return healthSchema.parse(await fetchJson('/api/health', signal))
}

export async function fetchSequenceSnapshot(
  levelId: string,
  signal?: AbortSignal,
): Promise<SequenceSnapshot> {
  return sequenceSnapshotSchema.parse(
    await fetchJson(`/api/sequence/${encodeURIComponent(levelId)}`, signal),
  )
}

export async function* streamChat(
  request: ChatRequest,
  signal?: AbortSignal,
): AsyncIterable<StreamEvent> {
  const parsedRequest = chatRequestSchema.parse(request)

  let response: Response
  try {
    const requestInit: RequestInit = {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify(parsedRequest),
      ...(signal === undefined ? {} : { signal }),
    }
    response = await fetch(`${API_BASE_URL}/api/chat`, requestInit)
  } catch (error: unknown) {
    if (error instanceof Error) {
      throw new ApiError(error.message)
    }
    throw new ApiError('request failed')
  }

  if (!response.ok) {
    throw new ApiError(`request failed: ${response.status}`, response.status)
  }

  if (response.body === null) {
    throw new ApiError('chat stream missing body', response.status)
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

      const chunk = decoder.decode(value, { stream: true })
      for (const frame of parser.push(chunk)) {
        yield streamEventSchema.parse(frame)
      }
    }

    const tail = decoder.decode()
    for (const frame of parser.push(`${tail}\n\n`)) {
      yield streamEventSchema.parse(frame)
    }
  } catch (error: unknown) {
    if (error instanceof Error) {
      throw new ApiError(error.message)
    }
    throw new ApiError('chat stream failed')
  } finally {
    reader.releaseLock()
  }
}
