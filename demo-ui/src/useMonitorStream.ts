import { useEffect, useState } from 'react'
import type { MonitorEvent } from './simContracts'
import { streamMonitor } from './simMonitorClient'

/** Subscribes to the read-only monitor stream for the caller's lifetime. */
export function useMonitorStream(): readonly MonitorEvent[] {
  const [events, setEvents] = useState<readonly MonitorEvent[]>([])

  useEffect(() => {
    const controller = new AbortController()

    void (async () => {
      try {
        for await (const event of streamMonitor(0, controller.signal)) {
          setEvents((previous) => [...previous, event])
        }
      } catch {
        // Observability only: a dropped monitor stream must never break the
        // console, and the panel keeps whatever it already received.
      }
    })()

    return () => controller.abort()
  }, [])

  return events
}
