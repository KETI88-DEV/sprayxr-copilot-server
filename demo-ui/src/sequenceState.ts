import type { SequenceUpdate, SequenceZone, ZoneStatus } from './contracts'

export function applySequenceUpdate(
  zones: readonly SequenceZone[],
  update: SequenceUpdate,
): SequenceZone[] {
  const nextIds = new Set(update.next_zone_ids)
  const completedIds = new Set(update.completed_zone_ids)

  return zones.map((zone) => {
    const status: ZoneStatus =
      completedIds.has(zone.zone_id)
        ? 'completed'
        : zone.zone_id === update.current_zone_id
          ? 'current'
          : nextIds.has(zone.zone_id)
            ? 'next'
            : 'pending'
    return { ...zone, status }
  })
}
