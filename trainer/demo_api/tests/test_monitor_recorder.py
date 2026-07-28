from __future__ import annotations

import anyio

from trainer.demo_api.monitor import MonitorRecorder


def test_record_assigns_monotonic_seq_and_snapshot_filters_by_cursor() -> None:
    recorder = MonitorRecorder(maxlen=10)

    first = recorder.record("s1", "start", "input", {"k": 1})
    second = recorder.record("s1", "start", "output", {"k": 2})
    assert (first.seq, second.seq) == (1, 2)

    snapshot = recorder.snapshot(cursor=1, limit=10)
    assert [event.seq for event in snapshot.events] == [2]
    assert snapshot.next_cursor == 2
    assert snapshot.dropped_before is None


def test_snapshot_reports_gap_when_cursor_predates_buffer() -> None:
    recorder = MonitorRecorder(maxlen=3)
    for index in range(5):
        _ = recorder.record("s1", "runtime", "input", {"i": index})

    snapshot = recorder.snapshot(cursor=0, limit=10)
    assert [event.seq for event in snapshot.events] == [3, 4, 5]
    assert snapshot.dropped_before == snapshot.events[0].seq
    assert snapshot.next_cursor == 5


def test_snapshot_respects_limit_and_keeps_cursor_when_nothing_new() -> None:
    recorder = MonitorRecorder(maxlen=10)
    for index in range(4):
        _ = recorder.record("s1", "runtime", "output", {"i": index})

    limited = recorder.snapshot(cursor=0, limit=2)
    assert [event.seq for event in limited.events] == [1, 2]
    assert limited.next_cursor == 2

    drained = recorder.snapshot(cursor=4, limit=10)
    assert drained.events == []
    assert drained.next_cursor == 4


def test_subscribe_receives_live_events_and_unsubscribe_stops_them() -> None:
    async def _run() -> tuple[list[int], int]:
        recorder = MonitorRecorder(maxlen=10)
        send, receive = recorder.subscribe()

        _ = recorder.record("s1", "start", "input", {"x": 1})
        _ = recorder.record("s1", "start", "output", {"x": 2})

        seqs: list[int] = []
        with anyio.fail_after(2.0):
            for _ in range(2):
                event = await receive.receive()
                seqs.append(event.seq)

        recorder.unsubscribe(send)
        _ = recorder.record("s1", "session_end", "output", {"x": 3})
        send.close()
        await receive.aclose()
        return seqs, len(recorder.snapshot(cursor=0, limit=10).events)

    seqs, recorded = anyio.run(_run)
    assert seqs == [1, 2]
    # The third event is still recorded; only the live fan-out stopped.
    assert recorded == 3
