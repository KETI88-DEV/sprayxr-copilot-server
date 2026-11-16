"""A model streaming frequently must still stop at its total operation budget."""
import time

import httpx
import pytest

from trainer.scripts.tests.test_openai_backend import _backend, _delta, _sse
from trainer.scripts.openai_backend import OpenAIBackendTimeoutError


def test_frequent_stream_fragments_do_not_reset_the_total_generation_deadline():
    from trainer.scripts.openai_backend import generation_deadline_scope
    class FrequentTokens(httpx.SyncByteStream):
        closed = False
        def __iter__(self):
            for _ in range(40):
                time.sleep(.005)  # Well below the configured seven-second read timeout.
                yield _sse(_delta('x'), done=False)
            yield _sse(_delta(finish_reason='stop'))
        def close(self):
            self.closed = True
    stream = FrequentTokens()
    backend = _backend(lambda request: httpx.Response(200, stream=stream))
    start = time.monotonic()
    try:
        with generation_deadline_scope(.04), pytest.raises(OpenAIBackendTimeoutError):
            backend.complete_chat([{'role':'user','content':'q'}], max_tokens=64, enable_thinking=False)
        assert stream.closed
        assert time.monotonic() - start < .15
    finally:
        backend.close()
