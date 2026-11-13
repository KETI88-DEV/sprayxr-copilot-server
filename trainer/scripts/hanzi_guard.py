"""생성 중에 한자가 나오지 못하게 막는다.

Qwen3.6은 한국어 답 중간에 한자로 넘어가는 일이 있다(2026-10-12 데모에서 관측). 문장을 다 만든 뒤
지우면 그 자리의 뜻이 사라지므로, 처음부터 그 토큰을 고르지 못하게 한다.

**어떤 토큰을 막는가.** 바이트 단위 BPE에서 CJK 통합 한자(U+4E00–U+9FFF)는 UTF-8 첫 바이트가
0xE4–0xE9다. 한글 음절(U+AC00–U+D7A3)은 0xEA–0xED라 겹치지 않는다. 그래서 원시 바이트에 0xE4–0xE9가
있는 토큰을 막으면 흔한 한자(한 토큰)와 드문 한자(바이트 토큰 여럿) 모두 시작조차 못 한다. 확장 A·호환
한자처럼 다른 첫 바이트를 쓰는 것은 문자 검사로 덧붙인다.

Qwen3.6-27B 어휘 실측(2026-10-12): 248,077개 중 한자를 포함하는 토큰 55,328개가 첫 바이트 규칙에
**전부** 걸리고(합집합 55,659), 막는 집합에 한글이 섞인 토큰은 **0개**다.

**적용 범위.** HF 백엔드는 `generation_config.suppress_tokens`로 모든 생성에 걸린다. vLLM(OpenAI 호환)
백엔드는 **막지 않는다.** 요청마다 5만 개를 보낼 수 없고, 지식 답 정규식에서 한자 범위를 빼 보았더니
Spark에서 띄어쓰기가 전부 사라져(2026-10-12) 되돌렸다. vLLM 쪽은 서버 측 logits processor 로 해야 하고,
그 전에 Spark 실물에서 확인해야 한다.
"""

from __future__ import annotations

import logging
import re
from typing import Any, Final, Protocol

logger = logging.getLogger(__name__)

# 정규식 문자 클래스 안에 그대로 넣는다(answer_grammar).
HANZI_CLASS: Final = "㐀-䶿一-鿿豈-﫿"
_HANZI: Final = re.compile(f"[{HANZI_CLASS}]")
_CJK_LEAD: Final = range(0xE4, 0xEA)


class _Tokenizer(Protocol):
    def get_vocab(self) -> dict[str, int]: ...
    def decode(self, token_ids: list[int]) -> str: ...


def _byte_decoder() -> dict[str, int]:
    """GPT-2 바이트↔유니코드 매핑의 역(바이트 단위 BPE 토큰 문자열 → 원시 바이트)."""
    bs = list(range(ord("!"), ord("~") + 1)) + list(range(ord("¡"), ord("¬") + 1)) + list(range(ord("®"), ord("ÿ") + 1))
    cs = bs[:]
    extra = 0
    for b in range(256):
        if b not in bs:
            bs.append(b)
            cs.append(256 + extra)
            extra += 1
    return {chr(c): b for b, c in zip(bs, cs)}


def hanzi_token_ids(tokenizer: _Tokenizer) -> list[int]:
    decoder = _byte_decoder()
    banned: set[int] = set()
    for piece, token_id in tokenizer.get_vocab().items():
        if all(ch in decoder for ch in piece) and any(decoder[ch] in _CJK_LEAD for ch in piece):
            banned.add(token_id)
        elif _HANZI.search(tokenizer.decode([token_id])):
            banned.add(token_id)
    return sorted(banned)


def suppress_hanzi(pipeline: Any) -> int:
    """라이브 파이프라인의 HF 모델이 한자 토큰을 고르지 못하게 한다. 막은 토큰 수를 돌려준다(못 막으면 0)."""
    model = getattr(pipeline, "model", None)
    tokenizer = getattr(pipeline, "tokenizer", None)
    generation_config = getattr(model, "generation_config", None)
    if generation_config is None or tokenizer is None:
        logger.info("hanzi guard: no HF generation_config; only regex-constrained answers are guarded")
        return 0
    ids = hanzi_token_ids(tokenizer)
    generation_config.suppress_tokens = ids
    logger.info("hanzi guard: suppressing %d hanzi tokens", len(ids))
    return len(ids)
