"""한자 토큰 차단: 규칙, 실제 어휘 실측, 그리고 HF generate 가 그 차단을 실제로 지키는지."""

from __future__ import annotations

import re
from pathlib import Path

import pytest

from trainer.scripts.hanzi_guard import _byte_decoder, hanzi_token_ids, suppress_hanzi

_ENCODER = {b: ch for ch, b in _byte_decoder().items()}


def _piece(text: str) -> str:
    """원시 UTF-8 바이트를 바이트 단위 BPE 토큰 문자열로."""
    return "".join(_ENCODER[b] for b in text.encode("utf-8"))


class _FakeTokenizer:
    def __init__(self, vocab: dict[str, int]) -> None:
        self._vocab = vocab
        self._decode = {i: bytes(_byte_decoder()[ch] for ch in p).decode("utf-8", "replace") for p, i in vocab.items()}

    def get_vocab(self) -> dict[str, int]:
        return self._vocab

    def decode(self, token_ids: list[int]) -> str:
        return "".join(self._decode[i] for i in token_ids)


def test_bans_whole_and_partial_hanzi_but_never_hangul_or_ascii() -> None:
    zhong = "中".encode("utf-8")  # E4 B8 AD
    vocab = {
        _piece("中"): 0,                                   # 한 토큰 한자
        "".join(_ENCODER[b] for b in zhong[:1]): 1,         # 드문 한자의 첫 바이트만
        _piece("도막"): 2,
        _piece("thickness"): 3,
        "".join(_ENCODER[b] for b in zhong[1:]): 4,         # 이어지는 바이트만 — 혼자서는 한자를 시작 못 한다
    }

    banned = hanzi_token_ids(_FakeTokenizer(vocab))

    assert banned == [0, 1]


def test_suppress_hanzi_is_a_no_op_without_an_hf_generation_config() -> None:
    """vLLM(OpenAI 호환) 백엔드에는 generation_config 가 없다 — 그 경로는 정규식이 맡는다."""

    class _Remote:
        model = object()
        tokenizer = _FakeTokenizer({_piece("中"): 0})

    assert suppress_hanzi(_Remote()) == 0


def test_hf_generate_honors_suppress_tokens_even_with_explicit_sampling_kwargs() -> None:
    """파이프라인은 generate 에 temperature·top_p 등을 직접 넘긴다. 그래도 generation_config 의 차단이 걸려야 한다.

    무작위 가중치라 막지 않으면 거의 확실히 금지 토큰을 낸다 — 단언이 공허하지 않다.
    """
    torch = pytest.importorskip("torch")
    transformers = pytest.importorskip("transformers")
    torch.manual_seed(0)
    model = transformers.GPT2LMHeadModel(transformers.GPT2Config(n_layer=1, n_embd=16, n_head=2, vocab_size=64))
    allowed = {5, 6, 7}
    model.generation_config.suppress_tokens = [i for i in range(64) if i not in allowed]

    out = model.generate(
        torch.tensor([[5]]),
        max_new_tokens=40,
        do_sample=True,
        temperature=1.0,
        top_p=1.0,
        top_k=0,
        pad_token_id=5,
    )

    assert set(out[0, 1:].tolist()) <= allowed


_QWEN = Path("/models/Qwen3.6-27B")


@pytest.mark.skipif(not (_QWEN / "tokenizer.json").exists(), reason="Qwen3.6 tokenizer is a mounted artifact")
def test_qwen_vocabulary_is_fully_covered_without_touching_hangul() -> None:
    """실측: 한자를 포함하는 토큰은 전부 막히고, 막힌 토큰에 한글은 없다."""
    transformers = pytest.importorskip("transformers")
    tokenizer = transformers.AutoTokenizer.from_pretrained(str(_QWEN))
    banned = set(hanzi_token_ids(tokenizer))
    hanzi = re.compile(r"[一-鿿]")
    hangul = re.compile(r"[가-힣]")

    with_hanzi = {i for i in tokenizer.get_vocab().values() if hanzi.search(tokenizer.decode([i]))}

    assert with_hanzi <= banned
    assert not [i for i in banned if hangul.search(tokenizer.decode([i]))]
