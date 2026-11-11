"""Measure how often the translator keeps glossary terms, with the glossary in the prompt and without.

Uses the production prompt, placeholder masking and validation from query_language.
Spark (quick 8B, loopback only):
    python3 -m trainer.multilingual.measure_glossary --base-url http://127.0.0.1:18001/v1 --model shipcoat-quick-8b --out glossary_8b.csv
Development host (local weights, no vLLM regex constraint):
    python3 -m trainer.multilingual.measure_glossary --hf /models/Qwen3-8B --out glossary_8b_hf.csv
"""
from __future__ import annotations

import argparse
import csv
import json
from collections import defaultdict
from unittest import mock

import httpx

from trainer.demo_api import query_language as ql
from trainer.multilingual.glossary import missing_terms, term_pairs

KOREAN = [  # answer direction: Korean answer -> question language
    "에폭시 프라이머의 권장 건도막 두께는 125 µm입니다.",
    "건도막 두께가 최소 기준보다 얇으면 덧칠이 필요합니다.",
    "도장 중에는 습도막 두께를 게이지로 수시로 확인하세요.",
    "습도막 두께 150 µm는 건도막 두께 약 100 µm에 해당합니다.",
    "에어리스 스프레이는 펌프 압력을 충분히 높게 유지해야 합니다.",
    "에어리스 스프레이 작업 전에 표면처리 상태를 점검하세요.",
    "최대 재도장 간격을 넘기면 표면을 다시 거칠게 만들어야 합니다.",
    "20 °C에서 최대 재도장 간격은 7일입니다.",
    "표면처리가 불량하면 도막이 쉽게 벗겨집니다.",
    "표면처리 후 4시간 안에 첫 도장을 시작하세요.",
]
FOREIGN = {  # question direction: question -> Korean, one per term
    "en": ["What dry film thickness is recommended for the primer?", "How do I check wet film thickness while painting?",
           "What pressure does airless spray need?", "What is the maximum over coating interval at 20 °C?",
           "What surface preparation is needed before painting?"],
    "zh": ["底漆推荐的干膜厚度是多少？", "涂装时如何检查湿膜厚度？", "无气喷涂需要多大的压力？",
           "20 °C时最大覆涂间隔是多少？", "涂装前需要什么样的表面处理？"],
    "ru": ["Какая толщина сухой плёнки рекомендуется для грунтовки?", "Как проверить толщину мокрой плёнки во время окраски?",
           "Какое давление нужно для безвоздушного распыления?", "Каков максимально допустимый межслойный интервал при 20 °C?",
           "Какая подготовка поверхности нужна перед окраской?"],
    "vi": ["Chiều dày màng sơn khô khuyến nghị cho sơn lót là bao nhiêu?", "Làm sao kiểm tra chiều dày màng sơn ướt khi sơn?",
           "Phun sơn áp lực cao cần áp suất bao nhiêu?", "Thời gian cách lớp tối đa ở 20 °C là bao lâu?",
           "Cần chuẩn bị bề mặt như thế nào trước khi sơn?"],
    "id": ["Berapa ketebalan cat kering yang direkomendasikan untuk primer?", "Bagaimana memeriksa ketebalan cat basah saat mengecat?",
           "Berapa tekanan yang dibutuhkan airless spray?", "Berapa interval pengecatan maksimum pada 20 °C?",
           "Persiapan permukaan apa yang diperlukan sebelum mengecat?"],
    "th": ["ความหนาฟิล์มสีแห้งที่แนะนำสำหรับสีรองพื้นคือเท่าไร", "ตรวจความหนาฟิล์มสีเปียกระหว่างพ่นสีอย่างไร",
           "เครื่องพ่นไฮดรอลิค (Airless Spray) ต้องใช้แรงดันเท่าไร", "ระยะเวลาสูงสุดที่ยอมรับได้ในการทาทับที่ 20 °C คือเท่าไร",
           "ต้องทำการเตรียมพื้นผิวอย่างไรก่อนพ่นสี"],
    "uz": ["Praymer uchun quruq plyonka qalinligi qancha bo'lishi kerak?"],
}


def cases():
    for target in FOREIGN:
        for text in KOREAN:
            yield "ko", target, text
    for source, texts in FOREIGN.items():
        for text in texts:
            yield source, "ko", text


def openai_generator(base_url: str, model: str):
    client = httpx.Client(timeout=300)

    def generate(messages, regex):
        # Same request as AgentPipeline.translate_copilot_text, without streaming.
        payload = {"model": model, "messages": messages, "max_tokens": 1024, "temperature": 0.0, "top_p": 1.0,
                   "repetition_penalty": 1.0, "chat_template_kwargs": {"enable_thinking": False},
                   "skip_special_tokens": True, "structured_outputs": {"regex": regex}}
        response = client.post(base_url.rstrip("/") + "/chat/completions", json=payload)
        response.raise_for_status()
        return response.json()["choices"][0]["message"]["content"]
    return generate


def hf_generator(path: str):
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer
    tokenizer = AutoTokenizer.from_pretrained(path)
    model = AutoModelForCausalLM.from_pretrained(path, dtype=torch.bfloat16, device_map="cuda")

    def generate(messages, regex):
        ids = tokenizer.apply_chat_template(messages, add_generation_prompt=True, enable_thinking=False,
                                            return_tensors="pt", return_dict=True).to(model.device)
        with torch.no_grad():
            out = model.generate(**ids, max_new_tokens=1024, do_sample=False, temperature=None, top_p=None,
                                 top_k=None, repetition_penalty=1.0)
        return tokenizer.decode(out[0][ids["input_ids"].shape[1]:], skip_special_tokens=True)
    return generate


class _Echo:
    """Lets translate_text run every production check on an output we already have."""
    def __init__(self, text):
        self.text = text

    def translate_copilot_text(self, *_args, **_kwargs):
        return self.text


def measure(generate, source, target, text, glossary):
    masked, prefix, tokens = ql.protect_translation(text)
    with mock.patch.object(ql, "term_pairs", term_pairs if glossary else (lambda *_: [])):
        messages = ql.translation_messages(masked, source_language=source, target_language=target)
    raw = generate(messages, ql.translation_regex(prefix, tokens))
    translated, error = "", ""
    try:
        translated = ql.restore_translation(ql.parse_translation(raw), prefix, tokens)
        with mock.patch.object(ql, "missing_terms", lambda *_: []):
            ql.translate_text(_Echo(translated), text, source_language=source, target_language=target)
    except ql.TranslationUnavailableError as exc:
        error = str(exc)
    expected = [pair["target"] for pair in term_pairs(text, source, target)]
    missed = missing_terms(text, translated, source, target) if translated else expected
    return {"source": source, "target": target, "glossary": "on" if glossary else "off", "text": text,
            "translation": translated or raw, "other_error": error, "expected_terms": len(expected),
            "kept_terms": len(expected) - len(missed), "missed": "; ".join(missed)}


def summarize(rows):
    summary = defaultdict(lambda: [0, 0, 0, 0])  # kept, expected, delivered, total
    for row in rows:
        cell = summary[(f"{row['source']}->{row['target']}", row["glossary"])]
        cell[0] += row["kept_terms"]
        cell[1] += row["expected_terms"]
        # With the glossary on, a missed term blocks the answer; off, it ships as is (pre-rc3 behaviour).
        cell[2] += not row["other_error"] and (row["glossary"] == "off" or not row["missed"])
        cell[3] += 1
    return {f"{d} {g}": {"terms_kept": f"{c[0]}/{c[1]}", "delivered": f"{c[2]}/{c[3]}"}
            for (d, g), c in sorted(summary.items())}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--base-url")
    parser.add_argument("--model", default="shipcoat-quick-8b")
    parser.add_argument("--hf", help="local model directory instead of an OpenAI-compatible server")
    parser.add_argument("--out", required=True)
    args = parser.parse_args()
    generate = hf_generator(args.hf) if args.hf else openai_generator(args.base_url, args.model)
    rows = [measure(generate, s, t, text, g) for s, t, text in cases() for g in (False, True)]
    with open(args.out, "w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    print(json.dumps(summarize(rows), ensure_ascii=False, indent=1))


if __name__ == "__main__":
    main()
