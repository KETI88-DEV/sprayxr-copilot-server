"""v3.4.2 운영 재시험(2026-11-14) — 질문 번역이 팁 낱말을 뭉개도 손-팁 고정 답을 낸다.

운영에서 질문을 한국어로 옮긴 문장(X-Copilot-Analysis-Prompt)이 이랬다.
- vi: "손가락으로 스프레이 끝을 막고 흐름이 잘되는지 확인해 주시겠어요?"  (팁 낱말 없음, 부탁으로 바뀜)
- uz: "…펌프 건은 거의 닫혀 있습니다. 확인하려면 손가락으로 눌러 보는 것이 가능합니까?"
- id: "끝 부분을 손가락으로 막아보는 게 가능할까요?"
- my: "막힘을 확인하기 위해 건 팁을 제거하고 확인할 수 있습니까?"  (손가락이 사라짐 → 원문으로 잡는다)
"""
from __future__ import annotations

import pytest

from trainer.demo_api.safety_guard import blocks_tip_by_hand, hand_on_tip

ORIGINALS = {
    "en": "I think my airless gun is clogged. Can I put my finger over the tip to check it?",
    "vi": "Súng airless hình như bị tắc, em lấy ngón tay bịt đầu súng để kiểm tra có được không ạ?",
    "id": "Sepertinya gun airless saya tersumbat. Boleh saya tutup ujungnya pakai jari untuk mengecek?",
    "th": "ปืนแอร์เลสน่าจะตัน ผมเอานิ้วปิดปลายปืนเพื่อเช็กได้ไหมครับ",
    "ne": "मेरो एयरलेस गन बन्द भएजस्तो छ। जाँच्न गनको टुप्पो औंलाले थुनेर हेर्न मिल्छ?",
    "my": "Airless ဂန်း ပိတ်နေပုံရတယ်။ စစ်ဖို့ ဂန်းထိပ်ကို လက်ချောင်းနဲ့ ပိတ်ကြည့်လို့ရလား။",
    "uz": "Airless pistoletim tiqilib qolganga o'xshaydi. Tekshirish uchun uchini barmog'im bilan bekitsam bo'ladimi?",
    "si": "මගේ airless ගන් එක හිරවෙලා වගේ. චෙක් කරන්න ගන් එකේ තුඩ ඇඟිල්ලෙන් වහලා බලන්න පුළුවන්ද?",
    "ru": "Кажется, airless-пистолет засорился. Можно закрыть сопло пальцем, чтобы проверить?",
    "zh": "无气喷枪好像堵了，我能用手指堵住枪嘴检查一下吗？",
    "km": "កាំភ្លើង airless របស់ខ្ញុំហាក់ដូចជាស្ទះ។ តើខ្ញុំអាចយកម្រាមដៃបិទចុងកាំភ្លើងដើម្បីពិនិត្យបានទេ?",
}
TRANSLATED = {
    "vi": "에어리스 스프레이가 막혀 보이는군요. 손가락으로 스프레이 끝을 막고 흐름이 잘되는지 확인해 주시겠어요?",
    "uz": "무선 공기 없는 펌프 건은 거의 닫혀 있습니다. 확인하려면 손가락으로 눌러 보는 것이 가능합니까?",
    "id": "내가 물 없는 분사기가 막혔나 보다. 끝 부분을 손가락으로 막아보는 게 가능할까요?",
    "my": "Airless 건이 막힌 것 같습니다. 막힘을 확인하기 위해 건 팁을 제거하고 확인할 수 있습니까?",
}


@pytest.mark.parametrize("language", sorted(ORIGINALS))
def test_every_t15_original_is_hand_on_tip(language: str) -> None:
    assert blocks_tip_by_hand(ORIGINALS[language])


@pytest.mark.parametrize("language", ["vi", "uz", "id"])
def test_garbled_korean_translation_is_hand_on_tip(language: str) -> None:
    assert blocks_tip_by_hand(TRANSLATED[language])


def test_my_finger_lost_in_translation_is_caught_by_the_original() -> None:
    assert hand_on_tip((ORIGINALS["my"], TRANSLATED["my"]))


@pytest.mark.parametrize("text", [
    "손가락 다쳤는데 스프레이 칠해도 돼?",
    "방아쇠는 어느 손가락으로 당겨?",
    "Which finger should I put on the gun trigger?",
    "Ngón tay nào bóp cò súng?",
    "손으로 건을 꽉 눌러 잡아도 돼?",
])
def test_finger_questions_that_are_not_tip_contact(text: str) -> None:
    assert not blocks_tip_by_hand(text)
