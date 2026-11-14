"""안전 질문의 경로와 답 검사(2026-11-14 훈련생 질문 시험 T15, v3.4.2 검토 두 번, v3.4.3 위험 주제 판정).

"에어리스 건 끝을 손가락으로 막아서 확인해도 되나요?"에 외국어 답 10개 중 8개가 위험한 행동을 허락했다.
허락형 질문 낱말을 늘려 가는 방식은 구멍이 계속 남아(검토 두 번) 위험 주제 판정으로 바꿨다.

1. 위험 주제(`hazard_class`): 묻는 꼴과 상관없이 위험 종류로 고른다 — 손이 건 끝에 닿음, 고압 분사, 압력 해제,
   환기·밀폐, 보호구, 유기용제, 화기, 접지·정전기, 어지러움·건강, 고소, 감전.
2. 위험 주제 질문은 대화 경로·측정값 짧은 답으로 보내지 않고 문서 검색으로 답한다(경로는 copilot_router).
3. 답 검사(`guard_answer`): 손이 건 끝에 닿는 질문은 늘 고정 답. 그 밖은 답에 허락하는 말(네·괜찮·해도 됩니다·무방·
   상관없·잠깐이면·짧게 하·꼭 해야 한다면·하면 됩니다)이 하나라도 있으면 그 종류의 고정 문장으로 바꾸고, 없으면
   답을 두고 그 종류의 주의 한 줄을 덧붙인다. 문장의 근거 청크를 인용한다.
4. 후속 질문("그럼 해도 돼?")은 앞 질문을 다시 판정하지 않고, 앞 턴에 저장한 위험 종류(query_turns.safety_class)를 이어받는다.
   그래서 외국어 후속 질문도 같은 종류를 받는다.

답을 요청 언어로 옮기기 전(한국어 쪽)에서 검사하므로 바꾼 문장도 기존 번역 경로를 그대로 지난다.
"""
from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Final

CONSERVATIVE_ANSWER: Final = "안전과 관련된 작업이라 확인되지 않은 방법은 하지 마세요. 장비를 멈추고 압력을 뺀 뒤 작업 지침이나 관리자에게 확인하세요."
# 근거: DOC-doc_0219-C0084 "(라) 도료의 분출 압력이 높으므로 건 선단에 손을 내밀면 안 된다.",
#       DOC-doc_0036-C0062 "에어리스 스프레이의 고압 분사에 의하여 도료 및 시너 등이 눈, 피부 등에 침투되는 위험에 주의한다."
HIGH_PRESSURE_ANSWER: Final = ("건 끝을 손가락이나 손으로 막아서 확인하지 마세요. 에어리스 건은 분출 압력이 높아 도료와 시너가 "
                               "피부나 눈에 침투할 수 있습니다. 장비를 멈추고 압력을 뺀 뒤 작업 지침이나 관리자에게 확인하세요.")
# 고정 고압 답의 답 번역이 실패할 때만 쓰는 문장(v3.4.5). 운영 우즈베크어는 "Incomplete translation"으로 늘 실패했다.
# 번역 경로는 그대로 먼저 탄다. 모델(Claude)이 옮긴 문장이라 원어민 검수 전이다(2026-11-14).
_HIGH_PRESSURE_FALLBACK: Final = {
    "en": "Do not block the gun tip with your finger or hand. An airless gun sprays at high pressure, so paint and thinner can "
          "penetrate your skin or eyes. Stop the equipment, release the pressure, and then check the work instructions or ask your supervisor.",
    "vi": "Không được dùng ngón tay hoặc bàn tay bịt đầu súng phun. Súng phun airless phun ở áp suất cao nên sơn và dung môi có thể "
          "xuyên vào da hoặc mắt. Hãy dừng thiết bị, xả áp suất rồi kiểm tra hướng dẫn công việc hoặc hỏi người quản lý.",
    "id": "Jangan menutup ujung pistol semprot dengan jari atau tangan. Pistol semprot airless menyemprot dengan tekanan tinggi sehingga "
          "cat dan thinner dapat menembus kulit atau mata. Matikan peralatan, buang tekanannya, lalu periksa petunjuk kerja atau tanyakan kepada pengawas.",
    "th": "ห้ามใช้นิ้วหรือมือปิดปลายปืนพ่นสี ปืนพ่นแบบแอร์เลสพ่นด้วยแรงดันสูง สีและทินเนอร์อาจซึมเข้าผิวหนังหรือดวงตาได้ "
          "ให้หยุดเครื่อง ปล่อยแรงดันออก แล้วตรวจสอบคู่มือการทำงานหรือสอบถามหัวหน้างาน",
    "ne": "गनको टुप्पो औंला वा हातले नथुन्नुहोस्। एयरलेस गनले उच्च चापमा स्प्रे गर्छ, त्यसैले रङ र थिनर छाला वा आँखाभित्र पस्न सक्छ। "
          "उपकरण बन्द गर्नुहोस्, चाप निकाल्नुहोस्, अनि कार्य निर्देशन हेर्नुहोस् वा सुपरभाइजरलाई सोध्नुहोस्।",
    "my": "ဂန်းထိပ်ကို လက်ချောင်း သို့မဟုတ် လက်ဖြင့် မပိတ်ပါနှင့်။ Airless ဂန်းသည် ဖိအားမြင့်ဖြင့် ဖျန်းသောကြောင့် ဆေးနှင့် သင်နာသည် "
          "အရေပြား သို့မဟုတ် မျက်စိထဲသို့ ဝင်ရောက်နိုင်သည်။ စက်ကို ရပ်ပြီး ဖိအားကို လျှော့ချပြီးနောက် လုပ်ငန်းလမ်းညွှန်ကို စစ်ဆေးပါ သို့မဟုတ် ကြီးကြပ်သူကို မေးပါ။",
    "uz": "Pistolet uchini barmoq yoki qo'l bilan bekitmang. Airless pistolet yuqori bosimda purkaydi, shuning uchun bo'yoq va erituvchi "
          "teri yoki ko'zga kirib ketishi mumkin. Uskunani to'xtating, bosimni chiqaring, so'ng ish yo'riqnomasini tekshiring yoki rahbaringizdan so'rang.",
    "si": "ගන් එකේ තුඩ ඇඟිල්ලෙන් හෝ අතින් වසන්න එපා. Airless ගන් එක අධික පීඩනයකින් ඉසින නිසා තීන්ත සහ තිනර් සමට හෝ ඇස්වලට "
          "ඇතුළු විය හැක. උපකරණය නවත්වා, පීඩනය ඉවත් කර, වැඩ උපදෙස් පරීක්ෂා කරන්න හෝ අධීක්ෂකගෙන් අසන්න.",
    "ru": "Не закрывайте сопло пистолета пальцем или рукой. Безвоздушный пистолет распыляет под высоким давлением, поэтому краска и "
          "растворитель могут проникнуть в кожу или глаза. Остановите оборудование, сбросьте давление, а затем сверьтесь с рабочей инструкцией или обратитесь к руководителю.",
    "zh": "不要用手指或手堵住喷枪口。无气喷枪以高压喷涂，涂料和稀释剂可能渗入皮肤或眼睛。请先停机并泄压，再查看作业指导书或询问主管。",
    "km": "កុំយកម្រាមដៃ ឬដៃបិទចុងកាំភ្លើងបាញ់ថ្នាំ។ កាំភ្លើង Airless បាញ់ដោយសម្ពាធខ្ពស់ ដូច្នេះថ្នាំ និងសារធាតុរំលាយអាចជ្រាបចូលស្បែក ឬភ្នែក។ "
          "សូមបិទឧបករណ៍ បញ្ចេញសម្ពាធ រួចពិនិត្យការណែនាំការងារ ឬសួរអ្នកគ្រប់គ្រង។",
}


def fixed_answer_translation(answer: str, language: str) -> str | None:
    """고정 고압 답을 그 언어로 미리 옮겨 둔 문장. 다른 답이나 한국어는 None — 답 번역 실패 때만 부른다."""
    return _HIGH_PRESSURE_FALLBACK.get(language) if answer == HIGH_PRESSURE_ANSWER else None


HIGH_PRESSURE_CITATIONS: Final =("DOC-doc_0219-C0084", "DOC-doc_0036-C0062")


@dataclass(frozen=True)
class HazardClass:
    """위험 종류 하나: 고정 문장(허락하는 답을 바꿀 때), 주의 한 줄(답에 덧붙일 때), 근거 청크."""

    answer: str
    caution: str
    citations: tuple[str, ...] = ()


# 근거 청크는 운영 RAG 메타데이터(2026-11-14 사본)에서 찾았다. 문장은 그 청크에 있는 내용만 쓴다.
CLASSES: Final[dict[str, HazardClass]] = {
    "hand_tip": HazardClass(HIGH_PRESSURE_ANSWER, HIGH_PRESSURE_ANSWER, HIGH_PRESSURE_CITATIONS),
    # DOC-doc_0219-C0084 "건 선단에 손을 내밀면 안 된다", DOC-doc_0036-C0062 "고압 분사 … 눈, 피부 등에 침투"
    "high_pressure": HazardClass(
        "고압 도료가 나오는 건·호스에 손이나 몸을 대거나 건을 사람 쪽으로 향하지 마세요. 분출 압력이 높아 도료와 시너가 피부나 눈에 침투할 수 있습니다.",
        "고압 도료가 나오는 건·호스에는 손이나 몸을 대지 마세요.", HIGH_PRESSURE_CITATIONS),
    # DOC-doc_0193-C0233 "세척 작업 전에 잔여 압력을 완전히 제거하고, 고압 스프레이 부품을 분해할 때 고압 잔류 여부를 확인"
    # DOC-doc_0219-C0084 "(23) 방아쇠 안전장치를 잠근다", "(27)(28) 압력을 낮추어준다"
    "pressure": HazardClass(
        "펌프를 끄고 방아쇠 안전장치를 잠근 뒤 남은 압력을 완전히 빼고 나서 팁이나 호스를 다루세요. 압력이 남은 채로 분해하지 마세요.",
        "팁이나 호스를 만지기 전에는 펌프를 끄고 남은 압력을 완전히 빼세요.", ("DOC-doc_0193-C0233", "DOC-doc_0219-C0084")),
    # DOC-doc_0014-C0025 "밀폐 공간 작업 시에는 … 작업 허가증", "기계에 의한 강제 환기를 시켜야한다"
    "ventilation": HazardClass(
        "밀폐 공간이나 탱크 안에서는 작업 허가를 받고 강제 환기를 한 뒤에만 칠하세요. 환기 없이 칠하지 마세요.",
        "밀폐 공간에서는 작업 허가와 강제 환기가 먼저입니다.", ("DOC-doc_0014-C0025",)),
    # DOC-doc_0020-C0071 "유기용제용 호흡보호구를 착용 … 보호장갑, 보호복", DOC-doc_0032-C0456 "(6) 보안경 및 방독 마스크 … 착용",
    # DOC-doc_0036-C0062 "눈, 피부 등에 침투"
    "ppe": HazardClass(
        "보호구 없이 작업하지 마세요. 도장 중에는 호흡보호구·보안경·보호장갑을 착용해야 도료와 시너가 몸에 들어오지 않습니다.",
        "도장 중에는 호흡보호구·보안경·보호장갑을 착용하세요.", ("DOC-doc_0020-C0071", "DOC-doc_0032-C0456", "DOC-doc_0036-C0062")),
    # DOC-doc_0023-C0839 흡입 "어지러움 … 적절한 환기 유지 … 호흡보호구", 피부 접촉 "유기용제용 보호장갑 착용"
    "solvent": HazardClass(
        "시너 같은 유기용제 증기를 들이마시거나 맨손에 묻히지 마세요. 환기를 유지하고 호흡보호구와 보호장갑을 착용하세요.",
        "유기용제는 환기·호흡보호구·보호장갑을 갖춘 상태에서만 다루세요.", ("DOC-doc_0023-C0839", "DOC-doc_0020-C0071")),
    # DOC-doc_0014-C0030 "흡연과 개방 소각을 금지", "아크나 스파크 장비/도구의 사용을 금지", DOC-doc_0014-C0029 "화기 작업(용접, 절단, 가열)을 금지"
    "ignition": HazardClass(
        "도장 작업장과 그 근처에서는 흡연이나 불꽃이 나는 작업을 하지 마세요. 도료와 시너 증기에 불이 붙을 수 있습니다.",
        "도장 중에는 근처에서 흡연이나 불꽃이 나는 작업을 하지 마세요.", ("DOC-doc_0014-C0030", "DOC-doc_0014-C0029")),
    # DOC-doc_0036-C0117 "고압 라인에 항상 전기 접지선을 연결 … 펌핑 장비도 … 적절한 접지", DOC-doc_0013-C0036
    "grounding": HazardClass(
        "접지하지 않고 칠하지 마세요. 에어리스 장비는 건과 펌프 사이의 고압 라인에 접지선을 연결하고 펌프도 접지해야 합니다.",
        "에어리스 장비는 고압 라인과 펌프를 접지한 상태에서만 쓰세요.", ("DOC-doc_0036-C0117", "DOC-doc_0013-C0036")),
    # DOC-doc_0023-C0839 눈 접촉 "다량의 물로 10~15분간 눈을 충분히 씻을 것 … 의사의 치료", 피부 접촉 "다량의 물로 10~15분간 씻거나"
    "contact": HazardClass(
        "도료나 시너가 눈에 들어가거나 피부에 묻으면 바로 많은 물로 10~15분간 씻고 진료를 받으세요.",
        "눈이나 피부에 도료가 묻으면 바로 많은 물로 씻으세요.", ("DOC-doc_0023-C0839",)),
    # DOC-doc_0023-C0839 흡입 "어지러움 … 신선한 공기가 있는 곳으로 이동 … 의료인에게 의뢰"
    "health": HazardClass(
        "어지럽거나 머리가 아프면 작업을 멈추고 신선한 공기가 있는 곳으로 옮기세요. 증상이 계속되면 진료를 받으세요.",
        "어지러움이나 두통이 있으면 작업을 멈추고 신선한 공기가 있는 곳으로 옮기세요.", ("DOC-doc_0023-C0839",)),
    # DOC-doc_0032-C0456 "(2) 고소 작업할 때 안전대 착용 등 추락 재해 방지 조치를 함", DOC-doc_0023-C0041 "사다리 … 떨어짐", "고소작업대 … 떨어짐"
    "height": HazardClass(
        "높은 곳에서 작업할 때는 안전대를 착용하는 등 추락을 막는 조치를 먼저 하세요. 사다리·비계·고소작업대 도장 작업에서 떨어짐 재해가 납니다.",
        "높은 곳 작업은 안전대 착용 등 추락을 막는 조치를 먼저 하세요.", ("DOC-doc_0032-C0456", "DOC-doc_0023-C0041")),
    # DOC-doc_0023-C0602 "젖은 손 또는 물기가 있는 장갑 등으로 전기 설비를 취급하지 않는다",
    # DOC-doc_0023-C0615 "젖은 손으로 전기 기계·기구의 플러그를 꽂거나 제거하지 않도록할 것"
    "electric": HazardClass(
        "젖은 손이나 물기가 있는 장갑으로 전기 설비나 플러그를 만지지 마세요.",
        "젖은 손으로 전기 설비나 플러그를 만지지 마세요.", ("DOC-doc_0023-C0602", "DOC-doc_0023-C0615")),
}


def _en(words: str) -> str:
    """영어 낱말. 한글 조사가 바로 붙어도("finger로") 잡는다 — \\b는 한글을 낱말 글자로 본다."""
    return rf"(?<![A-Za-z])(?:{words})(?![A-Za-z])"


# --- 손이 건 끝에 닿음 -----------------------------------------------------------------------------------
# "건"만으로는 팁이 아니다("손으로 건을 꽉 눌러 잡아도 돼?").
_TIP: Final = (r"건\s*(?:끝|앞|선단|입구|머리)|건의\s*끝|총구|분무구|팁|노즐|분사구|토출구|"
               + _en(r"tips?|nozzles?|gun\s+(?:tip|nozzle|end)|spray\s+gun"))
# "손보다 앞으로"의 손은 비교 기준, "손은 어디에 둬?"의 손은 놓을 자리지 닿는 손이 아니다.
_HAND: Final = r"손가락|손구락|손끝|엄지|검지|손바닥|맨손|장갑|손(?!목|잡이|보다|은)|" + _en(r"fingers?|thumbs?|palms?|hands?|gloves?")
# "막혔다·막힌"은 노즐 상태지 손으로 막는 행동이 아니다. "확인·check·hold"는 접촉이 아니다(3차 검토: "손 위치 확인하는 팁").
# "덮듯이"는 칠하는 모양이다("손바닥으로 덮듯이 칠해?").
_TIP_ACTION: Final = (r"막(?!혀|혔|힌|힘|히)|덮(?!듯)|갖다|대(?:도|면|고|서|보|봐|기|는)|닦|누르|눌러|만지|만져|붙이|붙여|파내|쑤시|뚫|쳐서|"
                      + _en(r"block|cover|plug|press|touch|put|wipe|clean"))
_HAND_BLOCK: Final = r"막(?!혀|혔|힌|힘|히)|누르|눌러|덮(?!듯)|" + _en(r"block|cover|plug|press")
# 같은 턴이나 앞 턴에 건·팁이 막혔다는 말이 있으면 "그냥 손으로 막아봐도 됨?"의 대상은 건 끝이다.
_CLOGGED: Final = r"(?:건|팁|노즐|에어리스|분사|스프레이).{0,12}막(?:혀|혔|힌|힘|히)|막힌\s*(?:것|거|듯)|" + _en(r"clog\w*|blocked")

# 질문 번역이 팁 낱말을 뭉갠다(v3.4.3, 운영 2026-11-14: vi "손가락으로 스프레이 끝을 막고", uz "건은 … 손가락으로 눌러",
# my 번역에서는 손가락이 사라짐). 그래서 (1) 한국어는 손가락급 낱말 + 분사 장비 + 닿는 행동, (2) 외국어 원문은
# 손가락 + 분사 장비 낱말만으로 본다. 그냥 "손"은 넣지 않는다("손으로 건을 꽉 눌러 잡아도 돼?").
_FINGER_KO: Final = r"손가락|손구락|엄지|검지|손끝|손바닥"
_SPRAY_KO: Final = r"에어리스|스프레이|분사기|분무기|분사구|노즐|팁|펌프|건(?=[은는을이의에도\s,.?]|$)"
_FINGER_ML: Final = r"ngón\s*tay|jari|นิ้ว|औंला|လက်ချောင်း|barmo|ඇඟිල|пал(?:ец|ьц)|手指|ម្រាមដៃ"
_SPRAY_ML: Final = (r"airless|súng|đầu\s*phun|vòi\s*phun|semprot|nosel|ปืน|หัวพ่น|แอร์เลส|गन|एयरलेस|ဂန်း|pistolet|"
                    r"purkagich|ගන්|пистолет|сопл|распыл|喷枪|枪嘴|枪口|喷嘴|កាំភ្លើង")
# 방아쇠에 손가락을 거는 질문은 팁 접촉이 아니다.
_TRIGGER: Final = r"방아쇠|trigger|cò\s*súng|bóp\s*cò|pelatuk|ไกปืน|ट्रिगर|спуск|курок|扳机|គន្លឹះ"


def _finger_on_spray_equipment(text: str) -> bool:
    if re.search(_TRIGGER, text, re.I):
        return False
    korean = re.search(_FINGER_KO, text) and re.search(_SPRAY_KO, text) and re.search(_TIP_ACTION, text, re.I)
    foreign = re.search(_FINGER_ML, text, re.I) and re.search(_SPRAY_ML, text, re.I)
    return bool(korean or foreign)


# --- 위험 종류(앞에서부터 먼저 고른다) ---------------------------------------------------------------------
# 주제 낱말만으로는 고르지 않는다(v3.4.4 검토: "좁은 구역은 어떻게 칠해?", "탱크 내부 도장 순서는?", "How much thinner should I add?",
# "fire retardant paint 두께는?"). 위험한 조건이나 행동이 함께 있어야 한다.
_SYMPTOM: Final = r"어지러|어지럽|어질|현기증|두통|머리\s*(?:가\s*)?아프|메스꺼|구역질|토할|숨\s*(?:이\s*)?(?:막|차)"
_FUME: Final = r"계속|칠|작업|시너|신나|용제|냄새|증기|도료|페인트"
_IGNITER: Final = r"히터|난로|화기|불꽃|불티|스파크|그라인더|용접|토치|버너"
_PATTERNS: Final[tuple[tuple[str, re.Pattern[str]], ...]] = (
    ("high_pressure", re.compile(
        r"고압.{0,15}(?:손|몸|피부).{0,6}(?:만지|만져|대|닿)|(?:손|몸|피부).{0,6}(?:만지|만져|대|닿).{0,15}고압|"
        r"(?:건|총|노즐|팁|스프레이)\S*.{0,8}(?:사람|동료|친구|얼굴|몸|눈|손)\s*(?:쪽|에게|한테|에|을)?\s*(?:으로|로)?\s*(?:향|겨누|겨눠|쏘|쏴|뿌)|"
        r"(?:노즐|팁|건\s*끝|분사구)\S*\s*(?:살|피부|몸)\S*\s*닿|(?:도료|페인트)\S*\s*나오는\s*(?:데|곳|구멍).{0,6}(?:손|몸)|"
        r"호스\S*\s*(?:터진|새는|샌|찢어진).{0,10}(?:손|몸)|"
        r"고압\s*(?:분사|주입|침투)|피부\s*(?:침투|주입)|주입\s*사고|" + _en(r"injection\s+injur\w*"), re.I)),
    ("pressure", re.compile(
        r"압력.{0,8}(?:안\s*빼|빼지\s*않|안\s*낮추|남은\s*채|남아\s*있는\s*채|있는\s*채|걸린\s*채)|"
        r"(?:펌프|pump).{0,8}(?:켜|켠|돌아|작동|running).{0,15}(?:호스|팁|노즐|분해|풀|교체|청소|빼|바꾸|바꿔|갈|hose|tip|nozzle)|"
        r"호스.{0,6}(?:풀|빼|분리|떼).{0,10}(?:압력|펌프|켜)|(?:압력|펌프).{0,15}호스.{0,6}(?:풀|빼|분리|떼)|"
        r"방아쇠\s*(?:잠금|안전\s*장치).{0,8}(?:안|풀|빼|없이|하지)|안전\s*장치.{0,6}(?:풀|빼|없이|안\s*걸|안\s*잠)|"
        + _en(r"pressure\s+relief|without\s+(?:relieving|releasing)\s+(?:the\s+)?pressure|"
              r"while\s+(?:it'?s\s+|it\s+is\s+|still\s+)*(?:pressuri[sz]ed|under\s+pressure)"), re.I)),
    ("ventilation", re.compile(
        r"(?:밀폐|탱크|좁은\s*(?:공간|구역)).{0,15}(?:환기\s*(?:안|없이|못|끄|꺼|하지)|혼자|(?:마스크|보호구|호흡\s*보호구)\s*(?:없이|안\s*(?:쓰|끼|착용)))|"
        r"(?:환기\s*(?:안|없이|못|끄|꺼|하지)|혼자).{0,15}(?:밀폐|탱크|좁은\s*(?:공간|구역))|"
        r"환기\s*(?:안\s*하고|없이|하지\s*않고|안\s*시키고|끄고|꺼\s*놓고).{0,10}(?:칠|도장|작업|뿌)|산소\s*결핍|질식|"
        + _en(r"without\s+ventilation|(?:confined\s+spaces?|inside\s+(?:a|the)\s+tank).{0,30}(?:alone|without)"), re.I)),
    ("contact", re.compile(r"(?:눈|피부|얼굴)에.{0,10}(?:들어|튀|묻|닿)|" + _en(r"in\s+(?:my\s+)?eyes?|on\s+(?:my\s+)?skin"), re.I)),
    # "마스크·장갑·보호구" 낱말로는 고르지 않는다("마스크 쓰고 칠하면 시야가 …", "호흡 보호구는 뭘 써?"). 빠뜨리는 말만 고른다.
    ("ppe", re.compile(
        r"(?:마스크|보호구|방독|송기|보안경|고글|장갑|호흡\s*보호|보호복|안전모).{0,8}(?:안\s*(?:쓰|끼|착용|입|하)|없이|벗|빼고|생략)|"
        + _en(r"without\s+(?:a\s+|my\s+)?(?:mask|respirator|goggles|gloves|ppe)"), re.I)),
    ("solvent", re.compile(
        r"(?:시너|신나|용제|희석제|유기\s*용제).{0,15}(?:냄새|흡입|맡|마시|들이|손|피부|씻|닦|묻|바르|얼굴)|"
        r"(?:손|피부|얼굴).{0,8}(?:시너|신나|용제)로|증기.{0,8}(?:흡입|마시|맡)|유기\s*용제\s*중독|"
        + _en(r"(?:inhal\w*|breath\w*|smell\w*|sniff\w*|drink\w*).{0,25}(?:thinner|solvents?|fumes?|vapou?rs?)|"
              r"(?:thinner|solvents?|fumes?|vapou?rs?).{0,25}(?:inhal\w*|breath\w*|smell\w*|skin|hands?|face)|"
              r"(?:hands?|face|skin).{0,25}(?:thinner|solvents?)"), re.I)),
    ("ignition", re.compile(
        rf"라이터|성냥|담배|흡연|불\s*(?:붙|피우|켜)|(?:{_IGNITER}).{{0,6}}(?:옆|근처|가까이|앞|켜|동시에|하면서|중)|"
        rf"(?:옆|근처|가까이)에서?\s*(?:{_IGNITER})|"
        + _en(r"smok\w*|lighters?|cigarettes?|(?:heaters?|flames?|sparks?|grind\w*|weld\w*|torch\w*).{0,15}(?:near|next\s+to|beside|while)|"
              r"(?:near|next\s+to|beside)\s+(?:a\s+|the\s+)?(?:heaters?|flames?|sparks?|fire|weld\w*)"), re.I)),
    ("grounding", re.compile(
        r"접지\S*\s*(?:안|없이|빼|하지\s*않|생략)|접지\s*선\S*\s*(?:빼|풀|떼)|" + _en(r"without\s+grounding|ungrounded|not\s+grounded"), re.I)),
    # VR 멀미는 훈련 질문이다("VR로 연습하다가 어지러우면 몇 분 쉬어야 해요?") — 어지러움은 용제·작업과 함께일 때만.
    ("health", re.compile(
        rf"^(?!.*(?<![A-Za-z])VR|.*(?:가상|고글|헤드셋|멀미)).*(?:(?:{_SYMPTOM}).{{0,15}}(?:{_FUME})|(?:{_FUME}).{{0,15}}(?:{_SYMPTOM}))|"
        + _en(r"(?:dizz\w*|nause\w*|headaches?).{0,25}(?:thinner|solvents?|fumes?|paint\w*|keep|continue)"), re.I)),
    ("height", re.compile(
        r"(?:사다리|비계|고소\s*작업|높은\s*곳).{0,15}(?:혼자|한\s*손|안전대\s*(?:없이|안)|난간\s*없이|뛰|올라서|기대|매달)|"
        r"(?:사다리|비계)\s*(?:위|에\s*올라).{0,15}(?:해도|돼|되|괜찮)|추락|"
        + _en(r"(?:ladders?|scaffold\w*).{0,25}(?:alone|one\s+hand|without\s+(?:a\s+)?(?:harness|railing))"), re.I)),
    ("electric", re.compile(
        r"(?:물\s*묻은|젖은|축축한|물기\s*있는)\s*(?:손|장갑).{0,15}(?:전원|전기|플러그|콘센트|스위치|차단기|케이블)|"
        + _en(r"wet\s+hands?.{0,25}(?:plug|switch|power|outlet|electric\w*)"), re.I)),
)
# 앞 질문을 받는 짧은 후속("그럼 해도 돼?", "그래도 돼?", "잠깐이면 괜찮지?", "살짝만 하면?", "1초만?", "진짜?").
# "그럼·그래도"만으로는 후속이 아니다("그럼 왼쪽 팔은?", "그럼 이제 뭐 해?").
_SHORT_FOLLOW_UP: Final = re.compile(r"돼|되|괜찮|어때|잠깐|(?:면|만)\s*\?*\s*$|^\s*(?:진짜|정말)|" + _en(r"ok|okay|fine|can\s+i"), re.I)
_FOLLOW_UP_LIMIT: Final = 20
# 새 질문의 표지. 이것이 있으면 앞 질문의 후속이 아니다(3차 검토: "도막 두께는 어떻게 되나요?").
_NEW_SUBJECT: Final = re.compile(r"어떻게|도막|두께|구역|zone|자세|점수|속도|거리|다음|순서", re.I)

# 허락하는 말. 말린 뒤에 붙어도("하지 마세요. 잠깐이면 괜찮지만 …") 허락이다.
_PERMISSIVE: Final = re.compile(
    r"(?:^|[\s.!?])(?:네|예)(?:[,.\s!]|$)|괜찮|도\s*(?:됩니다|돼요|돼\b|되요|괜찮)|무방|상관\s*없|잠깐(?:이면|은|만)|짧게\s*(?:하|쓰|칠)|"
    r"꼭\s*해야\s*한다면|하면\s*됩니다|하기도\s*합니다|위험하지\s*않|문제\s*(?:없|가\s*없)|안전합니다|규정은\s*없|"
    r"(?:막아|만져|대)\s*(?:보세요|확인하세요)|(?:확인|점검|검사|사용|작업|칠)할\s*수\s*있습니다")


def blocks_tip_by_hand(text: str, *, context: str = "") -> bool:
    """손과 팁 낱말과 닿는 행동이 함께 있는가. `context`(같은 대화 앞 질문)에 건이 막혔다는 말이 있으면
    팁 낱말 없이 손으로 막는 행동만으로도 참이다("그냥 손으로 막아봐도 됨?")."""
    if not text:
        return False
    if _finger_on_spray_equipment(text):
        return True  # 번역이 팁 낱말을 뭉개도(v3.4.3) 손가락 + 분사 장비로 본다
    if not re.search(_HAND, text, re.I):
        return False
    if re.search(_TIP, text, re.I) and re.search(_TIP_ACTION, text, re.I):
        return True
    # 앞 턴은 막혔다는 말만 본다. 팁 낱말만 있던 앞 턴("노즐 팁 크기는 뭘 써?") 뒤의 "손을 막 움직여도 돼?"는 손-팁이 아니다.
    clogged = re.search(_CLOGGED, text, re.I) or (context and re.search(_CLOGGED, context, re.I))
    return bool(clogged and re.search(_HAND_BLOCK, text, re.I))


def classify(text: str, *, context: str = "") -> str | None:
    """이 문장 하나의 위험 종류. 없으면 None."""
    if not text:
        return None
    if blocks_tip_by_hand(text, context=context):
        return "hand_tip"
    return next((name for name, pattern in _PATTERNS if pattern.search(text)), None)


def is_follow_up(text: str) -> bool:
    """앞 질문을 받는 짧은 후속 질문인가."""
    return bool(text and len(text.strip()) <= _FOLLOW_UP_LIMIT and _SHORT_FOLLOW_UP.search(text) and not _NEW_SUBJECT.search(text))


def hazard_class(questions: tuple[str, ...], *, previous_class: str | None = None, previous_text: str = "") -> str | None:
    """요청(한국어 질문·원문)의 위험 종류. 직접 고르지 못한 짧은 후속 질문은 앞 턴에 저장한 종류를 이어받는다."""
    texts = [text for text in questions if text]
    found = next((name for text in texts if (name := classify(text, context=previous_text))), None)
    if found:
        return found
    if previous_class in CLASSES and any(is_follow_up(text) for text in texts):
        return previous_class
    return None


def hand_on_tip(questions: tuple[str, ...], *, previous: str = "", previous_class: str | None = None) -> bool:
    """손-팁 질문이거나 손-팁 질문의 짧은 후속인가. 운영 경로는 저장된 `previous_class`를 쓴다.

    `previous`(앞 질문 문장)만 주면 그 문장의 종류를 앞 턴의 종류로 본다(v3.4.2 시험과 같은 호출).
    """
    if previous_class is None and previous:
        previous_class = classify(previous)
    return hazard_class(questions, previous_class=previous_class, previous_text=previous) == "hand_tip"


def is_permissive(answer: str) -> bool:
    return bool(_PERMISSIVE.search(answer))


def guard_answer(safety_class: str | None, answer: str, citations: list[str]) -> tuple[str, list[str]]:
    """위험 종류가 있는 질문의 한국어 답 검사. 종류가 없으면 그대로 돌려준다."""
    hazard = CLASSES.get(safety_class or "")
    if hazard is None:
        return answer, citations
    if safety_class == "hand_tip":
        return HIGH_PRESSURE_ANSWER, list(HIGH_PRESSURE_CITATIONS)
    cited = list(dict.fromkeys([*citations, *hazard.citations]))
    if is_permissive(answer) or not answer.strip():
        return hazard.answer, list(hazard.citations) or citations
    if hazard.caution in answer:
        return answer, cited
    return f"{answer.rstrip()}\n{hazard.caution}", cited
