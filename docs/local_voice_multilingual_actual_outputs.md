# 7개 언어 실제 모델 출력 — 2026-10-06

실행 모델: `qwen36-27b-shipcoat-v6-1-bf16`. Spark에서 실제 호출한 모델 응답 35건과 배포 API 응답 3건을 기록했다. **답변을 번역·교정·요약하거나 기대 답변으로 대체하지 않았다.** 아래 코드 블록은 응답 문자열 원문이다.

## 실행 조건

- 입력: 기존 질문 예시 문서의 5개 질문 × 7개 언어. 작성했던 기대 답변은 모델에 전달하지 않았다.
- 모델 직접 호출: 로컬 vLLM `/v1/chat/completions`. 각 언어의 질문과 해당 언어로 답하라는 시스템 지시만 전달했다. 검색 문서·대화 이력·번역 예시는 제공하지 않았다.
- 직접 호출 설정: temperature=0, top_p=1, top_k=-1, repetition_penalty=1.0, max_tokens=192, enable_thinking=false. 생성이 제한에 걸린 결과도 재시도하거나 잘라 숨기지 않았다.
- 배포 API: `/query`, knowledge_search, use_rag=Y. 입력·출력 언어를 동일하게 지정했다. 한국어·영어·중국어 각 1건이며, 기존 검색·한국어 생성·번역 경로를 사용했다.
- 직접 호출과 배포 API는 다른 경로다. 보류 언어의 직접 모델 출력이 있다는 이유로 배포 지원을 활성화한 것은 아니다.
- 모든 입력은 텍스트다. 이 기록은 음성 인식 시험이 아니다. 지연시간은 HTTP 요청부터 전체 응답 수신까지 측정한 값이며 p95 벤치마크가 아니다.

시스템 지시의 LANGUAGE 자리에 Korean / English / Simplified Chinese / Vietnamese / Uzbek in Latin script / Burmese / Nepali를 넣었다.

```text
You are a ship-coating assistant. Answer the user's question in {LANGUAGE}, using 1 to 3 sentences. Do not invent measurements or specification limits. If the information is insufficient to make a judgment, explain what information is needed.
```

## 확인된 오류

배포 API의 중국어 Q01 답변에는 `干燥度膜厚度（DFT）`가 출력됐다. 출처 대조표와 실행 용어집의 확인 표기는 `干膜厚度`이므로 용어집을 제공해도 정확한 용어 사용이 강제되지 않은 실제 사례다. 아래 API 응답은 이 오류를 수정하지 않은 원문이다.

Q04는 적용 규격 없이 A-1의 DFT가 80 µm라는 값만 제공한 질문이다. 직접 출력에서 중국어는 최소 75 µm, 우즈벡어는 120–160 µm, 미얀마어는 75 µm ±25%를 임의로 만들어 판정했다. 영어는 대부분의 선박 도장에 통상 허용된다는 근거 없는 일반화를 했고, 네팔어도 기준 없이 허용된다고 답했다. 해당 출력은 수치 판정의 기대 조건을 충족하지 못했다. 이 결과는 검색 문서와 판정 도구가 없는 직접 모델 호출 결과다.

Q05의 우즈벡어 출력에는 건조 시간을 뜻하는 표현 뒤에 DFT를 붙인 `qurish vaqti (DFT)`가 있고, 제품을 지정하지 않았는데도 7–14일이라는 예시 범위를 제시했다. 용어 및 조건 검수가 필요한 실제 출력이다.

따라서 특정 언어의 문장을 생성할 수 있다는 사실을 도장 답변 정확도나 배포 품질 승인으로 해석할 수 없다. 아래에서 실패 응답도 원문 그대로 확인할 수 있다.

## 실제 배포 API 응답

사용자가 배포 서비스에서 받는 응답이다. 아래에는 표시 답변과 한국어 정본, 근거 ID, 현지화 상태를 구분했다.

### API / 한국어

질문: 건조도막두께, DFT가 뭐예요?

HTTP 200 · 전체 15978 ms · 현지화 상태 `unchanged`

표시 답변 원문:

```text
DFT(Dry Film Thickness)는 도료가 완전히 건조한 상태에서의 도막 두께를 의미합니다. 이 값은 도장 품질 평가에서 가장 중요한 기준 중 하나로, 도장이 충분한 방청 성능을 발휘하는지를 판단하는 척도로 사용됩니다.
```

한국어 정본:

```text
DFT(Dry Film Thickness)는 도료가 완전히 건조한 상태에서의 도막 두께를 의미합니다. 이 값은 도장 품질 평가에서 가장 중요한 기준 중 하나로, 도장이 충분한 방청 성능을 발휘하는지를 판단하는 척도로 사용됩니다.
```

근거 ID: `["DOC-doc_0036-C0083", "DOC-doc_0045-C0127"]`

단계 시간: `{"input_translation_ms": 0, "copilot_ms": 15974, "output_translation_ms": 0, "total_ms": 15974}`

실패 사유: `None`

### API / 영어

질문: What is dry film thickness, or DFT?

HTTP 200 · 전체 15598 ms · 현지화 상태 `translated`

표시 답변 원문:

```text
DFT refers to the thickness of the coating remaining after drying, which is an important indicator in paint quality control.
```

한국어 정본:

```text
DFT는 도료가 건조되어 남은 도막의 두께를 의미하며, 도장 품질 관리에서 중요한 지표입니다.
```

근거 ID: `["DOC-doc_0033-C0015"]`

단계 시간: `{"input_translation_ms": 3285, "copilot_ms": 9022, "output_translation_ms": 3287, "total_ms": 15595}`

실패 사유: `None`

### API / 중국어

질문: 什么是干膜厚度（DFT）？

HTTP 200 · 전체 14909 ms · 현지화 상태 `translated`

표시 답변 원문:

```text
干燥度膜厚度（DFT）是指涂料干燥后残留的涂层厚度。
```

한국어 정본:

```text
건조도막두께(DFT)는 도료가 건조되어 남은 막의 두께를 의미합니다.
```

근거 ID: `["DOC-doc_0033-C0015"]`

단계 시간: `{"input_translation_ms": 3549, "copilot_ms": 8301, "output_translation_ms": 3056, "total_ms": 14906}`

실패 사유: `None`

## 모델 직접 출력 35건

요청한 언어와 실제 출력 언어가 다르거나 답이 틀려도 원문을 그대로 남겼다. `finish_reason=length`는 최대 생성 길이 도달이며 완결된 답변으로 평가하지 않는다.

### Q01. 건조도막두께

#### 한국어 (ko)

질문: 건조도막두께, DFT가 뭐예요?

HTTP 200 · 6164 ms · finish_reason `stop`

```text
DFT는 Dry Film Thickness의 약자로, 도장 후 건조된 도막의 두께를 의미합니다. 이 수치는 도장 품질과 내구성을 확인하는 핵심 기준 중 하나입니다.
```

#### 영어 (en)

질문: What is dry film thickness, or DFT?

HTTP 200 · 6030 ms · finish_reason `stop`

```text
Dry film thickness (DFT) is the thickness of a coating after it has fully cured and the solvents have evaporated. It is a critical parameter for ensuring the coating provides adequate protection and meets performance specifications.
```

#### 중국어 (zh)

질문: 什么是干膜厚度（DFT）？

HTTP 200 · 4426 ms · finish_reason `stop`

```text
干膜厚度（DFT）是指涂层完全干燥固化后形成的薄膜厚度。它是衡量涂装质量的关键指标，直接影响涂层的防腐性能和使用寿命。
```

#### 베트남어 (vi)

질문: Chiều dày màng sơn khô (DFT) là gì?

HTTP 200 · 7648 ms · finish_reason `stop`

```text
Chiều dày màng sơn khô (DFT) là độ dày của lớp sơn sau khi đã khô hoàn toàn, được đo bằng micromet (µm) hoặc milimet (mm). Đây là thông số quan trọng để đảm bảo hiệu suất bảo vệ và tuổi thọ của lớp phủ.
```

#### 우즈벡어 (uz)

질문: Quruq plyonka qalinligi (DFT) nima?

HTTP 200 · 8346 ms · finish_reason `stop`

```text
DFT (Dry Film Thickness) — bu qo'llanilgan qoplamaning quruq holatdagi qalinligi. Bu ko'rsatkich qoplamaning himoya xususiyatlarini ta'minlash uchun muhim ahamiyatga ega.
```

#### 미얀마어 (my)

질문: DFT ဆိုတာ ဘာလဲ။

HTTP 200 · 11119 ms · finish_reason `stop`

```text
DFT (DFT) ဆိုသည်မှာ Dry Film Thickness (အလွှာထူထဲမှု) ကို ဆိုလိုသည်။
သင်္ဘောအလွှာအတွက် အလွှာထူထဲမှုသည် အရေးကြီးသော ကိုယ်လက်အစိတ်အပိုင်းဖြစ်သည်။
```

#### 네팔어 (ne)

질문: DFT भनेको के हो?

HTTP 200 · 10199 ms · finish_reason `stop`

```text
DFT (Dry Film Thickness) ले कोटिङको सुख्खा परतको मोटाईलाई जनाउँछ। यो मापनले कोटिङले अपेक्षित संरक्षण प्रदान गरिरहेको छ कि छैन भन्ने कुरा पत्ता लगाउन मद्दत गर्छ।
```

### Q02. 저도막

#### 한국어 (ko)

질문: 저도막이 뭐예요?

HTTP 200 · 10425 ms · finish_reason `stop`

```text
[분석] 저도막은 도장 공정에서 도료의 두께가 과도하게 낮아 표면이 노출되는 현상을 의미합니다.
[판정] 이상 탐지
[원인] 측정값 편차 또는 작업 조건 이탈이 확인되었습니다.
[조치]
1. 해당 구역을 재작업 후 재측정하십시오.
```

#### 영어 (en)

질문: What does it mean when a coating is too thin?

HTTP 200 · 7435 ms · finish_reason `stop`

```text
A coating that is too thin may not provide adequate protection against corrosion, UV exposure, or chemical attack. It can also lead to premature failure, such as peeling, flaking, or blistering, because the film lacks the necessary thickness to perform its intended function.
```

#### 중국어 (zh)

질문: 涂层太薄是什么意思？

HTTP 200 · 6973 ms · finish_reason `stop`

```text
“涂层太薄”是指涂层厚度未达到规定标准，可能导致防腐性能不足。通常由喷涂距离过远、走枪速度过快或稀释过度引起。建议检查施工参数并补涂至合格范围。
```

#### 베트남어 (vi)

질문: Lớp sơn quá mỏng có nghĩa là gì?

HTTP 200 · 10900 ms · finish_reason `stop`

```text
Lớp sơn quá mỏng là tình trạng độ dày lớp sơn đo được thấp hơn giá trị quy định trong tiêu chuẩn hoặc bản vẽ kỹ thuật. Điều này thường do pha sơn sai tỷ lệ, kỹ thuật phun không đều hoặc bề mặt nền không được xử lý sạch sẽ. Hậu quả là khả năng chống ăn mòn và bảo vệ kết cấu bị giảm đáng kể.
```

#### 우즈벡어 (uz)

질문: Bo‘yoq qatlami juda yupqa degani nima?

HTTP 200 · 9054 ms · finish_reason `stop`

```text
Bu holat bo‘yoq qatlami yetarli himoya qatlami hosil qilmaganini anglatadi. Natijada, korroziya tezroq boshlanishi mumkin. Qo‘shimcha qatlam surtish tavsiya etiladi.
```

#### 미얀마어 (my)

질문: ဆေးအလွှာ ပါးလွန်းတယ်ဆိုတာ ဘာကို ဆိုလိုတာလဲ။

HTTP 200 · 15994 ms · finish_reason `stop`

```text
DFT (Dry Film Thickness) တိုင်းတာမှုအရ လိုအပ်သောအထူထက် ပိုမိုပါးလွှာနေခြင်းကို ဆိုလိုသည်။ ဤကဲ့သို့ပါးလွှာနေပါက ကာကွယ်မှုမလုံလောက်တော့ဘဲ အောက်ခြေသံသတ္တုပေါ်သို့ ရေနှင့် အောက်ဆီဂျင် ဝင်ရောက်နိုင်ခြေရှိသည်။
```

#### 네팔어 (ne)

질문: रङको तह धेरै पातलो हुनु भनेको के हो?

HTTP 200 · 9751 ms · finish_reason `stop`

```text
रङको तह धेरै पतलो भएमा सतहको संरक्षण (protection) अपर्याप्त हुन्छ र जंग लाग्ने सम्भावना बढ्छ। यसले गर्दा रङको आयु र प्रभावकारिता घट्छ।
```

### Q03. 표면처리

#### 한국어 (ko)

질문: 도장 전에 하는 표면처리는 뭐예요?

HTTP 200 · 8135 ms · finish_reason `stop`

```text
도장 전에 하는 표면처리는 표면의 이물질, 녹, 습기 등을 제거하여 도장 부착력을 높이는 작업입니다. 일반적으로 블라스팅, 샌딩, 세정 등을 통해 표면을 깨끗하고 균일한 상태로 만드는 것이 중요합니다.
```

#### 영어 (en)

질문: What is surface preparation before painting?

HTTP 200 · 7209 ms · finish_reason `stop`

```text
Surface preparation is the process of cleaning and treating a substrate to ensure proper paint adhesion and long-term durability. It typically involves removing contaminants like oil, rust, and old paint, followed by profiling the surface to create a suitable anchor pattern for the coating.
```

#### 중국어 (zh)

질문: 涂装前的表面处理是什么？

HTTP 200 · 7677 ms · finish_reason `stop`

```text
涂装前的表面处理是指通过喷砂、打磨或化学清洗等方式，清除基材表面的油污、锈迹和杂质，并达到规定的清洁度与粗糙度标准。这一工序旨在增强涂层与基材的附着力，是确保防腐涂层长期有效性的关键前提。
```

#### 베트남어 (vi)

질문: Chuẩn bị bề mặt trước khi sơn là gì?

HTTP 200 · 12070 ms · finish_reason `stop`

```text
Chuẩn bị bề mặt là quá trình làm sạch, xử lý và tạo độ nhám cho bề mặt kim loại để đảm bảo lớp sơn bám dính tốt nhất.
Các bước thường bao gồm loại bỏ dầu mỡ, bụi bẩn, rỉ sét và kiểm tra độ sạch theo tiêu chuẩn ISO 8501-1.
Việc này giúp tăng tuổi thọ và hiệu quả bảo vệ của lớp phủ chống ăn mòn.
```

#### 우즈벡어 (uz)

질문: Bo‘yashdan oldin sirtni tayyorlash nima?

HTTP 200 · 16718 ms · finish_reason `stop`

```text
Bo‘yashdan oldin sirtni tayyorlash — bu yuzaning tozaligi, quruqligi va bir jilgili holatga keltirilishi orqali qoplamaning yopishishini ta’minlash jarayonidir. Bu bosqichda chang, yog‘, tuz va boshqa ifloslantiruvchilarni olib tashlash hamda zarur bo‘lsa, sirtning shilliq qilib tekislash ishlari amalga oshiriladi.
```

#### 미얀마어 (my)

질문: ဆေးမသုတ်ခင် မျက်နှာပြင်ကို ပြင်ဆင်တာ ဘာလဲ။

HTTP 200 · 10245 ms · finish_reason `stop`

```text
ပေးထားသော အချက်အလက်တွင် မျက်နှာပြင်သန့်ရှင်းရေးဆိုင်ရာ အချက်အလက် မပါဝင်သဖြင့် မည်သည့်အဆင့်ကို အသုံးပြုရမည်ကို ခန့်မှန်း၍မရပါ။
```

#### 네팔어 (ne)

질문: रङ लगाउनुअघि सतह तयार गर्नु भनेको के हो?

HTTP 200 · 7447 ms · finish_reason `stop`

```text
रङ लगाउनुअघि सतह तयार गरिनुपर्छ भन्ने कुरा हो। यसले सतहको सफाइ र उपचार सुनिश्चित गर्छ।
```

### Q04. A-1 / 80 µm 판정

#### 한국어 (ko)

질문: A-1 구역의 DFT가 80 µm인데 합격인가요?

HTTP 200 · 6990 ms · finish_reason `stop`

```text
제공된 데이터에서 기준 범위와 측정값의 편차 기준이 명시되지 않았습니다. 합격 여부를 판단하려면 사양서 기준(예: 90/10 규칙)과 실제 측정값의 비교가 필요합니다.
```

#### 영어 (en)

질문: The DFT in zone A-1 is 80 µm. Does it pass?

HTTP 200 · 9545 ms · finish_reason `stop`

```text
The DFT of 80 µm is within the typical acceptable range for most ship coatings, but you should verify against the specific specification for your project. If the spec allows a tolerance (e.g., 90/10 rule), ensure at least 90% of readings are above the minimum.
```

#### 중국어 (zh)

질문: A-1 区域的 DFT 是 80 µm，合格吗？

HTTP 200 · 3763 ms · finish_reason `stop`

```text
合格。该区域 DFT 为 80 µm，满足最低 75 µm 的要求。
```

#### 베트남어 (vi)

질문: DFT tại khu vực A-1 là 80 µm. Như vậy có đạt không?

HTTP 200 · 9533 ms · finish_reason `stop`

```text
Để xác định xem DFT 80 µm có đạt hay không, tôi cần biết thông số kỹ thuật (spec) của lớp sơn, bao gồm giá trị DFT mục tiêu và dung sai cho phép (tolerance). Vui lòng cung cấp thêm thông tin này để tôi có thể đánh giá chính xác.
```

#### 우즈벡어 (uz)

질문: A-1 hududida DFT 80 µm. Bu talabga javob beradimi?

HTTP 200 · 8849 ms · finish_reason `stop`

```text
A-1 hududi uchun DFT talabi odatda 120–160 µm oralig'ida bo'ladi. 80 µm qiymati bu talabdan past, shuning uchun qoplama qayta qo'llanilishi kerak.
```

#### 미얀마어 (my)

질문: A-1 ဧရိယာမှာ DFT က 80 µm ပါ။ သတ်မှတ်ချက်နဲ့ ကိုက်ညီပါသလား။

HTTP 200 · 10710 ms · finish_reason `stop`

```text
A-1 ဧရိယာအတွက် DFT သတ်မှတ်ချက်မှာ 75 µm ± 25% ဖြစ်သည်။ 80 µm သည် 56.25 µm မှ 93.75 µm ကြားတွင်ရှိသဖြင့် သတ်မှတ်ချက်နှင့် ကိုက်ညီပါသည်။
```

#### 네팔어 (ne)

질문: A-1 क्षेत्रमा DFT 80 µm छ। के यो स्वीकार्य छ?

HTTP 200 · 4449 ms · finish_reason `stop`

```text
हो, यो सङ्केत अनुसार A-1 कक्षेत्रको लागि DFT 80 µm स्वीकार्य छ।
```

### Q05. 재도장 시점

#### 한국어 (ko)

질문: 다음 도장층은 언제 칠할 수 있나요?

HTTP 200 · 3062 ms · finish_reason `stop`

```text
도장 가능 여부를 확인하려면 현재 도장층의 종류와 건조 상태를 알려주세요.
```

#### 영어 (en)

질문: When can I apply the next coat?

HTTP 200 · 4669 ms · finish_reason `stop`

```text
The recoating interval depends on the specific paint system and environmental conditions. Please provide the paint product name and the current temperature and humidity for a precise answer.
```

#### 중국어 (zh)

질문: 什么时候可以涂下一道涂层？

HTTP 200 · 6534 ms · finish_reason `stop`

```text
需要知道当前涂层的类型、施工环境（温度、湿度）以及产品技术数据表（TDS）中规定的重涂间隔时间。请提供这些信息，以便我为您判断是否已达到可重涂状态。
```

#### 베트남어 (vi)

질문: Khi nào tôi có thể sơn lớp tiếp theo?

HTTP 200 · 6307 ms · finish_reason `stop`

```text
Thời gian chờ giữa các lớp sơn (overcoat interval) phụ thuộc vào loại sơn và điều kiện môi trường. Bạn cần kiểm tra kỹ Technical Data Sheet (TDS) của nhà sản xuất để biết khoảng thời gian chính xác.
```

#### 우즈벡어 (uz)

질문: Keyingi bo‘yoq qatlamini qachon surtish mumkin?

HTTP 200 · 19268 ms · finish_reason `stop`

```text
Keyingi qatlamni surtish uchun oldingi qatlamning qurish vaqti (DFT) va qayta bo‘yash oralig‘i (overcoat interval) muhim. Odatda, bu ma’lumotlar texnik xulosa (TDS) da ko‘rsatilgan. Masalan, ba’zi epoksi bo‘yoqlarda qayta bo‘yash oralig‘i 7–14 kun bo‘lishi mumkin. Shuning uchun, aniq javob olish uchun bo‘yoqning TDS hujjatini tekshiring.
```

#### 미얀마어 (my)

질문: နောက်ဆေးအလွှာကို ဘယ်အချိန်မှာ သုတ်လို့ရမလဲ။

HTTP 200 · 11398 ms · finish_reason `stop`

```text
နော့ခ်ဆေးအလွှာကို ခြောက်သွေ့ပြီးမှသာ သုတ်လိမ်းနိုင်ပါသည်။ အလွှာတစ်လွှာစီ သုတ်လိမ်းပြီးနောက် ခြောက်သွေ့ချိန်ကို စောင့်ဆိုင်းရပါမည်။
```

#### 네팔어 (ne)

질문: अर्को तहको रङ कहिले लगाउन मिल्छ?

HTTP 200 · 10709 ms · finish_reason `stop`

```text
अर्को तहको रङ लग्नको लागि पहिलो तहको रङ पूर्ण रूपमा सुख्खा हुनुपर्छ। सुख्खा हुनको समय रङको प्रकार र मौसममा फरक हुन सक्छ, त्यसैले निर्माताको निर्देशन पालना गर्नुहोस्।
```

## 원시 기록과 해석 범위

- 직접 호출 완료 사유 집계: `{"stop": 35}`.
- 원시 JSONL: Spark `/home/keti/shipcoat/releases/multilingual-rc1/actual-qa-20261006.jsonl`. 요청 본문·응답 본문·호출 시간·모델 ID를 함께 저장했다.
- 수집 스크립트: 같은 디렉터리의 `shipcoat_collect_actual_qa.py`.
- 이 문서는 실제 출력 증거다. 현지 전문가의 기술 정확도·자연스러움 검수 결과는 아니다. HTTP 200을 답변 품질 합격으로 해석하지 않는다.
- 이전 [35쌍 문서](local_voice_multilingual_qa_examples.md)는 작성된 기대 답변이며 이 실행 결과와 구분한다.
