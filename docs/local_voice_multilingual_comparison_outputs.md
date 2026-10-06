# 세 가지 생성 방식의 실제 출력·항목별 판정

2026-10-06 · 모델 `qwen36-27b-shipcoat-v6-1-bf16` · [비교 결과](local_voice_multilingual_method_comparison.md) · [사전 조건](local_voice_multilingual_comparison_protocol.md)

105개 비교 항목이다. A 35건은 기존 실측 재사용, B/C 한국어 5건은 공유, B의 Q03 외국어 6건은 원답 실패로 미호출이다. 출력이 없는 항목에 답변을 작성해 넣지 않았다. 코드 블록은 실제 문자열이며 오탈자·한국어 혼입·반복·잘림을 수정하지 않았다. 내용 판정과 요청 언어 준수를 분리했다.

P=기준 충족(잠정), F=미충족, U=판정 유보, —=미호출. 언어 P는 요청 언어로 답변했다는 뜻이지 원어민 수준의 유창성 인증이 아니다. 기술 P와 언어 P를 모두 만족하고 정상 종료한 경우만 종합 P다. 판정자는 AI이며 독립 전문가 검수 전이다.

### A / Q01 / ko

- 방식: RAG 없이 직접 답변
- 질문: 건조도막두께, DFT가 뭐예요?
- 기술 내용: P · 요청 언어: P · 종합: P · 검토 확신도: AI 잠정
- 판정 사유/주요 오류: 건조 도막 두께 정의 충족
- HTTP: 200 · 종료: stop · 생성 경로 시간: 6164 ms

```text
DFT는 Dry Film Thickness의 약자로, 도장 후 건조된 도막의 두께를 의미합니다. 이 수치는 도장 품질과 내구성을 확인하는 핵심 기준 중 하나입니다.
```

### B / Q01 / ko

- 방식: 한국어 RAG 답변 후 번역
- 질문: 건조도막두께, DFT가 뭐예요?
- 기술 내용: P · 요청 언어: P · 종합: P · 검토 확신도: AI 잠정
- 판정 사유/주요 오류: 건조 후 도막 두께 정의 충족
- HTTP: 200 · 종료: stop · 생성 경로 시간: 8395 ms

```text
DFT(Dry Film Thickness)는 도료가 건조된 후의 도막 두께를 의미합니다. 이 수치는 도장 품질 관리에서 규정 준수 여부를 확인하기 위해 반드시 측정해야 하는 핵심 항목입니다.
```

### C / Q01 / ko

- 방식: RAG 근거로 대상 언어 직접 생성
- 질문: 건조도막두께, DFT가 뭐예요?
- 기술 내용: P · 요청 언어: P · 종합: P · 검토 확신도: AI 잠정
- 판정 사유/주요 오류: 건조 후 도막 두께 정의 충족
- HTTP: 200 · 종료: stop · 생성 경로 시간: 8395 ms

```text
DFT(Dry Film Thickness)는 도료가 건조된 후의 도막 두께를 의미합니다. 이 수치는 도장 품질 관리에서 규정 준수 여부를 확인하기 위해 반드시 측정해야 하는 핵심 항목입니다.
```

### A / Q01 / en

- 방식: RAG 없이 직접 답변
- 질문: What is dry film thickness, or DFT?
- 기술 내용: P · 요청 언어: P · 종합: P · 검토 확신도: AI 잠정
- 판정 사유/주요 오류: 건조 후 두께 정의 충족
- HTTP: 200 · 종료: stop · 생성 경로 시간: 6030 ms

```text
Dry film thickness (DFT) is the thickness of a coating after it has fully cured and the solvents have evaporated. It is a critical parameter for ensuring the coating provides adequate protection and meets performance specifications.
```

### B / Q01 / en

- 방식: 한국어 RAG 답변 후 번역
- 질문: What is dry film thickness, or DFT?
- 기술 내용: P · 요청 언어: P · 종합: P · 검토 확신도: AI 잠정
- 판정 사유/주요 오류: 한국어 DFT 정의를 대상 언어로 전달
- HTTP: 200 · 종료: stop · 생성 경로 시간: 13325 ms

```text
DFT (Dry Film Thickness) refers to the thickness of the coating after it has dried. This value is a critical item that must be measured to verify compliance in coating quality control.
```

### C / Q01 / en

- 방식: RAG 근거로 대상 언어 직접 생성
- 질문: What is dry film thickness, or DFT?
- 기술 내용: P · 요청 언어: P · 종합: P · 검토 확신도: AI 잠정
- 판정 사유/주요 오류: 건조 후 도막 두께 정의 충족
- HTTP: 200 · 종료: stop · 생성 경로 시간: 7028 ms

```text
DFT stands for Dry Film Thickness, which is the thickness of the coating after it has dried. It is a critical parameter in ship coating to ensure proper protection and performance of the coating system.
```

### A / Q01 / zh

- 방식: RAG 없이 직접 답변
- 질문: 什么是干膜厚度（DFT）？
- 기술 내용: P · 요청 언어: P · 종합: P · 검토 확신도: AI 잠정
- 판정 사유/주요 오류: 건조 도막 두께 정의 충족
- HTTP: 200 · 종료: stop · 생성 경로 시간: 4426 ms

```text
干膜厚度（DFT）是指涂层完全干燥固化后形成的薄膜厚度。它是衡量涂装质量的关键指标，直接影响涂层的防腐性能和使用寿命。
```

### B / Q01 / zh

- 방식: 한국어 RAG 답변 후 번역
- 질문: 什么是干膜厚度（DFT）？
- 기술 내용: P · 요청 언어: P · 종합: P · 검토 확신도: AI 잠정
- 판정 사유/주요 오류: 한국어 DFT 정의를 대상 언어로 전달
- HTTP: 200 · 종료: stop · 생성 경로 시간: 12868 ms

```text
DFT（Dry Film Thickness）是指涂料干燥后的涂层厚度。该数值是涂装质量控制中必须测量的关键项目，用于确认是否符合规定。
```

### C / Q01 / zh

- 방식: RAG 근거로 대상 언어 직접 생성
- 질문: 什么是干膜厚度（DFT）？
- 기술 내용: P · 요청 언어: P · 종합: P · 검토 확신도: AI 잠정
- 판정 사유/주요 오류: 건조 후 도막 두께 정의 충족
- HTTP: 200 · 종료: stop · 생성 경로 시간: 11467 ms

```text
干膜厚度（DFT）是指涂层完全干燥固化后的最终厚度。它可以通过湿膜厚度（WFT）乘以体积固体份（SV%）再除以100来估算，即 DFT = WFT × SV% / 100。DFT是评估涂层是否满足防腐保护要求的关键指标。
```

### A / Q01 / vi

- 방식: RAG 없이 직접 답변
- 질문: Chiều dày màng sơn khô (DFT) là gì?
- 기술 내용: P · 요청 언어: P · 종합: P · 검토 확신도: AI 잠정
- 판정 사유/주요 오류: 건조 도막 두께 정의 충족
- HTTP: 200 · 종료: stop · 생성 경로 시간: 7648 ms

```text
Chiều dày màng sơn khô (DFT) là độ dày của lớp sơn sau khi đã khô hoàn toàn, được đo bằng micromet (µm) hoặc milimet (mm). Đây là thông số quan trọng để đảm bảo hiệu suất bảo vệ và tuổi thọ của lớp phủ.
```

### B / Q01 / vi

- 방식: 한국어 RAG 답변 후 번역
- 질문: Chiều dày màng sơn khô (DFT) là gì?
- 기술 내용: P · 요청 언어: P · 종합: P · 검토 확신도: AI 잠정
- 판정 사유/주요 오류: 한국어 DFT 정의를 대상 언어로 전달
- HTTP: 200 · 종료: stop · 생성 경로 시간: 14725 ms

```text
DFT (Dry Film Thickness) là độ dày lớp sơn sau khi sơn đã khô. Số liệu này là một mục tiêu quan trọng phải được đo lường để đảm bảo tuân thủ quy định trong kiểm soát chất lượng sơn.
```

### C / Q01 / vi

- 방식: RAG 근거로 대상 언어 직접 생성
- 질문: Chiều dày màng sơn khô (DFT) là gì?
- 기술 내용: P · 요청 언어: P · 종합: P · 검토 확신도: AI 잠정
- 판정 사유/주요 오류: 건조 후 도막 두께 정의 충족
- HTTP: 200 · 종료: stop · 생성 경로 시간: 7956 ms

```text
DFT (Dry Film Thickness) là chiều dày của lớp sơn sau khi đã khô hoàn toàn. Đây là thông số quan trọng để đánh giá chất lượng và hiệu quả bảo vệ của lớp phủ.
```

### A / Q01 / uz

- 방식: RAG 없이 직접 답변
- 질문: Quruq plyonka qalinligi (DFT) nima?
- 기술 내용: P · 요청 언어: P · 종합: P · 검토 확신도: AI 잠정
- 판정 사유/주요 오류: 건조 상태의 도막 두께 정의 충족
- HTTP: 200 · 종료: stop · 생성 경로 시간: 8346 ms

```text
DFT (Dry Film Thickness) — bu qo'llanilgan qoplamaning quruq holatdagi qalinligi. Bu ko'rsatkich qoplamaning himoya xususiyatlarini ta'minlash uchun muhim ahamiyatga ega.
```

### B / Q01 / uz

- 방식: 한국어 RAG 답변 후 번역
- 질문: Quruq plyonka qalinligi (DFT) nima?
- 기술 내용: P · 요청 언어: P · 종합: P · 검토 확신도: AI 잠정
- 판정 사유/주요 오류: 한국어 DFT 정의를 대상 언어로 전달
- HTTP: 200 · 종료: stop · 생성 경로 시간: 20266 ms

```text
DFT (Dry Film Thickness) — quruq qatlam qalinligi — bo‘yoq quriganidan keyingi qatlam qalinligini bildiradi. Bu qiymat qoplama sifatini nazorat qilishda talablarga muvofiqlikni tekshirish uchun o‘lchash talab qilinadigan asosiy ko‘rsatkichdir.
```

### C / Q01 / uz

- 방식: RAG 근거로 대상 언어 직접 생성
- 질문: Quruq plyonka qalinligi (DFT) nima?
- 기술 내용: P · 요청 언어: P · 종합: P · 검토 확신도: AI 잠정
- 판정 사유/주요 오류: 건조 후 도막 두께 정의 충족
- HTTP: 200 · 종료: stop · 생성 경로 시간: 12407 ms

```text
DFT (Dry Film Thickness) — bu quruq plyonka qalinligi bo‘lib, qoplam qatlami quriganidan keyin o‘lchanadigan qalinlikdir. Bu ko‘rsatkich qoplamning himoya xususiyatlarini baholashda muhim ahamiyatga ega.
```

### A / Q01 / my

- 방식: RAG 없이 직접 답변
- 질문: DFT ဆိုတာ ဘာလဲ။
- 기술 내용: F · 요청 언어: P · 종합: F · 검토 확신도: AI 잠정
- 판정 사유/주요 오류: 두께를 중요한 신체 부위로 표현하는 무관한 문장 포함
- HTTP: 200 · 종료: stop · 생성 경로 시간: 11119 ms

```text
DFT (DFT) ဆိုသည်မှာ Dry Film Thickness (အလွှာထူထဲမှု) ကို ဆိုလိုသည်။
သင်္ဘောအလွှာအတွက် အလွှာထူထဲမှုသည် အရေးကြီးသော ကိုယ်လက်အစိတ်အပိုင်းဖြစ်သည်။
```

### B / Q01 / my

- 방식: 한국어 RAG 답변 후 번역
- 질문: DFT ဆိုတာ ဘာလဲ။
- 기술 내용: P · 요청 언어: F · 종합: F · 검토 확신도: AI 잠정
- 판정 사유/주요 오류: 요청 언어 대신 한국어 출력; DFT 정의 내용 자체는 맞음
- HTTP: 200 · 종료: stop · 생성 경로 시간: 14709 ms

```text
DFT(건조 필름 두께)는 도료가 건조된 후의 도막 두께를 의미합니다. 이 수치는 도장 품질 관리에서 규정 준수 여부를 확인하기 위해 반드시 측정해야 하는 핵심 항목입니다.
```

### C / Q01 / my

- 방식: RAG 근거로 대상 언어 직접 생성
- 질문: DFT ဆိုတာ ဘာလဲ။
- 기술 내용: U · 요청 언어: P · 종합: U · 검토 확신도: 유보
- 판정 사유/주요 오류: 건조 후 두께 취지는 있으나 원어 용어 검수 유보; 한국어 근거 인용 혼재
- HTTP: 200 · 종료: stop · 생성 경로 시간: 14266 ms

```text
DFT (Dry Film Thickness) ဆိုသည်မှာ အလွှာခန်းပြီးနောက် ကျန်ရှိသော အလွှာအထူကို ဆိုလိုသည်။
[DOC-doc_0045-C0127] အရ DFT သည် "DFT 측정, 도막 건조 후 도막의 두께 측정(대표적인 검사 장비)" ဟု ဖော်ပြထားသည်။
```

### A / Q01 / ne

- 방식: RAG 없이 직접 답변
- 질문: DFT भनेको के हो?
- 기술 내용: U · 요청 언어: P · 종합: U · 검토 확신도: 유보
- 판정 사유/주요 오류: 건조 도막 두께 취지는 보이나 원어 기술 의미 검수 유보
- HTTP: 200 · 종료: stop · 생성 경로 시간: 10199 ms

```text
DFT (Dry Film Thickness) ले कोटिङको सुख्खा परतको मोटाईलाई जनाउँछ। यो मापनले कोटिङले अपेक्षित संरक्षण प्रदान गरिरहेको छ कि छैन भन्ने कुरा पत्ता लगाउन मद्दत गर्छ।
```

### B / Q01 / ne

- 방식: 한국어 RAG 답변 후 번역
- 질문: DFT भनेको के हो?
- 기술 내용: P · 요청 언어: F · 종합: F · 검토 확신도: AI 잠정
- 판정 사유/주요 오류: 요청 언어 대신 한국어 출력; DFT 정의 내용 자체는 맞음
- HTTP: 200 · 종료: stop · 생성 경로 시간: 14706 ms

```text
DFT(건조 필름 두께)는 도료가 건조된 후의 도막 두께를 의미합니다. 이 수치는 도장 품질 관리에서 규정 준수 여부를 확인하기 위해 반드시 측정해야 하는 핵심 항목입니다.
```

### C / Q01 / ne

- 방식: RAG 근거로 대상 언어 직접 생성
- 질문: DFT भनेको के हो?
- 기술 내용: U · 요청 언어: P · 종합: U · 검토 확신도: 유보
- 판정 사유/주요 오류: 건조 상태의 두께 취지는 있으나 원어 기술 의미 검수 유보
- HTTP: 200 · 종료: stop · 생성 경로 시간: 11930 ms

```text
DFT (Dry Film Thickness) भनेको कोटिङ (coating) को सुख्खा अवस्थामा मापन गरिएको मोटाइ हो। यसले सतहमा लागू भएको रङ वा कोटिङको वास्तविक मोटाइलाई जनाउँछ।
```

### A / Q02 / ko

- 방식: RAG 없이 직접 답변
- 질문: 저도막이 뭐예요?
- 기술 내용: F · 요청 언어: P · 종합: F · 검토 확신도: AI 잠정
- 판정 사유/주요 오류: 개념 질문을 이상 탐지로 바꾸고 측정 편차를 확인했다고 허위 단정
- HTTP: 200 · 종료: stop · 생성 경로 시간: 10425 ms

```text
[분석] 저도막은 도장 공정에서 도료의 두께가 과도하게 낮아 표면이 노출되는 현상을 의미합니다.
[판정] 이상 탐지
[원인] 측정값 편차 또는 작업 조건 이탈이 확인되었습니다.
[조치]
1. 해당 구역을 재작업 후 재측정하십시오.
```

### B / Q02 / ko

- 방식: 한국어 RAG 답변 후 번역
- 질문: 저도막이 뭐예요?
- 기술 내용: F · 요청 언어: P · 종합: F · 검토 확신도: AI 잠정
- 판정 사유/주요 오류: 저도막 대신 가열 건조 도료의 정의를 인용
- HTTP: 200 · 종료: stop · 생성 경로 시간: 9413 ms

```text
[DOC-doc_0032-C0042] 소지에 칠하고 나서 가열하여 도막이 형성되도록 만든 도료이 다. 가열하는 온도는 보통 100°C 이상으로 한다. ASTM, BS에서 는 강제 건조(66°C 이하)보다 높은 온도로 한다.
```

### C / Q02 / ko

- 방식: RAG 근거로 대상 언어 직접 생성
- 질문: 저도막이 뭐예요?
- 기술 내용: F · 요청 언어: P · 종합: F · 검토 확신도: AI 잠정
- 판정 사유/주요 오류: 저도막 대신 가열 건조 도료의 정의를 인용
- HTTP: 200 · 종료: stop · 생성 경로 시간: 9413 ms

```text
[DOC-doc_0032-C0042] 소지에 칠하고 나서 가열하여 도막이 형성되도록 만든 도료이 다. 가열하는 온도는 보통 100°C 이상으로 한다. ASTM, BS에서 는 강제 건조(66°C 이하)보다 높은 온도로 한다.
```

### A / Q02 / en

- 방식: RAG 없이 직접 답변
- 질문: What does it mean when a coating is too thin?
- 기술 내용: F · 요청 언어: P · 종합: F · 검토 확신도: AI 잠정
- 판정 사유/주요 오류: 보호 성능 영향만 설명하고 요구 두께 미달이라는 정의 누락
- HTTP: 200 · 종료: stop · 생성 경로 시간: 7435 ms

```text
A coating that is too thin may not provide adequate protection against corrosion, UV exposure, or chemical attack. It can also lead to premature failure, such as peeling, flaking, or blistering, because the film lacks the necessary thickness to perform its intended function.
```

### B / Q02 / en

- 방식: 한국어 RAG 답변 후 번역
- 질문: What does it mean when a coating is too thin?
- 기술 내용: F · 요청 언어: P · 종합: F · 검토 확신도: AI 잠정
- 판정 사유/주요 오류: 한국어 원답의 무관한 가열 건조 도료 정의를 전달
- HTTP: 200 · 종료: stop · 생성 경로 시간: 16911 ms

```text
[DOC-doc_0032-C0042] This is a coating applied to the substrate and then heated to form a film. The heating temperature is usually above 100°C. ASTM and BS specify temperatures higher than forced drying (below 66°C).
```

### C / Q02 / en

- 방식: RAG 근거로 대상 언어 직접 생성
- 질문: What does it mean when a coating is too thin?
- 기술 내용: P · 요청 언어: P · 종합: P · 검토 확신도: AI 잠정
- 판정 사유/주요 오류: 규정·요구 두께와 보호 성능을 연결하여 설명
- HTTP: 200 · 종료: stop · 생성 경로 시간: 7585 ms

```text
A coating that is too thin may not provide adequate protection or coverage, potentially leading to issues like poor adhesion, reduced durability, or insufficient barrier properties. It's important to ensure the coating meets the specified thickness requirements for optimal performance.
```

### A / Q02 / zh

- 방식: RAG 없이 직접 답변
- 질문: 涂层太薄是什么意思？
- 기술 내용: P · 요청 언어: P · 종합: P · 검토 확신도: AI 잠정
- 판정 사유/주요 오류: 규정 두께 미달 설명
- HTTP: 200 · 종료: stop · 생성 경로 시간: 6973 ms

```text
“涂层太薄”是指涂层厚度未达到规定标准，可能导致防腐性能不足。通常由喷涂距离过远、走枪速度过快或稀释过度引起。建议检查施工参数并补涂至合格范围。
```

### B / Q02 / zh

- 방식: 한국어 RAG 답변 후 번역
- 질문: 涂层太薄是什么意思？
- 기술 내용: F · 요청 언어: P · 종합: F · 검토 확신도: AI 잠정
- 판정 사유/주요 오류: 한국어 원답의 무관한 가열 건조 도료 정의를 전달
- HTTP: 200 · 종료: stop · 생성 경로 시간: 17376 ms

```text
[DOC-doc_0032-C0042] 这是一种在基材上涂布后通过加热形成涂膜的涂料。加热温度通常高于100°C。ASTM和BS标准规定，加热温度应高于强制干燥温度（66°C以下）。
```

### C / Q02 / zh

- 방식: RAG 근거로 대상 언어 직접 생성
- 질문: 涂层太薄是什么意思？
- 기술 내용: P · 요청 언어: P · 종합: P · 검토 확신도: AI 잠정
- 판정 사유/주요 오류: 규정·요구 두께와 보호 성능을 연결하여 설명
- HTTP: 200 · 종료: stop · 생성 경로 시간: 8505 ms

```text
“涂层太薄”是指涂层厚度未达到规定标准，可能导致防护性能不足。在船舶涂装中，过薄的涂层无法有效隔绝腐蚀介质，易引发锈蚀或剥落。需通过测厚仪检测并补涂至合格范围。
```

### A / Q02 / vi

- 방식: RAG 없이 직접 답변
- 질문: Lớp sơn quá mỏng có nghĩa là gì?
- 기술 내용: P · 요청 언어: P · 종합: P · 검토 확신도: AI 잠정
- 판정 사유/주요 오류: 규정 또는 도면 두께 미달 설명
- HTTP: 200 · 종료: stop · 생성 경로 시간: 10900 ms

```text
Lớp sơn quá mỏng là tình trạng độ dày lớp sơn đo được thấp hơn giá trị quy định trong tiêu chuẩn hoặc bản vẽ kỹ thuật. Điều này thường do pha sơn sai tỷ lệ, kỹ thuật phun không đều hoặc bề mặt nền không được xử lý sạch sẽ. Hậu quả là khả năng chống ăn mòn và bảo vệ kết cấu bị giảm đáng kể.
```

### B / Q02 / vi

- 방식: 한국어 RAG 답변 후 번역
- 질문: Lớp sơn quá mỏng có nghĩa là gì?
- 기술 내용: F · 요청 언어: P · 종합: F · 검토 확신도: AI 잠정
- 판정 사유/주요 오류: 한국어 원답의 무관한 가열 건조 도료 정의를 전달
- HTTP: 200 · 종료: stop · 생성 경로 시간: 19231 ms

```text
[DOC-doc_0032-C0042] Đây là loại sơn được tạo thành bằng cách quét lên bề mặt và sau đó gia nhiệt để hình thành màng sơn. Nhiệt độ gia nhiệt thường trên 100°C. Theo ASTM và BS, nhiệt độ này cao hơn so với sấy cưỡng bức (dưới 66°C).
```

### C / Q02 / vi

- 방식: RAG 근거로 대상 언어 직접 생성
- 질문: Lớp sơn quá mỏng có nghĩa là gì?
- 기술 내용: P · 요청 언어: P · 종합: P · 검토 확신도: AI 잠정
- 판정 사유/주요 오류: 규정·요구 두께와 보호 성능을 연결하여 설명
- HTTP: 200 · 종료: stop · 생성 경로 시간: 9209 ms

```text
Lớp sơn quá mỏng là tình trạng độ dày lớp sơn không đạt yêu cầu kỹ thuật, khiến khả năng bảo vệ và chống ăn mòn của lớp phủ bị suy giảm. Điều này thường xảy ra khi lượng sơn áp dụng không đủ hoặc quá trình phun sơn không đồng đều.
```

### A / Q02 / uz

- 방식: RAG 없이 직접 답변
- 질문: Bo‘yoq qatlami juda yupqa degani nima?
- 기술 내용: F · 요청 언어: P · 종합: F · 검토 확신도: AI 잠정
- 판정 사유/주요 오류: 보호 부족과 재도장 권고만 있고 요구 두께 미달 정의 누락
- HTTP: 200 · 종료: stop · 생성 경로 시간: 9054 ms

```text
Bu holat bo‘yoq qatlami yetarli himoya qatlami hosil qilmaganini anglatadi. Natijada, korroziya tezroq boshlanishi mumkin. Qo‘shimcha qatlam surtish tavsiya etiladi.
```

### B / Q02 / uz

- 방식: 한국어 RAG 답변 후 번역
- 질문: Bo‘yoq qatlami juda yupqa degani nima?
- 기술 내용: F · 요청 언어: P · 종합: F · 검토 확신도: AI 잠정
- 판정 사유/주요 오류: 한국어 원답의 무관한 가열 건조 도료 정의를 전달
- HTTP: 200 · 종료: stop · 생성 경로 시간: 24081 ms

```text
[DOC-doc_0032-C0042] Bu material yuzaga surtilgandan so'ng qizdirib qoplamani hosil qilish uchun ishlatiladi. Qizdirish harorati odatda 100°C dan yuqori bo'ladi. ASTM va BS standartlarida majburiy quritish (66°C dan past) dan yuqori haroratda amalga oshiriladi.
```

### C / Q02 / uz

- 방식: RAG 근거로 대상 언어 직접 생성
- 질문: Bo‘yoq qatlami juda yupqa degani nima?
- 기술 내용: P · 요청 언어: P · 종합: P · 검토 확신도: AI 잠정
- 판정 사유/주요 오류: 규정·요구 두께와 보호 성능을 연결하여 설명
- HTTP: 200 · 종료: stop · 생성 경로 시간: 14090 ms

```text
O‘qilgan hujjatlar bo‘yicha, "yupqa qatlam" (low film thickness) deganda, qoplama qalinligi tavsiya etilgan minimal chegaradan past bo‘lgan holat tushuniladi. Bu holatda qoplamaning himoya xususiyatlari yetarli darajada ta’minlanmaydi.
```

### A / Q02 / my

- 방식: RAG 없이 직접 답변
- 질문: ဆေးအလွှာ ပါးလွန်းတယ်ဆိုတာ ဘာကို ဆိုလိုတာလဲ။
- 기술 내용: U · 요청 언어: P · 종합: U · 검토 확신도: 유보
- 판정 사유/주요 오류: 필요 두께 미달이라는 취지는 보이나 원어 기술 의미 검수 유보
- HTTP: 200 · 종료: stop · 생성 경로 시간: 15994 ms

```text
DFT (Dry Film Thickness) တိုင်းတာမှုအရ လိုအပ်သောအထူထက် ပိုမိုပါးလွှာနေခြင်းကို ဆိုလိုသည်။ ဤကဲ့သို့ပါးလွှာနေပါက ကာကွယ်မှုမလုံလောက်တော့ဘဲ အောက်ခြေသံသတ္တုပေါ်သို့ ရေနှင့် အောက်ဆီဂျင် ဝင်ရောက်နိုင်ခြေရှိသည်။
```

### B / Q02 / my

- 방식: 한국어 RAG 답변 후 번역
- 질문: ဆေးအလွှာ ပါးလွန်းတယ်ဆိုတာ ဘာကို ဆိုလိုတာလဲ။
- 기술 내용: F · 요청 언어: F · 종합: F · 검토 확신도: AI 잠정
- 판정 사유/주요 오류: 한국어 원답의 무관한 가열 건조 도료 정의를 전달; 요청 언어 대신 한국어 출력
- HTTP: 200 · 종료: stop · 생성 경로 시간: 17851 ms

```text
[DOC-doc_0032-C0042] 소지에 칠하고 나서 가열하여 도막이 형성되도록 만든 도료이다. 가열하는 온도는 보통 100°C 이상으로 한다. ASTM, BS에서는 강제 건조(66°C 이하)보다 높은 온도로 한다.
```

### C / Q02 / my

- 방식: RAG 근거로 대상 언어 직접 생성
- 질문: ဆေးအလွှာ ပါးလွန်းတယ်ဆိုတာ ဘာကို ဆိုလိုတာလဲ။
- 기술 내용: F · 요청 언어: F · 종합: F · 검토 확신도: AI 잠정
- 판정 사유/주요 오류: 근거 ID만 반복 출력하고 토큰 상한 도달; 답변 없음
- HTTP: 200 · 종료: length · 생성 경로 시간: 24108 ms

```text
[DOC-doc_0032-C0061]
[DOC-doc_0032-C0255]
[DOC-doc_0032-C0105]
[DOC-doc_0032-C0131]
[DOC-doc_0032-C0244]
[DOC-doc_0032-C0042]
[DOC-doc_0047-C0132]
[DOC-doc_0039-C0040]
[DOC-doc_0032-C0061]
[DOC-doc_0032-C0255]
[DOC-doc_0032-C0105]
[DOC-doc_0032-C0131]
[DOC-doc_0032-C024
```

### A / Q02 / ne

- 방식: RAG 없이 직접 답변
- 질문: रङको तह धेरै पातलो हुनु भनेको के हो?
- 기술 내용: F · 요청 언어: P · 종합: F · 검토 확신도: AI 잠정
- 판정 사유/주요 오류: 보호 부족의 영향만 설명하고 요구 두께 미달 정의 누락
- HTTP: 200 · 종료: stop · 생성 경로 시간: 9751 ms

```text
रङको तह धेरै पतलो भएमा सतहको संरक्षण (protection) अपर्याप्त हुन्छ र जंग लाग्ने सम्भावना बढ्छ। यसले गर्दा रङको आयु र प्रभावकारिता घट्छ।
```

### B / Q02 / ne

- 방식: 한국어 RAG 답변 후 번역
- 질문: रङको तह धेरै पातलो हुनु भनेको के हो?
- 기술 내용: F · 요청 언어: F · 종합: F · 검토 확신도: AI 잠정
- 판정 사유/주요 오류: 한국어 원답의 무관한 가열 건조 도료 정의를 전달; 요청 언어 대신 한국어 출력
- HTTP: 200 · 종료: stop · 생성 경로 시간: 17604 ms

```text
[DOC-doc_0032-C0042] 소지에 칠하고 나서 가열하여 도막이 형성되도록 만든 도료이다. 가열하는 온도는 보통 100°C 이상으로 한다. ASTM, BS에서는 강제 건조(66°C 이하)보다 높은 온도로 한다.
```

### C / Q02 / ne

- 방식: RAG 근거로 대상 언어 직접 생성
- 질문: रङको तह धेरै पातलो हुनु भनेको के हो?
- 기술 내용: F · 요청 언어: P · 종합: F · 검토 확신도: AI 잠정
- 판정 사유/주요 오류: 저도막을 Tint/Weak Color와 혼동
- HTTP: 200 · 종료: stop · 생성 경로 시간: 9227 ms

```text
[DOC-doc_0032-C0061] मा 'Tint' (Tint, Weak Color, Tint Color) लाई 'Weak Color' भनिएको छ। यसले सानो रङ तहको अवस्था देखाउँछ।
```

### A / Q03 / ko

- 방식: RAG 없이 직접 답변
- 질문: 도장 전에 하는 표면처리는 뭐예요?
- 기술 내용: P · 요청 언어: P · 종합: P · 검토 확신도: AI 잠정
- 판정 사유/주요 오류: 오염 제거와 부착력 목적 설명
- HTTP: 200 · 종료: stop · 생성 경로 시간: 8135 ms

```text
도장 전에 하는 표면처리는 표면의 이물질, 녹, 습기 등을 제거하여 도장 부착력을 높이는 작업입니다. 일반적으로 블라스팅, 샌딩, 세정 등을 통해 표면을 깨끗하고 균일한 상태로 만드는 것이 중요합니다.
```

### B / Q03 / ko

- 방식: 한국어 RAG 답변 후 번역
- 질문: 도장 전에 하는 표면처리는 뭐예요?
- 기술 내용: F · 요청 언어: P · 종합: F · 검토 확신도: AI 잠정
- 판정 사유/주요 오류: 용접 비드 연마 반복으로 토큰 상한 도달; 완결 답변 실패
- HTTP: 200 · 종료: length · 생성 경로 시간: 26083 ms

```text
도장 전 표면 처리는 블라스팅, 동력 공구 연마, 수동 연마, 용제 세정, 용접선 연마, 용접 비드 제거, 용접 스패터 제거, 용접 비드 연마, 용접 비드 연마, 용접 비드 연마, 용접 비드 연마, 용접 비드 연마, 용접 비드 연마, 용접 비드 연마, 용접 비드 연마, 용접 비드 연마, 용접 비드 연마, 용접 비드 연마, 용접 비드 연마, 용접 비드 연마, 용접 비드 연마, 용접 비드 연마, 용접 비드 연마, 용접 비드 연마, 용접 비드 연마, 용접 비드 연마, 용접 비드 연마, 용접 비드 연
```

### C / Q03 / ko

- 방식: RAG 근거로 대상 언어 직접 생성
- 질문: 도장 전에 하는 표면처리는 뭐예요?
- 기술 내용: F · 요청 언어: P · 종합: F · 검토 확신도: AI 잠정
- 판정 사유/주요 오류: 용접 비드 연마 반복으로 토큰 상한 도달; 완결 답변 실패
- HTTP: 200 · 종료: length · 생성 경로 시간: 26083 ms

```text
도장 전 표면 처리는 블라스팅, 동력 공구 연마, 수동 연마, 용제 세정, 용접선 연마, 용접 비드 제거, 용접 스패터 제거, 용접 비드 연마, 용접 비드 연마, 용접 비드 연마, 용접 비드 연마, 용접 비드 연마, 용접 비드 연마, 용접 비드 연마, 용접 비드 연마, 용접 비드 연마, 용접 비드 연마, 용접 비드 연마, 용접 비드 연마, 용접 비드 연마, 용접 비드 연마, 용접 비드 연마, 용접 비드 연마, 용접 비드 연마, 용접 비드 연마, 용접 비드 연마, 용접 비드 연마, 용접 비드 연
```

### A / Q03 / en

- 방식: RAG 없이 직접 답변
- 질문: What is surface preparation before painting?
- 기술 내용: P · 요청 언어: P · 종합: P · 검토 확신도: AI 잠정
- 판정 사유/주요 오류: 청정·표면 형상·부착력 설명
- HTTP: 200 · 종료: stop · 생성 경로 시간: 7209 ms

```text
Surface preparation is the process of cleaning and treating a substrate to ensure proper paint adhesion and long-term durability. It typically involves removing contaminants like oil, rust, and old paint, followed by profiling the surface to create a suitable anchor pattern for the coating.
```

### B / Q03 / en

- 방식: 한국어 RAG 답변 후 번역
- 질문: What is surface preparation before painting?
- 기술 내용: F · 요청 언어: — · 종합: F · 검토 확신도: AI 잠정
- 판정 사유/주요 오류: 한국어 생성이 반복·잘림으로 실패하여 번역 호출하지 않음
- HTTP: — · 종료: KoreanGenerationFailed · 생성 경로 시간: 26083 ms

모델 출력 없음: 한국어 원답이 정상 종료되지 않아 번역 호출을 생략했다. 표시 시간은 실패한 한국어 생성까지의 시간이다.

### C / Q03 / en

- 방식: RAG 근거로 대상 언어 직접 생성
- 질문: What is surface preparation before painting?
- 기술 내용: P · 요청 언어: P · 종합: P · 검토 확신도: AI 잠정
- 판정 사유/주요 오류: 청정·거칠기·오염 제거·부착력 설명
- HTTP: 200 · 종료: stop · 생성 경로 시간: 10304 ms

```text
Surface preparation is the process of cleaning and roughening a metal surface to ensure proper paint adhesion and remove contaminants like oil, grease, and rust. It is critical because even high-quality coatings will fail if the underlying surface is not properly treated. Common methods include abrasive blasting or power tool cleaning to achieve the required surface profile.
```

### A / Q03 / zh

- 방식: RAG 없이 직접 답변
- 질문: 涂装前的表面处理是什么？
- 기술 내용: P · 요청 언어: P · 종합: P · 검토 확신도: AI 잠정
- 판정 사유/주요 오류: 오염 제거·청정도·조도·부착력 설명
- HTTP: 200 · 종료: stop · 생성 경로 시간: 7677 ms

```text
涂装前的表面处理是指通过喷砂、打磨或化学清洗等方式，清除基材表面的油污、锈迹和杂质，并达到规定的清洁度与粗糙度标准。这一工序旨在增强涂层与基材的附着力，是确保防腐涂层长期有效性的关键前提。
```

### B / Q03 / zh

- 방식: 한국어 RAG 답변 후 번역
- 질문: 涂装前的表面处理是什么？
- 기술 내용: F · 요청 언어: — · 종합: F · 검토 확신도: AI 잠정
- 판정 사유/주요 오류: 한국어 생성이 반복·잘림으로 실패하여 번역 호출하지 않음
- HTTP: — · 종료: KoreanGenerationFailed · 생성 경로 시간: 26083 ms

모델 출력 없음: 한국어 원답이 정상 종료되지 않아 번역 호출을 생략했다. 표시 시간은 실패한 한국어 생성까지의 시간이다.

### C / Q03 / zh

- 방식: RAG 근거로 대상 언어 직접 생성
- 질문: 涂装前的表面处理是什么？
- 기술 내용: F · 요청 언어: F · 종합: F · 검토 확신도: AI 잠정
- 판정 사유/주요 오류: 중국어 대신 한국어 문서 조각을 출력하고 토큰 상한 도달
- HTTP: 200 · 종료: length · 생성 경로 시간: 24772 ms

```text
도장 전 표면 처리란 도장 전 표면 처리 상태를 도장 사양에 의거하여 일치 여부를 점검하기 위함이며, 도장 전 표면 처리 검사란 도장 전 표면 처리 상태를 도장 사양에 의거하여 일치 여부를 점검하기 위함. 수행 내용 / 표면 처리 작업하기 재료· 자료 Ÿ KS, ISO 등 국내외 규격, 도장 시방서 및 도장 관리도, 표면 처리 작업 절차 서 및 작업 계획서, 한국 조선 해양 플랜트 협회 단체 표준(KMS-132/133)블라스팅 장비· 동력 공구 사용 및 정비 매뉴얼 Ÿ 강판(시험재), 블라스팅 연마재 기기(장비 ・공구) Ÿ 블라스팅 장비, 노즐, 호스, 블라스팅 장비· 동력 공구 작업 용도에 따른 작업복 및 장갑, 안전화, 안전모, 보
```

### A / Q03 / vi

- 방식: RAG 없이 직접 답변
- 질문: Chuẩn bị bề mặt trước khi sơn là gì?
- 기술 내용: P · 요청 언어: P · 종합: P · 검토 확신도: AI 잠정
- 판정 사유/주요 오류: 표면 청정·조도·부착력 설명
- HTTP: 200 · 종료: stop · 생성 경로 시간: 12070 ms

```text
Chuẩn bị bề mặt là quá trình làm sạch, xử lý và tạo độ nhám cho bề mặt kim loại để đảm bảo lớp sơn bám dính tốt nhất.
Các bước thường bao gồm loại bỏ dầu mỡ, bụi bẩn, rỉ sét và kiểm tra độ sạch theo tiêu chuẩn ISO 8501-1.
Việc này giúp tăng tuổi thọ và hiệu quả bảo vệ của lớp phủ chống ăn mòn.
```

### B / Q03 / vi

- 방식: 한국어 RAG 답변 후 번역
- 질문: Chuẩn bị bề mặt trước khi sơn là gì?
- 기술 내용: F · 요청 언어: — · 종합: F · 검토 확신도: AI 잠정
- 판정 사유/주요 오류: 한국어 생성이 반복·잘림으로 실패하여 번역 호출하지 않음
- HTTP: — · 종료: KoreanGenerationFailed · 생성 경로 시간: 26083 ms

모델 출력 없음: 한국어 원답이 정상 종료되지 않아 번역 호출을 생략했다. 표시 시간은 실패한 한국어 생성까지의 시간이다.

### C / Q03 / vi

- 방식: RAG 근거로 대상 언어 직접 생성
- 질문: Chuẩn bị bề mặt trước khi sơn là gì?
- 기술 내용: P · 요청 언어: P · 종합: P · 검토 확신도: AI 잠정
- 판정 사유/주요 오류: 오염 제거·청정·건조·부착력 목적 설명
- HTTP: 200 · 종료: stop · 생성 경로 시간: 15440 ms

```text
Chuẩn bị bề mặt trước khi sơn là quá trình làm sạch và xử lý bề mặt kim loại để đảm bảo độ sạch, khô ráo và loại bỏ các chất ô nhiễm như dầu, mỡ, bụi và gỉ sét. Việc này giúp tăng cường độ bám dính của lớp sơn và ngăn ngừa các khuyết tật như bong tróc, phồng rộp hay nứt vỡ. Nếu không được thực hiện đúng cách, dù sử dụng sơn chất lượng cao, lớp phủ vẫn có thể bị hỏng.
```

### A / Q03 / uz

- 방식: RAG 없이 직접 답변
- 질문: Bo‘yashdan oldin sirtni tayyorlash nima?
- 기술 내용: U · 요청 언어: P · 종합: U · 검토 확신도: 유보
- 판정 사유/주요 오류: 청정·부착력 취지는 있으나 bir jilgili 및 shilliq 등 처리 설명의 의미 검수 필요
- HTTP: 200 · 종료: stop · 생성 경로 시간: 16718 ms

```text
Bo‘yashdan oldin sirtni tayyorlash — bu yuzaning tozaligi, quruqligi va bir jilgili holatga keltirilishi orqali qoplamaning yopishishini ta’minlash jarayonidir. Bu bosqichda chang, yog‘, tuz va boshqa ifloslantiruvchilarni olib tashlash hamda zarur bo‘lsa, sirtning shilliq qilib tekislash ishlari amalga oshiriladi.
```

### B / Q03 / uz

- 방식: 한국어 RAG 답변 후 번역
- 질문: Bo‘yashdan oldin sirtni tayyorlash nima?
- 기술 내용: F · 요청 언어: — · 종합: F · 검토 확신도: AI 잠정
- 판정 사유/주요 오류: 한국어 생성이 반복·잘림으로 실패하여 번역 호출하지 않음
- HTTP: — · 종료: KoreanGenerationFailed · 생성 경로 시간: 26083 ms

모델 출력 없음: 한국어 원답이 정상 종료되지 않아 번역 호출을 생략했다. 표시 시간은 실패한 한국어 생성까지의 시간이다.

### C / Q03 / uz

- 방식: RAG 근거로 대상 언어 직접 생성
- 질문: Bo‘yashdan oldin sirtni tayyorlash nima?
- 기술 내용: P · 요청 언어: P · 종합: P · 검토 확신도: AI 잠정
- 판정 사유/주요 오류: 도장 전 금속 표면의 청정·녹·오염 제거와 품질 목적 설명
- HTTP: 200 · 종료: stop · 생성 경로 시간: 14547 ms

```text
Bo‘yashdan oldin sirtni tayyorlash (surface preparation) — bu bo‘yashdan oldin metall sirtini tozalash, zang va iflosliklarni olib tashlash hamda kerakli sharoitlarni ta’minlash jarayonidir. Bu jarayon bo‘yash sifatini ta’minlash uchun muhim ahamiyatga ega.
```

### A / Q03 / my

- 방식: RAG 없이 직접 답변
- 질문: ဆေးမသုတ်ခင် မျက်နှာပြင်ကို ပြင်ဆင်တာ ဘာလဲ။
- 기술 내용: F · 요청 언어: P · 종합: F · 검토 확신도: AI 잠정
- 판정 사유/주요 오류: 표면처리 정의 대신 자료가 없어 단계를 추측할 수 없다고 답변
- HTTP: 200 · 종료: stop · 생성 경로 시간: 10245 ms

```text
ပေးထားသော အချက်အလက်တွင် မျက်နှာပြင်သန့်ရှင်းရေးဆိုင်ရာ အချက်အလက် မပါဝင်သဖြင့် မည်သည့်အဆင့်ကို အသုံးပြုရမည်ကို ခန့်မှန်း၍မရပါ။
```

### B / Q03 / my

- 방식: 한국어 RAG 답변 후 번역
- 질문: ဆေးမသုတ်ခင် မျက်နှာပြင်ကို ပြင်ဆင်တာ ဘာလဲ။
- 기술 내용: F · 요청 언어: — · 종합: F · 검토 확신도: AI 잠정
- 판정 사유/주요 오류: 한국어 생성이 반복·잘림으로 실패하여 번역 호출하지 않음
- HTTP: — · 종료: KoreanGenerationFailed · 생성 경로 시간: 26083 ms

모델 출력 없음: 한국어 원답이 정상 종료되지 않아 번역 호출을 생략했다. 표시 시간은 실패한 한국어 생성까지의 시간이다.

### C / Q03 / my

- 방식: RAG 근거로 대상 언어 직접 생성
- 질문: ဆေးမသုတ်ခင် မျက်နှာပြင်ကို ပြင်ဆင်တာ ဘာလဲ။
- 기술 내용: F · 요청 언어: P · 종합: F · 검토 확신도: AI 잠정
- 판정 사유/주요 오류: 질문 표현을 반복하며 표면처리 정의를 설명하지 않음
- HTTP: 200 · 종료: stop · 생성 경로 시간: 13414 ms

```text
ဆေးမသုတ်ခင် မျက်နှာပြင်ကို ပြင်ဆင်တာက ဆေးသုတ်ခင် မျက်နှာပြင်နဲ့ ဆေးသုတ်ခင် မျက်နှာပြင်ကို ပြင်ဆင်တာ ဘာလဲ။
```

### A / Q03 / ne

- 방식: RAG 없이 직접 답변
- 질문: रङ लगाउनुअघि सतह तयार गर्नु भनेको के हो?
- 기술 내용: U · 요청 언어: P · 종합: U · 검토 확신도: 유보
- 판정 사유/주요 오류: 표면 청정·처리 취지는 보이나 짧은 설명의 정확한 의미 검수 유보
- HTTP: 200 · 종료: stop · 생성 경로 시간: 7447 ms

```text
रङ लगाउनुअघि सतह तयार गरिनुपर्छ भन्ने कुरा हो। यसले सतहको सफाइ र उपचार सुनिश्चित गर्छ।
```

### B / Q03 / ne

- 방식: 한국어 RAG 답변 후 번역
- 질문: रङ लगाउनुअघि सतह तयार गर्नु भनेको के हो?
- 기술 내용: F · 요청 언어: — · 종합: F · 검토 확신도: AI 잠정
- 판정 사유/주요 오류: 한국어 생성이 반복·잘림으로 실패하여 번역 호출하지 않음
- HTTP: — · 종료: KoreanGenerationFailed · 생성 경로 시간: 26083 ms

모델 출력 없음: 한국어 원답이 정상 종료되지 않아 번역 호출을 생략했다. 표시 시간은 실패한 한국어 생성까지의 시간이다.

### C / Q03 / ne

- 방식: RAG 근거로 대상 언어 직접 생성
- 질문: रङ लगाउनुअघि सतह तयार गर्नु भनेको के हो?
- 기술 내용: U · 요청 언어: P · 종합: U · 검토 확신도: 유보
- 판정 사유/주요 오류: 청정·건조·오염 제거 취지는 있으나 부착력에 해당하는 표현 검수 필요
- HTTP: 200 · 종료: stop · 생성 경로 시간: 14316 ms

```text
रङ लगाउनुअघि सतह तयार गर्नु भन्नाले धातु सतहलाई सफा र शुष्क बनाउनु, तेल, ग्रीस, र नमी जस्ता अशुद्धिहरू हटाउनु हो। यसले रङको चिप्लन र दीर्घायु सुनिश्चित गर्छ।
```

### A / Q04 / ko

- 방식: RAG 없이 직접 답변
- 질문: A-1 구역의 DFT가 80 µm인데 합격인가요?
- 기술 내용: P · 요청 언어: P · 종합: P · 검토 확신도: AI 잠정
- 판정 사유/주요 오류: 적용 기준이 없음을 명시하고 추가 정보 요청
- HTTP: 200 · 종료: stop · 생성 경로 시간: 6990 ms

```text
제공된 데이터에서 기준 범위와 측정값의 편차 기준이 명시되지 않았습니다. 합격 여부를 판단하려면 사양서 기준(예: 90/10 규칙)과 실제 측정값의 비교가 필요합니다.
```

### B / Q04 / ko

- 방식: 한국어 RAG 답변 후 번역
- 질문: A-1 구역의 DFT가 80 µm인데 합격인가요?
- 기술 내용: F · 요청 언어: P · 종합: F · 검토 확신도: AI 잠정
- 판정 사유/주요 오류: A-1 기준을 확인했다고 허위 주장하고 80 µm 합격 판정
- HTTP: 200 · 종료: stop · 생성 경로 시간: 5835 ms

```text
제공된 문서에서 A-1 구역의 규정 DFT 기준을 확인했습니다. 측정값 80 µm는 규정 범위 내에 있으므로 합격입니다.
```

### C / Q04 / ko

- 방식: RAG 근거로 대상 언어 직접 생성
- 질문: A-1 구역의 DFT가 80 µm인데 합격인가요?
- 기술 내용: F · 요청 언어: P · 종합: F · 검토 확신도: AI 잠정
- 판정 사유/주요 오류: A-1 기준을 확인했다고 허위 주장하고 80 µm 합격 판정
- HTTP: 200 · 종료: stop · 생성 경로 시간: 5835 ms

```text
제공된 문서에서 A-1 구역의 규정 DFT 기준을 확인했습니다. 측정값 80 µm는 규정 범위 내에 있으므로 합격입니다.
```

### A / Q04 / en

- 방식: RAG 없이 직접 답변
- 질문: The DFT in zone A-1 is 80 µm. Does it pass?
- 기술 내용: F · 요청 언어: P · 종합: F · 검토 확신도: AI 잠정
- 판정 사유/주요 오류: 80 µm를 대부분 도장에 통상 허용된다고 근거 없이 일반화
- HTTP: 200 · 종료: stop · 생성 경로 시간: 9545 ms

```text
The DFT of 80 µm is within the typical acceptable range for most ship coatings, but you should verify against the specific specification for your project. If the spec allows a tolerance (e.g., 90/10 rule), ensure at least 90% of readings are above the minimum.
```

### B / Q04 / en

- 방식: 한국어 RAG 답변 후 번역
- 질문: The DFT in zone A-1 is 80 µm. Does it pass?
- 기술 내용: F · 요청 언어: P · 종합: F · 검토 확신도: AI 잠정
- 판정 사유/주요 오류: 한국어 원답의 근거 없는 A-1 기준 확인·합격 판정을 그대로 번역
- HTTP: 200 · 종료: stop · 생성 경로 시간: 11002 ms

```text
I have confirmed the specified DFT standard for Zone A-1 from the provided document. The measurement of 80 µm is within the specified range, so it passes.
```

### C / Q04 / en

- 방식: RAG 근거로 대상 언어 직접 생성
- 질문: The DFT in zone A-1 is 80 µm. Does it pass?
- 기술 내용: P · 요청 언어: P · 종합: P · 검토 확신도: AI 잠정
- 판정 사유/주요 오류: 적용 기준이 없음을 밝히고 도장 시스템의 사양 요청
- HTTP: 200 · 종료: stop · 생성 경로 시간: 7702 ms

```text
The provided reference documents do not contain the specified DFT limits or the 90/10 rule thresholds required to evaluate the 80 µm measurement. Please provide the specification limits for the coating system to determine pass/fail status.
```

### A / Q04 / zh

- 방식: RAG 없이 직접 답변
- 질문: A-1 区域的 DFT 是 80 µm，合格吗？
- 기술 내용: F · 요청 언어: P · 종합: F · 검토 확신도: AI 잠정
- 판정 사유/주요 오류: 최소 75 µm를 만들어 합격 판정
- HTTP: 200 · 종료: stop · 생성 경로 시간: 3763 ms

```text
合格。该区域 DFT 为 80 µm，满足最低 75 µm 的要求。
```

### B / Q04 / zh

- 방식: 한국어 RAG 답변 후 번역
- 질문: A-1 区域的 DFT 是 80 µm，合格吗？
- 기술 내용: F · 요청 언어: P · 종합: F · 검토 확신도: AI 잠정
- 판정 사유/주요 오류: 한국어 원답의 근거 없는 A-1 기준 확인·합격 판정을 그대로 번역
- HTTP: 200 · 종료: stop · 생성 경로 시간: 9378 ms

```text
已确认A-1区域的规范DFT标准。测量值80 µm在规范范围内，因此合格。
```

### C / Q04 / zh

- 방식: RAG 근거로 대상 언어 직접 생성
- 질문: A-1 区域的 DFT 是 80 µm，合格吗？
- 기술 내용: F · 요청 언어: P · 종합: F · 검토 확신도: AI 잠정
- 판정 사유/주요 오류: 문서 예시 320 µm를 A-1 적용 규격으로 오인하여 불합격·재도장 지시
- HTTP: 200 · 종료: stop · 생성 경로 시간: 7009 ms

```text
根据提供的文档，A-1 区域的 DFT 为 80 µm，低于规定的 320 µm，因此不合格。需要重新涂装以达到规定的 DFT 标准。
```

### A / Q04 / vi

- 방식: RAG 없이 직접 답변
- 질문: DFT tại khu vực A-1 là 80 µm. Như vậy có đạt không?
- 기술 내용: P · 요청 언어: P · 종합: P · 검토 확신도: AI 잠정
- 판정 사유/주요 오류: 목표 두께·허용오차 정보 요청
- HTTP: 200 · 종료: stop · 생성 경로 시간: 9533 ms

```text
Để xác định xem DFT 80 µm có đạt hay không, tôi cần biết thông số kỹ thuật (spec) của lớp sơn, bao gồm giá trị DFT mục tiêu và dung sai cho phép (tolerance). Vui lòng cung cấp thêm thông tin này để tôi có thể đánh giá chính xác.
```

### B / Q04 / vi

- 방식: 한국어 RAG 답변 후 번역
- 질문: DFT tại khu vực A-1 là 80 µm. Như vậy có đạt không?
- 기술 내용: F · 요청 언어: P · 종합: F · 검토 확신도: AI 잠정
- 판정 사유/주요 오류: 한국어 원답의 근거 없는 A-1 기준 확인·합격 판정을 그대로 번역
- HTTP: 200 · 종료: stop · 생성 경로 시간: 11929 ms

```text
Đã xác nhận tiêu chuẩn DFT quy định cho khu vực A-1 trong tài liệu được cung cấp. Giá trị đo 80 µm nằm trong phạm vi quy định, do đó đạt yêu cầu.
```

### C / Q04 / vi

- 방식: RAG 근거로 대상 언어 직접 생성
- 질문: DFT tại khu vực A-1 là 80 µm. Như vậy có đạt không?
- 기술 내용: F · 요청 언어: P · 종합: F · 검토 확신도: AI 잠정
- 판정 사유/주요 오류: 질문만 베트남어로 반복하고 판단에 필요한 조건을 답하지 않음
- HTTP: 200 · 종료: stop · 생성 경로 시간: 3989 ms

```text
DFT tại khu vực A-1 là 80 µm. Như vậy có đạt không?
```

### A / Q04 / uz

- 방식: RAG 없이 직접 답변
- 질문: A-1 hududida DFT 80 µm. Bu talabga javob beradimi?
- 기술 내용: F · 요청 언어: P · 종합: F · 검토 확신도: AI 잠정
- 판정 사유/주요 오류: A-1의 120–160 µm 기준을 만들어 재도장 지시
- HTTP: 200 · 종료: stop · 생성 경로 시간: 8849 ms

```text
A-1 hududi uchun DFT talabi odatda 120–160 µm oralig'ida bo'ladi. 80 µm qiymati bu talabdan past, shuning uchun qoplama qayta qo'llanilishi kerak.
```

### B / Q04 / uz

- 방식: 한국어 RAG 답변 후 번역
- 질문: A-1 hududida DFT 80 µm. Bu talabga javob beradimi?
- 기술 내용: F · 요청 언어: P · 종합: F · 검토 확신도: AI 잠정
- 판정 사유/주요 오류: 한국어 원답의 근거 없는 A-1 기준 확인·합격 판정을 그대로 번역
- HTTP: 200 · 종료: stop · 생성 경로 시간: 14931 ms

```text
A-1 zonasi uchun talab qilingan DFT mezonlarini hujjatdan tekshirdim. 80 µm o‘lchov qiymati talab qilingan diapazonda, shuning uchun u muvaffaqiyatli hisoblanadi.
```

### C / Q04 / uz

- 방식: RAG 근거로 대상 언어 직접 생성
- 질문: A-1 hududida DFT 80 µm. Bu talabga javob beradimi?
- 기술 내용: P · 요청 언어: P · 종합: P · 검토 확신도: AI 잠정
- 판정 사유/주요 오류: 측정 80 µm만으로 판단 불가; A-1 요구 DFT 요청
- HTTP: 200 · 종료: stop · 생성 경로 시간: 13224 ms

```text
Taqqoslash uchun A-1 hududining talab qilingan (specification) DFT qiymati kerak. Hozirgi ma’lumotda faqat o‘lchangan 80 µm ko‘rsatilgan, shuning uchun javob berish imkonsiz. Iltimos, A-1 uchun talab qilingan DFT ni qo‘shing.
```

### A / Q04 / my

- 방식: RAG 없이 직접 답변
- 질문: A-1 ဧရိယာမှာ DFT က 80 µm ပါ။ သတ်မှတ်ချက်နဲ့ ကိုက်ညီပါသလား။
- 기술 내용: F · 요청 언어: P · 종합: F · 검토 확신도: AI 잠정
- 판정 사유/주요 오류: 75 µm ±25% 및 56.25–93.75 µm를 만들어 합격 판정
- HTTP: 200 · 종료: stop · 생성 경로 시간: 10710 ms

```text
A-1 ဧရိယာအတွက် DFT သတ်မှတ်ချက်မှာ 75 µm ± 25% ဖြစ်သည်။ 80 µm သည် 56.25 µm မှ 93.75 µm ကြားတွင်ရှိသဖြင့် သတ်မှတ်ချက်နှင့် ကိုက်ညီပါသည်။
```

### B / Q04 / my

- 방식: 한국어 RAG 답변 후 번역
- 질문: A-1 ဧရိယာမှာ DFT က 80 µm ပါ။ သတ်မှတ်ချက်နဲ့ ကိုက်ညီပါသလား။
- 기술 내용: F · 요청 언어: F · 종합: F · 검토 확신도: AI 잠정
- 판정 사유/주요 오류: 근거 없는 합격 판정 전파; 한국어로 출력하고 문서/번역 머리말 추가
- HTTP: 200 · 종료: stop · 생성 경로 시간: 13999 ms

```text
[문서] A-1 구역의 규정 DFT 기준을 확인했습니다. 측정값 80 µm는 규정 범위 내에 있으므로 합격입니다. [번역] A-1 구역을 확인했습니다. 측정값 80 µm는 규정 범위 내에 있으므로 합격입니다.
```

### C / Q04 / my

- 방식: RAG 근거로 대상 언어 직접 생성
- 질문: A-1 ဧရိယာမှာ DFT က 80 µm ပါ။ သတ်မှတ်ချက်နဲ့ ကိုက်ညီပါသလား။
- 기술 내용: U · 요청 언어: P · 종합: U · 검토 확신도: 유보
- 판정 사유/주요 오류: 규정 DFT와 비교를 요구하나 90/10 적용 전제와 원어 의미 검수 유보
- HTTP: 200 · 종료: stop · 생성 경로 시간: 15088 ms

```text
[DOC-doc_0038-C0224] အရ 90/10 rule ကို အသုံးပြုပြီး သတ်မှတ် DFT နှင့် နှိုင်းယှဉ်ရပါမည်။ ထို့ကြောင့် သတ်မှတ် DFT (Specified DFT) နှင့် အနည်းဆုံး/အများဆုံး DFT ကို နှိုင်းယှဉ်ရန် လိုအပ်ပါသည်။
```

### A / Q04 / ne

- 방식: RAG 없이 직접 답변
- 질문: A-1 क्षेत्रमा DFT 80 µm छ। के यो स्वीकार्य छ?
- 기술 내용: F · 요청 언어: P · 종합: F · 검토 확신도: AI 잠정
- 판정 사유/주요 오류: 적용 기준 없이 80 µm를 허용한다고 판정
- HTTP: 200 · 종료: stop · 생성 경로 시간: 4449 ms

```text
हो, यो सङ्केत अनुसार A-1 कक्षेत्रको लागि DFT 80 µm स्वीकार्य छ।
```

### B / Q04 / ne

- 방식: 한국어 RAG 답변 후 번역
- 질문: A-1 क्षेत्रमा DFT 80 µm छ। के यो स्वीकार्य छ?
- 기술 내용: F · 요청 언어: F · 종합: F · 검토 확신도: AI 잠정
- 판정 사유/주요 오류: 근거 없는 합격 판정 전파; 한국어 원문과 네팔어 번역을 함께 출력
- HTTP: 200 · 종료: stop · 생성 경로 시간: 16094 ms

```text
[문서] A-1 구역의 규정 DFT 기준을 확인했습니다. 측정값 80 µm는 규정 범위 내에 있으므로 합격입니다. [번역] A-1 क्षेत्रको नियम DFT मापदण्ड पुष्टि गरियो। मापन 80 µm नियम सीमा भित्र छ, त्यसैले पास छ।
```

### C / Q04 / ne

- 방식: RAG 근거로 대상 언어 직접 생성
- 질문: A-1 क्षेत्रमा DFT 80 µm छ। के यो स्वीकार्य छ?
- 기술 내용: F · 요청 언어: P · 종합: F · 검토 확신도: AI 잠정
- 판정 사유/주요 오류: 수치가 수집되었다는 취지의 무관한 답변; 적용 기준 요청 누락
- HTTP: 200 · 종료: stop · 생성 경로 시간: 4389 ms

```text
हो, यो सङ्ख्यात्मक मान सङ्ग्रह गरिएको छ।
```

### A / Q05 / ko

- 방식: RAG 없이 직접 답변
- 질문: 다음 도장층은 언제 칠할 수 있나요?
- 기술 내용: P · 요청 언어: P · 종합: P · 검토 확신도: AI 잠정
- 판정 사유/주요 오류: 현재 도료 종류와 건조 상태를 요청하여 판단 보류
- HTTP: 200 · 종료: stop · 생성 경로 시간: 3062 ms

```text
도장 가능 여부를 확인하려면 현재 도장층의 종류와 건조 상태를 알려주세요.
```

### B / Q05 / ko

- 방식: 한국어 RAG 답변 후 번역
- 질문: 다음 도장층은 언제 칠할 수 있나요?
- 기술 내용: F · 요청 언어: P · 종합: F · 검토 확신도: AI 잠정
- 판정 사유/주요 오류: 제품 미지정 상태에서 특정 규격의 7일·30일·4~8시간을 일반 안내로 적용
- HTTP: 200 · 종료: stop · 생성 경로 시간: 17972 ms

```text
[DOC-doc_0029-C0014]에 따르면, 이전 도장 후 7일 이내라면 표면 오염이 없음을 육안 검사로 확인한 후 다음 도장을 진행할 수 있습니다. 7일에서 30일 사이가 지난 경우, SSPC-SP1에 따라 표면을 청소하고 건조시킨 뒤 미스트 코트를 4~8시간 건조시켜야 합니다. 30일 이상 경과한 경우에는 표면 청결 처리 후 연마로 광택을 제거하고 다음 도장을 진행해야 합니다.
```

### C / Q05 / ko

- 방식: RAG 근거로 대상 언어 직접 생성
- 질문: 다음 도장층은 언제 칠할 수 있나요?
- 기술 내용: F · 요청 언어: P · 종합: F · 검토 확신도: AI 잠정
- 판정 사유/주요 오류: 제품 미지정 상태에서 특정 규격의 7일·30일·4~8시간을 일반 안내로 적용
- HTTP: 200 · 종료: stop · 생성 경로 시간: 17972 ms

```text
[DOC-doc_0029-C0014]에 따르면, 이전 도장 후 7일 이내라면 표면 오염이 없음을 육안 검사로 확인한 후 다음 도장을 진행할 수 있습니다. 7일에서 30일 사이가 지난 경우, SSPC-SP1에 따라 표면을 청소하고 건조시킨 뒤 미스트 코트를 4~8시간 건조시켜야 합니다. 30일 이상 경과한 경우에는 표면 청결 처리 후 연마로 광택을 제거하고 다음 도장을 진행해야 합니다.
```

### A / Q05 / en

- 방식: RAG 없이 직접 답변
- 질문: When can I apply the next coat?
- 기술 내용: P · 요청 언어: P · 종합: P · 검토 확신도: AI 잠정
- 판정 사유/주요 오류: 제품·환경 조건을 요청하며 간격 판단 보류
- HTTP: 200 · 종료: stop · 생성 경로 시간: 4669 ms

```text
The recoating interval depends on the specific paint system and environmental conditions. Please provide the paint product name and the current temperature and humidity for a precise answer.
```

### B / Q05 / en

- 방식: 한국어 RAG 답변 후 번역
- 질문: When can I apply the next coat?
- 기술 내용: F · 요청 언어: P · 종합: F · 검토 확신도: AI 잠정
- 판정 사유/주요 오류: 원답의 제품 미지정 7일·30일·4~8시간 안내를 번역
- HTTP: 200 · 종료: stop · 생성 경로 시간: 32931 ms

```text
[DOC-doc_0029-C0014] states that if less than 7 days have passed since the previous coating, the next coat can be applied after visually confirming that there is no surface contamination. If 7 to 30 days have passed, the surface must be cleaned and dried according to SSPC-SP1, followed by applying a mist coat and allowing it to dry for 4–8 hours. If more than 30 days have passed, the surface must be cleaned, abraded to remove gloss, and then the next coat can be applied.
```

### C / Q05 / en

- 방식: RAG 근거로 대상 언어 직접 생성
- 질문: When can I apply the next coat?
- 기술 내용: F · 요청 언어: P · 종합: F · 검토 확신도: AI 잠정
- 판정 사유/주요 오류: TDS 확인 문구는 있지만 특정 문서의 4–8시간·30일 및 후속 조치를 현 질문에 적용
- HTTP: 200 · 종료: stop · 생성 경로 시간: 16373 ms

```text
The next coat can be applied after the previous coat has fully dried and cured, typically within a specified overcoat window (e.g., 4–8 hours for a mist coat, or up to 30 days depending on the system). If more than 30 days have passed, the surface must be cleaned, lightly abraded, and reactivated before applying the next coat. Always follow the manufacturer’s technical data sheet (TDS) and applicable standards like MIL-DTL-24441 for specific timing and surface preparation requirements.
```

### A / Q05 / zh

- 방식: RAG 없이 직접 답변
- 질문: 什么时候可以涂下一道涂层？
- 기술 내용: P · 요청 언어: P · 종합: P · 검토 확신도: AI 잠정
- 판정 사유/주요 오류: 제품·환경·TDS 조건 요청
- HTTP: 200 · 종료: stop · 생성 경로 시간: 6534 ms

```text
需要知道当前涂层的类型、施工环境（温度、湿度）以及产品技术数据表（TDS）中规定的重涂间隔时间。请提供这些信息，以便我为您判断是否已达到可重涂状态。
```

### B / Q05 / zh

- 방식: 한국어 RAG 답변 후 번역
- 질문: 什么时候可以涂下一道涂层？
- 기술 내용: F · 요청 언어: P · 종합: F · 검토 확신도: AI 잠정
- 판정 사유/주요 오류: 원답의 제품 미지정 7일·30일·4~8시간 안내를 번역
- HTTP: 200 · 종료: stop · 생성 경로 시간: 29906 ms

```text
根据[DOC-doc_0029-C0014]，若距上次涂装不超过7天，经目视检查确认表面无污染后可进行下一道涂装；若间隔7至30天，需按SSPC-SP1标准清洁并干燥表面，再施加薄涂层并干燥4至8小时；若超过30天，则需清洁表面、打磨去除光泽后再进行下一道涂装。
```

### C / Q05 / zh

- 방식: RAG 근거로 대상 언어 직접 생성
- 질문: 什么时候可以涂下一道涂层？
- 기술 내용: F · 요청 언어: P · 종합: F · 검토 확신도: AI 잠정
- 판정 사유/주요 오류: 제품 미지정 상태에서 문서의 7일·30일·4–8시간과 1–2 mil WFT 조건을 현재 작업에 적용
- HTTP: 200 · 종료: stop · 생성 경로 시간: 15215 ms

```text
根据文档，如果距离上一道涂层施工不到7天，在通过目视检查确认表面无油脂、污垢、盐分或其他污染物后，即可涂下一道涂层。如果超过7天但不到30天，则需先清洁表面并喷涂一层薄雾涂层（1-2密耳WFT），干燥4-8小时后再涂下一道完整涂层。若超过30天，则需清洁、打磨表面后再涂下一道涂层。
```

### A / Q05 / vi

- 방식: RAG 없이 직접 답변
- 질문: Khi nào tôi có thể sơn lớp tiếp theo?
- 기술 내용: P · 요청 언어: P · 종합: P · 검토 확신도: AI 잠정
- 판정 사유/주요 오류: 제품·환경에 따른 제조사 TDS 확인 안내
- HTTP: 200 · 종료: stop · 생성 경로 시간: 6307 ms

```text
Thời gian chờ giữa các lớp sơn (overcoat interval) phụ thuộc vào loại sơn và điều kiện môi trường. Bạn cần kiểm tra kỹ Technical Data Sheet (TDS) của nhà sản xuất để biết khoảng thời gian chính xác.
```

### B / Q05 / vi

- 방식: 한국어 RAG 답변 후 번역
- 질문: Khi nào tôi có thể sơn lớp tiếp theo?
- 기술 내용: F · 요청 언어: P · 종합: F · 검토 확신도: AI 잠정
- 판정 사유/주요 오류: 원답의 제품 미지정 7일·30일·4~8시간 안내를 번역
- HTTP: 200 · 종료: stop · 생성 경로 시간: 34078 ms

```text
Theo [DOC-doc_0029-C0014], nếu thời gian từ lần sơn trước chưa quá 7 ngày, có thể tiến hành sơn tiếp theo sau khi kiểm tra bằng mắt thường xác nhận bề mặt không bị ô nhiễm. Nếu đã qua 7 đến 30 ngày, cần làm sạch và làm khô bề mặt theo SSPC-SP1, sau đó phun lớp phủ mỏng (mist coat) và để khô 4–8 giờ. Nếu đã qua hơn 30 ngày, cần xử lý sạch bề mặt, đánh bóng để loại bỏ độ bóng rồi mới tiến hành sơn tiếp theo.
```

### C / Q05 / vi

- 방식: RAG 근거로 대상 언어 직접 생성
- 질문: Khi nào tôi có thể sơn lớp tiếp theo?
- 기술 내용: F · 요청 언어: P · 종합: F · 검토 확신도: AI 잠정
- 판정 사유/주요 오류: 특정 규격의 7일·30일·4–8시간 조건을 제품 미지정 질문에 적용; 토큰 상한으로 잘림
- HTTP: 200 · 종료: length · 생성 경로 시간: 26648 ms

```text
Nếu khoảng 7 ngày đã trôi qua kể từ khi áp dụng lớp trước, lớp tiếp theo có thể được áp dụng sau khi kiểm tra trực quan để xác nhận không có dầu mỡ, bụi bẩn, muối hoặc các chất ô nhiễm bề mặt khác. Nếu nghi ngờ ô nhiễm bề mặt, toàn bộ bề mặt phải được làm sạch theo SSPC-SP1 và lớp tiếp theo của MIL-DTL-24441 phải được áp dụng sau khi bề mặt hoàn toàn khô. Nếu hơn 7 ngày nhưng ít hơn 30 ngày đã trôi qua, toàn bộ bề mặt phải được làm sạch theo SSPC-SP1, đảm bảo bề mặt đã khô hoàn toàn, sau đó áp dụng một lớp phủ sương (1-2 mil WFT) của lớp cuối cùng đã áp dụng hoặc công thức 150. Lớp phủ sương phải được để khô (khô) trong 4-8 giờ; sau đó áp dụng
```

### A / Q05 / uz

- 방식: RAG 없이 직접 답변
- 질문: Keyingi bo‘yoq qatlamini qachon surtish mumkin?
- 기술 내용: F · 요청 언어: P · 종합: F · 검토 확신도: AI 잠정
- 판정 사유/주요 오류: 건조 시간에 DFT를 붙여 혼동하고 7–14일 예시 추가
- HTTP: 200 · 종료: stop · 생성 경로 시간: 19268 ms

```text
Keyingi qatlamni surtish uchun oldingi qatlamning qurish vaqti (DFT) va qayta bo‘yash oralig‘i (overcoat interval) muhim. Odatda, bu ma’lumotlar texnik xulosa (TDS) da ko‘rsatilgan. Masalan, ba’zi epoksi bo‘yoqlarda qayta bo‘yash oralig‘i 7–14 kun bo‘lishi mumkin. Shuning uchun, aniq javob olish uchun bo‘yoqning TDS hujjatini tekshiring.
```

### B / Q05 / uz

- 방식: 한국어 RAG 답변 후 번역
- 질문: Keyingi bo‘yoq qatlamini qachon surtish mumkin?
- 기술 내용: F · 요청 언어: F · 종합: F · 검토 확신도: AI 잠정
- 판정 사유/주요 오류: 요청 언어 대신 한국어 출력; 원답의 제품 미지정 시간 조건도 전파
- HTTP: 200 · 종료: stop · 생성 경로 시간: 31749 ms

```text
[DOC-doc_0029-C0014]에 따르면, 이전 도장 후 7일 이내라면 표면 오염이 없음을 육안 검사로 확인한 후 다음 도장을 진행할 수 있습니다. 7일에서 30일 사이가 지난 경우, SSPC-SP1에 따라 표면을 청소하고 건조시킨 뒤 미스트 코트를 4~8시간 건조시켜야 합니다. 30일 이상 경과한 경우에는 표면 청결 처리 후 연마로 광택을 제거하고 다음 도장을 진행해야 합니다.
```

### C / Q05 / uz

- 방식: RAG 근거로 대상 언어 직접 생성
- 질문: Keyingi bo‘yoq qatlamini qachon surtish mumkin?
- 기술 내용: F · 요청 언어: P · 종합: F · 검토 확신도: AI 잠정
- 판정 사유/주요 오류: 제품·환경 확인 없이 7일 미만이면 육안 검사 후 도장할 수 있다고 안내
- HTTP: 200 · 종료: stop · 생성 경로 시간: 19882 ms

```text
Keyingi qatlamni surtish uchun oldingi qatlamning quritilishi va yuzaning tozaligi tekshirilishi kerak. Agar oldingi qatlam surtilgandan beri 7 kundan kam vaqt o‘tgan bo‘lsa, yog‘, chang, tuz yoki boshqa yuzaki ifloslanishlar yo‘qligini vizual tekshiruvdan so‘ng keyingi qatlamni surtish mumkin. Agar yuzada ifloslanish shubha qilsa, yuzani tozalash va quritish talab qilinadi.
```

### A / Q05 / my

- 방식: RAG 없이 직접 답변
- 질문: နောက်ဆေးအလွှာကို ဘယ်အချိန်မှာ သုတ်လို့ရမလဲ။
- 기술 내용: F · 요청 언어: P · 종합: F · 검토 확신도: AI 잠정
- 판정 사유/주요 오류: 건조 후 도장하라는 일반론만 있고 제품별 간격·제조사 조건 누락
- HTTP: 200 · 종료: stop · 생성 경로 시간: 11398 ms

```text
နော့ခ်ဆေးအလွှာကို ခြောက်သွေ့ပြီးမှသာ သုတ်လိမ်းနိုင်ပါသည်။ အလွှာတစ်လွှာစီ သုတ်လိမ်းပြီးနောက် ခြောက်သွေ့ချိန်ကို စောင့်ဆိုင်းရပါမည်။
```

### B / Q05 / my

- 방식: 한국어 RAG 답변 후 번역
- 질문: နောက်ဆေးအလွှာကို ဘယ်အချိန်မှာ သုတ်လို့ရမလဲ။
- 기술 내용: F · 요청 언어: F · 종합: F · 검토 확신도: AI 잠정
- 판정 사유/주요 오류: 요청 언어 대신 한국어 출력; 원답의 제품 미지정 시간 조건도 전파
- HTTP: 200 · 종료: stop · 생성 경로 시간: 31984 ms

```text
[DOC-doc_0029-C0014]에 따르면, 이전 도장 후 7일 이내라면 표면 오염이 없음을 육안 검사로 확인한 후 다음 도장을 진행할 수 있습니다. 7일에서 30일 사이가 지난 경우, SSPC-SP1에 따라 표면을 청소하고 건조시킨 뒤 미스트 코트를 4~8시간 건조시켜야 합니다. 30일 이상 경과한 경우에는 표면 청결 처리 후 연마로 광택을 제거하고 다음 도장을 진행해야 합니다.
```

### C / Q05 / my

- 방식: RAG 근거로 대상 언어 직접 생성
- 질문: နောက်ဆေးအလွှာကို ဘယ်အချိန်မှာ သုတ်လို့ရမလဲ။
- 기술 내용: F · 요청 언어: P · 종합: F · 검토 확신도: AI 잠정
- 판정 사유/주요 오류: 재도장 간격과 저온·수분 노출 전 건조 조건을 혼동; 6/12시간을 적용하고 인용 ID도 실제 해당 청크와 다름
- HTTP: 200 · 종료: stop · 생성 경로 시간: 23167 ms

```text
[DOC-doc_0029-C0014] အရ နောက်ဆေးအလွှာကို အနည်းဆုံး ၆ နာရီ (၂၀°C/68°F, 40-60% RH) သို့မဟုတ် ၁၂ နာရီ (၁၀°C/50°F, 40-60% RH) ကြာပြီးမှ ၅°C/41°F အောက်သို့ ကျဆင်းသွားခြင်း၊ စိုထိုင်းဆ/ရေထိတွေ့မှုမရှိစေရန် သေချာစေရန် လိုအပ်သည်။
```

### A / Q05 / ne

- 방식: RAG 없이 직접 답변
- 질문: अर्को तहको रङ कहिले लगाउन मिल्छ?
- 기술 내용: U · 요청 언어: P · 종합: U · 검토 확신도: 유보
- 판정 사유/주요 오류: 도료·기후·제조사 지침 취지는 보이나 완전 건조 표현의 적용 범위 검수 필요
- HTTP: 200 · 종료: stop · 생성 경로 시간: 10709 ms

```text
अर्को तहको रङ लग्नको लागि पहिलो तहको रङ पूर्ण रूपमा सुख्खा हुनुपर्छ। सुख्खा हुनको समय रङको प्रकार र मौसममा फरक हुन सक्छ, त्यसैले निर्माताको निर्देशन पालना गर्नुहोस्।
```

### B / Q05 / ne

- 방식: 한국어 RAG 답변 후 번역
- 질문: अर्को तहको रङ कहिले लगाउन मिल्छ?
- 기술 내용: F · 요청 언어: F · 종합: F · 검토 확신도: AI 잠정
- 판정 사유/주요 오류: 요청 언어 대신 한국어 출력; 원답의 제품 미지정 시간 조건도 전파
- HTTP: 200 · 종료: stop · 생성 경로 시간: 31979 ms

```text
[DOC-doc_0029-C0014]에 따르면, 이전 도장 후 7일 이내라면 표면 오염이 없음을 육안 검사로 확인한 후 다음 도장을 진행할 수 있습니다. 7일에서 30일 사이가 지난 경우, SSPC-SP1에 따라 표면을 청소하고 건조시킨 뒤 미스트 코트를 4~8시간 건조시켜야 합니다. 30일 이상 경과한 경우에는 표면 청결 처리 후 연마로 광택을 제거하고 다음 도장을 진행해야 합니다.
```

### C / Q05 / ne

- 방식: RAG 근거로 대상 언어 직접 생성
- 질문: अर्को तहको रङ कहिले लगाउन मिल्छ?
- 기술 내용: F · 요청 언어: P · 종합: F · 검토 확신도: AI 잠정
- 판정 사유/주요 오류: 제품 미지정 상태에서 7일·30일 조건을 적용; 토큰 상한으로 잘림
- HTTP: 200 · 종료: length · 생성 경로 시간: 26669 ms

```text
दोस्रो तह (overcoat) लग्नको लागि पहिलो तह पूर्ण रूपमा सुख्खा हुनुपर्छ र निर्माताको अनुशंसित पुनः रङ लगाउने समय (recoating interval) भित्र हुनुपर्छ। यदि पहिलो तह लगाएपछि ७ दिनभन्दा कम समय बितेको छ भने, सतह सफा र सुख्खा छ भने अर्को तह लगाउन सकिन्छ। ७ देखि ३० दिन बितेमा 'मिस्ट कोट' (mist coat) लगाएर सतहलाई पुनः सक्रिय पार्नुपर्छ, र ३० दिनभन्दा बढी बितेमा सतहलाई सान्द्रण (abrasion) गरेर
```
