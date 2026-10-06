# 7개 언어 질문·답변 예시 35쌍

작성일: 2026-10-06. 도장 지식 질문 5개를 7개 언어로 구성했다. 질문과 답변은 같은 언어로 작성했으며, 각 표의 한국어 행이 의미 비교 기준이다. **모델을 실제 호출해 얻은 응답이 아니라 시연·번역 검수·평가에 사용할 기대 답변 작성본**이다. 실제 모델 응답과 처리 시간은 [배포 검증 기록](local_voice_multilingual_release_evidence.md)에 있다.

| 언어 | 현재 평가 API | 아래 문장 상태 |
|---|---|---|
| 한국어 | `ko` 활성 | 한국어 의미 기준 |
| 영어 | `en` 활성 | 기대 답변 작성본, 전체 문장 전문가 검수 전 |
| 중국어 | `zh` 활성, 간체 | 기대 답변 작성본, 전체 문장 전문가 검수 전 |
| 베트남어 | 비활성 | 번역 검수용 초안 |
| 우즈벡어 | 비활성, 라틴 문자 초안 | 번역 검수용 초안 |
| 미얀마어 | 비활성 | 번역 검수용 초안 |
| 네팔어 | 비활성 | 번역 검수용 초안 |

기술 용어의 실제 문서 표기와 출처는 [현지 문서 대조표](local_voice_multilingual_terminology.md)에 있다. 아래 외국어 문장 전체가 그 문서에서 인용됐거나 현지 전문가에게 승인됐다는 뜻은 아니다. 특히 미얀마어·네팔어의 DFT는 확정되지 않은 전문 명칭을 새로 붙이지 않고 약어와 설명식 문장으로 제시했다.

## Q01. 건조도막두께의 뜻

평가할 의미: 건조된 도막의 두께를 설명한다. 젖은 상태의 두께나 도료 사용량으로 바꾸지 않는다. 용어 근거: 대조표 T01, KO1/EN1/ZH1/VI1/UZ1.

| 언어 | 질문 | 답변 예시 |
|---|---|---|
| 한국어 | 건조도막두께, DFT가 뭐예요? | 도료가 건조된 뒤 형성된 도막의 두께입니다. |
| 영어 | What is dry film thickness, or DFT? | It is the thickness of the coating after the paint has dried. |
| 중국어 | 什么是干膜厚度（DFT）？ | 干膜厚度是涂料干燥后形成的涂层厚度。 |
| 베트남어 | Chiều dày màng sơn khô (DFT) là gì? | Đó là chiều dày của lớp sơn sau khi khô. |
| 우즈벡어 | Quruq plyonka qalinligi (DFT) nima? | Bu bo‘yoq qurigandan keyin hosil bo‘lgan qatlamning qalinligidir. |
| 미얀마어 | DFT ဆိုတာ ဘာလဲ။ | ဆေးခြောက်ပြီးနောက် ဖြစ်ပေါ်လာတဲ့ ဆေးအလွှာရဲ့ အထူကို ဆိုလိုပါတယ်။ |
| 네팔어 | DFT भनेको के हो? | यो रङ सुकिसकेपछि बनेको तहको मोटाइ हो। |

## Q02. 저도막의 뜻

평가할 의미: 적용 규격의 최소 요구 두께에 미달한 상태다. 단순히 다른 곳보다 얇다는 사실만으로 불합격을 확정하거나 모든 제품에 공통인 최소 수치를 만들어 내면 안 된다. 외국어 질문은 표준 용어라고 확정되지 않은 명칭 대신 “너무 얇은 도막”이라는 설명을 사용했다.

| 언어 | 질문 | 답변 예시 |
|---|---|---|
| 한국어 | 저도막이 뭐예요? | 도막 두께가 적용 규격에서 요구하는 최소 두께보다 얇은 상태입니다. |
| 영어 | What does it mean when a coating is too thin? | It means the coating thickness is below the minimum required by the applicable specification. |
| 중국어 | 涂层太薄是什么意思？ | 指涂层厚度低于适用规格要求的最小厚度。 |
| 베트남어 | Lớp sơn quá mỏng có nghĩa là gì? | Nghĩa là chiều dày lớp sơn thấp hơn mức tối thiểu mà quy cách áp dụng yêu cầu. |
| 우즈벡어 | Bo‘yoq qatlami juda yupqa degani nima? | Bu qatlam qalinligi tegishli texnik talabda belgilangan eng kam qalinlikdan past ekanini bildiradi. |
| 미얀마어 | ဆေးအလွှာ ပါးလွန်းတယ်ဆိုတာ ဘာကို ဆိုလိုတာလဲ။ | သက်ဆိုင်ရာ သတ်မှတ်ချက်အရ လိုအပ်တဲ့ အနည်းဆုံးအထူထက် ဆေးအလွှာက ပိုပါးနေတာကို ဆိုလိုပါတယ်။ |
| 네팔어 | रङको तह धेरै पातलो हुनु भनेको के हो? | यसको अर्थ तहको मोटाइ लागू मापदण्डले तोकेको न्यूनतम मोटाइभन्दा कम हुनु हो। |

## Q03. 표면처리의 뜻

평가할 의미: 도장 전 피도면을 준비하는 작업을 설명한다. 구체적인 제품·표면 정보 없이 특정 화학약품이나 작업 방법을 지시하지 않는다. 용어 근거: 대조표 T11, KO1/EN1/ZH1/VI1. 우즈벡어·미얀마어·네팔어는 문장 검수 대상이다.

| 언어 | 질문 | 답변 예시 |
|---|---|---|
| 한국어 | 도장 전에 하는 표면처리는 뭐예요? | 도장할 표면의 먼지, 기름, 녹 등을 제거하고 도료를 칠할 수 있도록 표면을 준비하는 작업입니다. |
| 영어 | What is surface preparation before painting? | It means removing dust, oil, rust and similar contamination, and preparing the surface for painting. |
| 중국어 | 涂装前的表面处理是什么？ | 是清除待涂表面的灰尘、油污、锈蚀等，并使表面适合涂装的准备工作。 |
| 베트남어 | Chuẩn bị bề mặt trước khi sơn là gì? | Đó là việc loại bỏ bụi, dầu mỡ, gỉ sét và các chất bẩn tương tự, đồng thời chuẩn bị bề mặt để sơn. |
| 우즈벡어 | Bo‘yashdan oldin sirtni tayyorlash nima? | Bu sirtdagi chang, yog‘, zang va shunga o‘xshash iflosliklarni olib tashlab, sirtni bo‘yashga tayyorlashdir. |
| 미얀마어 | ဆေးမသုတ်ခင် မျက်နှာပြင်ကို ပြင်ဆင်တာ ဘာလဲ။ | ဆေးသုတ်မယ့် မျက်နှာပြင်ပေါ်က ဖုန်၊ ဆီ၊ သံချေးနဲ့ အလားတူ အညစ်အကြေးတွေကို ဖယ်ရှားပြီး ဆေးသုတ်ဖို့ ပြင်ဆင်တာပါ။ |
| 네팔어 | रङ लगाउनुअघि सतह तयार गर्नु भनेको के हो? | सतहको धुलो, तेल, खिया र यस्तै फोहोर हटाएर रङ लगाउन मिल्ने गरी तयार गर्नु हो। |

## Q04. 수치만 있고 판정 기준이 없는 경우

평가할 의미: `A-1`, `80 µm`를 정확히 유지하면서 추가 기준을 요청한다. 이 예시는 실제 합부 판정 기능의 구현을 의미하지 않으며, 정보가 부족한 지식 질문에 어떻게 답해야 하는지 보여준다. 한 번의 두께 측정만으로 구역 전체 합격을 선언하면 실패다.

| 언어 | 질문 | 답변 예시 |
|---|---|---|
| 한국어 | A-1 구역의 DFT가 80 µm인데 합격인가요? | A-1의 측정값 80 µm만으로는 합격 여부를 판정할 수 없습니다. 적용 규격의 두께 기준과 검사 규칙이 필요합니다. |
| 영어 | The DFT in zone A-1 is 80 µm. Does it pass? | The 80 µm reading in A-1 alone is not enough to determine acceptance. The thickness criteria and inspection rules in the applicable specification are needed. |
| 중국어 | A-1 区域的 DFT 是 80 µm，合格吗？ | 仅凭 A-1 的 80 µm 测量值无法判定是否合格，还需要适用规格中的厚度要求和检验规则。 |
| 베트남어 | DFT tại khu vực A-1 là 80 µm. Như vậy có đạt không? | Chỉ với giá trị đo 80 µm tại A-1 thì chưa thể kết luận đạt hay không. Cần có yêu cầu về chiều dày và quy tắc kiểm tra trong quy cách áp dụng. |
| 우즈벡어 | A-1 hududida DFT 80 µm. Bu talabga javob beradimi? | A-1 hududidagi 80 µm o‘lchovning o‘zi bilan talabga mosligini aniqlab bo‘lmaydi. Tegishli texnik hujjatdagi qalinlik talablari va tekshirish qoidalari kerak. |
| 미얀마어 | A-1 ဧရိယာမှာ DFT က 80 µm ပါ။ သတ်မှတ်ချက်နဲ့ ကိုက်ညီပါသလား။ | A-1 မှာ တိုင်းတာထားတဲ့ 80 µm တန်ဖိုးတစ်ခုတည်းနဲ့ ကိုက်ညီမှုကို မဆုံးဖြတ်နိုင်ပါဘူး။ သက်ဆိုင်ရာ သတ်မှတ်ချက်ထဲက အထူစံနှုန်းနဲ့ စစ်ဆေးရေးစည်းမျဉ်းတွေ လိုပါတယ်။ |
| 네팔어 | A-1 क्षेत्रमा DFT 80 µm छ। के यो स्वीकार्य छ? | A-1 को 80 µm मापन मात्रबाट स्वीकार्य छ कि छैन भनेर निर्णय गर्न सकिँदैन। लागू प्राविधिक मापदण्डको मोटाइसम्बन्धी सीमा र निरीक्षण नियमहरू चाहिन्छ। |

## Q05. 다음 도장층을 칠할 수 있는 시간

평가할 의미: 제품과 작업 조건에 맞는 최소·최대 재도장 간격을 확인해야 한다. 근거 없이 “몇 분 후”라고 답하지 않는다. 용어·개념 근거: 대조표 T10, EN1/ZH1/VI2의 재도장 조건. 외국어 답변 문장은 작성본이며 제조사 문서의 인용문이 아니다.

| 언어 | 질문 | 답변 예시 |
|---|---|---|
| 한국어 | 다음 도장층은 언제 칠할 수 있나요? | 제품 기술자료(TDS)와 작업 온도에 따른 최소·최대 재도장 간격을 확인해야 합니다. 지금 정보만으로는 시간을 정할 수 없습니다. |
| 영어 | When can I apply the next coat? | Check the minimum and maximum recoating intervals for the product and working temperature in the technical data sheet (TDS). The information provided is not enough to specify a time. |
| 중국어 | 什么时候可以涂下一道涂层？ | 需要查阅产品技术数据表（TDS），确认相应施工温度下的最短和最长覆涂间隔。现有信息不足以确定具体时间。 |
| 베트남어 | Khi nào tôi có thể sơn lớp tiếp theo? | Cần kiểm tra thời gian tối thiểu và tối đa giữa các lớp sơn theo sản phẩm và nhiệt độ thi công trong tài liệu kỹ thuật (TDS). Thông tin hiện có chưa đủ để xác định thời gian cụ thể. |
| 우즈벡어 | Keyingi bo‘yoq qatlamini qachon surtish mumkin? | Mahsulotning texnik ma’lumot varag‘ida (TDS) ish haroratiga mos eng qisqa va eng uzoq qayta bo‘yash oralig‘ini tekshirish kerak. Hozirgi ma’lumotlar bilan aniq vaqtni belgilab bo‘lmaydi. |
| 미얀마어 | နောက်ဆေးအလွှာကို ဘယ်အချိန်မှာ သုတ်လို့ရမလဲ။ | ထုတ်ကုန်နည်းပညာအချက်အလက်စာရွက် (TDS) မှာ အလုပ်လုပ်တဲ့အပူချိန်အလိုက် နောက်ဆေးအလွှာသုတ်ဖို့ အနည်းဆုံးနဲ့ အများဆုံး စောင့်ရမယ့်အချိန်ကို စစ်ဆေးရပါမယ်။ အခုအချက်အလက်နဲ့ အချိန်အတိအကျ မသတ်မှတ်နိုင်ပါဘူး။ |
| 네팔어 | अर्को तहको रङ कहिले लगाउन मिल्छ? | उत्पादनको प्राविधिक विवरणपत्र (TDS) मा काम गर्ने तापक्रमअनुसार अर्को तह लगाउने न्यूनतम र अधिकतम समय अन्तराल जाँच्नुपर्छ। अहिलेको जानकारीबाट निश्चित समय तोक्न सकिँदैन। |

## 예시 사용 방법

- 시연: 현재 활성 언어인 한국어·영어·중국어 질문부터 사용한다. `input_language`와 `output_language`를 동일하게 지정하면 질문 언어로 답변을 받는다.
- 검수: 보류 언어 4종은 이 문서를 원어민·도장 담당자에게 전달해 질문의 자연스러움과 답변 의미를 확인한다. API 성공 사례로 소개하지 않는다.
- 평가: 정답 문장과 글자 단위 일치를 요구하지 않는다. 정의·부정·조건·숫자·단위·구역 ID가 유지되는지 확인한다. Q04의 합격 단정, Q05의 임의 시간 제시는 실패로 기록한다.
- 음성 시험: 질문 문장을 실제 화자가 읽고, 전사 원문부터 확인한다. 인식 오류와 답변·번역 오류를 따로 기록한다.

현재 배포 API에서 Q01 영어 질문을 입력하고 영어로 받는 요청 예시:

```json
{"query":{"user_id":"USER-001","prompt":"What is dry film thickness, or DFT?","task_type":"knowledge_search","use_rag":"Y"},"input_language":"en","output_language":"en"}
```

위 요청의 실제 응답 문장과 처리 시간은 실행마다 달라질 수 있다. 이 문서의 35쌍을 학습 데이터나 확정 평가 정답으로 사용하기 전에는 언어별 검수를 완료한다.
