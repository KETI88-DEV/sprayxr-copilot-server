# 도장 전문용어 현지 문서 대조표

2026-11-06, v0.3. Spark 본 Co-Pilot API가 받던 12개 언어 코드를 대상으로 핵심 5개 용어(T01 건조도막두께, T02 습도막두께, T09 에어리스 스프레이, T10 최대 재도장 간격, T11 표면처리)를 언어별 현지 1차 문서로 다시 조사했다(§6). 질의 언어는 12개를 모두 유지하고, 다섯 용어가 모두 문서로 확인된 7개 언어에만 런타임 용어집 rc3를 적용한다(§4).

2026-10-06, v0.2. 기존 v0.1의 작성자 번역 후보를 재검토했다. **문서에서 실제 표기를 찾은 항목만 용례 확인으로 표시한다.** 현지 문서 확인은 전문 번역가의 전체 문장 품질 승인과는 별개다. 현지 영어 문서가 존재하는 것을 그 국가 언어의 용어 근거로 삼지 않는다.

평가용 게이트웨이(rc.1)는 한국어·영어·중국어다. 본 Co-Pilot API는 12개 언어를 받는다. v0.3 용어집은 한국어·영어·중국어·러시아어·베트남어·인도네시아어·태국어에 적용한다. 베트남어는 v0.2에서 실제 번역 실패가 재현돼 보류했던 언어다. 용어 근거와 번역 품질은 별개이므로 번역 품질은 실측으로 따로 확인한다. 우즈벡어는 확인된 건조도막두께 하나만 적용한다. 미얀마어·네팔어·싱할라어·크메르어는 핵심 용어 근거를 찾지 못해 용어집 없이 기존 번역 검증만 거친다.

## 1. 직접 확인한 1차 자료

| ID | 발행 주체 / 문서 / 언어 | 확인 위치와 근거 | 링크 |
|---|---|---|---|
| KO1 | KCC, 가루멜 EX8816(S) 기술자료 / 한국어 | p.1 이론도포율·내식성·사용방법. 건조도막두께, 표면처리 | [한국 제조사 PDF](https://www.kccworld.co.kr/techdoc/c01/cvt/C01F201510050008636kor.pdf) |
| EN1 | Jotun, Hardtop WF Application Guide / 영국 영어판 | p.1 Surface preparation, p.4 Film thickness, 건조·재도장 절 | [영어판 PDF](https://jotundatasheetsprod.blob.core.windows.net/ags/AG%C2%A441048%C2%A4Hardtop%20WF%C2%A4Euk%C2%A4GB.pdf) |
| ZH1 | 佐敦/Jotun, Hardtop WF 施工指南 / 중국 간체판 | p.1 표면 처리, p.4 두께 표·측정 절. 발행 2025-01-09 | [중국어판 PDF](https://jotundatasheetsprod.blob.core.windows.net/ags/AG%C2%A441048%C2%A4Hardtop%20WF%C2%A4Zho%C2%A4CN.pdf) |
| VI1 | Jotun Việt Nam, SteelMaster 1200HPE / 베트남어 | p.2 건조·습도막 및 표면 준비 절 | [베트남어 기술자료](https://jotundatasheetsprod.blob.core.windows.net/tds/TDS%C2%A443902%C2%A4SteelMaster%201200HPE%C2%A4vi-VN%C2%A4VN.pdf) |
| VI2 | Jotun Việt Nam, Jotashield Colour Extreme / 베트남어 | p.2 분사 장비·건조, p.3 표면 준비. 장식용 제품이므로 수치는 선박 도장에 전용하지 않음 | [베트남어 기술자료](https://jotundatasheetsprod.blob.core.windows.net/tds/TDS%C2%A431642%C2%A4Jotashield%20Colour%20Extreme%C2%A4vi-VN%C2%A4VN.pdf) |
| UZ1 | Silkcoat Uzbekistan, ClassPlastic / 우즈벡어 | 제품 사양의 Quruq plyonka qalinligi. 이전 초안의 설명식 번역과 실제 제조사 표기가 다름 | [우즈베키스탄 제조사](https://silkcoat.uz/uz/product/classplastic-2/) |
| MY1 | Yangon City Development Committee, Myanmar National Building Code 2025 Part 6 / 미얀마어+영어 | PDF p.71–74에 도막 시험과 paint film formation 등. 추출 문자 손상으로 세부 미얀마어 철자 확정 보류 | [미얀마 공공기관 PDF](https://ybps.ycdc.gov.mm/gwt_assets/files/mnbc_part_6.pdf) |
| MY2 | Nippon Paint Myanmar / 미얀마어 | 제품 분류 သင်္ဘောသုတ်ဆေးများ는 선박용 도료의 현지 사용 근거. DFT 용어 근거는 아님 | [현지 제조사](https://www.nipponpaint.com.mm/my/) |
| NE1 | CTEVT, Assistant House Painter / 네팔어판 | 표지·교육과정 확인. 레거시 글꼴이 로마자 기호로 추출돼 전문용어 유니코드 철자 확정 보류 | [네팔 직업교육기관 PDF](https://www.ctevt.org.np/public/uploads/docs/2021-02-26_Assistant%20Housepainter.pdf) |
| NE2 | CTEVT, Junior House Painter 2004 / 영어판 | p.3 surface preparation 교육내용. 네팔어 번역의 증거는 아님 | [동일 기관 영어판](https://ctevt.org.np/uploads/docs/2021-02-26_Junior%20House%20Painter%2C%202004.pdf) |
| KO2 | NCS 선박도장(15080203) 학습모듈 1508020305 / 한국어 | p.20 건도막 두께·에어리스 스프레이, p.29 습도막 두께, p.25 건조도막두께. RAG 지식 원본 | 저장소 밖 RAG 원본(`data/raw/pdf`) |
| KO3 | NCS 선박도장(15080203) 학습모듈 1508020302 / 한국어 | p.88 최대 재도장 간격, p.83 에어리스 스프레이. RAG 지식 원본 | 저장소 밖 RAG 원본(`data/raw/pdf`) |
| KO4 | Jotun Korea, Hardtop WF 기술 자료 시트 / 한국어 | p.2 건조 도막 두께·습도막 두께·표면 처리 요약 테이블, p.3 최대 재도장간격 | [한국어 기술자료](https://jotundatasheetsprod.blob.core.windows.net/tds/TDS%C2%A441048%C2%A4Hardtop%20WF%C2%A4Kor%C2%A4KR.pdf) |
| KO5 | KCC, 코레폭스-EH2350(GF) 데이터시트 / 한국어 | p.1 건조도막두께·도장 방법 에어리스 스프레이, p.2 최대 재도장 간격 | [한국 제조사 PDF](https://www.kccworld.co.kr/techdoc/c01/cvt/C01F201503255524108kor.pdf) |
| RU1 | Jotun, Hardtop XP 시공 지침(Rus/KZ) / 러시아어 | p.1 Подготовка поверхности, p.4 Толщина сухой·мокрой пленки, p.7 максимально допустимый межслойный интервал | [러시아어 시공 지침](https://jotundatasheetsprod.blob.core.windows.net/ags/AG%C2%A43140%C2%A4Hardtop%20XP%C2%A4Rus%C2%A4KZ.pdf) |
| RU3 | Jotun, Hardtop XP 기술자료(Rus/RU) / 러시아어 | p.2 Толщина сырой пленки, p.3 безвоздушное распыление | [러시아어 기술자료](https://jotundatasheetsprod.blob.core.windows.net/tds/TDS%C2%A43140%C2%A4Hardtop%20XP%C2%A4Rus%C2%A4RU.pdf) |
| VI3 | Jotun, Jotafloor EP Coating 시공 지침 / 베트남어 | p.6 Thời gian cách lớp tối đa. 영어판 같은 쪽 Maximum over coating intervals와 대응. 콘크리트 바닥재 제품 | [베트남어 시공 지침](https://jotundatasheetsprod.blob.core.windows.net/ags/AG%C2%A447884%C2%A4Jotafloor%20EP%20Coating%C2%A4vi-VN%C2%A4GB.pdf) |
| VI4 | Jotun Việt Nam, Hardtop XP / 베트남어 | p.2 máy phun sơn áp lực cao (airless), Chiều dày khô·ướt, p.3 thời gian tối đa để sơn lớp kế | [베트남어 기술자료](https://jotundatasheetsprod.blob.core.windows.net/tds/TDS%C2%A43140%C2%A4Hardtop%20XP%C2%A4vi-VN%C2%A4VN.pdf) |
| VI5 | Nippon Paint Việt Nam, NIPPON 52-01 Epoxy Topcoat / 베트남어 | p.2 Phun áp lực cao, độ dày màng sơn khô | [베트남 제조사 PDF](https://professional.nipponpaint.com.vn/sites/default/files/2025-06/NIPPON%2052-01%20EPOXY%20TOPCOAT%20-%20TDS%20-%20VN.pdf) |
| VI6 | Jotun Việt Nam, Penguard Express MIO 80 / 베트남어 | p.2 máy phun sơn yếm khí (airless spray) | [베트남어 기술자료](https://jotundatasheetsprod.blob.core.windows.net/tds/TDS%C2%A410680%C2%A4Penguard%20Express%20MIO%2080%C2%A4vi-VN%C2%A4VN.pdf) |
| ID1 | Jotun Indonesia, Gardex Primer / 인도네시아어 | p.1 airless spray(영어 그대로), persiapan permukaan | [인도네시아어 기술자료](https://jotundatasheetsprod.blob.core.windows.net/tds/TDS%C2%A410740%C2%A4Gardex%20Primer%C2%A4id-ID%C2%A4ID.pdf) |
| ID2 | 인도네시아 공공사업부 도로총국(Bina Marga), 2018 일반 시방서 8.7절 강재 도장 / 인도네시아어 | p.5 ketebalan cat kering, p.12 ketebalan cat basah. 강교량 시방서 | [중부 자바 주 공공사업국 사본](https://siwasit.dpubinmarcipka.jatengprov.go.id/SiWasit/assets/dokumen/dokumen_BinarMarga-Spek%202018%20Rev.2%20-%20SEKSI%208-7%20CAT%20BAJA.pdf) |
| ID3 | Nippon Paint Indonesia, EA4 Finish / 인도네시아어 | p.1 Persiapan Permukaan, interval pengecatan maksimum | [인도네시아 제조사 PDF](https://www.nipponpaint-indonesia.com/data/products/TDS_NIPPON%20EA4%20FINISH_12042023.pdf) |
| TH1 | Jotun Thailand, Penguard Express / 태국어 | p.2 ความหนาฟิล์มสีแห้ง·เปียก, การเตรียมพื้นผิว, p.3 เครื่องพ่นไฮดรอลิค (Airless Spray), ระยะเวลาสูงสุดที่ยอมรับได้ในการทาทับ. 글자 층이 깨져 렌더링으로 판독 | [태국어 기술자료](https://jotundatasheetsprod.blob.core.windows.net/tds/TDS%C2%A42400%C2%A4Penguard%20Express%C2%A4th-TH%C2%A4TH.pdf) |
| NE3 | 네팔 해외고용위원회(FEB), 하우스 페인터 교육과정 / 네팔어 | p.12 सतह तयारी. 글자 층 대부분이 깨져 이 단어만 렌더링과 대조 | [웹 아카이브 사본](https://web.archive.org/web/20230604222214id_/https://www.feb.gov.np/extra_upload/60669819d71c7_5c5021be8b9ff_1.%20Final%20training_guide__house%20painter.pdf) |

웹 뷰어로 열리지 않는 제조사 자료는 원본을 내려받아 PDF 본문 또는 HTML을 직접 확인했다. 제품 수치·압력·건조시간은 용어 근거일 뿐 본 프로젝트의 판정 기준이 아니다.

## 2. 기존 16개 용어의 항목별 판정

`확인`은 문서의 표기를 찾았다는 뜻이다. `설명`은 관련 개념만 확인했으며 초안 용어의 정확한 현지 사용은 미확인이다. `미확인`은 이번 조사 자료에서 근거를 확보하지 못했다는 뜻으로, 그 언어에 용어가 없다는 뜻이 아니다.

| ID / 한국어 개념 | 영어 문서 표기 / 판정 | 중국어 문서 표기 / 판정 | 베트남어 문서 표기 / 판정 |
|---|---|---|---|
| T01 건조도막두께 | dry film thickness / 확인 EN1 p.4 | 干膜厚度 / 확인 ZH1 p.4 | chiều dày màng sơn khô / 확인 VI1 p.2 |
| T02 습도막두께 | wet film thickness / 확인 EN1 p.4 | 湿膜厚度 / 확인 ZH1 p.4 | chiều dày màng sơn ướt / 확인 VI1 p.2 |
| T03 저도막 | 초안 명칭 미확인 | 干膜厚度偏低 / 설명 ZH1 p.4. 초안 膜厚不足는 미확인 | 초안 명칭 미확인 |
| T04 과도막 | 관련 현상 설명만 확인 | 膜厚过高 / 확인 ZH1 p.4 | 초안 명칭 미확인 |
| T05 분사거리 | 건과 피도면 간 거리 설명 EN1. 초안 명칭 미확인 | 초안 명칭 미확인 | 초안 명칭 미확인 |
| T06 분사각도 | 각도 설명은 있으나 기준축 의미 대조 필요 | 초안 명칭 미확인 | Góc phun / 확인 VI2 p.2. 노즐 팬 각도와 자세각의 의미 대조 필요 |
| T07 분사건 이동속도 | 미확인 | 미확인 | 미확인 |
| T08 중첩률 | 미확인 | 미확인 | 미확인 |
| T09 에어리스 스프레이 | airless spray / 확인 EN1 p.4 | 无气喷涂 / 확인 ZH1 p.4 | máy phun sơn áp lực cao (airless) / 확인 VI4 p.2. 문서마다 표기가 다름(§6) |
| T10 재도장 간격 | Maximum over coating intervals / 확인 EN1. 최대 간격 | 最大覆涂间隔 / 확인 ZH1. 최대 간격 | Thời gian cách lớp tối đa / 확인 VI3 p.6(바닥재 지침). VI2 p.3의 최소 건조시간과 치환하지 않음 |
| T11 표면처리 | surface preparation / 확인 EN1 p.1 | 表面处理 / 확인 ZH1 p.1 | Chuẩn bị bề mặt / 확인 VI1 p.2 |
| T12 도장 구역 | 프로젝트 정의 개념. 현지 표준 전문용어 미확인 | 동일 | 동일 |
| T13 자세 유사도 | 프로젝트 분석 개념. 현지 도장 표준 용어 미확인 | 동일 | 동일 |
| T14 도장 순서 | 미확인 | 미확인 | 미확인 |
| T15 보호구 | 기존 초안 표기를 해당 자료로 확정하지 않음 | 동일 | 동일 |
| T16 작업 중지 | 안전 지시문. 전문용어와 별도 문장 검수 필요 | 동일 | 동일 |

한국어 T01·T02·T09·T10·T11은 KO1~KO5에서 확인했다(§6). T09의 한국어는 v0.2의 프로젝트 명칭 '무공기 분사' 대신 문서 표기 '에어리스 스프레이'로 바꿨다. '무공기 분사'는 확인한 한국어 문서 어디에도 없다. 나머지 한국어는 프로젝트 개념명으로 유지하되 외부 표준 확인 완료로 표시하지 않는다. T06은 노즐 팬 각도와 분사건 자세각을 혼동할 수 있어 강제 용어집에서 제외한다. T10의 최소·최대 조건도 서로 치환하지 않는다.

## 3. 우즈벡어·미얀마어·네팔어 판정

| 항목 | 우즈벡어 | 미얀마어 | 네팔어 |
|---|---|---|---|
| T01 DFT | **Quruq plyonka qalinligi**, 확인 UZ1. 기존 quruq bo‘yoq qatlami qalinligi는 문서 근거 없는 후보로 철회 | MY1에서 도막 시험 개념 확인. 기존 초안 철자는 미확인 | NE1/NE2 확인. 기존 네팔어 초안 철자는 미확인 |
| T02–T16 | 기존 15개 후보 모두 정확한 용례 미확인 | 기존 15개 후보 모두 정확한 용례 미확인 | 기존 15개 후보 모두 정확한 용례 미확인 |
| 추가 근거 | 제조사 우즈벡어 사양 확보 | MY2의 선박용 도료 분류 표기 확인 | 공공 직업교육기관 네팔어 도장 교육과정 확보 |
| 다음 작업 | 산업·선박 시공 지침 추가 대조 | 원본 렌더링·현지 검수로 철자 복원 | 레거시 글꼴 변환·현지 검수 |
| 1차 활성화 | 보류 | 보류 | 보류 |

v0.1의 근거 없는 설명식 후보를 삭제했다. 국내 자료를 기계 번역해 현지 문서인 것처럼 출처를 붙이지 않는다. v0.3 재조사 결과(싱할라어·크메르어 포함)는 §6에 있다.

## 4. 런타임 용어집 (rc3)

`trainer/multilingual/glossary.json`(2026-11-06, rc3)은 핵심 5개 용어의 언어별 표기를 담는다. 본 Co-Pilot API와 평가용 게이트웨이가 같은 파일을 `trainer/multilingual/glossary.py`로 읽는다. 용어 식별자는 `term`이다. 인도네시아어 코드 `id`와 겹치지 않게 하려고 바꿨다. 언어별 대표 표기는 `renderings`, 같은 용어의 다른 문서 표기와 격변화형은 `variants`에 둔다. 약어(DFT 등)는 넣지 않는다. 번역문이 약어를 그대로 쓸 수 있기 때문이다.

- 한국어 대표 표기는 지식 코퍼스와 학습 데이터가 실제로 쓰는 형태로 정했다. RAG 청크 10,829개에서 '건조도막두께'는 0회, '건도막 두께'는 252회다. 그래서 '건도막 두께'를 대표로, KCC 표기 '건조도막두께'를 변형으로 둔다. T09는 '에어리스 스프레이'(59회)이고 '무공기'는 0회다.
- 용어를 찾을 때는 띄어쓰기·대소문자·전각 문자·ё/е 차이를 무시한다. 한국어 답변은 '표면 처리'처럼 띄어 쓰는 경우가 많다(RAG 2,118회, '표면처리' 203회). v0.2 런타임은 띄어쓰기까지 같은 문자열만 찾아서 이런 답변에는 용어집이 걸리지 않았다.
- 원문에 용어가 있고 대상 언어 표기가 있으면 번역 요청에 대역어를 함께 준다. 번역 결과에는 대표 표기나 변형 가운데 하나가 들어 있어야 한다. 빠지면 번역을 내보내지 않는다. 그때 본 Co-Pilot API는 오류(503)를 돌려준다. 평가용 게이트웨이는 출력 번역이면 한국어 원문으로 대체하고(`fallback_original`), 입력 번역이면 질의를 실행하지 않는다.
- 디코딩 단계에서 강제하지는 않는다. 번역 호출은 반복 벌점 없이 greedy로 생성한다. 반복 벌점은 프롬프트에 들어 있는 대역어를 그대로 쓰는 것을 불리하게 만들기 때문이다.

`source-attested`는 문서 확인 상태다. 전체 번역의 전문가 승인이나 안전 정확도를 보장하지 않는다. 본 API는 지식·자세·순서 답변을 모두 번역하므로, 모든 응답 언어의 안전 문장이 원어민 검수 대상이다.

## 5. 원본 확인 기록

확인일은 KO1~NE1이 2026-10-06이고 KO2 이후는 2026-11-06이다. VI1과 UZ1은 2026-11-06에 다시 내려받아 해시가 같음을 확인했다. SHA256은 내려받아 확인한 원본 바이트 기준이다. 발행자가 파일을 갱신하면 달라질 수 있다.

| ID | SHA256 |
|---|---|
| KO1 | `ae513bd7c0c09350f75fbb5f2559a181ca0d24e71bc0f6e0162d7e3a84a82e3a` |
| EN1 | `92ac031b0d6275fc950a4a636a896e71c3033ba159dcde5419c4dff55926ec4a` |
| ZH1 | `3bf9ad49c4fe76d8985591ba03720f21371eaeedc94bdaad17d2156dcdedaf0a` |
| VI1 | `094fe0b7e21cdb521842d01b75664b6f300dd55c533b0cec6038b56647330e66` |
| UZ1 | `234c6a22ddde3dae3c22f006cff18839b0141c371d40c2c2aed4fdcb5a2014d9` |
| NE1 | `9aee12a040b287b91b3a89d40acabe9422e17519858c1e060692ee221435baf7` |
| KO2 | `dbf9da48c135a2fcdf55c00170b75834e0d0a4f5cedcf45b8b0ae155b1a51033` |
| KO3 | `96c86fad2e3169f1a3a526907813d890d36fafdb22a30584ca8ed57e77dee06a` |
| KO4 | `50a85b89cf78547ceeb29fe69c60f96308c31a96d6cfb9d9d2179b7f86b728d1` |
| KO5 | `69cb1b74c4eb6f674caba4a2665a435d6a70c47ffa4021ee85c45500a663651b` |
| RU1 | `85c06071c16f4182efb2bfa30d3d81a3621658ff7aff586c5e7612c8d1999c97` |
| RU3 | `dffaf2f9ab3861c1435edc0ab69404c764ee086ec51b8df02d42ab25b1ef7d91` |
| VI3 | `f2c07138a48a0c442c2f3bf9277f42bfc5a251384731fb6068429a7fa1eaa22a` |
| VI4 | `7402b0c0f8b759c2299dd79807be395d62ed5731a5efd21ac4c2b3989b9dc8c6` |
| VI5 | `3f2f68fc154b3971cc725c69ca96d7e293b74bb6da39f7ca2caab437ee14e100` |
| VI6 | `a3fbb8c7847fb7121f75b2a75f33c40b2fc05d90a9a69096d1b49cd20999ad94` |
| ID1 | `e5aa3aeead4acef19ad5808944ba93df8c7b4a36f7f21b8d8271143aad10da79` |
| ID2 | `1fecf02dc25b8de9a7f996c0685db558bab0265e39a37574e9803c7eb1a0304b` |
| ID3 | `f6d498b15f3f5381c5d6b0ece86cd43374c1b6cd63993ebeefbebda86f26be12` |
| TH1 | `874bc232f7afd3b4ec1b88a595846f4400d93019ea338c34ae7a689afb566a5b` |
| NE3 | `5427e918bf8cc5c5389093477704a13c5947397c68dcce6705b6a1d9476533a6` |

## 6. 질의 언어별 핵심 5개 용어 판정 (v0.3)

2026-11-06에 Spark 본 Co-Pilot API가 받던 12개 언어 코드를 대상으로 했다. 언어마다 제조사·공공기관이 낸 현지어 문서를 내려받아 SHA-256과 쪽을 기록했다(§1, §5). 판정 규칙은 v0.2와 같다. **다섯 용어가 모두 확인된 언어에만 용어집을 적용한다.** 질의 언어 12개는 모두 유지한다(2026-11-09 결정, 11-06의 숨김 결정을 바꿈).

| 언어 | T01 건조도막두께 | T02 습도막두께 | T09 에어리스 스프레이 | T10 최대 재도장 간격 | T11 표면처리 | 용어집 |
|---|---|---|---|---|---|---|
| 한국어 | 건도막 두께 KO2 p.20, 건조도막두께 KO5 p.1 | 습도막 두께 KO2 p.29, KO4 p.2 | 에어리스 스프레이 KO2 p.20, KO5 p.1 | 최대 재도장 간격 KO3 p.88, KO5 p.2 | 표면처리 KO1, 표면 처리 KO4 p.2 | 정본 |
| 영어 | dry film thickness EN1 p.4 | wet film thickness EN1 p.4 | airless spray EN1 p.4 | maximum over coating interval EN1 | surface preparation EN1 p.1 | 적용 |
| 중국어 | 干膜厚度 ZH1 p.4 | 湿膜厚度 ZH1 p.4 | 无气喷涂 ZH1 p.4 | 最大覆涂间隔 ZH1 | 表面处理 ZH1 p.1 | 적용 |
| 러시아어 | толщина сухой плёнки RU1 p.4 | толщина мокрой плёнки RU1 p.4 | безвоздушное распыление RU3 p.3 | максимально допустимый межслойный интервал RU1 p.7 | подготовка поверхности RU1 p.1 | 적용 |
| 베트남어 | chiều dày màng sơn khô VI1 p.2 | chiều dày màng sơn ướt VI1 p.2 | máy phun sơn áp lực cao (airless) VI4 p.2 ¹ | thời gian cách lớp tối đa VI3 p.6 ² | chuẩn bị bề mặt VI1 p.2 | 적용 |
| 인도네시아어 | ketebalan cat kering ID2 p.5 ³ | ketebalan cat basah ID2 p.12 ³ | airless spray ID1 p.1 ⁴ | interval pengecatan maksimum ID3 p.1 | persiapan permukaan ID3 p.1 | 적용 |
| 태국어 | ความหนาฟิล์มสีแห้ง TH1 p.2 ⁵ | ความหนาฟิล์มสีเปียก TH1 p.2 | เครื่องพ่นไฮดรอลิค (Airless Spray) TH1 p.3 ⁵ | ระยะเวลาสูงสุดที่ยอมรับได้ในการทาทับ TH1 p.3 ⁵ | การเตรียมพื้นผิว TH1 p.2 ⁵ | 적용 |
| 우즈벡어 | quruq plyonka qalinligi UZ1 | 미확인 | 미확인 | 바닥 피복 건축 규범의 문장만 ⁶ | 설명만 | T01만 |
| 네팔어 | 미확인 | 미확인 | 미확인 | 미확인 | सतह तयारी, 철자 확정 보류 NE3 p.12 ⁷ | 미적용 |
| 미얀마어 | 일반 'Film thickness' 제목만, 추출 손상 MY1 p.71 | 미확인 | 미확인 | 미확인 | 미확인 | 미적용 |
| 싱할라어 | 미확인 | 미확인 | 미확인 | 미확인 | 미확인 | 미적용 |
| 크메르어 | 미확인 | 미확인 | 미확인 | 미확인 | 미확인 | 미적용 |

1. 베트남 문서마다 표기가 다르다. VI4는 máy phun sơn áp lực cao (airless), VI6은 máy phun sơn yếm khí (airless spray), VI5는 Phun áp lực cao다. 용어집은 세 표기를 모두 받는다.
2. Jotun 콘크리트 바닥재 시공 지침의 제목으로, 영어판 같은 쪽의 Maximum over coating intervals와 대응한다. 선박용 기술자료는 이 개념을 문장으로만 쓴다(thời gian tối đa để sơn lớp kế, VI4 p.3).
3. 인도네시아 공공사업부 도로총국의 강교량 도장 시방서다. 선박 도장 문서는 아니다.
4. 인도네시아 제조사 문서는 airless를 번역하지 않고 영어로 쓴다(ID1, ID3).
5. PDF 글자 층이 깨져 있어(성조 부호 손상) 원본을 렌더링해 판독했다.
6. SHNQ 3.04.01-21의 바닥 보호 폴리머 피복 조항이다. 강재·선박 분야가 아니고 러시아어 직역형이라 채택하지 않았다.
7. NE3 문서는 글자 층 대부분이 깨져 이 단어만 렌더링과 대조했다. CTEVT 교육과정(레거시 Preeti 글꼴)에도 같은 단어가 있다.

Jotun 공개 자료 저장소 전체 목록(기술자료 18,292건, 시공 지침 1,677건)에 크메르어·미얀마어·싱할라어·네팔어판은 없다. 캄보디아·미얀마 기술자료도 모두 영어판이다. 러시아어 근거는 모두 Jotun 문서라 다른 제조사와 교차 확인하지 않았다. 용어 근거는 번역 품질 승인이 아니다. 문장 번역은 원어민 검수(PRD 출시 기준)를 아직 거치지 않았다.
