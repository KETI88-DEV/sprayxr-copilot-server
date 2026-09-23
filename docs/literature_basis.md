# 교정 설계의 문헌 근거

이 문서는 **자세 교정 파이프라인의 설계 결정마다 근거를 붙이고, 그 근거가 우리를 지지하는지
반박하는지 아무 말도 하지 않는지를 적는다.** 교정 문구·발부 규칙·채점 정규화·지표 이름을 두고
논의가 생기면 여기를 먼저 본다.

조사 2026-09-23. 그날 기준의 상태다.

**쓰는 법.** 결정이 바뀌면 해당 절의 "우리 설계" 줄을 고치고 문헌 줄은 그대로 둔다 — 문헌은
우리가 무엇을 하든 바뀌지 않는다. 새 근거를 넣을 때는 **실제로 받아 본 검색 결과에서만** 가져온다.

**상태 표기**

| 표기 | 뜻 |
|---|---|
| **지지** | 문헌이 우리 선택을 뒷받침한다 |
| **반박** | 문헌이 우리 선택과 반대 방향을 가리킨다 |
| **미검증** | 이 질문을 다룬 연구를 찾지 못했다 — 우리 데이터로만 설 수 있다 |
| **조건부** | 지지하지만 우리 상황에 그대로 옮기기 어려운 단서가 붙는다 |

**출처 등급**: **[PR]** 동료심사 · **[PP]** 프리프린트 · **[T]** 업계 기술문헌(핸드북·컨설팅) · **[V]** 벤더 홍보.

---

## 1. 진행축 이탈 — 업계는 이것을 "arcing"이라 부른다

**우리 설계.** 머리의 **진행축(lateral) 변위**를 자세 교정 후보에 넣는다(C1, 커밋 `86032af`).
이 축은 몸의 좌우가 아니라 **스트로크 진행 방향**이고, 크게 어긋났다는 것은 작업자가 스트로크를
따라 이동하지 않고 제자리에서 팔만 휘둘렀다는 뜻이다.

| 출처 | 내용 |
|---|---|
| **[T]** KTA-Tator, *Spray Application — Techniques to Optimize Film Build*. https://kta.com/spray-application-techniques/ | "Arcing occurs when an applicator turns their wrist only, instead of moving their entire arm/shoulder" — 패스 **가운데는 두껍고 양 끝은 얇아진다**. 처방은 손목을 고정하고 팔·어깨·몸을 함께 옮겨 건을 수직·등거리로 유지하는 것. |
| **[T]** *Products Finishing*, *For Quality Spray Painting, Remember the Fundamentals*. https://www.pfonline.com/articles/for-quality-spray-painting-remember-the-fundamentals | arcing을 명백한 결함으로 규정하고, 몸이 닿는 범위를 넘으면 등거리를 유지할 수 없으므로 패스 길이를 약 3 ft로 제한하라고 한다. |
| **[T]** Airblast Ltd., *How Operator Technique Influences Coating Quality*. https://airblast.co.uk/operator-technique-and-coating-quality/ | "gun to arc … changes the distance between the nozzle and the surface, causing uneven film thickness." 같은 장비·같은 도료에서도 작업자마다 품질이 다르다고 명시. |
| **[T]** MnTAP (Univ. of Minnesota), *Transfer Efficiency*. http://www.mntap.umn.edu/industries/facility/paint/efficiency/ | **작업자 기법만으로 전이효율이 10–30% 움직인다.** "사람이 1차 변수"의 수치 근거. |
| **[PR]** Björing, G., & Hägg, G.M. (2000). *Musculoskeletal exposure of manual spray painting in the woodworking industry.* Int. J. Industrial Ergonomics. https://www.sciencedirect.com/science/article/abs/pii/S0169814100000263 | 도장공 35명. 작업물이 나쁜 상대 위치를 강요하면 **오른 위팔이 심하게 외전**한다 — 몸의 배치가 측정 가능한 문제라는 증거(다만 결과 변수는 어깨 건염 위험이지 도막이 아니다). |

**상태: 조건부 지지.** 메커니즘은 업계 기술문헌이 독립적으로 같은 결론에 도달해 있다 — 진행축
머리 변위는 **arcing의 운동학적 서명**이다. 그래서 "새로운 생체역학 법칙"이 아니라 "도장 검사원이면
아는 결함을 머리 좌표로 검출한 것"이라고 설명할 수 있다.

**단서 둘.** (1) 전부 동료심사가 아닌 업계 문헌이다. (2) 우리는 아직 **진행축 변위가 도막 두께
편차와 상관된다는 것을 재지 않았다.** 그 측정 전까지 말할 수 있는 것은 "가장 어긋난 축이고,
교정 대상에 넣었더니 지시 품질 상한이 올랐다"까지다. **"도막 품질이 좋아진다"는 우리 데이터에도
문헌에도 근거가 없다.**

---

## 2. 교정 문구는 몸이 아니라 도구·표면에 걸어야 한다

**우리 설계(현재).** 문구가 몸 부위를 부른다 — "오른손(건)을 23.3cm 올리세요", "진행 방향 뒤로
30.7cm 옮겨 서세요"(`instructions.HEAD_PHRASES`, `_right_text`).

| 출처 | 내용 |
|---|---|
| **[PR]** Chua, L.-K., Jimenez-Diaz, J., Lewthwaite, R., Kim, T., & Wulf, G. (2021). *Superiority of external attentional focus for motor performance and learning: Systematic reviews and meta-analyses.* Psychological Bulletin 147(6), 618–645. https://doi.org/10.1037/bul0000335 | 약 143개 연구 메타분석. 움직임의 **효과·도구**에 주의를 거는 외적 초점이 신체 부위에 거는 내적 초점보다 수행·학습 모두에서 낫고, **나이·숙련도·건강 상태와 무관**하다. |
| **[PR]** Wulf, G. (2013). *Attentional focus and motor learning: a review of 15 years.* Int. Review of Sport and Exercise Psychology 6(1), 77–104. https://doi.org/10.1080/1750984X.2012.723728 | 과제·숙련도·연령을 가로질러 재현. "단 하나의 지시 단서가 전신 협응을 바꿀 수 있다"고 명시. |

**상태: 반박.** 우리 문구는 교과서적 **내적 초점**이다. 같은 기하를 도구와 벽에 걸면 공짜로
외적 초점이 된다 — "건을 벽에서 20cm로 유지하세요", "분사 팬을 벽에 수직으로 유지하세요".
**측정 파이프라인은 한 줄도 바뀌지 않는다.** §1의 업계 처방("몸을 스트로크와 함께 옮기고 건을
수직으로")도 같은 형태다. 미결정 — 화면 문구는 소유자 판단 사항이다.

---

## 3. 한 번에 몇 개를 말하는가

**우리 설계.** 머리는 **문장 하나**(σ가 가장 큰 자세 축). 오른손은 **임계를 넘은 축 전부**(최대 3).
합치면 한 스트로크에 최대 4개가 나간다.

| 출처 | 내용 |
|---|---|
| **[PR]** Raisbeck, L.D., & Diekfuss, J.A. (2017). *Verbal Cues and Attentional Focus: A Simulated Target-Shooting Experiment.* J. Motor Learning and Development 5(1), 148–159. https://doi.org/10.1123/jmld.2016-0017 | 1개 대 3개 직접 조작. **단서 1개 집단이 즉시 파지에서 유의하게 나았다.** 결론은 "언어 단서 수 최소화의 중요성". |
| **[PR]** Niźnikowski, T., et al. (2025). *Enhancing the learning of sports skills through verbal feedback.* Frontiers in Sports and Active Living 7, 1519365. https://doi.org/10.3389/fspor.2025.1519365 | "핵심 요소의 오류에만 피드백을 줄이는 것이 모든 오류에 주는 것보다 효과적"(체조 연구 인용). |
| **[PR]** Becker, K.A., & Fairbrother, J.T. (2019). *The use of multiple externally directed attentional focus cues facilitates motor learning.* Int. J. Sports Science & Coaching 14(5), 651–657. https://doi.org/10.1177/1747954119870172 | 단서 3개 조건에서도 외적 초점 이점이 살아남는다. **주의**: 3개 안에서 외적 대 내적을 비교한 것이지 1개 대 3개가 아니다 — "3개가 1개만큼 좋다"의 근거가 되지 않는다. |
| **[PR]** Wulf, G., & Shea, C.H. (2002). *Principles derived from the study of simple skills do not generalize to complex skill learning.* Psychonomic Bulletin & Review 9(2), 185–211. https://doi.org/10.3758/BF03196276 | **반대 무게.** 이런 처방은 대개 단순 실험실 과제에서 나왔고, 복합 전신 과제(스키 시뮬레이터)에서는 성립하지 않았다. |

**상태: 머리 규칙은 지지, 오른손 규칙은 반박.** 머리 한 문장은 Raisbeck & Diekfuss가 사실상
직접 검정한 조건이다. 오른손 다축은 그 실험에서 **진 조건보다 많은** 단서를 만든다.

**그래서 "머리도 다축으로 늘린다"(천장 100%가 되는 안)는 착수하지 않는다.** 재야 할 것은 그
반대 방향, 곧 **오른손을 단일 문장으로 줄이는 안**이다. Wulf & Shea의 단서 때문에 문헌만으로
잘라서는 안 되고 우리 파이프라인에서 재야 한다.

---

## 4. 우리 지표의 정체 — "개선율"이 아니다

**우리 설계.** G = (S1 − S0) / (1 − S0). **S1은 관측값이 아니다** — 교정점을 지시한 만큼
평행이동해 다시 채점한 계산값이다. 아무도 다시 뿌리지 않는다.

### 4.1 두 가정

| 출처 | 내용 |
|---|---|
| **[PR]** Murray, E.J., & Hernán, M.A. (2016). *Adherence adjustment in the Coronary Drug Project.* Clinical Trials 13(4), 372–378. https://doi.org/10.1177/1740774516634335 | 우리가 계산하는 양의 이름은 **per-protocol(전면 순응) 추정량**이고, 실제 효과(ITT)와의 차이가 곧 불순응이다. |
| **[PR]** Karimi, A.-H., Schölkopf, B., & Valera, I. (2021). *Algorithmic Recourse: from Counterfactual Explanations to Interventions.* FAccT '21. https://arxiv.org/abs/2002.06278 | 입력을 권고 델타만큼 고쳐 다시 채점하는 방식은 **변수들이 독립적으로 조작 가능**하다고 가정한다 — 인과로 얽힌 변수에서는 거짓. |
| **[PR]** Barocas, S., Selbst, A.D., & Raghavan, M. (2020). *The hidden assumptions behind counterfactual explanations and principal reasons.* FAT* '20. https://doi.org/10.1145/3351095.3372830 | 나머지 가정: 권고된 특징 변화가 **주체가 실제로 취할 수 있는 행동으로 깨끗이 대응된다**는 것. |
| **[PR]** Gottesman, O., et al. (2019). *Guidelines for reinforcement learning in healthcare.* Nature Medicine 25, 16–18. https://doi.org/10.1038/s41591-018-0310-5 | 손으로 만든 모형으로 로그 데이터 위에서 정책을 평가하면 **모형의 오차를 그대로 물려받은 낙관적 추정**이 나온다. |
| **[PP]** Hakim, T. (2024). *Evaluation Framework for Feedback Generation Methods in Skeletal Movement Assessment.* xAI4Biometrics @ ECCV 2024. https://arxiv.org/abs/2404.09359 | 이 분야의 유일한 피드백 평가 프레임워크인데, **주석된 실수 라벨**에 대고 채점하지 교정을 시뮬레이션해 재채점하지 않는다. |

**상태: 명명된 결함.** 우리 G는 off-policy 추정량이 **아니다** — IPS/DR류는 관측된 결과를
재가중하고 겹침(overlap)을 요구하는데, 우리는 교정된 스트로크를 아예 관측하지 않는다.
그 행동의 지지집합은 정확히 0이므로 **데이터만으로는 식별되지 않는다.** 두 가정 위에서만 정의된다.

1. **완전 순응** — 작업자가 지시 벡터를 정확히 실행한다. 인적 요인 근거 없음.
2. **가산 평행이동** — 교정점을 옮겨도 나머지 기하가 그대로다. **이 리그에서 명백히 거짓**이다.
   손은 운동 사슬 위에 있고, 손을 옮기면 노즐 축 기하가 바뀐다.

**따라서 정확한 이름은 "완전 순응과 가산 평행이동 운동모형 아래에서의 지시 품질 상한"이다.**
"개선", "학습", "격차를 메웠다"로 부르지 않는다. 화면 라벨과 `instructions.IMPROVEMENT_LABELS`가
이 조건을 달고 다닌다(커밋에서 고쳤다).

### 4.2 G라는 형식 자체의 결함 셋

| 출처 | 내용 |
|---|---|
| **[PR]** Hake, R.R. (1998). *Interactive-engagement versus traditional methods.* American J. Physics 66(1), 64–74. https://doi.org/10.1119/1.18809 (PhysPort 정리: https://www.physport.org/expert/gain/) | 정규화 이득 g = (post−pre)/(100−pre)의 원전. 우리 G는 시험 점수 자리에 유사도를 넣은 같은 식이다. |
| **[PR]** Nissen, J.M., Talbot, R.M., Thompson, A.N., & Van Dusen, B. (2018). *Comparison of normalized gain and Cohen's d.* Phys. Rev. Phys. Educ. Res. 14(1), 010115. https://doi.org/10.1103/PhysRevPhysEducRes.14.010115 | g는 **사전점수가 높은 집단에 유리하게 편향**된다 — 보통 g를 고르는 이유(기준선 독립성)와 반대다. |
| **[PR]** Christman, E., Miller, P., & Stewart, J. (2024). *Beyond normalized gain.* Phys. Rev. Phys. Educ. Res. 20(1), 010123. https://doi.org/10.1103/PhysRevPhysEducRes.20.010123 | g도 Cohen's d도 집단 간 변이를 제거하지 못한다. 단일 요약 대신 사후점수를 사전점수의 함수로 모형화하라. |
| **[PR]** Tu, Y.-K., Gilthorpe, M.S., & Griffiths, G.S. (2002). *Is reduction of pocket probing depth correlated with the baseline value or is it "mathematical coupling"?* J. Dental Research 81(10), 722–726. https://doi.org/10.1177/154405910208101013 | **수학적 결합**: 변화량이 기준선을 포함하므로 둘 사이의 상관·회귀는 일부가 산술 아티팩트다. |
| **[PR]** Bowman, H., Bonkhoff, A., Hope, T., Grefkes, C., & Price, C. (2021). *Inflated Estimates of Proportional Recovery From Stroke.* Stroke 52(5), 1915–1920. https://doi.org/10.1161/STROKEAHA.120.033031 | **우리 지표의 정확한 쌍둥이**("잃은 것의 약 70%를 회복한다")가 수학적 결합과 천장 압축으로 부풀려진 것임을 보였다. |
| **[PR]** Marx, J.D., & Cummings, K. (2007). *Normalized change.* American J. Physics 75(1), 87–91. | g가 비대칭이고 아래로 무계라서 대안(normalized change c)을 제안. |

**상태: 명명된 결함 셋.**

- **수학적 결합**: S0가 분자와 분모에 모두 있다. G와 S0의 관계를 보고할 때 일부는 산술이다.
- **기준선 분포 의존**: "G ≥ 0.70 비율"은 교정만의 성질이 아니라 **교정 + 이 표본의 S0 분포**의
  성질이다. **훈련생 집단으로 그대로 옮겨가지 않는다.**
- **천장 압축**: S = exp(−W²/2)이라 W가 작으면 S0 → 1이고 1 − S0 → 0이라 이미 좋은 스트로크에서
  G가 폭발한다. 계약 문구의 70%도 뇌졸중 문헌과 같은 형태다.

**해야 할 보고 보강**(미착수): 190쌍의 S0 히스토그램, 원시 ΔS = S1 − S0 병기, S0 구간별 층화 G,
그리고 σ 게이트가 사실상 천장 필터로 작동하는지 명시.

**우리에게 유리한 점 하나**: S1이 시뮬레이션이고 독립적 측정 잡음을 갖지 않으므로 **평균 회귀가
G를 부풀리지 않는다.** 이것은 방어가 아니라 §4.1 문제의 뒷면이다.

---

## 5. 전문가 산포로 나누는 것(z점수 채점)

**우리 설계.** 축별 이탈을 그 축의 전문가 산포로 나눠 σ를 만들고, 점수 S = exp(−W²/2),
W는 점별 결합 σ의 가중 평균(`measured_score.score_track`).

| 출처 | 내용 |
|---|---|
| **[PR]** Scholz, J.P., & Schöner, G. (1999). *The uncontrolled manifold concept.* Experimental Brain Research 126(3), 289–306. https://doi.org/10.1007/s002210050738 | UCM 원전. 분산을 과제 결과를 보존하는 다양체 안 성분(V_UCM)과 직교 성분(V_ORT)으로 가르고, **통제되는 변수는 V_UCM/V_ORT를 최대화하는 것**이다. |
| **[PR]** Latash, M.L., Scholz, J.P., & Schöner, G. (2002). *Motor control strategies revealed in the structure of motor variability.* Exercise and Sport Sciences Reviews 30(1), 26–31. https://doi.org/10.1097/00003677-200201000-00006 | 운동 과잉(abundance) 관점, 재활·기능 습득 응용을 명시. |
| **[PR]** Müller, H., & Sternad, D. (2004). *Decomposition of variability in the execution of goal-oriented tasks.* JEP:HPP 30(1), 212–233. https://doi.org/10.1037/0096-1523.30.1.212 | TNC 분해. 연습은 **Tolerance**(과제가 변동을 용서하는 영역으로 이동)·Noise·Covariation으로 오차를 줄인다. **전문가 산포는 조작적으로 그 축의 tolerance다.** |
| **[PR]** Sternad, D. (2018). *It's not (only) the mean that matters.* Current Opinion in Behavioral Sciences 20, 183–195. https://doi.org/10.1016/j.cobeha.2018.01.004 | 변동성의 **구조**가 통제 우선순위를 드러내고 개입 설계에 쓸 수 있다. |
| **[PR]** Ewerton, M., et al. (2018). *Assisting Movement Training and Execution With Visual and Haptic Feedback.* Frontiers in Neurorobotics 12, 24. https://doi.org/10.3389/fnbot.2018.00024 | **직접적인 공학 선례.** 전문가 시연의 ProMP 분포를 짓고 **전문가 표준편차에 반비례해 햅틱 강성**을 준다 — 산포가 크면 부드럽게, 작으면 단단하게. 우리 채점 함수의 햅틱 판이다. |
| **[PR]** Toth, A.J., & Rodger, M. (2025). *A Motor Abundance Approach to Augmented Feedback for Skill Learning.* Ecological Psychology. https://doi.org/10.1080/10407413.2025.2582869 | UCM에서 피드백을 설계한 첫 시도. **다양체로부터의 이탈**을 소리로 준 쪽이 변수별로 준 쪽보다 나은 경향이었으나 **통계적으로 결정적이지 않았다**. |

**상태: 지지, 단서 둘.**

1. **UCM은 과제 결과로 중요도를 정하지 전문가 산포로 정하지 않는다.** 전문가가 어떤 축에서 좁은
   것은 그 축이 과제에 묶여 있다는 *증거*이지 정의가 아니다 — 도료가 상관하지 않는 생체역학적
   제약일 수도 있다. UCM 자신의 방법이라면 **어느 축이 실제로 도막을 움직이는지 먼저 확인**하라고
   한다(§1의 미측정 항목과 같은 것).
2. **축별 z점수는 공분산을 대각으로 가정한 마할라노비스다.** 실제 축들은 상관돼 있을 것이다
   (머리가 건을 따라가고, 스트로크를 따라 wall과 lateral이 함께 움직인다). 상관이 있으면 **가장
   큰 주변 z가 과제 관련 오차에 가장 크게 기여하는 축이 아닐 수 있다.** 전체 공분산으로 기여를
   재는 것이 문헌에 충실한 개선이고, 전문가 분포를 이미 갖고 있으므로 싼 변경이다.

**부수**: √2 배율 보정도 같은 이유로 옳은 본능이다 — Ewerton도 차이의 분포가 아니라 **위치의
분포**와 비교한다.

---

## 6. 무엇을 말할지 고르는 규칙

**우리 설계.** σ(= 이탈 ÷ 전문가 산포)가 가장 큰 축을 고른다(argmax-σ).

| 출처 | 내용 |
|---|---|
| **[PR]** Niźnikowski et al. (2025), §3과 같음 | 실험적 지지가 있는 선택 기준은 **과제 중요도**("핵심 요소", 전문가가 지정)이지 오차 크기가 아니다. |
| **[PR]** Chiviacowsky, S., & Wulf, G. (2007). *Feedback After Good Trials Enhances Learning.* RQES 78(1), 40–47. https://doi.org/10.1080/02701367.2007.10599402 | **가장 좋은** 시행 뒤 피드백을 받은 쪽이 **가장 나쁜** 시행 뒤 받은 쪽보다 잘 배웠다 — 최대 오차로 고르는 규칙이 진 설계다(시행 수준이지 축 수준은 아니다). |
| **[PR]** Hülsmann, F., et al. (2018). *Classification of motor errors to provide real-time feedback for sports coaching in VR.* Computers & Graphics 76, 47–59. https://doi.org/10.1016/j.cag.2018.08.003 | 10가지 오류 패턴을 독립적으로 검출하지만 **여러 오류가 동시에 발화할 때의 우선순위 기제는 제안하지 않는다.** |
| **[PR]** Ma, D., et al. (2026). *AgentCoach: LLM-Based Adaptive Coaching Feedback for Motor Skill Learning.* CHI '26. https://doi.org/10.1145/3772318.3791652 | 시스템 수준의 가장 가까운 선례. "coaching point"를 측정 가능한 파라미터에 매핑하고 이력에 맞춰 적응. **주의: 선택 규칙은 초록에서 확인되지 않았다 — 전문 필요.** |

**상태: 미검증.** 오차 크기 기반 선택을 다른 규칙과 비교한 연구를 찾지 못했다. 반박도 없다.

**우리 데이터가 이 자리를 메운다.** 같은 후보 집합에서 σ 대신 **난수**로 고른 대조군은
split ≥0.70이 78.42%로 **기준선 82.63%보다 낮았다**(커밋 `cb9a527`). 곧 argmax-σ는 난수보다
확실히 낫다. 다만 **argmax-σ 대 과제 중요도 가중**은 아직 아무도 재지 않았다 — 도장 물리가
wall 축을 더 중요하게 만든다면 고정 중요도 사전분포가 문헌상 더 나은 근거를 갖는다.

**C1 결과의 다른 해석 하나를 적어 둔다**: lateral이 도움이 된 것은 argmax-σ가 옳아서가 아니라
**과제에 관련된 축을 배제한 것이 어떤 선택 규칙 아래서도 틀렸기** 때문일 수 있다. 두 해석은
아직 구별되지 않았다.

---

## 7. 피드백 빈도와 파지 — 우리가 안고 있는 위험

**우리 설계.** `CORRECTION_SIGMA_THRESHOLD = 0.1`. 이 코퍼스에서 발부율이 사실상 100%다.

| 출처 | 내용 |
|---|---|
| **[PR]** Salmoni, A.W., Schmidt, R.A., & Walter, C.B. (1984). *Knowledge of results and motor learning: A review and critical reappraisal.* Psychological Bulletin 95(3), 355–386. https://doi.org/10.1037/0033-2909.95.3.355 | **guidance hypothesis**: 잦은 증강 피드백은 연습 중 수행을 떠받치면서 학습자 자신의 **오차 탐지 능력 발달을 막고**, 피드백을 걷으면 무너진다. |
| **[PR]** Winstein, C.J., & Schmidt, R.A. (1990). *Reduced frequency of knowledge of results enhances motor skill learning.* JEP:LMC 16(4), 677–691. https://doi.org/10.1037/0278-7393.16.4.677 | 50% **페이딩** KR이 습득기에는 100%와 같았으나 **무피드백 파지(5분·24시간)에서 유의하게 나았다.** |
| **[PR]** Lee, T.D., & Carnahan, H. (1990). *Bandwidth Knowledge of Results and Motor Learning.* QJEP-A 42(4), 777–789. https://doi.org/10.1080/14640749008401249 | **대역폭 KR**(허용대를 넘을 때만 피드백)이 단순 빈도 감소 대조군보다 오차 감소·안정화·파지에서 낫다 — 이득은 "덜 주기" 때문이 아니다. |
| **[PR]** Soderstrom, N.C., & Bjork, R.A. (2015). *Learning Versus Performance: An Integrative Review.* Perspectives on Psychological Science 10(2), 176–199. https://doi.org/10.1177/1745691615569000 | 습득기 수행 향상이 학습으로 이어지지 않는 일이 흔하고, 일부 조작은 둘을 **반대 방향**으로 움직인다. |
| **[PP]** Hsieh, T.-H., et al. (2026). *Skill-Adaptive Ghost Instructors.* CHI '26. https://arxiv.org/abs/2603.06253 | VR 피아노. "상시 큐잉은 과의존을 유발하고 기억·전이를 잠식한다." 수행 적응형 페이딩이 무유도 파지를 개선. |
| **[PR]** Sigrist, R., Rauter, G., Riener, R., & Wolf, P. (2013). *Augmented visual, auditory, haptic, and multimodal feedback in motor learning: A review.* Psychonomic Bulletin & Review 20(1), 21–53. https://doi.org/10.3758/s13423-012-0333-8 | 기술 매개 훈련에서 피드백 양식·빈도·과제 복잡도를 묶은 표준 참고문헌. |
| **[PR]** Wulf & Shea (2002), §3과 같음 | **반대 무게.** 페이딩·빈도 감소의 이점은 대개 단순 과제에서 나왔고 복합 전신 과제에서는 재현되지 않았다. |

**상태: 명명된 위험.** 좋은 소식은 σ 임계 자체가 **대역폭 피드백**이라는 것이고 Lee & Carnahan이
그 기제를 지지한다. 나쁜 소식은 **우리가 그 대역폭을 0에 가깝게 두었다**는 것이다 — 곧 Salmoni와
Winstein & Schmidt가 지목한 100% 피드백 조건이고, Hsieh가 VR에서 과의존으로 재현한 조건이다.

게다가 §4의 G는 **교정한 그 패스 위에서 재는 수행 지표**다. Soderstrom & Bjork가 말하는
"수행은 오르는데 학습은 아닌" 경우를 우리 설계는 **구별할 수 없다.**

**그러므로 82.63% → 91.58%(C1)는 "무엇을 말할지"에 대한 증거이지 "얼마나 자주 말할지가 옳다"는
증거가 아니다.** 부호검정 74:7도 파지에 대해 아무 말도 하지 않는다.

**결정적 검정(미실시)**: 무피드백 파지 시험, 가능하면 24시간 후, σ=0.1 대 더 넓은/페이딩 임계.
훈련생 데이터가 없어 지금은 불가능하다. 문헌이 권하는 처방은 "덜 주기"가 아니라 **대역폭 확대 +
페이딩**, 또는 **자기통제 전달**(훈련생이 교정을 요청)이다.

σ를 2.0에서 0.1로 내린 결정(2026-09-04)은 여전히 유효하다 — 그때는 미발부가 G를 정의상 0으로
박고 있었다. 다만 그 판단도 같은 수행 지표 위에서 내려졌다.

---

## 8. 평가 설계 — 누수·군집·검정

| 출처 | 내용 |
|---|---|
| **[PR]** Saeb, S., Lonini, L., Jayaraman, A., Mohr, D.C., & Kording, K.P. (2017). *The need to approximate the use-case in clinical machine learning.* GigaScience 6(5), gix019. https://doi.org/10.1093/gigascience/gix019 | 레코드 단위 분할은 **피험자 단위** 분할보다 성능을 크게 과대평가한다(같은 데이터에서 약 2% 대 9–27% 오차). |
| **[PR]** Little, M.A., et al. (2017). *Using and understanding cross-validation strategies.* GigaScience 6(5), gix020. https://doi.org/10.1093/gigascience/gix020 | 분할은 **주장하려는 일반화**에 맞춰 골라야지 편의로 고르면 안 된다. |
| **[PR]** Hammerla, N.Y., & Plötz, T. (2015). *Let's (not) stick together: pairwise similarity biases cross-validation in activity recognition.* UbiComp '15, 1041–1051. https://doi.org/10.1145/2750858.2807551 | 시간적으로 인접하거나 같은 세션의 동작 윈도는 쌍별로 유사해, 다른 폴드에 넣으면 결과가 부풀려진다. |
| **[PR]** Kapoor, S., & Narayanan, A. (2023). *Leakage and the reproducibility crisis in ML-based science.* Patterns 4(9), 100804. https://doi.org/10.1016/j.patter.2023.100804 | 8가지 누수 유형. 우리 위험은 "훈련·시험 간 비독립"과 "정규화기를 풀링된 데이터에서 추정". |
| **[PR]** Kaufman, S., Rosset, S., Perlich, C., & Stitelman, O. (2012). *Leakage in data mining.* ACM TKDD 6(4), 15. https://doi.org/10.1145/2382577.2382579 | 학습–예측 분리 규율의 원전. |
| **[PR]** Lazic, S.E. (2010). *The problem of pseudoreplication in neuroscientific studies.* BMC Neuroscience 11, 5. https://doi.org/10.1186/1471-2202-11-5 | 같은 피험자에서 나온 비독립 관측을 독립 반복으로 세면 유효 n이 부풀고 거짓 양성이 난다. |
| **[PR]** Ho, J., Tumkaya, T., Aryal, S., Choi, H., & Claridge-Chang, A. (2019). *Moving beyond P values: data analysis with estimation graphics.* Nature Methods 16, 565–566. https://doi.org/10.1038/s41592-019-0470-3 | 짝지은 효과크기의 **부트스트랩 신뢰구간**을 이분법적 p값보다 앞세우는 것이 권고 표준. |
| **[PP]** *Another Argument in Favour of Wilcoxon's Signed Rank Test.* https://arxiv.org/abs/1311.5354 | Wilcoxon 부호순위의 ARE는 연속 대칭 분포에서 ≥0.864, 정규에서 0.955 — 부호검정(0.637)보다 효율적. |
| **[PR]** Frykberg, G.E., Grip, H., & Alt Murphy, M. (2021). *How many trials are needed in kinematic analysis of reach-to-grasp?* J. NeuroEngineering and Rehabilitation 18, 101. https://doi.org/10.1186/s12984-021-00895-3 | ICC 기반으로 참가자당 3–5시행이면 운동학 평균이 안정(타이밍·매끄러움은 8–9). σ의 표본 수를 정당화하는 방법. |

### 8.1 우리가 직접 확인한 것 (2026-09-23)

- **σ 누수 없음.** 전처리 누수 의심(z점수 분모를 A∪B에서 추정)은 **사실이 아니다.**
  `service.improvement`가 `split`의 S0와 S1을 **둘 다 B 반쪽으로만** 재고, 교정 선택만 A에서 한다.
  풀링 경로가 없다.
- **군집 구조.** 190쌍의 독립 작업자는 **6명**이다(작업자×세션 144칸, 세션당 패스 중앙값 1개·최대 4;
  작업자×날짜 17칸; 레벨 10개). "새 작업자에게 일반화된다"는 주장의 정직한 n은 **6**이다.
- **유사복제 보정 후에도 결론이 산다.** 작업자 단위로 다시 재면 **6명 전원**이 ≥0.70과 중앙값
  양쪽에서 좋아졌고(작업자 단위 부호검정 p = 0.0156), 작업자를 복원추출한 **군집 부트스트랩**의
  Δ≥0.70 90% 구간은 **[+5.78, +11.80] %p**로 패스 단위 [+5.26, +12.63]과 사실상 같다.
  **앞으로는 군집 버전을 표준으로 보고한다.**
- **날짜 분할의 한계.** 날짜 분할은 *선택* 순환은 막지만 *정체* 누수는 막지 못한다 — 같은 전문가가
  양쪽 반쪽에 나온다. Saeb/Little 기준으로는 피험자 단위보다 레코드 단위에 가깝다. 정직한 주장은
  **"이 전문가의 새 세션에 일반화된다"**이지 "전문가 일반의 폼에 일반화된다"가 아니다.

### 8.2 들어맞지 않은 비판 하나 (기록해 둔다)

"부호검정이 실패할 수 없는 단언이다 — S1은 구성상 S0보다 크니까"라는 지적이 있었다.
**우리 부호검정은 S1 대 S0가 아니라 같은 쌍 위에서 두 정책의 G를 비교한 것이다.** 실패할 수 있다는
증거도 있다 — 난수 축 대조군이 46:71로 졌다. 다만 **바탕의 지적은 다른 자리에서 들어맞는다**:
G > 0 자체는 거의 자동이므로 "G ≥ 0.70 비율"을 귀무 검정처럼 제시해서는 안 되고, 실제로 반증
가능했던 양은 **대조군 비교뿐이므로 그것을 헤드라인으로 올려야 한다.**

---

## 9. 이 분야의 기존 시스템과 전이

| 출처 | 내용 |
|---|---|
| **[PR]** Lee, G.A., Yang, U., Son, W., Kim, Y., Jo, D., Kim, K.-H., & Choi, J. (2010). *VR Content-Based Training for Spray Painting Tasks in the Shipbuilding Industry.* ETRI Journal 32(5). https://doi.org/10.4218/etrij.10.1510.0105 | **가장 가까운 선행 연구** — 한국 조선소 도장 VR 훈련, 실제 야드 배치. 평가는 선체 블록 위의 도막 침착이지 자세가 아니다. **전문 미확보 — 인용 전 확인 필요.** |
| **[PR]** Xu, H., Wu, D., Yu, W., Zhao, Y., & Yu, Q. (2023). *Artificial Spray Modeling of Ships for Personnel Training.* Coatings 13(5), 838. https://doi.org/10.3390/coatings13050838 | Unity 기반 선박 도장 훈련기. 채점이 **건 파라미터·위치·방향과 그 결과 도막 두께**로 구성된다 — 자세 항 없음. |
| **[PR]** Mulders, M., Buchner, J., & Kerres, M. (2022). *VR in Vocational Training: … Vehicle Painting Simulator.* Technology, Knowledge and Learning. https://doi.org/10.1007/s10758-022-09630-w | 견습생 47명. VR은 부스 실습의 **보완이지 대체가 아니다**라고 결론. **전문 미확보.** |
| **[PP]** Nasri, M., et al. (2024). *Designing a VR Training Apprenticeship for Cold Spray Advanced Manufacturing.* arXiv:2411.08859. https://arxiv.org/abs/2411.08859 | 22명. VR 훈련 후 **실제 장비에서의 과제 수행**을 측정. 전문가 예시·자세 채점 없음. |
| **[PR]** Kaplan, A.D., et al. (2021). *The Effects of VR, AR, and MR as Training Enhancement Methods: A Meta-Analysis.* Human Factors 63(4). https://doi.org/10.1177/0018720820904229 | **중요한 귀무 결과**: XR 훈련은 비시뮬레이션 대조 환경과 **다르지 않다.** 가치는 안전·비용에 있다. |
| **[PR]** Stone, R.T., Watts, K.P., & Zhong, P. (2011). *Virtual Reality Integrated Welder Training.* Welding Journal 90(7), 136–141. | 22명. **50% VR 대체** 집단이 네 가지 용접 자격에서 전통 훈련 집단을 앞섰다. 소표본, 비맹검. |
| **[PR]** Ipsita, A., et al. (2022). *Towards Modeling of VR Welding Simulators.* CHI '22. https://doi.org/10.1145/3491102.3517696 | 24명. VR 용접이 영상 대조군보다 전이가 좋았고 **시각-햅틱 유도가 결과를 개선**했다 — 몰입이 아니라 **유도 신호의 질**이 전이를 만든다는 가장 강한 통제 증거. |
| **[PR]** Vaughan, N., & Gabrys, B. (2020). *Scoring and Assessment in Medical VR Training Simulators with Dynamic Time Series Classification.* Eng. Applications of AI. https://arxiv.org/abs/2006.12366 | 271명. **DTW-1NN 60%, SoftDTW 최근접중심 77.5%, 학습 ResNet 85%** — 평범한 DTW 최근접이웃이 가장 약했다. |
| **[PR]** *Procrustes Dynamic Time Warping Analysis for Automated Surgical Skill Evaluation.* ASTESJ 6(1), 100. https://www.astesj.com/v06/i01/p100/ | Procrustes 정렬 후 DTW로 JIGSAWS에서 약 97% 분류. **저자 미확인.** |
| **[V]** SimSpray https://simspray.net/ · VirtualPaint(IWRC) https://iwrc.uni.edu/painter-training/virtualpaint · Skillveri https://skillveri.com/ · Seabery https://seaberyat.com/ar-spray-painting-simulator/ | 상용. 공통적으로 **건**(거리·각도·속도·중첩)과 **도막**(두께·커버리지·전이효율)을 채점한다. 방법론 미공개. |

**상태: 자세 채점은 미검증 영역이다.** 논문이든 상용이든 **몸을 채점하는 시스템을 찾지 못했다.**
신규성이자 위험이다 — 빌려 쓸 임계값도 검증된 문구도 없다. 리뷰어가 견줄 기준선을 위해 건 자세·
도막 지표를 자세 점수 **옆에** 두는 것이 좋다.

**우리에게 겨눠진 경고**: 숙련도 평가의 표준 기법인 **Procrustes 정렬 후 DTW는 전역 평행이동을
지운다** — 우리가 가장 정보량이 많다고 찾아낸 바로 그 오프셋이다. 우리 방법이 전역 평행이동을
**일부러 정규화하지 않는다는 것과 그 이유**를 명시해야 한다. 또 우리 매칭이 DTW급 엄밀성을
갖는다고 주장해서는 안 된다(Vaughan & Gabrys).

**포지셔닝**: Kaplan의 귀무 결과 때문에 "VR이 실물보다 낫다"고 말하면 안 된다. Stone과 Ipsita가
가리키는 것은 **혼합 훈련**과 **실제 부스가 줄 수 없는 유도 신호**다. 우리 자세 교정의 자리가
바로 거기다.

---

## 10. 확인하지 못한 것 (인용 전 처리)

- **Lee et al. 2010(ETRI)**, **Mulders et al. 2022** — 전문 미확보. 지표 서술이 2차 요약 기반이다.
- **AgentCoach(CHI '26)** — 단서 선택 규칙이 초록에서 확인되지 않았다.
- **2016 Procedia DTW 논문**, **ASTESJ Procrustes-DTW** — 저자 확인 불가.
- **Hake 1998 · Marx & Cummings 2007** — 권·쪽 정보가 PhysPort 정리 페이지 기준이다.
- **부호검정 ARE 0.637** — 벤더 문서(NCSS) 기준. 출판 전 비모수 표준 교재로 대체할 것.
- §1의 arcing 문헌은 전부 **업계 기술문헌**이고 동료심사가 아니다.

---

## 11. 여기서 나온, 아직 재지 않은 측정

우선순위 순.

1. **진행축 머리 변위 ↔ 패스 내 도막 두께 편차의 상관.** 성립하면 지금 학계에 없는 결과다.
   성립하지 않으면 우리 주장의 범위가 좁아진다. §1·§5 양쪽이 이것을 지목한다.
2. **전체 공분산 마할라노비스로 축 순위 매기기** 대 현행 argmax-σ. §5 단서 2. 싼 변경.
3. **오른손을 단일 문장으로 줄이기.** §3. 머리를 다축으로 늘리는 안의 반대 방향이다.
4. **교정 문구를 외적 초점으로 다시 쓰기.** §2. 측정 무관, 문구만.
5. **보고 보강** — S0 히스토그램, 원시 ΔS 병기, S0 층화 G. §4.2.
6. **무피드백 파지 시험.** §7. 훈련생 데이터 확보 전까지 불가능.
