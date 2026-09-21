// 3D 뷰. compare()가 낸 `view` 블록만 그린다 — 지표를 브라우저에서 다시 계산하지
// 않고, 3D 장면 안에는 숫자를 아예 걸지 않는다(교정 화살표의 번호 배지만 예외다).
// 수치는 전부 왼쪽 패널이 서버의 표시 문자열로 낸다(app.js 머리말).
//
// 무엇을 그리나(서버 응답 계약):
//   - **두 스켈레톤**(`layers.query.joints`/`bones`와 `layers.reference.joints`/`bones`): 대상은 파랑,
//     기준은 보라. 둘 다 시뮬레이터 아바타 추정이라 관절 표식은 속 빈(윤곽선) 구다.
//   - 관절 진하기 = `view.deviation.joints[name]` / `view.deviation.joint_max_m`. **서버가 잰 값**이고
//     화면은 다시 계산하지 않는다. 사용자가 요구한 그림이 바로 이것이다: "기준 스켈레톤과 질의
//     스켈레톤을 같이 그리고 유사도가 떨어지는 조인트일수록 색상을 진하게".
//   - 대표 프레임 실측 마커(`fit.anchors[].measured_marker`): 층 색으로 **채운** 구체 — 실측 부호다.
//     자기 궤적 선 위의 **한 순간**이므로 선을 가로지르는 고리(`moment-ring`)를 두른다.
//   - 장치 모양(`fit.anchors[].device`): 몸(추정 관절) → 장치(실측 구). 건은 굵은 막대(도구), HMD·
//     플래쉬는 가는 막대. 이름과 거리 글자는 서버 문자열(`device.label`) 그대로다.
//   - 벽 위 분사 스트로크(`view.stroke` · `view.reference_stroke`): 몸 움직임 궤적과 **다른 색 계열**
//     (따뜻한 주황·노랑)로 그린다. 사용자가 둘을 혼동한 것이 이 화면을 다시 만든 출발점이다.
//   - 움직임 궤적(`layers.*.head/right/left`): 대상·기준 둘 다, 층 색(차가운 파랑·보라)의 가는 관.
//     채점이 실제로 쓰는 실측이라 읽혀야 한다 — 다만 따뜻한 스트로크 색 계열로는 절대 가지 않는다.
//   - 교정 화살표: `from_frame`→`to_frame`(없으면 `from`→`to`). 3D 안 글자는 **번호뿐**이다.
//   - 검증된 벽면 메시(`mesh.available`일 때만). 양면(DoubleSide) 반투명.
//
// 3D 안에는 숫자를 걸지 않는다 — 수치는 전부 왼쪽 패널이다. 예전에는 보정 오프셋 선 위에
// "보정 오프셋 0.57m(적합값)", 어긋남 점선 위에 "12.8cm" 같은 라벨이 떠 있었고, 사용자가
// "숫자며 수치 무슨 말인지를 모르겠고 보정 오프셋은 왜 표시했는지를 모르겠음"이라고 했다.
//
// three.js는 cdnjs에서 버전 고정으로 이 스크립트보다 먼저 실린다(index.html).
// 못 실리면 `createSkeletonView`가 아무것도 던지지 않고 안내만 남긴다.

const POINT_LABELS = { right: "오른손", head: "머리", left: "왼손", hit: "착탄점" };

// 층별 기본색. σ 3단(ok/warn/bad)과 겹치지 않는 색. style.css의 --layer-*와 같은 값.
const LAYER_COLOR = { query: 0x5b9bff, reference: 0xa684e0, corrected: 0x34c2c2 };
const MESH_COLOR = 0x9aa5b1;
// 추정 뼈는 실측 마커(층 색 100%)보다 한 단계 조용해야 한다 — 채도를 뺀 밝은 회청색.
// 이 색은 이제 **벌어짐 0**의 색이다: 관절이 기준과 멀어질수록 층 색 쪽으로 진해진다.
const BONE_COLOR = 0xc8d0dc;

// 벽 위 분사 스트로크 색. 움직임 궤적(파랑·보라)과 **다른 색 계열**이어야 한다 — 사용자가 둘을
// 혼동했고("기준 스트로크가 보이지 않음", 궤적을 스트로크로 읽음) 그것이 이 재작업의 출발점이다.
// 따뜻한 쪽(주황·노랑)으로 몰아 층 색과 겹치지 않게 한다.
const STROKE_COLOR = { query: 0xff7a1a, reference: 0xffd23f };
const STROKE_LEGEND = { query: "대상 스트로크(벽면 착탄)", reference: "기준 스트로크(벽면 착탄)" };

const SIGMA_OK = 0x2f855a;
const SIGMA_WARN = 0xb7791f;
const SIGMA_BAD = 0xc53030;
const SIGMA_UNKNOWN = 0x718096;

// 크기 상수(m). 화살표 굵기는 교정 크기와 분리한다 — 크기는 길이와 ③의 수치로 전달한다.
const MEASURED_MARKER_RADIUS = 0.035;
const RING_OUTER = 0.05;
const RING_INNER = 0.04;
// 머리 구체 반지름 폴백(m). 서버가 `fit.head_diameter_m`을 못 주는 드문 경로(스켈레톤은 있는데
// fit이 없는 경우)에서만 쓴다 — 정상 경로는 서버가 이 리그의 골반→머리 길이 × BVH 실측 비례로
// 낸 지름을 쓴다(`buildSkeletonGroup`, `body_proportions.head_diameter_m`). 예전에는 고정
// 0.09m(지름 0.18m)를 항상 썼는데, 그려지는 골반→머리 0.409m의 44%라 풍선처럼 보였다(사람은
// 17~22%). 이 폴백 값은 그 44%가 아니라 서버가 보통 내는 값에 맞춘 근사치일 뿐이다.
const HEAD_SPHERE_RADIUS_FALLBACK = 0.036;
// 뼈는 장면 안에서 읽혀야 한다(2라운드: 1~2px 회색 선이 벽 메시 위에서 가장 약한 요소였다). 실측 마커(층 색
// 구체 0.035m)와는 채도로 위계를 둔다 — 뼈는 무채색, 실측은 층 색.
const BONE_RADIUS = 0.009;
const BONE_OPACITY = 0.9;
const BONE_OPACITY_FADED = 0.45;
// IK 다리는 **흐리게** 그린다(S4) — 추적 장치가 없고 시뮬레이터가 지은 자세다. 기본 그림은
// **자리표시 다리**(점선)이고 IK는 토글이다(B) — 자리표시는 늘 두 층에 그려지므로 다리가 통째로
// 사라지는 화면은 없다. 해부학 범위를 벗어난 IK 다리도 토글로 볼 수 있고(B: "IK는 토글로 볼 수
// 있게"), 그때는 한 단계 더 흐리다 — 어느 검사를 얼마나 벗어났는지는 서버 고지가 숫자로 말한다.
const LEG_OPACITY_FACTOR = 0.5;
const LEG_OUT_OF_RANGE_OPACITY_FACTOR = 0.6;
// 오른 전완·손 저신뢰(S5). 아바타 손 IK와 트래킹된 건이 원본에서 중앙값 571mm 어긋나 있다.
const LOW_CONFIDENCE_OPACITY_FACTOR = 0.45;
// 계산된 층(교정 후·예상)의 선. 실측 궤적의 실선과 한눈에 구분되게 끊어 그린다.
const COMPUTED_DASH = 0.02;
const COMPUTED_GAP = 0.02;
const ARROW_SHAFT_RADIUS = 0.006;
const ARROW_HEAD_RADIUS = 0.016;
// 벽 위 분사 스트로크의 굵기(m). 뼈(0.009)보다 가늘게 둬 몸과 위계를 지키면서, 1px 선과 달리
// 반투명 벽 위에서 읽힌다.
const STROKE_RADIUS = 0.006;

const BADGE_MIN_PX = 26;
// 글자 라벨을 몸통 화면 위치에서 바깥으로 미는 첫 거리(px).
const LABEL_PUSH_PX = 16;
// 맞춤 목표 = 맞춤 대상이 캔버스 반폭·반높이에서 차지할 비율. 남기는 14%는 마커 구체 반지름·
// 화살표 촉·글자 라벨 몫이다(0.75는 그만큼을 그냥 비워 뒀다).
//   pose(기본·'자세 맞춤'): **그려지는 것을 하나도 자르지 않는다** — 두 몸·자리표시 다리·교정
//     화살표·벽면 스트로크가 전부 캔버스 안이다. 발이 바닥면에 선 것을 보는 자리이고(B), 기준
//     벽면 스트로크가 보이는 자리다(S2). 레벨과 무관하게 이 한 규칙이다(F).
//   torso("상체 확대")만 0.98이다 — 상체를 캔버스에 꽉 채우려고 일부러 다리·교정 화살표·착탄점을
//     맞춤 대상에서 뺐고, 그래서 그것들이 화면 밖으로 나갈 수 있다. 나간 것의 이름은 캔버스 아래에 적는다.
const FRAMING_TARGET = { pose: 0.86, tracks: 0.75, wall: 0.9, torso: 0.98 };

// F 합격선: 상체(머리 **실측** 마커~골반)가 캔버스 높이에서 차지할 최소 비율. 기본 뷰는 이 선을
// **지키지 못할 때 그 사실을 적는다**(`describeTorsoShortfall`) — 당겨서 맞추지 않는다.
//
// 한때 여기서 합격선까지 당겼고(`pullInToTorsoFloor`), 여섯 조합 전부 0.4000이 나왔다. 그 값은
// 숫자만 맞은 값이었다 — 검증이 카메라 행렬로 직접 투영해 재니 같은 화면에서 **자리표시 다리
// 관절 16개 중 L1은 2개만** 캔버스 안이었고(발 y=942~1030, 캔버스 높이 600), 대상·기준 벽면
// 스트로크 끝점 3개도 밖이었다. 사용자 정책은 "자리표시 다리를 기본으로 그린다"(B)이고 첫 불만은
// "기준 스트로크가 보이지 않는다"(S2)였으므로, 그 둘을 화면 밖으로 밀어내고 얻은 합격은 합격이 아니다.
//
// 둘은 한 카메라에서 동시에 참일 수 없다 — 이것은 취향이 아니라 기하다. 상체는 머리~발 높이의
// 일부이므로 전신을 담으면 상체 비율의 천장이 (상체 길이 ÷ 머리~발 높이)다: 실측 김경호 L1 첫
// 패스에서 머리 실측 마커가 바닥 위 1.5083m, 골반→머리 0.4091m라 천장이 0.27이고 합격선 0.40에
// 닿지 못한다. 그래서 기본 뷰는 **자르지 않는 쪽**을 택하고, 못 미친 값과 합격선을 화면이 숫자로
// 말하며, 그 선을 넘겨 보려는 사람은 '상체 확대'를 누른다(그 프리셋은 일부러 자른다).
//
// 고친 뒤 실측(헤드리스 CDP, 김경호 L1·L3·L6 첫 패스, 기준 자동, 1875×1000·840×1050 여섯 조합):
// 기본 뷰의 상체 비율은 L1 0.1517 · L3 0.3309 · L6 0.2103(창 크기와 무관)이고, 화면 밖으로 나간
// 맞춤 대상은 **0개**, 카메라 행렬로 직접 투영한 자리표시 다리 관절은 여섯 조합 모두 16/16이
// 캔버스 안이다. '상체 확대'는 같은 패스에서 0.6448 · 0.7675 · 0.5266이라 합격선을 넘는다 —
// 고지의 마지막 문장이 주장하는 그 값이 실제로 그 값이다.
const MIN_TORSO_FRACTION = 0.4;

// 움직임 궤적은 **차분한 한 색**이다(S3). 예전에는 `view.weights`(채점 가중치)로 진하기를 정했는데,
// 사용자가 원한 강조는 가중치가 아니라 **기준과의 차이**였다("유사도가 떨어지는 조인트일수록 색상을
// 진하게"). 진해지는 것은 이제 관절뿐이고, 궤적은 스트로크보다 눈에 덜 띄어야 한다.
const TRACK_OPACITY = 0.3;
const TRACK_MARKER_RADIUS = 0.006;
// 실측 궤적(질의·기준)은 **읽혀야** 한다 — 채점이 실제로 쓰는 값이다. 사용자가 "진한 구와 연한 선"이
// 무엇인지 몰랐다: 선이 0.3 불투명도 1px이라 구와 같은 장치라는 것이 안 보였다. 1px 선만으로는 WebGL에서
// 굵기를 못 키우므로 스트로크처럼 가는 관(실린더)을 잇되, **색은 층 색(차가운 계열) 그대로**라 따뜻한
// 벽면 스트로크(주황·노랑)와 섞이지 않는다. 관은 스트로크(0.006)보다 가늘다.
const TRACK_LINE_OPACITY = 0.9;
const TRACK_TUBE_RADIUS = 0.0035;
const TRACK_TUBE_OPACITY = 0.75;
// "이 순간" 고리: 채운 구체(대표 프레임 실측)를 둘러 궤적을 가로지르는 고리. 축은 그 자리의 궤적
// 방향이다 — 구가 선 위의 한 점이라는 것이 모양으로 보인다. 속 빈 고리라 실측 부호(채운 구)와 다르다.
const MOMENT_RING_RADIUS = 0.055;
const MOMENT_RING_TUBE = 0.005;
// 장치 모양(몸 → 장치). 건은 손에 든 도구라 굵게, HMD·플래쉬는 가늘게. 색은 층 색을 흰색 쪽으로
// 씻은 반투명 — 채운 층 색 구(실측 부호)와도, 무채색 뼈(추정)와도 한눈에 갈린다.
const DEVICE_TOOL_RADIUS = 0.012;
const DEVICE_LINK_RADIUS = 0.004;
const DEVICE_WASH = 0.45;
const DEVICE_OPACITY = 0.6;
// 장치 이름표 끈(px). 구 중심에서 이만큼 앞에서 멈추고(고리를 가리지 않게), 이보다 짧으면 달지 않는다.
const LEADER_STOP_PX = 8;
const LEADER_MIN_PX = 6;

// 벌어짐 0에서도 **층 색이 남아 있어야** 두 몸이 누구인지 구별된다. 무채색에서 시작하면 잘 맞은
// 관절끼리는 대상·기준이 같은 회색이 되어 두 자세가 한 덩어리로 뭉친다(첫 실측 화면이 그랬다).
// 그래서 t=0은 층 색을 회색 쪽으로 씻은 색이고, t=1이 층 색 100%다 — 색조는 층, 진하기는 벌어짐.
const DEVIATION_WASH = 0.72;

/** 벌어짐 t(0~1)를 진하기로. 눈금(최대 편차)은 서버가 준다 — 화면은 나누기만 하고 거리를 다시 재지 않는다. */
function deviationStyle(metres, maxMetres, layerColor) {
  const span = typeof maxMetres === "number" && maxMetres > 0 ? maxMetres : null;
  const t = span === null || typeof metres !== "number" ? 0 : clamp(metres / span, 0, 1);
  const pale = mixColor(layerColor, BONE_COLOR, DEVIATION_WASH);
  return { t, color: mixColor(pale, layerColor, t), opacity: 0.4 + 0.55 * t };
}

/** 두 16진 색 사이 선형 보간. */
function mixColor(from, to, t) {
  const out = [];
  for (let shift = 16; shift >= 0; shift -= 8) {
    const a = (from >> shift) & 0xff;
    const b = (to >> shift) & 0xff;
    out.push(Math.round(a + (b - a) * t));
  }
  return (out[0] << 16) | (out[1] << 8) | out[2];
}

function clamp(value, min, max) {
  return Math.min(max, Math.max(min, value));
}

// --- 휠 확대 ---------------------------------------------------------------------------
//
// 옛 핸들러는 `event.deltaY > 0 ? 1.1 : 1/1.1`이라 **부호만** 쓰고 크기를 버렸다. 실측(CDP로
// deltaY를 4·53·100·120·240·1000으로 바꿔 32회): 배율이 전부 1.1000 / 0.90909였다 — 마우스 한
// 노치(deltaY=100, deltaMode=0)도 트랙패드의 deltaY=4도 똑같이 10%를 먹는다. 트랙패드는 한 번
// 쓸기에 그런 이벤트를 수십 건 내므로 1.1^50 = 117배가 한 번에 걸려 장면이 상·하한에 붙었다.
//
// 이제 deltaY를 **픽셀로 정규화**해 지수에 넣는다. 한 노치(WHEEL_NOTCH_PX)당 WHEEL_NOTCH_FACTOR,
// 그 사이는 연속이다: deltaY=4면 1.05^0.04 = +0.20%. 한 이벤트가 WHEEL_PIXELS_MAX를 넘지 못하게
// 잘라 관성 스크롤의 큰 한 방이 화면을 날려 보내지 못하게 한다.
const WHEEL_NOTCH_PX = 100;      // 마우스 휠 한 노치(관측: deltaY=100, deltaMode=0, wheelDelta=-120)
const WHEEL_NOTCH_FACTOR = 1.05; // 한 노치당 +5% / -4.76%(옛 값의 절반)
const WHEEL_PIXELS_MAX = 300;    // 한 이벤트 상한 = 3노치(약 ±15%)
const WHEEL_LINE_PX = 16;        // deltaMode=1(줄) 한 줄을 픽셀로 볼 때의 값
// 거리 한계는 **장면 크기에서** 나온다. 옛 값 clamp(0.05, 60)은 L3 기본 거리 0.80m의 1/16배·75배로
// 비대칭이었다. 이제 기본 프리셋('자세 맞춤')이 푼 거리를 기준으로 안쪽 1/8배·바깥쪽 24배까지 본다.
// 실측 프리셋 거리(김경호 L1/L2/L3/L6, 1875×1000, m):
//   자세 맞춤 1.012 / 1.260 / 0.802 / 1.237 · 상체 확대 0.636 / 0.548 / 0.708 / 0.809
//   궤적 포함 2.836 / 3.134 / 2.391 / 2.992 · 벽 포함 3.152 / 2.908 / 3.758 / 3.491
// 가장 가까운 프리셋(0.548)도 가장 먼 프리셋(3.758)도 그때의 한계 안이다 — 안쪽 한계가 0.100~0.158m,
// 바깥쪽이 19.2~30.2m였다. 절대 한계는 카메라 근·원 평면(0.02m·200m) 안쪽으로 남겨 둔다.
const RADIUS_MIN_ABS = 0.05;
const RADIUS_MAX_ABS = 60;
const RADIUS_IN_FACTOR = 8;
const RADIUS_OUT_FACTOR = 24;

/** 휠 이벤트 한 건의 세기를 **픽셀**로. deltaMode는 브라우저·장치마다 다른 단위를 쓴다. */
function wheelPixels(event, viewportHeight) {
  const mode = event.deltaMode ?? 0;
  const delta = event.deltaY ?? 0;
  if (mode === 1) return delta * WHEEL_LINE_PX;
  if (mode === 2) return delta * viewportHeight;
  return delta;
}

/** 픽셀 세기 → 카메라 거리에 곱할 배율. 한 노치가 정확히 WHEEL_NOTCH_FACTOR가 되는 지수. */
function wheelZoomFactor(pixels) {
  const px = clamp(pixels, -WHEEL_PIXELS_MAX, WHEEL_PIXELS_MAX);
  return Math.exp((px / WHEEL_NOTCH_PX) * Math.log(WHEEL_NOTCH_FACTOR));
}

/** 이 장면에서 허용할 카메라 거리 범위. `base`는 기본 프리셋이 푼 거리(없으면 절대 한계만). */
function radiusLimitsFor(base) {
  if (!(typeof base === "number") || !(base > 0)) return { min: RADIUS_MIN_ABS, max: RADIUS_MAX_ABS };
  return {
    min: Math.max(RADIUS_MIN_ABS, base / RADIUS_IN_FACTOR),
    max: Math.min(RADIUS_MAX_ABS, base * RADIUS_OUT_FACTOR),
  };
}

function scaleColor(hex, factor) {
  const r = Math.round(((hex >> 16) & 255) * factor);
  const g = Math.round(((hex >> 8) & 255) * factor);
  const b = Math.round((hex & 255) * factor);
  return (r << 16) | (g << 8) | b;
}

function sigmaColor(sigma) {
  // 근거가 없으면 회색이다 — 없는 σ를 0σ(초록)로 그리면 "근거가 강하다"는 거짓 인상을 준다.
  if (typeof sigma !== "number") return SIGMA_UNKNOWN;
  if (sigma <= 1) return SIGMA_OK;
  if (sigma <= 2) return SIGMA_WARN;
  return SIGMA_BAD;
}

function isVec3(value) {
  return Array.isArray(value) && value.length === 3 && value.every((x) => typeof x === "number");
}

/** 씬 그래프에서 이 그룹의 후손 수. 다시 그릴 때 층이 **쌓이는지**를 재는 계수기다 —
 * `resetScene()`이 층을 비우지 않으면 [평가]를 누를 때마다 이 수가 늘어난다. */
function countDescendants(group) {
  let n = 0;
  group.traverse(() => { n += 1; });
  return n - 1; // 자기 자신은 빼고 센다
}

/** 이 그룹 안에서 `userData.kind`가 주어진 값인 객체 수. */
function countKind(group, kind) {
  let n = 0;
  group.traverse((child) => { if (child.userData.kind === kind) n += 1; });
  return n;
}

function disposeObject3D(object) {
  object.traverse((child) => {
    if (child.geometry) child.geometry.dispose();
    if (child.material) {
      if (Array.isArray(child.material)) child.material.forEach((m) => m.dispose());
      else child.material.dispose();
    }
  });
}

function buildDom(container) {
  container.replaceChildren();
  container.classList.add("view3d");

  const toolbar = document.createElement("div");
  toolbar.className = "view3d-toolbar";
  const title = document.createElement("h2");
  title.className = "view3d-title";
  title.textContent = "3D 뷰";
  const controls = document.createElement("div");
  controls.className = "view3d-controls";
  controls.innerHTML = `
    <label><input id="view3dLayerQuery" type="checkbox" checked> 대상(파랑)</label>
    <label><input id="view3dLayerReference" type="checkbox" checked> 기준(보라)</label>
    <label><input id="view3dLayerLegs" type="checkbox" checked> 다리(IK)</label>
    <label><input id="view3dLayerCorrected" type="checkbox"> 교정 후(예상)</label>
    <label><input id="view3dLayerWall" type="checkbox" checked> 벽면</label>
    <span class="view3d-controls-sep" aria-hidden="true"></span>
    <button id="view3dPresetFit" type="button">자세 맞춤</button>
    <button id="view3dPresetTorso" type="button">상체 확대</button>
    <button id="view3dPresetFront" type="button">벽 정면</button>
    <button id="view3dPresetSide" type="button">벽 측면</button>
    <button id="view3dPresetTracks" type="button">궤적 포함</button>
    <button id="view3dPresetWall" type="button">벽 포함</button>
  `;
  toolbar.append(title, controls);

  const host = document.createElement("div");
  host.className = "view3d-host";
  const canvas = document.createElement("canvas");
  canvas.className = "view3d-canvas";
  const labelsHost = document.createElement("div");
  labelsHost.className = "view3d-labels";
  // 벽면을 그리지 않는 사유는 캔버스 왼쪽 위에 겹쳐 띄운다 — 캔버스 아래 마지막 줄에 두면 첫눈에 안 보였다.
  const wallNote = document.createElement("p");
  wallNote.className = "view3d-wall-note";
  host.append(canvas, labelsHost, wallNote);

  const legend = document.createElement("div");
  legend.className = "view3d-legend";

  const notes = document.createElement("ul");
  notes.className = "view3d-notes";
  const notesMore = document.createElement("details");
  notesMore.className = "view3d-notes-more";
  notesMore.hidden = true;
  const errors = document.createElement("p");
  errors.className = "view3d-errors";
  const note = document.createElement("p");
  note.className = "view3d-note";
  const correctedNote = document.createElement("p");
  correctedNote.className = "view3d-note view3d-corrected-note";
  // 상체 확대 프리셋이 화면 밖으로 내보낸 맞춤 대상의 이름. 자른 사실을 화면이 스스로 말한다.
  const framingNote = document.createElement("p");
  framingNote.className = "view3d-note view3d-framing-note";

  // wallNote는 위에서 host(캔버스 위 겹침)에 넣었다 — 여기서 다시 넣으면 host 밖으로 옮겨진다.
  container.append(toolbar, host, legend, notes, notesMore, errors, note, correctedNote, framingNote);
  return { canvas, host, labelsHost, legend, notes, notesMore, wallNote, errors, note, correctedNote, framingNote, controls, toolbar };
}

/** `note`에 문구와 상태 톤을 함께 건다. 빈 문구는 CSS `:empty`가 숨긴다. */
function setNote(el, text, tone) {
  el.textContent = text ?? "";
  if (tone) el.dataset.tone = tone;
  else el.removeAttribute("data-tone");
}

function showUnavailable(container, message) {
  const dom = buildDom(container);
  setNote(dom.note, message, "error");
  dom.toolbar.hidden = true;
  dom.host.hidden = true;
  dom.legend.hidden = true;
}

function legendItem(swatchClass, text) {
  const item = document.createElement("span");
  item.className = "legend-item";
  if (swatchClass) {
    const swatch = document.createElement("i");
    swatch.className = `swatch ${swatchClass}`;
    item.append(swatch);
  }
  item.append(document.createTextNode(text));
  return item;
}

function renderLegend(legend, view) {
  legend.replaceChildren();
  // 범례는 **최소한**만 남긴다(S6): 무엇이 실측이고 무엇이 추정인지, 진하기가 무엇을 뜻하는지,
  // 어느 색이 스트로크인지. 사용자가 "텍스트가 너무 많아서 무슨 말을 하고 싶은지 모르겠음"이라고 했다.
  const title = document.createElement("span");
  title.className = "legend-title";
  title.textContent = "화살표 색 = 근거 σ";
  legend.append(
    title,
    legendItem("ok", "≤1σ"),
    legendItem("warn", "≤2σ"),
    legendItem("bad", ">2σ"),
  );
  // 스트로크 색은 궤적 색과 **다른 계열**이다 — 그 사실을 범례가 말한다(S2).
  for (const layer of ["query", "reference"]) {
    const item = legendItem(`stroke-${layer}`, STROKE_LEGEND[layer]);
    item.dataset.role = `stroke-${layer}`;
    legend.append(item);
  }
  // 진하기 눈금 문구는 **서버가 통째로** 짓는다(`deviation.legend`). 화면이 이름과 숫자를
  // 조립하면 눈금과 문구가 갈린다 — 이전 판은 "가장 진한 곳 79.9cm"라고 적었는데 그 79.9cm짜리
  // 관절(LeftCalf)은 다리를 숨겨서 화면 어디에도 없었다. 이제 서버가 눈금을 그리는 관절에서만
  // 잡고, 뺀 관절과 몸 전체 평행이동 몫까지 같은 자리에서 말한다.
  for (const text of view?.deviation?.legend ?? []) {
    if (typeof text !== "string" || text.length === 0) continue;
    const item = legendItem("deviation", text);
    item.dataset.role = "deviation";
    legend.append(item);
  }
  for (const text of view?.legend_notes ?? []) {
    if (typeof text === "string") legend.append(legendItem(null, text));
  }
  // 스트로크 문구도 서버 것이다 — 몇 곳에서 끊었는지·얼마나 떴는지는 서버가 재고 화면은 옮기기만 한다.
  if (typeof view?.stroke?.note === "string" && view.stroke.note.length > 0) {
    const item = legendItem("stroke-query", view.stroke.note);
    item.dataset.role = "stroke";
    legend.append(item);
  }
}

function mount(THREE, container) {
  const dom = buildDom(container);
  renderLegend(dom.legend, null);
  setNote(dom.note, "패스를 고르고 평가를 누르면 여기에 뜹니다.", "idle");

  const scene = new THREE.Scene();
  const camera = new THREE.PerspectiveCamera(50, 1, 0.02, 200);
  let renderer;
  try {
    renderer = new THREE.WebGLRenderer({ canvas: dom.canvas, antialias: true, preserveDrawingBuffer: true });
  } catch (err) {
    showUnavailable(container, "이 브라우저에서 WebGL을 쓸 수 없어 3D 뷰를 그리지 않습니다.");
    return { render() {}, clear() {}, loading() {} };
  }
  renderer.setPixelRatio(Math.min(window.devicePixelRatio || 1, 2));
  scene.background = new THREE.Color(0x070a0f);

  scene.add(new THREE.AmbientLight(0xffffff, 0.7));
  const key = new THREE.DirectionalLight(0xffffff, 0.9);
  key.position.set(2, 3, 2);
  scene.add(key);

  const layerGroups = {
    query: new THREE.Group(),
    reference: new THREE.Group(),
    corrected: new THREE.Group(),
  };
  layerGroups.corrected.visible = false;
  for (const group of Object.values(layerGroups)) scene.add(group);
  const correctionsGroup = new THREE.Group();
  scene.add(correctionsGroup);
  let meshObject = null;
  let skeletonLegs = null;
  let referenceLegs = null;
  // 자리표시 다리(B) — 기본으로 보이는 쪽이다. IK 다리와 **한 층에서 동시에** 보이지 않는다.
  let skeletonPlaceholderLegs = null;
  let referencePlaceholderLegs = null;
  // 자리표시 발이 바닥면에 못 닿은 층의 **간격 보조선**. 자리표시 다리와 한 몸으로 켜고 끈다.
  let skeletonFloorGaps = null;
  let referenceFloorGaps = null;

  const labelRegistry = [];
  // 자리표시 발↔바닥면 **간격** 글자. IK 토글을 켠 층에서는 보조선과 함께 사라져야 한다 —
  // 선 없는 숫자가 남으면 화면이 그리지 않는 것을 말하게 된다.
  const gapLabels = [];
  function clearLabels() {
    for (const entry of labelRegistry) {
      entry.el.remove();
      if (entry.leader) entry.leader.remove();
    }
    labelRegistry.length = 0;
    gapLabels.length = 0;
    deviceLabels.length = 0;
  }
  function registerLabel(position, text, extraClass, extra) {
    const el = document.createElement("div");
    el.className = extraClass ? `view3d-label ${extraClass}` : "view3d-label";
    el.textContent = text;
    dom.labelsHost.appendChild(el);
    const entry = { position: position.clone(), el, ...(extra ?? {}) };
    labelRegistry.push(entry);
    return entry;
  }

  // 카메라 상태(구면좌표) — 드래그 회전·휠 확대가 이 값만 바꾼다.
  const camState = { target: new THREE.Vector3(0, 1, 0), radius: 3, theta: Math.PI / 4, phi: Math.PI / 2.5 };
  // 자동 맞춤 상태. 사용자가 드래그·휠로 카메라를 만지면 창 크기가 바뀌어도 다시 맞추지 않는다.
  // `baseRadius`는 이 뷰의 기본 프리셋('자세 맞춤')이 푼 카메라 거리다 — 휠 한계가 장면 크기를 따른다.
  const framing = { mode: "pose", auto: true, ratio: null, baseRadius: null };
  // 마지막으로 그린 view. resize()가 첫 호출부터 읽으므로 그보다 먼저 선언한다.
  let lastView = null;

  function applyCamera() {
    const sinPhi = Math.sin(camState.phi);
    camera.position.set(
      camState.target.x + camState.radius * sinPhi * Math.sin(camState.theta),
      camState.target.y + camState.radius * Math.cos(camState.phi),
      camState.target.z + camState.radius * sinPhi * Math.cos(camState.theta),
    );
    camera.up.set(0, 1, 0);
    camera.lookAt(camState.target);
    camera.updateMatrixWorld();
    // 검증이 **자기 손으로** 투영할 수 있도록 카메라의 날 상태만 내놓는다(계산된 측정값이 아니다).
    container.__view3dCameraDebug = {
      theta: camState.theta,
      phi: camState.phi,
      radius: camState.radius,
      target: [camState.target.x, camState.target.y, camState.target.z],
      fov: camera.fov,
      near: camera.near,
    };
  }
  applyCamera();

  let dragging = false;
  let lastX = 0;
  let lastY = 0;
  dom.canvas.style.touchAction = "none";
  dom.canvas.addEventListener("pointerdown", (event) => {
    dragging = true;
    lastX = event.clientX;
    lastY = event.clientY;
    dom.canvas.setPointerCapture(event.pointerId);
  });
  dom.canvas.addEventListener("pointerup", () => { dragging = false; });
  dom.canvas.addEventListener("pointercancel", () => { dragging = false; });
  dom.canvas.addEventListener("pointermove", (event) => {
    if (!dragging) return;
    const dx = event.clientX - lastX;
    const dy = event.clientY - lastY;
    lastX = event.clientX;
    lastY = event.clientY;
    camState.theta -= dx * 0.008;
    camState.phi = clamp(camState.phi - dy * 0.008, 0.05, Math.PI - 0.05);
    framing.auto = false;
    applyCamera();
    scheduleFramingNote();
  });
  dom.canvas.addEventListener(
    "wheel",
    (event) => {
      event.preventDefault();
      const { h } = hostSize();
      const factor = wheelZoomFactor(wheelPixels(event, h));
      const limits = radiusLimitsFor(framing.baseRadius);
      camState.radius = clamp(camState.radius * factor, limits.min, limits.max);
      framing.auto = false;
      applyCamera();
      scheduleFramingNote();
    },
    { passive: false },
  );

  function hostSize() {
    return { w: dom.host.clientWidth || 1, h: dom.host.clientHeight || 1 };
  }

  function resize() {
    const { w, h } = hostSize();
    renderer.setSize(w, h, false);
    camera.aspect = w / h;
    camera.updateProjectionMatrix();
    if (lastView && framing.auto) frameTo(framing.mode);
  }
  const resizeObserver = new ResizeObserver(() => resize());
  resizeObserver.observe(dom.host);
  resize();

  function toPixel(vector) {
    const { w, h } = hostSize();
    const p = vector.clone().project(camera);
    return { x: ((p.x + 1) / 2) * w, y: ((1 - p.y) / 2) * h, z: p.z };
  }

  /** 월드 반경 r인 구(원추 밑면)가 화면에서 차지하는 지름(px). 카메라 오른쪽 축으로 잰다. */
  function screenDiameter(center, radius) {
    const right = new THREE.Vector3().setFromMatrixColumn(camera.matrixWorld, 0).normalize();
    const a = toPixel(center);
    const b = toPixel(center.clone().addScaledVector(right, radius));
    return 2 * Math.hypot(b.x - a.x, b.y - a.y);
  }

  const debugLive = {
    badgeCenters: [],
    arrowConeScreenDiameters: [],
    markerScreenDiameters: {},
    handMarkerScreenDiameter: null,
    arrowStarts: [],
  };
  let coneRegistry = [];
  let markerRegistry = [];
  // 두 층의 실측 구 중심과 장치 막대(관절 → 구). 글자 라벨이 피해야 할 장애물이다.
  let sphereObstacles = [];
  let deviceBars = [];
  // 장치 이름표. 층 토글을 끄면 그 층의 이름표도 사라져야 한다(그림 없는 글자가 남으면 안 된다).
  const deviceLabels = [];

  // --- 검증 계측(헤드리스 브라우저가 읽는다) ---------------------------------------------
  // 몸통 마디: 라벨을 밀어낼 장애물이자, 라벨-몸통 겹침을 세는 대상.
  // 몸통 **줄기**(골반-척추-가슴-머리)만. `torsoLabelOverlaps`가 세는 대상이 정확히 이 셋이다 —
  // 팔까지 포함한 넓은 판정은 `limbScreenRects()`/`boneLabelOverlaps`가 따로 낸다.
  const TORSO_BONES = [["Pelvis", "Spine"], ["Spine", "Chest"], ["Chest", "Head"]];
  const TORSO_HALF_PX = 11;
  // 화살표 기둥 반두께(px). 기둥은 몸통보다 얇으니 몸통(11px)보다 좁게 잡는다.
  const ARROW_HALF_PX = 8;
  // 팔·어깨 마디 반두께(px). 몸통 줄기보다 얇게 그려지므로 좁게 잡는다.
  const LIMB_HALF_PX = 8;
  // 보정 오프셋 선·어긋남 점선 반두께(px). 1px 가는 선이지만 글자 사이를 지나면 글자가 갈라져 보인다.
  const THIN_LINE_HALF_PX = 5;
  // 배지(.view3d-label-arrow) 반폭(px): 24px 원 + 1px 테두리. 중심거리 26px 검사는 대각선에서
  // 모서리 겹침을 놓친다(검증 라운드: ④↔① 20.4px²) — 사각형으로 겹침을 본다.
  const BADGE_HALF_PX = 13;
  const isVec3Like = (v) => v instanceof THREE.Vector3;

  /** 화면 선분 하나를 사각형 여러 개로 쪼갠다(라벨 회피용 장애물).
   * 표본 수는 화면 길이에서 뽑는다 — 고정 步수(옛 코드: 팔 6步)는 긴 마디에서 사각형 사이에 틈을 남겨
   * 라벨이 그 틈으로 뼈 위에 앉았다(840×1050의 '10.5cm' × LeftUpperArm>LeftForeArm). */
  function segmentRects(p0, p1, half, into) {
    const rects = into ?? [];
    const steps = Math.max(2, Math.ceil(Math.hypot(p1.x - p0.x, p1.y - p0.y) / half));
    for (let i = 0; i <= steps; i += 1) {
      const t = i / steps;
      const x = p0.x + (p1.x - p0.x) * t;
      const y = p0.y + (p1.y - p0.y) * t;
      rects.push({ x0: x - half, x1: x + half, y0: y - half, y1: y + half });
    }
    return rects;
  }

  /** 몸통 **줄기** 마디(골반-척추-가슴-머리)를 화면 사각형으로 쪼갠다. */
  function torsoScreenRects() {
    const joints = lastView?.layers?.query?.joints ?? {};
    const rects = [];
    for (const [a, b] of TORSO_BONES) {
      if (!isVec3(joints[a]) || !isVec3(joints[b])) continue;
      segmentRects(toPixel(new THREE.Vector3(...joints[a])), toPixel(new THREE.Vector3(...joints[b])), TORSO_HALF_PX, rects);
    }
    return rects;
  }

  /** 몸통 줄기를 뺀 **나머지 상체 마디**(어깨·팔·손)를 화면 사각형으로 쪼갠다.
   * 다리는 뺀다 — 다리 표시 방식은 아직 정해지지 않았고, 다리를 장애물로 넣으면 라벨이 위로만 몰린다.
   * 줄기만 장애물로 두면 거리 라벨이 팔뼈를 덮었다(검증 라운드: L3 '10.5cm' × LeftForeArm>LeftHand). */
  function limbScreenRects() {
    const joints = lastView?.layers?.query?.joints ?? {};
    const bones = lastView?.layers?.query?.bones ?? [];
    const legSet = new Set(lastView?.fit?.legs?.joints ?? []);
    const stem = new Set(TORSO_BONES.map(([a, b]) => `${a}>${b}`));
    const rects = [];
    for (const [a, b] of bones) {
      if (legSet.has(a) || legSet.has(b) || stem.has(`${a}>${b}`)) continue;
      if (!isVec3(joints[a]) || !isVec3(joints[b])) continue;
      segmentRects(toPixel(new THREE.Vector3(...joints[a])), toPixel(new THREE.Vector3(...joints[b])), LIMB_HALF_PX, rects);
    }
    return rects;
  }

  /** 교정 화살표 기둥(시작→끝)을 화살표별로 화면 사각형으로 쪼갠다.
   * 반투명 라벨 띠가 기둥을 가로지르면 화살표가 끊겨 보인다(L6의 '보정 오프셋' 라벨).
   * 배지는 **자기** 화살표 위에 앉아야 하므로 `n`으로 자기 기둥을 골라낼 수 있게 나눠 담는다. */
  function arrowScreenRectsByN() {
    const byN = [];
    for (const entry of labelRegistry) {
      if (entry.kind !== "badge" || !isVec3Like(entry.from) || !isVec3Like(entry.to)) continue;
      byN.push({ n: entry.n, rects: segmentRects(toPixel(entry.from), toPixel(entry.to), ARROW_HALF_PX) });
    }
    return byN;
  }

  function arrowScreenRects() {
    const rects = [];
    for (const arrow of arrowScreenRectsByN()) rects.push(...arrow.rects);
    return rects;
  }

  /** 장치 막대(몸의 추정 관절 → 실측 장치 구, 두 층 다)를 화면 사각형으로 쪼갠다.
   * 검증 라운드에서 옛 관절→구 선의 라벨 글자 **사이로** 바로 그 선이 지나갔다 — 막대도 장애물이다. */
  function thinLineScreenRects() {
    const rects = [];
    for (const bar of deviceBars) segmentRects(toPixel(bar.from), toPixel(bar.to), THIN_LINE_HALF_PX, rects);
    return rects;
  }

  /** 실측 마커 구체가 화면에서 차지하는 사각형. 라벨이 실측 부호를 스치면 안 된다
   * (검증 라운드: L3에서 오프셋 라벨 아래 모서리가 왼손 실측 마커를 약 4px 스쳤다). */
  function measuredMarkerScreenRects() {
    const rects = [];
    for (const center of sphereObstacles) {
      const px = toPixel(center);
      // 구를 두른 "이 순간" 고리까지 피한다 — 고리가 구보다 크다.
      const half = screenDiameter(center, MOMENT_RING_RADIUS) / 2;
      rects.push({ x0: px.x - half, x1: px.x + half, y0: px.y - half, y1: px.y + half });
    }
    return rects;
  }

  /** 넓은 정의: 다리를 뺀 **모든** 관절(골반·척추·머리·어깨·팔·손)과 **모든** 실측/예측 마커를
   * 화면에 투영한 바운딩 박스. 세로 양끝은 흔히 손 관절이나 손 마커다. 좁은 정의는
   * `torsoScreenSpan()`이 따로 낸다. */
  function upperBodyScreen() {
    if (!lastView) return null;
    const joints = lastView.layers?.query?.joints ?? {};
    const legSet = new Set(lastView.fit?.legs?.joints ?? []);
    const points = [];
    for (const [name, value] of Object.entries(joints)) if (!legSet.has(name) && isVec3(value)) points.push(value);
    for (const anchor of lastView.fit?.anchors ?? []) if (isVec3(anchor.measured_marker)) points.push(anchor.measured_marker);
    if (points.length === 0) return null;
    let top = Infinity; let bottom = -Infinity; let left = Infinity; let right = -Infinity;
    for (const value of points) {
      const px = toPixel(new THREE.Vector3(...value));
      top = Math.min(top, px.y); bottom = Math.max(bottom, px.y);
      left = Math.min(left, px.x); right = Math.max(right, px.x);
    }
    const { w, h } = hostSize();
    return {
      topPx: top, bottomPx: bottom, heightPx: bottom - top, widthPx: right - left,
      hostHeightPx: h, hostWidthPx: w, heightFraction: (bottom - top) / h,
    };
  }

  /** 좁은 정의: 머리 **실측** 마커(HMD)와 골반 관절, 이 두 점만의 화면 세로 거리와 캔버스 대비 비율.
   * 합격선 문구('상체(머리 마커~골반) ≥ 캔버스 40%')가 가리키는 바로 그 두 점이고, 기본 뷰는 이
   * 값을 **재서 말한다**(`describeTorsoShortfall`) — 이 값에 맞춰 당기지는 않는다.
   * 맞춤 상자가 머리~바닥을 담으므로 이 값은 자세에만 달려 있고 레벨마다 갈린다(실측: 김경호
   * L1 0.1517 · L3 0.3309 · L6 0.2103, 1875×1000과 840×1050 동일 — 창 크기와 무관하다).
   * 갈리는 것은 프레이밍 규칙이 아니라 자세다: 쭈그린 패스일수록 상체가 화면에서 크다.
   * 넓은 정의(`upperBodyScreen`)는 손 관절·손 마커가 양끝일 때가 많아 훨씬 큰 값이 나온다 —
   * 그 값에 이 문구의 이름을 붙이면 출처와 다른 라벨이 된다. 두 값을 따로 내보내는 이유다. */
  function torsoScreenSpan() {
    if (!lastView) return null;
    const head = (lastView.fit?.anchors ?? []).find((a) => a.joint === "Head" && isVec3(a.measured_marker));
    const pelvis = lastView.layers?.query?.joints?.Pelvis;
    if (!head || !isVec3(pelvis)) return null;
    const a = toPixel(new THREE.Vector3(...head.measured_marker));
    const b = toPixel(new THREE.Vector3(...pelvis));
    const { h } = hostSize();
    return {
      topPx: Math.min(a.y, b.y), bottomPx: Math.max(a.y, b.y),
      heightPx: Math.abs(b.y - a.y), hostHeightPx: h, heightFraction: Math.abs(b.y - a.y) / h,
      ends: ["marker:Head(HMD)", "joint:Pelvis"],
    };
  }

  /** 추정 머리가 채운 구가 아니라 윤곽선인지(범례 진실성). */
  function estimateHeadIsOutline() {
    let found = null;
    layerGroups.query.traverse((child) => {
      if (child.userData.kind === "estimate-head") found = child.material?.wireframe === true;
    });
    return found;
  }

  /** 계산된 층(교정 후·예상)의 표식이 채운 구가 아니라 윤곽선인지(범례 진실성).
   * 범례 ②는 "실측은 … 층 색으로 채운 구체"라고 선언한다 — 계산값이 같은 부호를 쓰면 범례가 거짓이다.
   * 표식이 하나도 없으면 null(판정 대상 없음). */
  function computedLayerIsOutline() {
    let markers = 0;
    let outlines = 0;
    layerGroups.corrected.traverse((child) => {
      if (child.userData.kind !== "computed-layer-marker") return;
      markers += 1;
      if (child.material?.wireframe === true) outlines += 1;
    });
    return markers === 0 ? null : { markers, outlines, allOutline: markers === outlines };
  }

  /** 스켈레톤 가슴(없으면 골반)의 화면 위치. 스켈레톤이 없으면 null. */
  function bodyScreenCenter() {
    const joints = lastView?.layers?.query?.joints ?? {};
    const anchor = isVec3(joints.Chest) ? joints.Chest : isVec3(joints.Pelvis) ? joints.Pelvis : null;
    return anchor ? toPixel(new THREE.Vector3(...anchor)) : null;
  }

  /** 이름표 사각형의 가장 가까운 가장자리에서 구 중심 쪽으로 가는 끈(px). 구 고리 안까지는 들어가지 않는다. */
  function placeLeader(leader, r, target) {
    const cx = clamp(target.x, r.x0, r.x1);
    const cy = clamp(target.y, r.y0, r.y1);
    const dx = target.x - cx;
    const dy = target.y - cy;
    const len = Math.hypot(dx, dy) - LEADER_STOP_PX;
    if (len < LEADER_MIN_PX) {
      leader.style.display = "none";
      return;
    }
    leader.style.display = "";
    leader.style.left = `${cx}px`;
    leader.style.top = `${cy}px`;
    leader.style.width = `${len}px`;
    leader.style.transform = `rotate(${Math.atan2(dy, dx)}rad)`;
  }

  function layoutLabels() {
    // 1) 글자 라벨(miss 거리 등)을 먼저 놓고 그 사각형을 장애물로 모은다.
    // 2) 번호 배지는 앞 배지(26px 안)와 글자 라벨 사각형을 피해, 자기 화살표의 화면
    //    방향으로 26px씩 민다. 배지가 거리 라벨을 덮으면 숫자를 읽을 수 없다.
    const obstacles = [];
    const visible = (entry) => {
      const projected = entry.position.clone().project(camera);
      const shown = projected.z <= 1 && projected.z >= -1;
      entry.el.style.display = shown ? "" : "none";
      return shown;
    };
    let labelOverlaps = 0;
    // 몸통 줄기는 장애물이다 — 글자 라벨이 몸통 위에 앉으면 뼈도 글자도 못 읽는다(보정 오프셋 라벨).
    const torsoRects = torsoScreenRects();
    // 어깨·팔·손 마디도 장애물이다(줄기만 넣었더니 거리 라벨이 팔뼈를 덮었다).
    const limbRects = limbScreenRects();
    // 교정 화살표 기둥도 장애물이다 — 반투명 라벨 띠가 기둥을 가로지르면 화살표가 끊겨 보인다.
    const arrowsByN = arrowScreenRectsByN();
    const arrowRects = [];
    for (const arrow of arrowsByN) arrowRects.push(...arrow.rects);
    // 보정 오프셋 선·어긋남 점선, 그리고 실측 마커 구체도 장애물이다.
    const thinLineRects = thinLineScreenRects();
    const markerRects = measuredMarkerScreenRects();
    // 글자 라벨은 몸통 화면 위치에서 **바깥쪽**으로 민다 — 몸 한가운데에 거리·오프셋 라벨이 쌓여 팔·몸통을
    // 가렸다(2라운드). 기준점은 스켈레톤 가슴(없으면 라벨 자리 그대로).
    const body = bodyScreenCenter();
    let torsoLabelOverlaps = 0;
    let limbLabelOverlaps = 0;
    let arrowLabelOverlaps = 0;
    let thinLineLabelOverlaps = 0;
    let markerLabelOverlaps = 0;
    let labelsClamped = 0;
    for (const entry of labelRegistry) {
      if (entry.kind === "badge") continue;
      if (entry.el.hidden || !visible(entry)) {
        if (entry.leader) entry.leader.style.display = "none";
        continue;
      }
      const px = toPixel(entry.position);
      const w = entry.el.offsetWidth;
      const h = entry.el.offsetHeight;
      // .view3d-label의 transform: translate(-50%, -140%) — 사각형은 (x, y) 기준 위쪽에 놓인다.
      let ox = 0;
      let oy = 0;
      if (body) {
        const dx = px.x - body.x;
        const dy = px.y - body.y;
        const len = Math.hypot(dx, dy);
        [ox, oy] = len < 1e-6 ? [1, 0] : [dx / len, dy / len];
      }
      const rect = (xx, yy) => ({ x0: xx - w / 2, x1: xx + w / 2, y0: yy - 1.4 * h, y1: yy - 0.4 * h });
      const hits = (r, list) => list.some((o) => r.x0 < o.x1 && r.x1 > o.x0 && r.y0 < o.y1 && r.y1 > o.y0);
      const area = (r, list) => list.reduce((sum, o) => {
        const dx = Math.min(r.x1, o.x1) - Math.max(r.x0, o.x0);
        const dy = Math.min(r.y1, o.y1) - Math.max(r.y0, o.y0);
        return sum + (dx > 0 && dy > 0 ? dx * dy : 0);
      }, 0);
      const clashes = (r) => hits(r, obstacles) || hits(r, torsoRects) || hits(r, limbRects)
        || hits(r, arrowRects) || hits(r, thinLineRects) || hits(r, markerRects);
      // 가릴 때의 손해가 다르다: 몸통·팔을 덮으면 뼈도 글자도 못 읽고, 화살표 기둥을 덮으면 교정 화살표가
      // 끊겨 보이며, 실측 마커를 덮으면 범례가 가리키는 바로 그 부호가 사라진다. 가는 오프셋 선과
      // 앞 라벨끼리 겹치는 것은 그보다 가볍다.
      const cost = (r) => 4 * area(r, torsoRects) + 3 * area(r, limbRects) + 2 * area(r, arrowRects)
        + 2 * area(r, markerRects) + area(r, thinLineRects) + area(r, obstacles);
      // 캔버스 밖으로 밀려나면 글자가 잘린다 — 후보 자리마다 캔버스 안으로 되돌린 뒤에 점수를 매긴다.
      const { w: hostW, h: hostH } = hostSize();
      const fit = (xx, yy) => ({
        x: clamp(xx, w / 2 + 2, hostW - w / 2 - 2),
        y: clamp(yy, 1.4 * h + 2, hostH - 0.4 * h - 2),
      });
      // 한 방향으로만 밀면 화살표 네 개가 모인 쪽(L6 오른손)에서 빠져나갈 자리를 못 찾는다. 바깥쪽
      // 방향을 첫 후보로 두고 ±30°씩 틀어 가며, 거리도 늘려 가며 훑는다. 전부 겹치면 손해가 가장 작은
      // 자리를 쓴다(겹침 수는 아래에서 그대로 센다).
      const turns = [0, 1, -1, 2, -2, 3, -3, 4, -4, 5, -5, 6];
      const radii = [LABEL_PUSH_PX, LABEL_PUSH_PX + 26, LABEL_PUSH_PX + 58, LABEL_PUSH_PX + 96, LABEL_PUSH_PX + 140];
      let best = null;
      for (const radius of radii) {
        for (const turn of turns) {
          const angle = (turn * Math.PI) / 6;
          const cos = Math.cos(angle);
          const sin = Math.sin(angle);
          const dx = ox * cos - oy * sin;
          const dy = ox * sin + oy * cos;
          const ux = px.x + dx * radius;
          const uy = px.y + dy * radius;
          const spot = fit(ux, uy);
          const score = cost(rect(spot.x, spot.y));
          const clamped = Math.abs(spot.x - ux) > 0.5 || Math.abs(spot.y - uy) > 0.5;
          if (best === null || score < best.score) best = { ...spot, score, clamped };
          if (score === 0) break;
        }
        if (best !== null && best.score === 0) break;
      }
      let x = best.x;
      let y = best.y;
      if (best.clamped) labelsClamped += 1;
      if (clashes(rect(x, y))) labelOverlaps += 1;
      if (hits(rect(x, y), torsoRects)) torsoLabelOverlaps += 1;
      if (hits(rect(x, y), limbRects)) limbLabelOverlaps += 1;
      if (hits(rect(x, y), arrowRects)) arrowLabelOverlaps += 1;
      if (hits(rect(x, y), thinLineRects)) thinLineLabelOverlaps += 1;
      if (hits(rect(x, y), markerRects)) markerLabelOverlaps += 1;
      entry.el.style.left = `${x}px`;
      entry.el.style.top = `${y}px`;
      obstacles.push(rect(x, y));
      if (entry.leader) placeLeader(entry.leader, rect(x, y), px);
    }
    // 이름이 재는 것과 정확히 같아야 한다(검증 라운드 지적): `torsoLabelOverlaps`는 줄기 셋만,
    // `boneLabelOverlaps`는 다리를 뺀 **모든** 뼈다. 배지는 글자 라벨과 따로 센다(아래).
    debugLive.labelOverlaps = labelOverlaps;
    debugLive.torsoLabelOverlaps = torsoLabelOverlaps;
    debugLive.limbLabelOverlaps = limbLabelOverlaps;
    debugLive.boneLabelOverlaps = torsoLabelOverlaps + limbLabelOverlaps;
    debugLive.arrowLabelOverlaps = arrowLabelOverlaps;
    debugLive.offsetLineLabelOverlaps = thinLineLabelOverlaps;
    debugLive.measuredMarkerLabelOverlaps = markerLabelOverlaps;
    debugLive.labelsClamped = labelsClamped;
    debugLive.upperBody = upperBodyScreen();
    debugLive.torsoSpan = torsoScreenSpan();
    debugLive.estimateHeadIsOutline = estimateHeadIsOutline();
    debugLive.correctedLayerIsOutline = computedLayerIsOutline();
    // 검증자가 **자기 손으로** 다시 투영할 수 있게 카메라 행렬과 캔버스 크기를 그대로 내보낸다.
    // (구현이 낸 사각형·계수기를 믿지 않고 서버 페이로드에서 픽셀을 직접 얻는 길.)
    camera.updateMatrixWorld();
    debugLive.camera = {
      projectionMatrix: camera.projectionMatrix.toArray(),
      matrixWorldInverse: camera.matrixWorldInverse.toArray(),
      host: hostSize(),
    };
    // 번호 배지. 옛 배치는 (1) 배지끼리를 중심거리 26px로만 봐서 대각선 모서리 겹침을 놓쳤고
    // (2) 회피 탐색에서 **남의 화살표 기둥**을 아예 안 봤다(배지 ④가 화살표 ①·②의 기둥에 앉았다).
    // 이제 사각형 겹침으로 보고, 자기 기둥만 빼고 남의 기둥도 장애물로 넣는다.
    const badgeRect = (x, y) => ({ x0: x - BADGE_HALF_PX, x1: x + BADGE_HALF_PX, y0: y - BADGE_HALF_PX, y1: y + BADGE_HALF_PX });
    const overlapArea = (r, list) => list.reduce((sum, o) => {
      const dx = Math.min(r.x1, o.x1) - Math.max(r.x0, o.x0);
      const dy = Math.min(r.y1, o.y1) - Math.max(r.y0, o.y0);
      return sum + (dx > 0 && dy > 0 ? dx * dy : 0);
    }, 0);
    const overlaps = (r, list) => list.some((o) => r.x0 < o.x1 && r.x1 > o.x0 && r.y0 < o.y1 && r.y1 > o.y0);
    const { w: badgeHostW, h: badgeHostH } = hostSize();
    const placedBadges = [];
    const placedBadgeRects = [];
    let obstacleOverlaps = 0;
    let badgeBadgeOverlaps = 0;
    let badgeArrowOverlaps = 0;
    for (const entry of labelRegistry) {
      if (entry.kind !== "badge" || !visible(entry)) continue;
      const px = toPixel(entry.position);
      const a = toPixel(entry.from);
      const b = toPixel(entry.to);
      let dx = b.x - a.x;
      let dy = b.y - a.y;
      const len = Math.hypot(dx, dy);
      if (len < 1e-6) { dx = 0; dy = -1; } else { dx /= len; dy /= len; }
      // 자기 기둥은 배지가 앉아야 할 자리다 — 장애물에서 뺀다.
      const otherArrowRects = [];
      for (const arrow of arrowsByN) if (arrow.n !== entry.n) otherArrowRects.push(...arrow.rects);
      const badgeScore = (r) => 8 * overlapArea(r, placedBadgeRects) + 4 * overlapArea(r, otherArrowRects) + overlapArea(r, obstacles);
      // 기둥 방향을 축으로 삼아 앞뒤·좌우로만 훑으면(옛 along/side 격자) 찾는 자리가 기둥을 낀 가는 띠
      // 안으로 갇힌다 — 상체 확대처럼 라벨이 촘촘한 화면에서는 그 띠 안에 0점 자리가 없을 수 있다(검증
      // 라운드 지적: 김경호 L2 1000390_1 #2, 1875×1000, 상체 확대에서 배지 ②가 "6.5cm" 라벨과 226px²
      // 겹쳤다). 라벨 배치와 같은 반경×회전 극좌표 훑기로 바꿔 기둥 주변 360°를 본다. radius=1×BADGE_MIN_PX,
      // turn=0(기둥 방향 그대로)이 첫 후보라 이미 0점을 찾던 자리는 그대로 유지된다.
      const turns = [0, 1, -1, 2, -2, 3, -3, 4, -4, 5, -5, 6];
      const radii = [1, 2, 3, 4, 5, 6, 7, 8].map((n) => n * BADGE_MIN_PX);
      let best = null;
      for (const radius of radii) {
        for (const turn of turns) {
          const angle = (turn * Math.PI) / 6;
          const cos = Math.cos(angle);
          const sin = Math.sin(angle);
          const rdx = dx * cos - dy * sin;
          const rdy = dx * sin + dy * cos;
          const ux = px.x + rdx * radius;
          const uy = px.y + rdy * radius;
          const x = clamp(ux, BADGE_HALF_PX + 1, badgeHostW - BADGE_HALF_PX - 1);
          const y = clamp(uy, BADGE_HALF_PX + 1, badgeHostH - BADGE_HALF_PX - 1);
          const value = badgeScore(badgeRect(x, y));
          if (best === null || value < best.value) best = { x, y, value };
          if (value === 0) break;
        }
        if (best !== null && best.value === 0) break;
      }
      const rect = badgeRect(best.x, best.y);
      if (overlaps(rect, obstacles)) obstacleOverlaps += 1;
      if (overlaps(rect, placedBadgeRects)) badgeBadgeOverlaps += 1;
      if (overlaps(rect, otherArrowRects)) badgeArrowOverlaps += 1;
      placedBadges.push({ n: entry.n, x: best.x, y: best.y });
      placedBadgeRects.push(rect);
      entry.el.style.left = `${best.x}px`;
      entry.el.style.top = `${best.y}px`;
    }
    // 이름이 재는 것과 같아야 한다: 아래 셋은 **배지** 전용 계수기다(글자 라벨은 위에서 따로 센다).
    debugLive.badgeObstacleOverlaps = obstacleOverlaps;
    debugLive.badgeBadgeOverlaps = badgeBadgeOverlaps;
    debugLive.badgeArrowOverlaps = badgeArrowOverlaps;
    debugLive.badgeCenters = placedBadges.map((p) => ({ n: p.n, x: Math.round(p.x * 10) / 10, y: Math.round(p.y * 10) / 10 }));
    debugLive.arrowConeScreenDiameters = coneRegistry.map((c) => screenDiameter(c.center, c.radius));
    const diameters = {};
    for (const m of markerRegistry) diameters[m.point] = screenDiameter(m.center, MEASURED_MARKER_RADIUS);
    debugLive.markerScreenDiameters = diameters;
    debugLive.handMarkerScreenDiameter = typeof diameters.right === "number" ? diameters.right : null;
  }

  function frame() {
    requestAnimationFrame(frame);
    renderer.render(scene, camera);
    layoutLabels();
  }
  requestAnimationFrame(frame);

  /** 층 궤적 하나를 그린다(S3 — 대상·기준 **둘 다**, 차분한 한 색).
   *
   * `measured`는 이 층의 점이 관측인지 계산인지다 — 범례가 "실측 = 층 색으로 채운 구체"라고
   * 선언했으므로, 계산된 층(교정 후·예상)이 같은 부호를 쓰면 범례가 거짓이 된다. 질의·기준 궤적은
   * 같은 세 강체의 실측이라 채운 구를 쓰고, 교정 후 층은 **끊어 그린 선 + 속 빈(윤곽선) 표식**으로
   * 그린다(추정 머리를 윤곽선 구로 바꾼 것과 같은 원칙).
   *
   * 진하기는 더 이상 채점 가중치에서 나오지 않는다 — 궤적은 한 색이고, 진해지는 것은 관절뿐이다. */
  function buildLayerGroup(layerData, layerColor, labelPointNames, measured) {
    const group = new THREE.Group();
    for (const name of Object.keys(layerData ?? {})) {
      // 착탄 격자(`hit`)는 호길이 리샘플이라 모서리를 가로지르면 벽에서 뜬다 — 선으로 그리지 않는다.
      // 분사 스트로크는 서버가 자른 연결 구간(`view.stroke`)을 `buildStrokeGroup`이 따로 그린다.
      if (name === "joints" || name === "bones" || name === "hit") continue;
      const points = (layerData[name] ?? [])
        .filter((p) => isVec3(p))
        .map((p) => new THREE.Vector3(p[0], p[1], p[2]));
      if (points.length === 0) continue;

      const lineMaterial = measured === false
        ? new THREE.LineDashedMaterial({
          color: layerColor,
          transparent: true,
          opacity: TRACK_OPACITY,
          dashSize: COMPUTED_DASH,
          gapSize: COMPUTED_GAP,
        })
        : new THREE.LineBasicMaterial({
          color: layerColor,
          transparent: true,
          opacity: TRACK_LINE_OPACITY,
        });
      const line = new THREE.Line(new THREE.BufferGeometry().setFromPoints(points), lineMaterial);
      if (measured === false) line.computeLineDistances();
      line.userData.kind = measured === false ? "computed-layer-line" : "measured-layer-line";
      line.userData.point = name;
      group.add(line);
      // 실측 궤적은 가는 관으로 한 번 더 — 1px 선은 벽 메시 위에서 안 읽혔다. 계산된 층(끊은 선)은 그대로 둔다.
      if (measured !== false) {
        const tubeMaterial = new THREE.MeshBasicMaterial({
          color: layerColor, transparent: true, opacity: TRACK_TUBE_OPACITY, depthWrite: false,
        });
        for (let i = 0; i + 1 < points.length; i += 1) {
          const piece = boneCylinder(points[i], points[i + 1], TRACK_TUBE_RADIUS, tubeMaterial, "measured-layer-tube", name);
          if (piece) group.add(piece);
        }
      }

      // 표식은 궤적의 **끝점 하나**다. 옛 코드는 20개 표본마다 구체를 찍어 한 층에 60개, 질의·기준
      // 둘이면 120개가 화면을 덮었다 — 사용자가 "부분적으로 뿌려진 건 다 없앴으면 함"이라고 한 바로
      // 그 알갱이다. 줄 자체는 그대로라 "연결된 하나의 선"이 남고, 범례가 선언한 부호(층 색으로 채운
      // 구체 = 실측, 속 빈 표식 = 계산)는 끝점 표식이 그대로 진다.
      const markerGeometry = new THREE.SphereGeometry(TRACK_MARKER_RADIUS * (measured === false ? 1.6 : 1), 10, 6);
      const markerMaterial = new THREE.MeshBasicMaterial({
        color: layerColor,
        transparent: true,
        opacity: TRACK_OPACITY + 0.2,
        wireframe: measured === false,
      });
      const marker = new THREE.Mesh(markerGeometry, markerMaterial);
      marker.position.copy(points[points.length - 1]);
      marker.userData.kind = measured === false ? "computed-layer-marker" : "measured-layer-marker";
      group.add(marker);
      if (labelPointNames) {
        registerLabel(points[points.length - 1], POINT_LABELS[name] ?? name, "view3d-label-point");
      }
    }
    return group;
  }

  /** 벽 위 분사 스트로크. 서버가 낸 연결 구간(`view.stroke.runs`)을 그대로 잇는다.
   *
   * 선(THREE.Line)은 WebGL에서 굵기를 키울 수 없어 1px로만 그려졌고, 반투명 벽 위에서 거의 보이지
   * 않았다(옛 값 opacity 0.45). 여기서는 뼈와 같은 방식으로 **실린더**를 이어 굵기를 준다 —
   * 벽에 붙은 한 줄로 읽힌다. 점을 옮기거나 만들지 않는다: 정점은 서버가 준 원본 착탄점 그대로다.
   * 표식(구체)은 찍지 않는다 — 사용자가 없애 달라고 한 "낱개 점"이 바로 그것이다. */
  function buildStrokeGroup(strokeBlock, color) {
    const group = new THREE.Group();
    const runs = Array.isArray(strokeBlock?.runs) ? strokeBlock.runs : [];
    const material = new THREE.MeshBasicMaterial({ color, transparent: true, opacity: 0.95 });
    let segments = 0;
    let drawnRuns = 0;
    for (const run of runs) {
      const points = (Array.isArray(run) ? run : []).filter(isVec3).map((p) => new THREE.Vector3(...p));
      if (points.length < 2) continue;
      const runGroup = new THREE.Group();
      runGroup.userData.kind = "surface-stroke-run";
      runGroup.userData.points = points.length;
      for (let i = 0; i + 1 < points.length; i += 1) {
        const piece = boneCylinder(points[i], points[i + 1], STROKE_RADIUS, material, "surface-stroke", "");
        if (!piece) continue;
        runGroup.add(piece);
        segments += 1;
      }
      if (runGroup.children.length === 0) continue;
      group.add(runGroup);
      drawnRuns += 1;
    }
    return { group, runs: drawnRuns, segments };
  }

  /** 실린더 하나. 뼈·스트로크·자리표시가 모두 이 하나를 쓴다. */
  function boneCylinder(start, end, radius, material, kind, name) {
    const dir = new THREE.Vector3().subVectors(end, start);
    const length = dir.length();
    if (length < 1e-6) return null;
    dir.normalize();
    const mesh = new THREE.Mesh(new THREE.CylinderGeometry(radius, radius, length, 8), material);
    mesh.quaternion.setFromUnitVectors(new THREE.Vector3(0, 1, 0), dir);
    mesh.position.copy(start).addScaledVector(dir, length / 2);
    mesh.userData.kind = kind;
    mesh.userData.name = name;
    mesh.userData.length = length;
    return mesh;
  }

  /** 추정 스켈레톤 하나. **관절마다 진하기가 다르다** — 서버가 잰 벌어짐(`view.deviation`)이 근거다.
   *
   * 뼈 색은 양 끝 관절 벌어짐의 평균이다(뼈 하나가 두 관절 사이를 잇는 물건이므로). 관절 표식은
   * 속 빈(윤곽선) 구다 — 범례가 "채운 구체 = 실측"이라고 선언했으니 추정이 그 부호를 쓰면 안 된다.
   *
   * 다리 관절 목록도, 저신뢰 구간(오른 전완·손) 이름도 **서버가** 준다 — 화면에서 부위 이름으로
   * 규칙을 지어내지 않는다. */
  function buildSkeletonGroup(joints, bones, faded, legJoints, headDiameterM, style) {
    const group = new THREE.Group();
    const legs = new THREE.Group();
    if (!joints || !bones || Object.keys(joints).length === 0) return { group, legs };
    const layerColor = style?.layerColor ?? LAYER_COLOR.query;
    const deviations = style?.deviations ?? {};
    const maxDeviation = style?.maxDeviation ?? null;
    const legSet = new Set(Array.isArray(legJoints) ? legJoints : []);
    const lowJoints = new Set(Array.isArray(style?.lowConfidenceJoints) ? style.lowConfidenceJoints : []);
    const lowBones = new Set(
      (Array.isArray(style?.lowConfidenceBones) ? style.lowConfidenceBones : []).map((pair) => pair.join(">")),
    );
    const base = faded ? BONE_OPACITY_FADED : BONE_OPACITY;
    // 해부학 범위 밖 IK 다리는 한 단계 더 흐리다(B). 판정은 **서버**가 준다(`fit.legs_within_range`) —
    // 화면이 각도를 다시 재서 정하지 않는다.
    const legFactor = style?.legsOutOfRange === true
      ? LEG_OPACITY_FACTOR * LEG_OUT_OF_RANGE_OPACITY_FACTOR
      : LEG_OPACITY_FACTOR;
    const devOf = (name) => deviationStyle(deviations[name], maxDeviation, layerColor);
    const materials = [];
    const material = (color, opacity) => {
      const made = new THREE.MeshBasicMaterial({ color, transparent: true, opacity, depthWrite: false });
      materials.push(made);
      return made;
    };
    for (const [parent, child] of bones) {
      const a = joints[parent];
      const b = joints[child];
      if (!isVec3(a) || !isVec3(b)) continue;
      const isLeg = legSet.has(child);
      // 저신뢰 구간(S5)은 더 흐리게. 아바타 손 IK와 트래킹된 건이 원본에서 이미 어긋나 있어
      // 이 구간의 자리를 믿을 수 없다 — 서버 고지가 그 이유를 글자로 말한다.
      const low = lowBones.has(`${parent}>${child}`);
      const dev = devOf(child);
      let opacity = base * dev.opacity;
      if (isLeg) opacity *= legFactor;
      if (low) opacity *= LOW_CONFIDENCE_OPACITY_FACTOR;
      const bone = boneCylinder(
        new THREE.Vector3(...a), new THREE.Vector3(...b),
        BONE_RADIUS, material(dev.color, opacity), "bone", `${parent}>${child}`,
      );
      if (!bone) continue;
      bone.userData.leg = isLeg;
      bone.userData.lowConfidence = low;
      bone.userData.deviation = dev.t;
      (isLeg ? legs : group).add(bone);
    }
    // 관절 표식: 속 빈 구. 벌어진 관절일수록 크고 진하다 — 한눈에 "여기가 다르다"로 읽히게.
    for (const [name, position] of Object.entries(joints)) {
      if (!isVec3(position)) continue;
      const isLeg = legSet.has(name);
      const low = lowJoints.has(name);
      const dev = devOf(name);
      let opacity = base * dev.opacity;
      if (isLeg) opacity *= legFactor;
      if (low) opacity *= LOW_CONFIDENCE_OPACITY_FACTOR;
      const radius = BONE_RADIUS * (1.6 + 1.4 * dev.t);
      const marker = new THREE.Mesh(
        new THREE.SphereGeometry(radius, 10, 8),
        new THREE.MeshBasicMaterial({
          color: dev.color, transparent: true, opacity, depthWrite: false, wireframe: true,
        }),
      );
      marker.position.set(position[0], position[1], position[2]);
      marker.userData.kind = "estimate-joint";
      marker.userData.name = name;
      marker.userData.deviation = dev.t;
      (isLeg ? legs : group).add(marker);
    }
    if (isVec3(joints.Head)) {
      // 추정 머리는 **윤곽선**이다. 채운 구로 그리면 화면에서 가장 큰 구가 되고, 범례가 층 색으로
      // 채운 구체를 실측이라 선언하는 것과 부호가 겹쳐 읽힌다(service.LEGEND_NOTES).
      // 지름은 서버 값(`fit.head_diameter_m`)을 쓴다 — 이 리그의 골반→머리 길이 × BVH 실측 비례로
      // 나온 값이라 고정 상수보다 이 몸에 맞는다. 못 받으면 폴백 반지름을 쓴다.
      const radius = typeof headDiameterM === "number" && headDiameterM > 0 ? headDiameterM / 2 : HEAD_SPHERE_RADIUS_FALLBACK;
      const dev = devOf("Head");
      const headMaterial = new THREE.MeshBasicMaterial({
        color: dev.color, transparent: true, opacity: base * dev.opacity * 0.6, depthWrite: false, wireframe: true,
      });
      const head = new THREE.Mesh(new THREE.SphereGeometry(radius, 12, 8), headMaterial);
      head.position.set(...joints.Head);
      head.userData.kind = "estimate-head";
      group.add(head);
    }
    group.add(legs);
    return { group, legs };
  }

  /** 이 층의 자리표시 다리 관절(월드). 서버가 준 것만 쓴다 — 화면이 다리를 지어내지 않는다. */
  function placeholderJointsOf(fitBlock) {
    const joints = fitBlock?.legs?.placeholder_joints;
    return joints && typeof joints === "object" ? joints : {};
  }

  /** 자리표시 다리 하나(**점선**). 측정이 아니다(B).
   *
   * 서버가 세운 관절(`fit.legs.placeholder_joints`)을 서버가 준 짝(`placeholder_bones`)으로 잇기만
   * 한다 — 어느 관절이 어디에 붙는지를 화면이 정하지 않는다. IK 다리의 실선 실린더도, 범례가 실측에
   * 준 부호(층 색으로 채운 구체)도 쓰지 않는다: 점선인 것이 범례의 "점선 다리 = 자리표시(측정 아님)"와
   * 짝이고, 그 짝이 어긋나면 화면이 재지 않은 것을 잰 것처럼 보이게 된다. */
  function buildPlaceholderLegGroup(fitBlock, layerColor, faded) {
    const group = new THREE.Group();
    const joints = placeholderJointsOf(fitBlock);
    const bones = fitBlock?.legs?.placeholder_bones;
    if (!Array.isArray(bones)) return group;
    const opacity = (faded ? BONE_OPACITY_FADED : BONE_OPACITY) * LEG_OPACITY_FACTOR;
    for (const pair of bones) {
      const [parent, child] = Array.isArray(pair) ? pair : [];
      const a = joints[parent];
      const b = joints[child];
      if (!isVec3(a) || !isVec3(b)) continue;
      const line = new THREE.Line(
        new THREE.BufferGeometry().setFromPoints([new THREE.Vector3(...a), new THREE.Vector3(...b)]),
        new THREE.LineDashedMaterial({
          color: layerColor,
          transparent: true,
          opacity,
          dashSize: COMPUTED_DASH,
          gapSize: COMPUTED_GAP,
        }),
      );
      line.computeLineDistances();
      line.userData.kind = "placeholder-bone";
      line.userData.name = `${parent}>${child}`;
      group.add(line);
    }
    return group;
  }

  /** 자리표시 발과 바닥면 사이의 **간격** 표시(사용자 결정: "리그 길이 지키고 발이 뜨게").
   *
   * 엉덩이가 리그 다리보다 높은 패스에서는 곧게 내린 자리표시 발이 바닥면에 닿지 않는다. 예전에는
   * 다리를 그 틈만큼 늘려 세웠는데, 늘린 다리는 없는 길이를 주장한다. 이제 뜬 채로 두고 **그 틈을
   * 점선과 글자로** 말한다 — 두 끝점도 글자도 서버가 준다(`fit.legs.placeholder_floor_gaps`).
   * 화면이 길이를 다시 재거나 cm로 다시 반올림하면 같은 틈이 두 숫자로 보인다.
   *
   * 선은 자리표시 뼈와 **같은 점선**이고 글자는 "간격"이라고 말한다 — 실측이 쓰는 부호(층 색으로
   * 채운 구체·실선)를 쓰지 않는다. 재지 않은 것을 잰 것처럼 보이게 하지 않는다. */
  function buildPlaceholderFloorGaps(fitBlock, layerColor, faded, layer) {
    const group = new THREE.Group();
    const guides = fitBlock?.legs?.placeholder_floor_gaps;
    if (!Array.isArray(guides)) return group;
    const opacity = (faded ? BONE_OPACITY_FADED : BONE_OPACITY) * LEG_OPACITY_FACTOR;
    for (const guide of guides) {
      const from = guide?.from;
      const to = guide?.to;
      if (!isVec3(from) || !isVec3(to)) continue;
      const start = new THREE.Vector3(...from);
      const end = new THREE.Vector3(...to);
      const line = new THREE.Line(
        new THREE.BufferGeometry().setFromPoints([start, end]),
        new THREE.LineDashedMaterial({
          color: layerColor,
          transparent: true,
          opacity,
          dashSize: COMPUTED_DASH,
          gapSize: COMPUTED_GAP,
        }),
      );
      line.computeLineDistances();
      line.userData.kind = "placeholder-floor-gap";
      line.userData.name = typeof guide.joint === "string" ? guide.joint : "";
      group.add(line);
      if (typeof guide.label === "string" && guide.label.length > 0) {
        const entry = registerLabel(start.clone().lerp(end, 0.5), guide.label, "view3d-label-gap");
        entry.layer = layer;
        gapLabels.push(entry);
      }
    }
    return group;
  }

  /** 간격 보조선이 실제로 씬에 있고 보이는가(검증용). 말이 아니라 이 값으로 잰다. */
  function floorGapDebug(group) {
    const names = [];
    if (group) {
      group.traverse((child) => {
        if (child.userData.kind === "placeholder-floor-gap") names.push(child.userData.name);
      });
    }
    return { lines: names.length, visible: group ? group.visible : false, joints: names };
  }

  /** 대표 프레임 실측 마커(채운 구체) + 그 자리의 "이 순간" 고리 + 장치 모양(몸 → 장치).
   *
   * 예전에는 여기서 관절→예측 마커 선과 예측→실측 점선을 그리고 "보정 오프셋 0.57m(적합값)"·"12.8cm"
   * 라벨을 달았다. 사용자가 "보정 오프셋은 왜 표시했는지를 모르겠음"이라고 해서 통째로 지웠는데, 그러자
   * 채운 구가 어느 관절의 무엇인지 말해 주는 것이 화면에서 사라졌다("진한 구 형태와 연한 선과 스켈레톤은
   * 왜 겹치는 부분이 없는지 모르겠어"). 문제는 선이 아니라 이름이었다. 이제 선은 **몸(추정 관절) →
   * 장치(실측 구)** 막대로 돌아오고, 이름은 서버가 준 장치 이름·거리 문자열(`anchor.device.label`)이다.
   *
   * 막대는 실측 부호(층 색으로 채운 구체)를 쓰지 않는다 — 한쪽 끝이 추정 관절이다. 층 색을 흰색 쪽으로
   * 씻은 반투명 실린더이고, 범례(서버 `legend_notes`)가 "장치 모양 = 몸(추정)에서 장치(실측)까지"라고 말한다.
   * 고리는 그 층의 궤적(`trackLayer[anchor.point]`)이 구 자리에서 가는 방향을 축으로 삼는다 — 방향은
   * 그림을 세우는 데만 쓰고, 화면에 숫자로 내지 않는다. */
  function buildAnchorGroup(fit, layerColor, register, layer, trackLayer) {
    const group = new THREE.Group();
    const counts = { measuredSpheres: 0, momentRings: 0, deviceShapes: 0 };
    const devices = [];
    if (!fit) return { group, counts, devices };
    const markerMaterial = new THREE.MeshStandardMaterial({ color: layerColor, roughness: 0.5 });
    const markerGeometry = new THREE.SphereGeometry(MEASURED_MARKER_RADIUS, 20, 14);
    const ringMaterial = new THREE.MeshBasicMaterial({ color: layerColor, transparent: true, opacity: 0.95 });
    const deviceMaterial = new THREE.MeshBasicMaterial({
      color: mixColor(layerColor, 0xffffff, DEVICE_WASH), transparent: true, opacity: DEVICE_OPACITY, depthWrite: false,
    });
    for (const anchor of fit.anchors ?? []) {
      if (!isVec3(anchor.measured_marker)) continue;
      const marker = new THREE.Mesh(markerGeometry, markerMaterial);
      marker.position.set(...anchor.measured_marker);
      marker.userData.kind = "measured-marker";
      group.add(marker);
      counts.measuredSpheres += 1;
      if (register) markerRegistry.push({ point: anchor.point, center: marker.position.clone() });
      sphereObstacles.push(marker.position.clone());

      const tangent = trackTangentAt(trackLayer?.[anchor.point], marker.position);
      if (tangent) {
        const ring = new THREE.Mesh(new THREE.TorusGeometry(MOMENT_RING_RADIUS, MOMENT_RING_TUBE, 8, 32), ringMaterial);
        ring.position.copy(marker.position);
        // TorusGeometry의 축은 +Z다 — 그 축을 궤적 방향에 맞춰 고리가 선을 가로지르게 한다.
        ring.quaternion.setFromUnitVectors(new THREE.Vector3(0, 0, 1), tangent.direction);
        ring.userData.kind = "moment-ring";
        ring.userData.point = anchor.point;
        ring.userData.layer = layer;
        ring.userData.trackGapM = tangent.gap;
        group.add(ring);
        counts.momentRings += 1;
      }

      const device = anchor.device;
      if (!device || !isVec3(anchor.joint_position)) continue;
      const joint = new THREE.Vector3(...anchor.joint_position);
      const radius = device.shape === "tool" ? DEVICE_TOOL_RADIUS : DEVICE_LINK_RADIUS;
      // 막대는 구 **표면**에서 멈춘다 — 구 안까지 파고들면 실측 구가 막대에 묻혀 보인다.
      const toJoint = new THREE.Vector3().subVectors(joint, marker.position);
      const span = toJoint.length();
      if (span <= MEASURED_MARKER_RADIUS) continue;
      const surface = marker.position.clone().addScaledVector(toJoint.normalize(), MEASURED_MARKER_RADIUS);
      const bar = boneCylinder(joint, surface, radius, deviceMaterial, "device-shape", anchor.point);
      if (!bar) continue;
      bar.userData.layer = layer;
      bar.userData.shape = device.shape;
      bar.userData.from = [joint.x, joint.y, joint.z];
      bar.userData.to = [...anchor.measured_marker];
      group.add(bar);
      counts.deviceShapes += 1;
      deviceBars.push({ from: joint.clone(), to: marker.position.clone() });
      let label = null;
      if (typeof device.label === "string" && device.label.length > 0) {
        label = registerLabel(marker.position, device.label, "view3d-label-device", { kind: "device", layer, point: anchor.point });
        label.el.dataset.layer = layer;
        // 이름표가 회피 배치로 구에서 멀리 밀리면 어느 구의 이름인지 잃는다 — 이름표에서 구까지 가는 끈을 단다.
        label.leader = document.createElement("div");
        label.leader.className = "view3d-label-leader";
        label.leader.dataset.layer = layer;
        dom.labelsHost.appendChild(label.leader);
        deviceLabels.push(label);
      }
      devices.push({ layer, point: anchor.point, joint: anchor.joint, name: device.name, shape: device.shape, bar, label });
    }
    return { group, counts, devices };
  }

  /** 궤적 폴리라인에서 `at`에 가장 가까운 선분의 진행 방향과 그 선분까지의 거리(m).
   * 고리를 세우는 데만 쓴다 — 화면에 숫자로 내지 않는다(거리 문자열은 서버 몫이다). */
  function trackTangentAt(track, at) {
    const points = (Array.isArray(track) ? track : []).filter(isVec3).map((p) => new THREE.Vector3(...p));
    if (points.length < 2) return null;
    let best = null;
    const segment = new THREE.Line3();
    const closest = new THREE.Vector3();
    for (let i = 0; i + 1 < points.length; i += 1) {
      if (points[i].equals(points[i + 1])) continue;
      segment.set(points[i], points[i + 1]);
      segment.closestPointToPoint(at, true, closest);
      const gap = closest.distanceTo(at);
      if (best === null || gap < best.gap) {
        best = { gap, direction: new THREE.Vector3().subVectors(points[i + 1], points[i]).normalize() };
      }
    }
    return best;
  }

  function buildArrow(from, to, color) {
    const start = new THREE.Vector3(...from);
    const end = new THREE.Vector3(...to);
    const dir = new THREE.Vector3().subVectors(end, start);
    const length = dir.length();
    if (length < 1e-6) return null;
    dir.normalize();

    const headLength = Math.min(0.04, 0.35 * length);
    const shaftLength = Math.max(length - headLength, 0);

    const material = new THREE.MeshStandardMaterial({ color, roughness: 0.45, metalness: 0.05 });
    const quat = new THREE.Quaternion().setFromUnitVectors(new THREE.Vector3(0, 1, 0), dir);

    const group = new THREE.Group();
    if (shaftLength > 1e-6) {
      const shaft = new THREE.Mesh(new THREE.CylinderGeometry(ARROW_SHAFT_RADIUS, ARROW_SHAFT_RADIUS, shaftLength, 10), material);
      shaft.quaternion.copy(quat);
      shaft.position.copy(start).addScaledVector(dir, shaftLength / 2);
      group.add(shaft);
    }
    const cone = new THREE.Mesh(new THREE.ConeGeometry(ARROW_HEAD_RADIUS, headLength, 16), material);
    cone.quaternion.copy(quat);
    cone.position.copy(start).addScaledVector(dir, shaftLength + headLength / 2);
    group.add(cone);
    const coneBase = start.clone().addScaledVector(dir, shaftLength);
    return { group, coneBase };
  }

  function buildMesh(meshBlock) {
    if (!meshBlock || meshBlock.available !== true) return null;
    const vertices = meshBlock.vertices ?? [];
    const triangles = meshBlock.triangles ?? [];
    if (vertices.length === 0 || triangles.length === 0) return null;
    const positions = new Float32Array(vertices.length * 3);
    vertices.forEach((v, i) => {
      positions[i * 3] = v[0];
      positions[i * 3 + 1] = v[1];
      positions[i * 3 + 2] = v[2];
    });
    const indices = new Uint32Array(triangles.length * 3);
    triangles.forEach((t, i) => {
      indices[i * 3] = t[0];
      indices[i * 3 + 1] = t[1];
      indices[i * 3 + 2] = t[2];
    });
    const geometry = new THREE.BufferGeometry();
    geometry.setAttribute("position", new THREE.BufferAttribute(positions, 3));
    geometry.setIndex(new THREE.BufferAttribute(indices, 1));
    geometry.computeVertexNormals();
    // 정렬 변환이 반사(det<0)라 삼각형 감김이 뒤집혀 있다 — 양면으로 그려야 법선 방향과
    // 무관하게 보인다. 반투명 벽이 뒤의 몸·마커를 깊이로 가리지 않게 depthWrite를 끈다.
    const material = new THREE.MeshStandardMaterial({
      color: MESH_COLOR,
      transparent: true,
      opacity: 0.32,
      side: THREE.DoubleSide,
      roughness: 0.9,
      depthWrite: false,
    });
    const group = new THREE.Group();
    const surface = new THREE.Mesh(geometry, material);
    group.add(surface);
    const edges = new THREE.LineSegments(
      new THREE.EdgesGeometry(geometry, 25),
      new THREE.LineBasicMaterial({ color: 0x6b7686, transparent: true, opacity: 0.7 }),
    );
    group.add(edges);
    group.userData.triangleCount = triangles.length;
    return group;
  }

  /** 벽면 줄. 문구는 서버(reason_text·handedness_note)의 것을 그대로 쓴다. */
  function wallNoteText(meshBlock) {
    if (meshBlock === null || meshBlock === undefined) {
      return "이 패스는 레벨 정보가 없어 벽면을 그리지 않습니다.";
    }
    if (meshBlock.available !== true) {
      // 캔버스 위 겹침은 한 줄로 — 세부 사유(reason_detail)는 캔버스 아래 '고지 더 보기'에 둔다.
      return [meshBlock.reason_text ?? "이 레벨은 벽면을 그리지 않습니다.", meshBlock.reason_hint]
        .filter((text) => typeof text === "string" && text.length > 0)
        .join(" ");
    }
    // 벽을 **그리는** 레벨에서 이 슬롯은 아무 말도 하지 않는다(E). 이 상자는 캔버스 위에 겹쳐
    // 뜨는 자리라 "왜 벽이 없는가"만 말해야 하는데, 손대칭 판정 근거는 벽면 상태가 아니라 내부
    // 사유다 — 벽을 그리는 유일한 레벨(L1)에서만 3D 위에 뜨던 것이 정확히 슬롯 용도의 정반대였다.
    // 같은 종류의 내부 사유인 `reason_detail`과 같은 자리(캔버스 아래 '고지 더 보기')로 보낸다.
    return "";
  }

  /** 맞춤 대상 점. pose: 스켈레톤·실측/예측 마커·교정 화살표·대표 프레임 착탄점.
   * 궤적·기준층·교정후층·메시는 뺀다(벽은 자세보다 훨씬 커서 넣으면 자세가 쪼그라든다). */
  function framingPoints(view, mode) {
    const points = [];
    const push = (p) => { if (isVec3(p)) points.push(p); };
    const joints = view.layers?.query?.joints ?? {};
    const hasSkeleton = Object.keys(joints).length > 0;
    // torso("상체 확대"): 맞춤 대상을 **상체만**으로 좁힌다 — 다리·교정 화살표·착탄점을 다 뺀다.
    // 다리 이름은 여기서 판단하지 않고 서버(`fit.legs.joints`)가 준 목록을 쓴다.
    if (mode === "torso") {
      for (const p of upperBodyPoints(view)) push(p);
      return points;
    }
    if (hasSkeleton) {
      // **그려지는 다리는 언제나 맞춤 대상이다**(F). 예전에는 이 한 줄이 `legsShown()`을 통해 서버
      // 해부학 판정을 따라 갈렸다: 같은 프레이밍 코드가 레벨마다 다른 점 집합을 받아, L1은 다리를
      // 넣어 상체가 캔버스의 0.240, L3·L6은 다리를 빼 0.554·0.369가 됐다. 프레이밍 규칙이 아니라
      // **입력 점 집합**이 레벨마다 달랐던 것이다. 자리표시 다리는 모든 레벨에 있으므로, 그리는 쪽
      // (IK 또는 자리표시)의 발을 늘 넣으면 맞춤 상자가 레벨 불문 "머리~바닥"이 되고 비율이 자세에만
      // 의존한다. 층마다 자기 판정을 쓴다 — 두 층의 IK 판정은 자주 갈린다.
      const legNames = new Set(legJointNames(view));
      const queryExcluded = legsShown() ? new Set() : legNames;
      for (const [name, p] of Object.entries(joints)) if (!queryExcluded.has(name)) push(p);
      // **기준 몸도 맞춤 대상이다**(S1). 두 자세를 겹쳐 보여 주는 것이 이 화면의 목적이므로, 기준이
      // 화면 밖으로 잘리면 그림이 성립하지 않는다.
      const referenceJoints = view.layers?.reference?.joints ?? {};
      const referenceExcluded = referenceLegsShown() ? new Set() : legNames;
      for (const [name, p] of Object.entries(referenceJoints)) if (!referenceExcluded.has(name)) push(p);
      // IK를 끈 층은 **자리표시 다리가 그려진다** — 그 점이 맞춤 대상이다.
      if (!legsShown()) for (const p of Object.values(placeholderJointsOf(view.fit))) push(p);
      if (!referenceLegsShown()) for (const p of Object.values(placeholderJointsOf(view.reference_fit))) push(p);
      for (const anchor of view.fit?.anchors ?? []) {
        push(anchor.measured_marker);
        push(anchor.predicted_marker);
      }
      for (const anchor of view.reference_fit?.anchors ?? []) push(anchor.measured_marker);
    } else {
      for (const name of ["head", "right", "left"]) for (const p of view.layers?.query?.[name] ?? []) push(p);
    }
    for (const correction of view.corrections ?? []) {
      if (isVec3(correction.from_frame) && isVec3(correction.to_frame)) {
        push(correction.from_frame);
        push(correction.to_frame);
      } else {
        push(correction.from);
        push(correction.to);
      }
    }
    // 대표 프레임 착탄점은 **그리지 않는다**(낱개 점이라 뺐다). 여기서는 카메라 거리 계산에만 쓴다 —
    // 궤적 프리셋이 몸에서 벽까지를 담게 하는 기준점이다.
    if (mode !== "pose") push(view.composite?.hit);
    // **그려지는 벽면 스트로크는 기본 뷰에서도 맞춤 대상이다**(S2). 사용자 불만 1번이 "기준
    // 스트로크(벽면 착탄점)이 보이지 않음"이었는데, 이전 판은 스트로크를 pose 맞춤에서 빼 두어
    // 사용자가 제일 먼저 보는 화면에서 두 줄이 가장자리에 걸리거나 밖으로 나갔다. 원본 착탄점
    // 전체가 아니라 **서버가 잘라 준 구간의 정점**만 넣는다 — 그리지 않는 점으로 카메라를 끌지 않는다.
    for (const p of strokePoints(view)) push(p);
    if (mode === "tracks" || mode === "wall") {
      for (const name of ["head", "right", "left"]) for (const p of view.layers?.query?.[name] ?? []) push(p);
    }
    if (mode === "wall") {
      // 원본 착탄점만 — 리샘플 격자는 벽에서 뜰 수 있다. 그려지는 스트로크의 정점이 이 점들이다.
      // 기준 스트로크도 그려지므로 맞춤 대상에 넣는다(S2) — 안 넣으면 기준 줄이 화면 밖으로 나간다.
      for (const p of view.surface_hits ?? []) push(p);
      for (const p of view.reference_surface_hits ?? []) push(p);
      if (view.mesh?.available === true) for (const p of view.mesh.vertices ?? []) push(p);
    }
    return points;
  }

  /** 화면에 **실제로 그려지는** 벽면 스트로크의 정점(대상·기준 둘 다).
   * 서버가 이미 잘라 준 구간(`stroke.runs` / `reference_stroke.runs`)만 본다 — 자르기 판정을
   * 화면이 다시 하지 않고, 잘려 나간 점으로 카메라를 끌지도 않는다. */
  function strokePoints(view) {
    const out = [];
    for (const block of [view?.stroke, view?.reference_stroke]) {
      for (const run of block?.runs ?? []) for (const p of run ?? []) if (isVec3(p)) out.push(p);
    }
    return out;
  }

  /** 상체 점(월드): 서버가 다리라고 이름 붙인 관절을 뺀 모든 관절 + 실측·예측 마커.
   * 화면에 이미 있는 "상체" 정의(`upperBodyScreen`)와 **같은 점 집합**이다 — 두 곳에서 다르게 정의하면
   * 프리셋이 키운 것과 계측이 재는 것이 달라진다. */
  function upperBodyPoints(view, options) {
    const legSet = new Set(view?.fit?.legs?.joints ?? []);
    const points = [];
    const pushJoints = (joints) => {
      for (const [name, value] of Object.entries(joints ?? {})) {
        if (!legSet.has(name) && isVec3(value)) points.push(value);
      }
    };
    pushJoints(view?.layers?.query?.joints);
    for (const anchor of view?.fit?.anchors ?? []) {
      if (isVec3(anchor.measured_marker)) points.push(anchor.measured_marker);
      if (isVec3(anchor.predicted_marker)) points.push(anchor.predicted_marker);
    }
    // 합격선 때문에 당길 때만 **기준 몸의 상체도** 넣는다(F). 두 자세를 겹쳐 보여 주는 화면이라
    // 당긴 자리에서 기준 머리가 잘리면 그림이 성립하지 않는다. 'torso' 프리셋은 예전 그대로
    // 질의 상체만 본다 — 그 프리셋의 비율을 재는 `upperBodyScreen`과 같은 점 집합이어야 한다.
    if (options?.bothLayers === true) {
      pushJoints(view?.layers?.reference?.joints);
      for (const anchor of view?.reference_fit?.anchors ?? []) {
        if (isVec3(anchor.measured_marker)) points.push(anchor.measured_marker);
      }
    }
    return points;
  }

  const ANCHOR_NAME = { head: "머리", right: "오른손", left: "왼손" };

  /** 기본(자세 맞춤) 프리셋이 자르지 않는 맞춤 대상에 **이름**을 붙여 돌려준다.
   * 상체 확대가 무엇을 화면 밖으로 내보냈는지 말하려면 점마다 사람이 읽는 이름이 있어야 한다. */
  function namedFitTargets(view) {
    const out = [];
    const add = (name, p) => { if (isVec3(p)) out.push({ name, p }); };
    const joints = view?.layers?.query?.joints ?? {};
    const legSet = new Set(view?.fit?.legs?.joints ?? []);
    for (const [name, value] of Object.entries(joints)) if (!legSet.has(name)) add(`${name} 관절`, value);
    for (const anchor of view?.fit?.anchors ?? []) {
      const who = ANCHOR_NAME[anchor.point] ?? anchor.point;
      add(`${who} 실측 마커`, anchor.measured_marker);
      add(`${who} 예측 마커`, anchor.predicted_marker);
    }
    for (const correction of view?.corrections ?? []) {
      const n = typeof correction.n === "number" ? correction.n : "?";
      const useFrame = isVec3(correction.from_frame) && isVec3(correction.to_frame);
      add(`교정 ${n}의 시작점`, useFrame ? correction.from_frame : correction.from);
      add(`교정 ${n}의 화살촉`, useFrame ? correction.to_frame : correction.to);
    }
    // 벽면 스트로크도 이제 기본 맞춤 대상이다(S2) — 그러니 무엇이 잘렸는지 셀 때도 세야 한다.
    // 정점 하나하나를 이름으로 늘어놓으면 문장이 못 읽게 되므로 줄 단위로 양 끝만 이름을 붙인다.
    for (const [key, who] of [["stroke", "대상"], ["reference_stroke", "기준"]]) {
      const runs = view?.[key]?.runs ?? [];
      runs.forEach((run, i) => {
        const points = (run ?? []).filter(isVec3);
        if (points.length === 0) return;
        const label = runs.length > 1 ? `${who} 벽면 스트로크 ${i + 1}번 줄` : `${who} 벽면 스트로크`;
        add(`${label}의 시작`, points[0]);
        add(`${label}의 끝`, points[points.length - 1]);
      });
    }
    return out;
  }

  const OFFSCREEN_NAMES_SHOWN = 4;

  /** 한글 주격 조사. 앞 글자에 받침이 있으면 "이", 없으면 "가"(숫자·영문은 "가"로 둔다). */
  function subjectParticle(text) {
    const code = text.charCodeAt(text.length - 1);
    if (code < 0xac00 || code > 0xd7a3) return "가";
    return (code - 0xac00) % 28 === 0 ? "가" : "이";
  }

  /** 지금 카메라에서 캔버스 밖에 있는 맞춤 대상의 이름. 프리셋이 무엇을 잘랐는지 화면이 스스로 말한다. */
  function offscreenFitTargets() {
    if (!lastView) return [];
    const { w, h } = hostSize();
    const out = [];
    for (const { name, p } of namedFitTargets(lastView)) {
      const px = toPixel(new THREE.Vector3(...p));
      if (px.z > 1 || px.x < 0 || px.y < 0 || px.x > w || px.y > h) out.push(name);
    }
    return out;
  }

  function describeTorsoFraming(mode) {
    if (mode !== "torso") return "";
    // 프리셋을 누른 그대로면 "상체 확대", 그 뒤 카메라를 만졌으면 프리셋 이름으로 부르지 않는다.
    const head = framing.auto ? "상체 확대" : "지금 화면(상체 확대에서 카메라를 만졌습니다)";
    const outside = offscreenFitTargets();
    if (outside.length === 0) return `${head}: 맞춤 대상이 모두 화면 안에 있습니다.`;
    const shown = outside.slice(0, OFFSCREEN_NAMES_SHOWN).join(" · ");
    const rest = outside.length > OFFSCREEN_NAMES_SHOWN ? ` 외 ${outside.length - OFFSCREEN_NAMES_SHOWN}개` : "";
    return `${head}: ${shown}${rest}${subjectParticle(rest || shown)} 화면 밖에 있습니다 — 기본 뷰는 '자세 맞춤'입니다.`;
  }

  /** 다리 관절 **이름**은 서버(`fit.legs.joints`)가 준다 — 화면이 리그 이름을 지어내면
   * 출처와 다른 라벨이 된다. */
  function legJointNames(view) {
    return Array.isArray(view?.fit?.legs?.joints) ? view.fit.legs.joints : [];
  }

  /** 지금 **대상** 층이 IK 다리를 그리고 있는가. 끄면 그 자리에 자리표시 다리가 있다(B). */
  function legsShown() {
    return checkboxes.legs.checked && legsAllowed.query;
  }

  /** **기준** 층이 IK 다리를 그리고 있는가. 두 층의 판정은 자주 갈리므로(자동 기준 24쌍 중 10쌍)
   * 질의 판정으로 기준을 대신하면 맞춤 대상이 그림과 어긋난다. */
  function referenceLegsShown() {
    return checkboxes.legs.checked && legsAllowed.reference;
  }

  /** 지금 화면에 **그려진** 발 관절 월드 좌표. IK를 켠 층은 IK 발, 아니면 자리표시 발이다 —
   * 이름은 어느 쪽이든 서버가 준다(`fit.legs.joints` / `fit.legs.placeholder_joints`). */
  function legFeetPoints(view) {
    if (legsShown()) {
      const joints = view?.layers?.query?.joints ?? {};
      return legJointNames(view)
        .filter((name) => name.endsWith("Foot"))
        .map((name) => joints[name])
        .filter(isVec3);
    }
    const placeholder = placeholderJointsOf(view?.fit);
    return Object.keys(placeholder)
      .filter((name) => name.endsWith("Foot"))
      .map((name) => placeholder[name])
      .filter(isVec3);
  }

  /** 발이 보이는지 재 볼 프리셋. 툴바 버튼과 같은 카메라를 푼다(라벨도 버튼과 같은 글자다). */
  const PRESET_PROBES = [
    { label: "자세 맞춤", mode: "pose", angles: "default" },
    { label: "상체 확대", mode: "torso", angles: "current" },
    { label: "벽 정면", mode: "pose", angles: "wall", offset: 0 },
    { label: "벽 측면", mode: "pose", angles: "wall", offset: Math.PI / 2 },
    { label: "궤적 포함", mode: "tracks", angles: "current" },
    { label: "벽 포함", mode: "wall", angles: "current" },
  ];

  function solvePreset(probe) {
    let theta = camState.theta;
    let phi = camState.phi;
    if (probe.angles === "default") {
      ({ theta, phi } = defaultAngles(lastView));
    } else if (probe.angles === "wall") {
      const normal = estimateWallNormal(lastView);
      if (!normal) return null;
      theta = Math.atan2(normal.x, normal.z) + probe.offset;
      phi = Math.PI / 2.2;
    }
    return solveFraming(probe.mode, theta, phi);
  }

  /** 발이 **이 프리셋에서** 화면 밖이면, 발이 보이는 프리셋의 이름을 같이 말한다.
   *
   * pose(자세 맞춤·기본)는 다리를 맞춤 대상에 넣고 자르지 않으므로 이 고지는 보통 비어 있다 —
   * 사람이 카메라를 만졌거나 '상체 확대'처럼 일부러 자르는 프리셋에서만 뜬다. 바닥 기준 머리 높이
   * 고지가 가리키는 발이 실제로 안 보이면 아무도 그 사실을 확인할 길이 없기 때문에 남겨 둔다.
   * 발이 보이는 프리셋의 이름도 **말로 정하지 않고** 각 프리셋의 카메라를 실제로 풀어서
   * (`solveFraming`) 투영해 확인한다. 다리를 그리지 않는 화면에서는 이 고지가 아예 없다. */
  function describeFeetFraming() {
    if (!lastView) return "";
    const feet = legFeetPoints(lastView);
    if (feet.length === 0) return "";
    if (feet.every((p) => onScreenNow(p))) return "";
    const shows = PRESET_PROBES.map((probe) => {
      const solved = solvePreset(probe);
      if (!solved) return null;
      return feet.every((p) => insideFrustum(p, solved)) ? probe.label : null;
    }).filter((name) => name !== null);
    const where = shows.length === 0 ? "" : ` — ${shows.map((n) => `'${n}'`).join(" · ")}에서 보입니다`;
    // 지금 그려지는 쪽의 이름을 쓴다(B). IK 발은 시뮬레이터가 지은 것이고, 자리표시 발은 서버가
    // 바닥면에 세운 것이다 — 둘을 한 이름으로 부르면 고지가 그림과 다른 것을 가리킨다.
    const what = legsShown() ? "IK 발" : "자리표시 발";
    return `${what}은 ${nowWord()} 화면 밖에 있습니다${where}.`;
  }

  /** 지금 화면이 프리셋 그대로인지에 따라 고지가 자기를 부르는 말. 드래그·휠로 카메라를 만진 뒤에
   * "이 프리셋에서"라고 말하면 프리셋이 방어해 주지 못한다 — 사람이 바꾼 것은 프리셋이 아니라 화면이다. */
  function nowWord() {
    return framing.auto ? "이 프리셋에서" : "지금";
  }

  /** 기본 뷰에서 **그려지는 것이 화면 밖에 있으면** 그 이름을 말한다(감시자).
   *
   * 기본 뷰의 약속은 "하나도 자르지 않는다"이므로 이 문장은 보통 비어 있다 — 비어 있는 것이 정상이고,
   * 떠 있으면 약속이 깨졌다는 뜻이다(사람이 드래그·휠로 카메라를 만진 뒤에는 정상적으로 뜬다).
   * 이름은 `offscreenFitTargets`가 **투영해서** 고른다(말로 정하지 않는다). 맞춤 규칙이 되돌아가
   * 다시 자르기 시작하면 이 줄이 그 사실을 화면에 적는다. */
  function describePoseFraming(mode) {
    if (mode !== "pose") return "";
    const outside = offscreenFitTargets();
    if (outside.length === 0) return "";
    const head = framing.auto ? "자세 맞춤" : "지금 화면(자세 맞춤에서 카메라를 만졌습니다)";
    const shown = outside.slice(0, OFFSCREEN_NAMES_SHOWN).join(" · ");
    const rest = outside.length > OFFSCREEN_NAMES_SHOWN ? ` 외 ${outside.length - OFFSCREEN_NAMES_SHOWN}개` : "";
    const how = framing.auto ? " — '자세 맞춤'을 다시 누르면 다 보입니다" : " — '자세 맞춤'으로 돌아가면 다 보입니다";
    return `${head}: ${shown}${rest}${subjectParticle(rest || shown)} 화면 밖에 있습니다${how}.`;
  }

  /** 기본 뷰의 상체가 합격선(F)에 **못 미치면 그 숫자를 적는다**.
   *
   * 합격선까지 당기는 것과 "그려지는 것을 자르지 않는 것"은 한 카메라에서 동시에 참일 수 없다 —
   * 상체(머리 마커~골반)는 머리~발 높이의 일부이고, L1 실측에서 그 비(0.4091m ÷ 1.5083m = 0.27)가
   * 합격선 0.40보다 작다. 여기서는 자르지 않는 쪽을 택했으므로(B·S2), **못 미친다는 사실과 지금
   * 값을 화면이 말한다**. 말하지 않으면 화면이 조용히 합격선을 어기게 된다. */
  function describeTorsoShortfall(mode) {
    if (mode !== "pose" || !framing.auto) return "";
    const span = torsoScreenSpan();
    if (span === null || span.heightFraction >= MIN_TORSO_FRACTION) return "";
    const now = Math.round(span.heightFraction * 100);
    const goal = Math.round(MIN_TORSO_FRACTION * 100);
    return (
      `자세 맞춤: 상체(머리 마커~골반)가 캔버스 높이의 ${now}%입니다(합격선 ${goal}%) — `
      + "더 당기면 그려진 다리와 벽면 스트로크가 화면 밖으로 나갑니다. '상체 확대'가 그 선을 넘습니다."
    );
  }

  function describeFraming(mode) {
    return [describeTorsoFraming(mode), describePoseFraming(mode), describeTorsoShortfall(mode), describeFeetFraming()]
      .filter((t) => t)
      .join(" ");
  }

  function onScreenNow(p) {
    const { w, h } = hostSize();
    const px = toPixel(new THREE.Vector3(...p));
    return !(px.z > 1 || px.x < 0 || px.y < 0 || px.x > w || px.y > h);
  }

  /** `solveFraming`이 낸 카메라(target·offsetDir·radius)로 점을 투영해 절두체 안인지 본다.
   * `camera`를 실제로 움직이지 않고 **가정한** 프리셋을 재는 길이다. */
  function insideFrustum(p, solved) {
    const ndc = ndcAt(p, solved.target, solved.offsetDir, solved.radius);
    return ndc !== null && Math.abs(ndc.x) <= 1 && Math.abs(ndc.y) <= 1;
  }

  function boundsOf(points) {
    const min = [Infinity, Infinity, Infinity];
    const max = [-Infinity, -Infinity, -Infinity];
    for (const p of points) {
      for (let i = 0; i < 3; i += 1) {
        if (p[i] < min[i]) min[i] = p[i];
        if (p[i] > max[i]) max[i] = p[i];
      }
    }
    return { min, max };
  }

  /** 카메라 거리 d에서 주어진 점들을 투영했을 때 max(높이비, 폭비). 뒤로 넘어가면 Infinity. */
  /** 카메라 오프셋 방향에서 (전방, 오른쪽, 위) 축. `applyCamera`의 lookAt(up=(0,1,0))과 같은 기저다. */
  function cameraBasis(offsetDir) {
    const forward = offsetDir.clone().negate();
    const right = new THREE.Vector3().crossVectors(forward, new THREE.Vector3(0, 1, 0));
    if (right.lengthSq() < 1e-12) right.set(1, 0, 0);
    right.normalize();
    const up = new THREE.Vector3().crossVectors(right, forward).normalize();
    return { forward, right, up };
  }

  /** 맞춤 대상의 화면(NDC) 범위. 뒤로 넘어간 점이 하나라도 있으면 null. */
  function projectedExtent(corners, target, offsetDir, distance) {
    const position = target.clone().addScaledVector(offsetDir, distance);
    const { forward, right, up } = cameraBasis(offsetDir);
    const tanV = Math.tan((camera.fov * Math.PI) / 360);
    const tanH = tanV * camera.aspect;
    let xmin = Infinity; let xmax = -Infinity; let ymin = Infinity; let ymax = -Infinity;
    for (const corner of corners) {
      const v = corner.clone().sub(position);
      const z = v.dot(forward);
      if (z <= camera.near) return null;
      const x = v.dot(right) / (z * tanH);
      const y = v.dot(up) / (z * tanV);
      xmin = Math.min(xmin, x); xmax = Math.max(xmax, x);
      ymin = Math.min(ymin, y); ymax = Math.max(ymax, y);
    }
    return { xmin, xmax, ymin, ymax, tanH, tanV, right, up };
  }

  function projectedRatio(corners, target, offsetDir, distance) {
    const box = projectedExtent(corners, target, offsetDir, distance);
    if (box === null) return Infinity;
    return Math.max((box.xmax - box.xmin) / 2, (box.ymax - box.ymin) / 2);
  }

  /** 맞춤 대상이 화면 중심에서 **가장 멀리** 나간 정도(NDC). 1.0이 캔버스 가장자리다.
   *
   * `projectedRatio`는 범위의 **크기**만 본다. 카메라가 보는 곳은 AABB 중심이라, 점 구름이
   * 한쪽으로 치우치면 크기가 목표 안이어도 한쪽이 화면 밖으로 밀린다 — 벽면 스트로크를 기본 맞춤
   * 대상에 넣은 뒤(S2) 이것이 드러났다(김경호 L6 1000394_1 #0, 1875×1000: 스트로크 정점 77개 중
   * 3개가 캔버스 위로 6px). 이 양을 목표 아래로 누르면 **모든** 맞춤 대상이 캔버스 안이 된다.
   * 거리가 멀어질수록 모든 점이 화면 중심(= AABB 중심)으로 모이므로 이 값은 거리에 단조롭다 —
   * 같은 이분법을 그대로 쓸 수 있다. 치우치지 않은 구름에서는 `projectedRatio`와 같은 값이다. */
  function projectedReach(corners, target, offsetDir, distance) {
    const box = projectedExtent(corners, target, offsetDir, distance);
    if (box === null) return Infinity;
    return Math.max(Math.abs(box.xmin), Math.abs(box.xmax), Math.abs(box.ymin), Math.abs(box.ymax));
  }

  /** AABB 중심을 보고, 세로·가로 FOV 모두에서 목표 비율이 되는 거리를 이분법으로 찾는다.
   *
   * 카메라를 **움직이지 않고** 답만 낸다 — `frameTo`가 적용하고, 발이 어느 프리셋에서 보이는지 재는
   * `describeFeetFraming`은 적용하지 않은 채로 쓴다(가정한 프리셋을 재는 데 화면을 흔들면 안 된다). */
  function solveFraming(mode, theta, phi) {
    if (!lastView) return null;
    const points = framingPoints(lastView, mode);
    if (points.length === 0) return null;
    const { min, max } = boundsOf(points);
    const target = new THREE.Vector3((min[0] + max[0]) / 2, (min[1] + max[1]) / 2, (min[2] + max[2]) / 2);
    const corners = [];
    for (const x of [min[0], max[0]]) for (const y of [min[1], max[1]]) for (const z of [min[2], max[2]]) {
      corners.push(new THREE.Vector3(x, y, z));
    }
    const sinPhi = Math.sin(phi);
    const offsetDir = new THREE.Vector3(
      sinPhi * Math.sin(theta),
      Math.cos(phi),
      sinPhi * Math.cos(theta),
    ).normalize();
    // pose(자세 맞춤·기본)와 torso(상체 확대)는 **실제 점**에 맞춘다. AABB 8꼭짓점에 맞추면 몸 앞뒤로
    // 부풀린 상자의 가까운 모서리가 크게 잡혀, 화면에 실제로 보이는 것보다 멀리 물러선다(L6에서 상체가
    // 59%만 차지해야 할 거리를 75%로 세어 뒤로 물러났다). torso에서도 같은 일이 일어나 **확대 프리셋이
    // 기본보다 작아졌다** — 실측: L3 상체 비율 0.6692(실제 점) vs 0.4864(꼭짓점), 기본이 0.5827이다.
    // 궤적·벽 프리셋은 점이 많아 예전대로 꼭짓점을 쓴다.
    const fitPoints =
      mode === "pose" || mode === "torso"
        ? points.map((p) => new THREE.Vector3(...p))
        : corners;
    const goal = FRAMING_TARGET[mode] ?? 0.75;
    // pose만 **가장 멀리 나간 점**을 목표 아래로 누른다 — 기본 뷰에서는 맞춤 대상이 하나도
    // 잘리면 안 된다(다리 B · 벽면 스트로크 S2). torso("상체 확대")는 **일부러** 맞춤 대상을
    // 상체로 좁혀 나머지를 화면 밖으로 내보낸다(그 프리셋의 약속). 궤적·벽은 점이 많아 예전대로
    // 범위 크기에 맞춘다.
    const measure = mode === "pose" ? projectedReach : projectedRatio;
    let lo = 0.01;
    let hi = 100;
    for (let i = 0; i < 60; i += 1) {
      const mid = (lo + hi) / 2;
      if (measure(fitPoints, target, offsetDir, mid) > goal) lo = mid;
      else hi = mid;
    }
    // 기본 뷰는 여기서 끝난다. 예전에는 이 자리에서 상체 합격선까지 **당겼는데**(`pullInToTorsoFloor`),
    // 당기면 그려진 다리와 벽면 스트로크가 화면 밖으로 나간다(위 `MIN_TORSO_FRACTION` 주석의 실측).
    // 합격선은 이제 화면이 **재서 말하는 값**이지 카메라가 맞추는 값이 아니다.
    return { target, offsetDir, radius: hi, ratio: projectedRatio(fitPoints, target, offsetDir, hi) };
  }

  /** **가정한** 카메라(target·offsetDir·distance)에서 점의 NDC 좌표. 뒤로 넘어가면 null.
   * 화면을 흔들지 않고 재는 길이다 — `camera`를 움직이지 않는다. */
  function ndcAt(p, target, offsetDir, distance) {
    const basis = cameraBasis(offsetDir);
    const position = target.clone().addScaledVector(offsetDir, distance);
    const v = new THREE.Vector3(...p).sub(position);
    const z = v.dot(basis.forward);
    if (z <= camera.near) return null;
    const tanV = Math.tan((camera.fov * Math.PI) / 360);
    return { x: v.dot(basis.right) / (z * tanV * camera.aspect), y: v.dot(basis.up) / (z * tanV) };
  }

  function frameTo(mode) {
    const solved = solveFraming(mode, camState.theta, camState.phi);
    if (!solved) return;
    camState.target.copy(solved.target);
    camState.radius = solved.radius;
    framing.mode = mode;
    framing.auto = true;
    framing.ratio = solved.ratio;
    applyCamera();
    // 자른 사실은 카메라를 적용한 **뒤에** 잰다 — 투영이 새 카메라를 써야 이름이 맞는다.
    writeFramingNote();
  }

  /** 고지를 **지금 카메라로** 다시 쓴다.
   *
   * 예전에는 `frameTo`에서만 썼다. 그래서 사람이 드래그·휠로 카메라를 만지면 문구가 그 자리에 굳어
   * 거짓이 됐다 — 실측(L3 '자세 맞춤', 호스트 1094×600): 캔버스 가운데를 위로 140px 끌고 휠을 12회
   * 내리면 두 발이 화면 안(LeftFoot (482.6,323.7) · RightFoot (425.5,340.9))인데 고지는 그대로
   * "…화면 밖에 있습니다"였다. 3D 뷰를 돌려 보는 것은 데모 관람자가 가장 먼저 하는 동작이다.
   * 그래서 카메라가 바뀌는 **모든** 길(프리셋·드래그·휠·창 크기)이 이 함수를 지난다. */
  function writeFramingNote() {
    if (!lastView) return;
    const text = describeFraming(framing.mode);
    dom.framingNote.textContent = text;
    if (container.__view3dDebug) {
      // 프리셋이 푼 비율은 그 카메라에서만 참이다. 사람이 카메라를 만진 뒤에는 내놓지 않는다 —
      // 지금 화면의 상체 비율은 아래 `upperBody`가 live 카메라로 잰다.
      container.__view3dDebug.framingRatio = framing.auto ? framing.ratio : null;
      container.__view3dDebug.framingMode = framing.mode;
      container.__view3dDebug.framingAuto = framing.auto;
      // 자른 것의 이름은 '상체 확대'와 기본 뷰 둘 다에서 잰다 — 기본 뷰에서는 **0이어야** 한다(B·S2).
      container.__view3dDebug.framingOffscreen =
        framing.mode === "torso" || framing.mode === "pose" ? offscreenFitTargets() : [];
      container.__view3dDebug.framingNote = text;
      container.__view3dDebug.torsoSpan = torsoScreenSpan();
      // 합격선과 지금 값을 **같이** 낸다 — 검증이 "0.4000"만 보고 합격 처리하지 않도록.
      container.__view3dDebug.torsoPassLine = MIN_TORSO_FRACTION;
      container.__view3dDebug.torsoMeetsPassLine =
        torsoScreenSpan() === null ? null : torsoScreenSpan().heightFraction >= MIN_TORSO_FRACTION;
      // 카메라 행렬은 `layoutLabels`가 이미 `debugLive.camera`로 내보낸다(projectionMatrix ·
      // matrixWorldInverse · host). 여기서 또 쓰면 같은 이름 아래 모양이 다른 두 값이 번갈아 실려,
      // 검증이 어느 쪽을 읽었는지에 따라 답이 달라진다 — 한 출처로 둔다.
      container.__view3dDebug.upperBody = upperBodyScreen();
      // D3 검증용: IK 발이 지금 보이는가, 어느 프리셋이 보여 주는가.
      const feet = legFeetPoints(lastView);
      container.__view3dDebug.feetOnScreen = feet.length > 0 && feet.every((q) => onScreenNow(q));
      container.__view3dDebug.feetPresets = feet.length === 0
        ? []
        : PRESET_PROBES.map((probe) => {
            const solved = solvePreset(probe);
            return solved && feet.every((q) => insideFrustum(q, solved)) ? probe.label : null;
          }).filter((n) => n !== null);
    }
  }

  // 드래그는 한 프레임에 여러 번 들어온다. 고지 다시 쓰기를 한 프레임에 한 번으로 모은다.
  let framingNotePending = false;
  function scheduleFramingNote() {
    if (framingNotePending) return;
    framingNotePending = true;
    requestAnimationFrame(() => {
      framingNotePending = false;
      writeFramingNote();
    });
  }

  /** 작업자 쪽(착탄점→골반 수평 방향)에서 옆으로 약 50° 비켜 약간 위에서 본다. */
  function defaultAngles(view) {
    const joints = view.layers?.query?.joints ?? {};
    const body = isVec3(joints.Pelvis) ? joints.Pelvis : (view.corrections ?? []).map((c) => c.from).find(isVec3);
    const hit = view.composite?.hit;
    if (isVec3(body) && isVec3(hit)) {
      const dx = body[0] - hit[0];
      const dz = body[2] - hit[2];
      if (Math.hypot(dx, dz) > 1e-6) {
        return { theta: Math.atan2(dx, dz) + 0.9, phi: Math.PI / 2.5 };
      }
    }
    return { theta: Math.PI / 4, phi: Math.PI / 2.5 };
  }

  /** 벽 법선의 최선 추정(카메라 프리셋 편의, 측정값 아님). 작업자 쪽을 향하게 부호를 맞춘다. */
  function estimateWallNormal(view) {
    let normal = null;
    const mesh = view.mesh;
    const hit = view.composite?.hit;
    if (mesh && mesh.available === true && mesh.triangles?.length > 0 && isVec3(hit)) {
      // 대표 착탄점에 가장 가까운 삼각형 무게중심의 법선.
      let best = null;
      let bestDist = Infinity;
      const h = new THREE.Vector3(...hit);
      for (const [i0, i1, i2] of mesh.triangles) {
        const a = new THREE.Vector3(...mesh.vertices[i0]);
        const b = new THREE.Vector3(...mesh.vertices[i1]);
        const c = new THREE.Vector3(...mesh.vertices[i2]);
        const d = a.clone().add(b).add(c).multiplyScalar(1 / 3).distanceTo(h);
        if (d < bestDist) { bestDist = d; best = [a, b, c]; }
      }
      if (best) {
        const n = new THREE.Vector3().subVectors(best[1], best[0]).cross(new THREE.Vector3().subVectors(best[2], best[0]));
        if (n.lengthSq() > 1e-12) normal = n.normalize();
      }
    }
    if (!normal) {
      const wallCorrection = (view.corrections ?? []).find((c) => c.axis === "wall" && isVec3(c.unit));
      if (wallCorrection) normal = new THREE.Vector3(...wallCorrection.unit).normalize();
    }
    if (!normal) return null;
    const joints = view.layers?.query?.joints ?? {};
    if (isVec3(joints.Pelvis) && isVec3(hit)) {
      const toBody = new THREE.Vector3(...joints.Pelvis).sub(new THREE.Vector3(...hit));
      if (normal.dot(toBody) < 0) normal.negate();
    }
    return normal;
  }

  const checkboxes = {
    query: dom.controls.querySelector("#view3dLayerQuery"),
    reference: dom.controls.querySelector("#view3dLayerReference"),
    corrected: dom.controls.querySelector("#view3dLayerCorrected"),
    wall: dom.controls.querySelector("#view3dLayerWall"),
    legs: dom.controls.querySelector("#view3dLayerLegs"),
  };
  checkboxes.query.addEventListener("change", () => {
    layerGroups.query.visible = checkboxes.query.checked;
    applyDeviceLabelVisibility();
  });
  /** 장치 이름표는 자기 층과 한 몸이다 — 층을 끄면 막대와 함께 이름표도 사라진다. */
  function applyDeviceLabelVisibility() {
    for (const entry of deviceLabels) {
      entry.el.hidden = !(entry.layer === "reference" ? checkboxes.reference.checked : checkboxes.query.checked);
    }
  }
  /** 조상까지 모두 보이는 뼈 이름(검증용). three.js는 부모가 숨으면 자식도 그리지 않는다. */
  function visibleBoneNames() {
    const names = [];
    layerGroups.query.traverseVisible((child) => {
      if (child.userData.kind === "bone") names.push(child.userData.name);
    });
    return layerGroups.query.visible ? names : [];
  }
  /** 관절 하나가 받은 진하기(0~1, 검증용). 서버가 준 벌어짐이 실제로 색에 닿았는지 이 값으로 잰다. */
  function visibleJointDeviations() {
    const out = {};
    layerGroups.query.traverseVisible((child) => {
      if (child.userData.kind === "estimate-joint") out[child.userData.name] = child.userData.deviation;
    });
    return layerGroups.query.visible ? out : {};
  }
  // 층마다 자기 IK 다리가 해부학 범위 안일 때만 보인다(S4). 한쪽 판정으로 양쪽을 켜면, 범위를
  // 벗어난 다리를 "기준은 이렇게 섰다"로 그리게 된다.
  let legsAllowed = { query: false, reference: false };
  function applyLegsToggle() {
    const on = checkboxes.legs.checked;
    if (skeletonLegs) skeletonLegs.visible = on && legsAllowed.query;
    if (referenceLegs) referenceLegs.visible = on && legsAllowed.reference;
    // 끄면 자리표시(점선), 켜면 IK — **한 층에서 둘이 겹쳐 보이지 않는다**(B). 겹치면 어느 쪽이
    // 측정인지 읽을 수 없다. IK를 켤 수 없는 층은 자리표시가 그대로 남는다(다리 없는 화면은 없다).
    const placeholderShown = {
      query: !(on && legsAllowed.query),
      reference: !(on && legsAllowed.reference),
    };
    if (skeletonPlaceholderLegs) skeletonPlaceholderLegs.visible = placeholderShown.query;
    if (referencePlaceholderLegs) referencePlaceholderLegs.visible = placeholderShown.reference;
    // 간격 보조선과 그 글자는 자리표시 다리와 **한 몸**이다: IK로 바꾼 층에서는 둘 다 사라진다.
    // 남겨 두면 IK 발 옆에 자리표시 발의 틈이 떠 있어 그림과 글자가 다른 것을 말하게 된다.
    if (skeletonFloorGaps) skeletonFloorGaps.visible = placeholderShown.query;
    if (referenceFloorGaps) referenceFloorGaps.visible = placeholderShown.reference;
    for (const entry of gapLabels) {
      entry.el.hidden = !(entry.layer === "reference" ? placeholderShown.reference : placeholderShown.query);
    }
  }
  checkboxes.legs.addEventListener("change", () => {
    applyLegsToggle();
    if (lastView && framing.auto) frameTo(framing.mode);
    if (container.__view3dDebug) {
      container.__view3dDebug.legsVisible = checkboxes.legs.checked;
      container.__view3dDebug.visibleBones = visibleBoneNames();
      // 토글이 바꾼 것은 **자리표시↔IK**다. 이 값을 여기서 다시 쓰지 않으면 헤드리스 검증이
      // 그리기 직후의 옛 값을 읽어 "IK를 켰는데도 자리표시가 보인다"고 잘못 읽는다.
      container.__view3dDebug.placeholderLegsShownByLayer = {
        query: skeletonPlaceholderLegs ? skeletonPlaceholderLegs.visible : false,
        reference: referencePlaceholderLegs ? referencePlaceholderLegs.visible : false,
      };
    }
  });
  checkboxes.reference.addEventListener("change", () => {
    layerGroups.reference.visible = checkboxes.reference.checked;
    applyDeviceLabelVisibility();
  });
  checkboxes.corrected.addEventListener("change", () => {
    layerGroups.corrected.visible = checkboxes.corrected.checked;
    dom.correctedNote.textContent = checkboxes.corrected.checked
      ? "교정 후 층은 관측이 아니라 발부된 교정을 따랐을 때의 예상입니다 — 끊어 그린 선과 속 빈 표식으로 그립니다."
      : "";
    if (container.__view3dDebug) container.__view3dDebug.correctedLayerIsOutline = computedLayerIsOutline();
  });
  checkboxes.wall.addEventListener("change", () => {
    if (meshObject) meshObject.visible = checkboxes.wall.checked;
  });

  function applyPreset(offset) {
    if (!lastView) return;
    const normal = estimateWallNormal(lastView);
    if (!normal) {
      setNote(dom.note, "벽 방향을 추정할 수 없어 프리셋을 적용하지 않았습니다.", "empty");
      return;
    }
    camState.theta = Math.atan2(normal.x, normal.z) + offset;
    camState.phi = Math.PI / 2.2;
    frameTo("pose");
  }
  dom.controls.querySelector("#view3dPresetFit").addEventListener("click", () => {
    if (!lastView) return;
    Object.assign(camState, defaultAngles(lastView));
    frameTo("pose");
  });
  dom.controls.querySelector("#view3dPresetTorso").addEventListener("click", () => {
    if (!lastView) return;
    frameTo("torso");
  });
  dom.controls.querySelector("#view3dPresetFront").addEventListener("click", () => applyPreset(0));
  dom.controls.querySelector("#view3dPresetSide").addEventListener("click", () => applyPreset(Math.PI / 2));
  dom.controls.querySelector("#view3dPresetTracks").addEventListener("click", () => frameTo("tracks"));
  dom.controls.querySelector("#view3dPresetWall").addEventListener("click", () => frameTo("wall"));

  function loading(message) {
    setNote(dom.note, message ?? "비교를 계산하는 중…", "loading");
    dom.host.classList.add("is-loading");
  }

  function resetScene() {
    for (const group of Object.values(layerGroups)) {
      disposeObject3D(group);
      group.clear();
    }
    disposeObject3D(correctionsGroup);
    correctionsGroup.clear();
    clearLabels();
    coneRegistry = [];
    markerRegistry = [];
    sphereObstacles = [];
    deviceBars = [];
    skeletonLegs = null;
    referenceLegs = null;
    skeletonPlaceholderLegs = null;
    referencePlaceholderLegs = null;
    skeletonFloorGaps = null;
    referenceFloorGaps = null;
    if (meshObject) {
      scene.remove(meshObject);
      disposeObject3D(meshObject);
      meshObject = null;
    }
  }

  function render(payload) {
    dom.host.classList.remove("is-loading");
    const view = payload?.view;
    if (!view) {
      clear("표시할 뷰 데이터가 없습니다.", "empty");
      return;
    }
    lastView = view;
    resetScene();

    const fit = view.fit ?? null;
    const referenceFit = view.reference_fit ?? null;
    const joints = view.layers?.query?.joints ?? {};
    const referenceJoints = view.layers?.reference?.joints ?? {};
    const hasSkeleton = Object.keys(joints).length > 0;

    // 관절 진하기의 근거는 서버가 잰 벌어짐 하나뿐이다(S1). 못 받았으면(한쪽 스켈레톤이 안 섰으면)
    // 전부 0으로 칠하지 않고 눈금을 비워 둔다 — `deviationStyle`이 눈금 없이는 t=0을 낸다.
    const deviation = view.deviation ?? null;
    const deviations = deviation?.joints ?? {};
    const maxDeviation = deviation?.joint_max_m ?? null;

    // 마지막 인자는 "이 층의 점이 실측인가". 질의·기준은 같은 세 강체의 실측 궤적이고, 교정 후 층은
    // 발부된 교정을 따랐을 때의 **계산값**이다 — 범례가 실측에 준 부호(층 색으로 채운 구체)를 쓰면 안 된다.
    layerGroups.query.add(buildLayerGroup(view.layers?.query, LAYER_COLOR.query, !hasSkeleton, true));
    layerGroups.reference.add(buildLayerGroup(view.layers?.reference, LAYER_COLOR.reference, false, true));
    layerGroups.corrected.add(buildLayerGroup(view.layers?.corrected, LAYER_COLOR.corrected, false, false));

    const skeletonParts = buildSkeletonGroup(
      joints, view.layers?.query?.bones, fit?.faded === true, fit?.legs?.joints, fit?.head_diameter_m,
      {
        layerColor: LAYER_COLOR.query,
        deviations,
        maxDeviation,
        lowConfidenceBones: fit?.low_confidence_bones,
        lowConfidenceJoints: fit?.low_confidence_joints,
        legsOutOfRange: fit?.legs_within_range === false,
      },
    );
    layerGroups.query.add(skeletonParts.group);
    // 기준도 **같은 함수**로 그린다 — 두 그리기 경로를 두면 같은 사람이 층에 따라 다른 몸이 된다.
    const referenceParts = buildSkeletonGroup(
      referenceJoints, view.layers?.reference?.bones, referenceFit?.faded === true,
      referenceFit?.legs?.joints, referenceFit?.head_diameter_m,
      {
        layerColor: LAYER_COLOR.reference,
        deviations,
        maxDeviation,
        lowConfidenceBones: referenceFit?.low_confidence_bones,
        lowConfidenceJoints: referenceFit?.low_confidence_joints,
        legsOutOfRange: referenceFit?.legs_within_range === false,
      },
    );
    layerGroups.reference.add(referenceParts.group);
    debugLive.headDiameterM = typeof fit?.head_diameter_m === "number" ? fit.head_diameter_m : null;
    skeletonLegs = skeletonParts.legs;
    referenceLegs = referenceParts.legs;
    // 자리표시 다리(B) — **기본으로 보이는 쪽**. 층마다 자기 fit의 것을 그린다.
    skeletonPlaceholderLegs = buildPlaceholderLegGroup(fit, LAYER_COLOR.query, fit?.faded === true);
    layerGroups.query.add(skeletonPlaceholderLegs);
    referencePlaceholderLegs = buildPlaceholderLegGroup(
      referenceFit, LAYER_COLOR.reference, referenceFit?.faded === true,
    );
    layerGroups.reference.add(referencePlaceholderLegs);
    // 못 닿은 발의 간격 보조선(점선 + "간격 NNcm"). 서버가 준 층만 그린다 — 닿은 층은 빈 목록이다.
    skeletonFloorGaps = buildPlaceholderFloorGaps(fit, LAYER_COLOR.query, fit?.faded === true, "query");
    layerGroups.query.add(skeletonFloorGaps);
    referenceFloorGaps = buildPlaceholderFloorGaps(
      referenceFit, LAYER_COLOR.reference, referenceFit?.faded === true, "reference",
    );
    layerGroups.reference.add(referenceFloorGaps);
    // 토글은 "다리가 있는가"가 아니라 **자리표시를 IK로 바꾸는가**다(B). 다리는 자리표시로 늘
    // 그려지므로 다리 없는 화면은 없다. 바꿀 IK가 해부학 범위 안인 층이 하나도 없을 때만 잠근다 —
    // 예전에는 바로 그때(L3·L6) 다리가 통째로 사라지고 토글까지 잠겨 볼 것도 바꿀 것도 없었다.
    legsAllowed = {
      query: fit?.legs_toggle_enabled === true,
      reference: referenceFit?.legs_toggle_enabled === true,
    };
    checkboxes.legs.disabled = !(legsAllowed.query || legsAllowed.reference);
    checkboxes.legs.checked = false;
    applyLegsToggle();

    // 벽 위 분사 스트로크. 서버(`composite.stroke_runs`)가 **이미 잘라서** 준 연결 구간만 그린다 —
    // 화면은 자르는 판정을 다시 하지 않는다. 대상·기준 둘 다 그리고, 몸 움직임 궤적과 다른 색 계열을
    // 쓴다(S2) — 사용자가 둘을 혼동한 것이 이 화면을 다시 만든 출발점이다.
    const strokeGroup = buildStrokeGroup(view.stroke, STROKE_COLOR.query);
    layerGroups.query.add(strokeGroup.group);
    const referenceStrokeGroup = buildStrokeGroup(view.reference_stroke, STROKE_COLOR.reference);
    layerGroups.reference.add(referenceStrokeGroup.group);

    const anchors = buildAnchorGroup(fit, LAYER_COLOR.query, true, "query", view.layers?.query);
    layerGroups.query.add(anchors.group);
    const referenceAnchors = buildAnchorGroup(referenceFit, LAYER_COLOR.reference, false, "reference", view.layers?.reference);
    layerGroups.reference.add(referenceAnchors.group);
    layerGroups.query.visible = checkboxes.query.checked;
    layerGroups.reference.visible = checkboxes.reference.checked;
    applyDeviceLabelVisibility();
    layerGroups.corrected.visible = checkboxes.corrected.checked;

    // 화살표: 대표 프레임 실측 마커에서 나간다(from_frame). 번호는 서버의 n.
    const arrowStarts = [];
    for (const correction of view.corrections ?? []) {
      const useFrame = isVec3(correction.from_frame) && isVec3(correction.to_frame);
      const from = useFrame ? correction.from_frame : correction.from;
      const to = useFrame ? correction.to_frame : correction.to;
      if (!isVec3(from) || !isVec3(to)) continue;
      const arrow = buildArrow(from, to, sigmaColor(correction.sigma));
      if (!arrow) continue;
      correctionsGroup.add(arrow.group);
      coneRegistry.push({ n: correction.n, center: arrow.coneBase, radius: ARROW_HEAD_RADIUS });
      arrowStarts.push({ n: correction.n, from, source: useFrame ? "from_frame" : "from" });
      const start = new THREE.Vector3(...from);
      const end = new THREE.Vector3(...to);
      const mid = start.clone().add(end).multiplyScalar(0.5);
      registerLabel(mid, typeof correction.n === "number" ? String(correction.n) : "—", "view3d-label-arrow", {
        kind: "badge", n: correction.n, from: start, to: end,
      });
    }

    const mesh = view.mesh ?? null;
    meshObject = buildMesh(mesh);
    if (meshObject) {
      meshObject.visible = checkboxes.wall.checked;
      scene.add(meshObject);
      checkboxes.wall.disabled = false;
    } else {
      checkboxes.wall.disabled = true;
    }

    renderLegend(dom.legend, view);
    dom.notes.replaceChildren();
    dom.notesMore.replaceChildren();
    const moreLines = [];
    if (fit) {
      // 전부 싣는다. 서버가 접으라고 한 고지(`notes_collapsed`, 이름은 `note_keys`)만 한 번 눌러 펼치는 칸에 둔다 —
      // 캔버스 아래 고지가 벽처럼 쌓여 3D를 밀어냈다(2라운드). 짝짓기는 위치가 아니라 이름이다.
      const keys = Array.isArray(fit.note_keys) ? fit.note_keys : [];
      const collapsed = new Set(Array.isArray(fit.notes_collapsed) ? fit.notes_collapsed : []);
      // 다리 고지는 질의 fit이 아니라 **뷰의 두 층 몫**(`view.leg_notes`)에서 온다(S4). 질의 것만
      // 실었을 때는 기준 층 다리가 그려져 있는데 "다리는 그리지 않았습니다"라고 적히는 일이
      // 자동 기준 24쌍 중 10쌍에서 났다. 서버가 층 이름을 붙여 주고 같은 문장이면 합쳐 준다.
      const legNotes = (view.leg_notes ?? []).filter((text) => typeof text === "string");
      (fit.notes ?? []).forEach((text, i) => {
        if (keys[i] === "legs") {
          for (const line of legNotes) {
            const own = document.createElement("li");
            own.textContent = line;
            own.dataset.note = "legs";
            dom.notes.append(own);
          }
          return;
        }
        if (collapsed.has(keys[i])) {
          moreLines.push(text);
          return;
        }
        const item = document.createElement("li");
        item.textContent = text;
        dom.notes.append(item);
      });
    } else if (typeof view.fit_note === "string") {
      const item = document.createElement("li");
      item.textContent = view.fit_note;
      dom.notes.append(item);
    }
    if (mesh && mesh.available !== true && typeof mesh.reason_detail === "string") moreLines.push(mesh.reason_detail);
    // 손대칭 판정 근거(E). 벽을 그리는 레벨에서만 있는 내부 사유라 캔버스 위 벽면 슬롯이 아니라
    // 여기(접힌 '고지 더 보기')에 둔다 — 위 `reason_detail`과 같은 종류, 같은 자리다.
    if (mesh && mesh.available === true && typeof mesh.handedness_note === "string") {
      const evidence = typeof mesh.handedness_evidence_display === "string" ? ` ${mesh.handedness_evidence_display}` : "";
      moreLines.push(`${mesh.handedness_note}${evidence}`);
    }
    if (moreLines.length > 0) {
      const summary = document.createElement("summary");
      summary.textContent = `고지 ${moreLines.length}건 더 보기`;
      const list = document.createElement("ul");
      for (const text of moreLines) {
        const item = document.createElement("li");
        item.textContent = text;
        list.append(item);
      }
      // 닫힌 <details> 안의 목록도 Chromium은 (0,0) 근처에 사각형을 남긴다 — 닫혀 있을 때는 목록 자체를 숨긴다.
      list.hidden = !dom.notesMore.open;
      dom.notesMore.ontoggle = () => { list.hidden = !dom.notesMore.open; };
      dom.notesMore.append(summary, list);
    }
    dom.notesMore.hidden = moreLines.length === 0;
    const wallText = wallNoteText(mesh);
    setNote(dom.wallNote, wallText, wallText && mesh?.available !== true ? "empty" : undefined);
    const errors = view.display_errors ?? [];
    setNote(
      dom.errors,
      errors.length > 0 ? `일부 표시 블록을 만들지 못했습니다: ${errors.map((e) => e.block).join(", ")}` : "",
      errors.length > 0 ? "error" : undefined,
    );
    setNote(dom.note, "", undefined);
    if (!checkboxes.corrected.checked) dom.correctedNote.textContent = "";

    const boneLengths = {};
    layerGroups.query.traverse((child) => {
      if (child.userData.kind === "bone") boneLengths[child.userData.name] = child.userData.length;
    });

    container.__view3dDebug = debugLive;
    Object.assign(debugLive, {
      arrowCount: correctionsGroup.children.length,
      arrowStarts,
      meshTriangleCount: meshObject ? meshObject.userData.triangleCount : 0,
      labelCount: labelRegistry.length,
      labels: labelRegistry.map((entry) => entry.el.textContent),
      jointCount: Object.keys(joints).length,
      markerKinds: {
        measuredSpheres: anchors.counts.measuredSpheres,
        referenceMeasuredSpheres: referenceAnchors.counts.measuredSpheres,
        momentRings: anchors.counts.momentRings,
        referenceMomentRings: referenceAnchors.counts.momentRings,
      },
      // 장치 모양(몸 → 장치). 양 끝점과 이름표 글자를 **씬과 DOM에서** 그대로 읽어 낸다 — 검증이 서버
      // 앵커(`joint_position`·`measured_marker`·`device.label`)와 대조한다.
      devices: [...anchors.devices, ...referenceAnchors.devices].map((d) => ({
        layer: d.layer,
        point: d.point,
        joint: d.joint,
        name: d.name,
        shape: d.shape,
        from: d.bar.userData.from,
        to: d.bar.userData.to,
        radius: d.shape === "tool" ? DEVICE_TOOL_RADIUS : DEVICE_LINK_RADIUS,
        label: d.label ? d.label.el.textContent : null,
      })),
      momentRings: (() => {
        const rings = [];
        for (const group of [layerGroups.query, layerGroups.reference]) {
          group.traverse((child) => {
            if (child.userData.kind !== "moment-ring") return;
            rings.push({
              layer: child.userData.layer,
              point: child.userData.point,
              center: [child.position.x, child.position.y, child.position.z],
              trackGapM: child.userData.trackGapM,
            });
          });
        }
        return rings;
      })(),
      trackTubes: {
        query: countKind(layerGroups.query, "measured-layer-tube"),
        reference: countKind(layerGroups.reference, "measured-layer-tube"),
      },
      legsVisible: checkboxes.legs.checked,
      legsToggleDisabled: checkboxes.legs.disabled,
      // 자리표시 다리(B)가 실제로 씬에 있고 보이는가 — 헤드리스 브라우저가 말이 아니라 이 값으로 잰다.
      placeholderLegsShownByLayer: {
        query: skeletonPlaceholderLegs ? skeletonPlaceholderLegs.visible : false,
        reference: referencePlaceholderLegs ? referencePlaceholderLegs.visible : false,
      },
      placeholderBoneCounts: {
        query: skeletonPlaceholderLegs ? skeletonPlaceholderLegs.children.length : 0,
        reference: referencePlaceholderLegs ? referencePlaceholderLegs.children.length : 0,
      },
      // 자리표시 발이 바닥에 못 닿은 층의 간격 보조선과 그 글자(사용자 결정: 발이 뜬다).
      // 서버가 준 목록과 씬에 실제로 올라간 선이 같은지를 헤드리스 브라우저가 이 값으로 잰다.
      placeholderFloorGaps: {
        query: floorGapDebug(skeletonFloorGaps),
        reference: floorGapDebug(referenceFloorGaps),
      },
      placeholderGapLabels: gapLabels.map((entry) => entry.el.textContent),
      // 층마다 다리가 실제로 보이는가. 한쪽 판정으로 양쪽을 켜면 없는 자세를 그림이 주장한다.
      legsShownByLayer: { query: legsShown(), reference: checkboxes.legs.checked && legsAllowed.reference },
      visibleBones: visibleBoneNames(),
      // 관절 진하기가 서버 값에 실제로 닿았는가(0~1). 눈금은 `deviation.joint_max_m`이다.
      jointDeviations: visibleJointDeviations(),
      deviation: deviation === null ? null : {
        basis: deviation.basis,
        jointMaxM: deviation.joint_max_m ?? null,
        markerMaxM: deviation.marker_max_m ?? null,
        jointCount: Object.keys(deviation.joints ?? {}).length,
        markers: deviation.markers ?? {},
      },
      referenceJointCount: Object.keys(referenceJoints).length,
      notesCollapsed: moreLines,
      // 층마다 씬에 실제로 올라간 객체 수. 같은 대상을 다시 그리면(기준을 넘기든 [평가]를 다시 누르든)
      // 이 수가 **변하지 않아야** 한다 — 변하면 이전 층을 지우지 않고 쌓은 것이다.
      layerObjectCounts: {
        query: countDescendants(layerGroups.query),
        reference: countDescendants(layerGroups.reference),
        corrected: countDescendants(layerGroups.corrected),
        corrections: countDescendants(correctionsGroup),
      },
      // 층마다 그려진 궤적 줄 수. 범례가 말하는 줄 수와 같아야 한다(서버 `legend_notes`).
      // 벽 위 스트로크: 그려진 연결 구간 수·실린더 수. 서버 `view.stroke`가 준 것과 같아야 한다.
      stroke: {
        runs: strokeGroup.runs,
        segments: strokeGroup.segments,
        serverRuns: Array.isArray(view.stroke?.runs) ? view.stroke.runs.length : 0,
        serverPoints: view.stroke?.point_count ?? null,
        serverBreaks: view.stroke?.break_count ?? null,
        serverMaxLiftMm: view.stroke?.max_lift_mm ?? null,
        serverMaxKeptLiftMm: view.stroke?.max_kept_lift_mm ?? null,
        note: view.stroke?.note ?? null,
      },
      // 기준 스트로크도 같은 규칙으로 그려졌는가(S2). 서버가 준 구간 수와 같아야 한다.
      referenceStroke: {
        runs: referenceStrokeGroup.runs,
        segments: referenceStrokeGroup.segments,
        serverRuns: Array.isArray(view.reference_stroke?.runs) ? view.reference_stroke.runs.length : 0,
        serverPoints: view.reference_stroke?.point_count ?? null,
      },
      strokeColors: { query: STROKE_COLOR.query, reference: STROKE_COLOR.reference },
      trackColors: { query: LAYER_COLOR.query, reference: LAYER_COLOR.reference },
      layerTrackLines: {
        query: countKind(layerGroups.query, "measured-layer-line"),
        reference: countKind(layerGroups.reference, "measured-layer-line"),
        corrected: countKind(layerGroups.corrected, "computed-layer-line"),
      },
      boneLengths,
      fit,
      framingRatio: null,
    });

    Object.assign(camState, defaultAngles(view));
    framing.baseRadius = null;
    frameTo("pose");
    // 휠 한계의 기준은 이 뷰의 기본 프리셋 거리다(프리셋 버튼을 눌러도 바뀌지 않는다).
    framing.baseRadius = camState.radius;
    debugLive.wheel = {
      notchPx: WHEEL_NOTCH_PX,
      notchFactor: WHEEL_NOTCH_FACTOR,
      baseRadius: framing.baseRadius,
      limits: radiusLimitsFor(framing.baseRadius),
    };
    layoutLabels();
  }

  function clear(message, tone) {
    dom.host.classList.remove("is-loading");
    resetScene();
    lastView = null;
    framing.baseRadius = null;
    setNote(dom.note, message ?? "패스를 고르고 평가를 누르면 여기에 뜹니다.", tone ?? "idle");
    dom.notes.replaceChildren();
    dom.notesMore.replaceChildren();
    dom.notesMore.hidden = true;
    setNote(dom.wallNote, "", undefined);
    setNote(dom.errors, "", undefined);
    renderLegend(dom.legend, null);
    dom.correctedNote.textContent = "";
    container.__view3dDebug = null;
  }

  return { render, clear, loading };
}

/** 3D 뷰를 `container`에 붙인다. three.js가 없거나 초기화가 실패하면 안내만 남기고 던지지 않는다. */
export function createSkeletonView(container) {
  if (!container) return { render() {}, clear() {}, loading() {} };
  if (typeof THREE === "undefined") {
    showUnavailable(container, "three.js를 불러오지 못해 3D 뷰 없이 숫자 패널만 씁니다(오프라인이거나 CDN 차단).");
    return { render() {}, clear() {}, loading() {} };
  }
  try {
    return mount(THREE, container);
  } catch (err) {
    console.error("[skeleton_view] 3D 뷰 초기화 실패", err);
    showUnavailable(container, "3D 뷰를 초기화하지 못했습니다 — 숫자 패널은 그대로 동작합니다.");
    return { render() {}, clear() {}, loading() {} };
  }
}
