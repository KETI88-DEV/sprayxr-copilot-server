// 서버 응답만으로 그린다. 지표 계산을 브라우저에서 다시 하지 않는다.
// 두 곳에서 계산하면 값이 갈라지고 어느 쪽이 맞는지 알 수 없게 된다.
//
// 화면 구성은 브리프(task-11-brief.md)가 고정한 것을 그대로 따른다:
//   ① 고르기(대상 패스만 — 기준은 고를 수 없다, 전형기로 자동 고정)
//   ② 얼마나 다른가 (이 화면의 답 — 가장 크게)
//   ③ 무엇을 고치나 (3D 뷰 + 발부된 교정 목록)
//   ④ 개선율
//   ⑤ 접이식 세부 수치

import { createSkeletonView } from "./skeleton_view.js";

const $ = (id) => document.getElementById(id);

// 3D 뷰. three.js가 못 실렸거나(오프라인 등) 초기화가 실패해도 createSkeletonView는
// 절대 던지지 않고 조용한 안내로만 떨어진다 — 아래 run()이 이 객체를 어떻게 쓰든
// 숫자 패널(renderAxis 등)은 그 성패와 무관하게 그대로 동작한다(요구사항 3).
const skeletonView = createSkeletonView($("view3d"));

// 게이트가 비어 있는 이유. 판정 자리에 이유를 적어 "아직 재지 않았다"를 드러낸다.
const GATE_REASONS = {
  no_reference_distribution: " (레퍼런스 분포 없음 — 판정하지 않음)",
  not_wired: " (미배선)",
};

// 구조화 근거의 축 이름. 서버는 영문 리터럴("wall"/"vertical"/"lateral")을 낸다.
const AXIS_LABELS = { wall: "벽", vertical: "수직", lateral: "진행축" };

// 발부한 축 표의 점 이름. 서버는 영문 리터럴("right"/"head"/"left")을 낸다.
const POINT_LABELS = { right: "오른손", head: "머리", left: "왼손" };

// 개선율(분리)이 산출되지 않은 이유. `service.py::compare`의 `split_reason` 문자열과
// 1:1이다 — 기본값은 "no_split_reference"다("no_reference_distribution"은 게이트가
// 쓰는 다른 이름이라 여기서 쓰면 값이 안 맞는다).
const SPLIT_REASONS = {
  insufficient_dates: "날짜가 둘로 갈리지 않음",
  insufficient_samples: "반쪽 표본이 2개 미만",
  no_split_reference: "실측 레퍼런스 없음",
};

// 정규화 딱지. measured/provisional 둘 다 아니면(관절 축처럼 아예 배선되지 않은
// 경우) 값을 지어내지 않고 "—"로 낸다.
const NORMALIZATION_LABELS = { measured: "측정된 분포", provisional: "임시" };
const NORMALIZATION_CLASS = { measured: "measured", provisional: "provisional-flag" };

// 비교를 만들지 않은 이유. 서버의 reason과 1:1이다 — 각각 다른 문장으로 읽혀야
// 한다(브리프: "레퍼런스 없음/동작키 불일치/데이터 없음이 각각 달리 읽혀야 한다").
const UNAVAILABLE_REASONS = {
  motion_key_mismatch: "동작 키가 달라 비교하지 않습니다.",
  insufficient_frames: "착탄점이 있는 분사 프레임이 모자라 지표를 만들지 못했습니다.",
  no_source: "실데이터 소스가 배선되지 않았습니다.",
  pass_not_found: "고른 패스를 원본에서 찾지 못했습니다.",
  no_expert_worker: "이 데이터셋에 전형기 작업자가 없습니다.",
};

// unavailable 사유 중 "정상적인 없음"(계산은 됐지만 결과가 없다)과 "시스템이 못
// 답했다"를 구별한다 — 전자는 empty 톤(차분), 후자는 error 톤(빨강)으로 낸다.
const EMPTY_REASONS = new Set([
  "motion_key_mismatch",
  "insufficient_frames",
  "no_expert_worker",
]);

/** `note` 엘리먼트에 4상태(아직 안 함/로딩 중/없음/오류) 중 하나를 건다.
 * "ok"는 note 자체를 숨기고(내용은 다른 곳에서 그린다), 나머지는 문구와 함께 보인다. */
function setState(note, tone, text) {
  if (tone === "ok") {
    note.hidden = true;
    note.removeAttribute("data-tone");
    return;
  }
  note.hidden = false;
  note.dataset.tone = tone;
  note.textContent = text ?? "";
}

function severityTier(sigma) {
  if (typeof sigma !== "number") return "unknown";
  if (sigma <= 1) return "ok";
  if (sigma <= 2) return "warn";
  return "bad";
}

function severityColor(sigma) {
  const tier = severityTier(sigma);
  return tier === "unknown" ? "var(--text-faint)" : `var(--${tier})`;
}

function fmtCm(value) {
  // 0도 값이다. `??`/`typeof` 없이 `||`를 쓰면 진짜 0cm 편차가 "값 없음"으로 보인다.
  return typeof value === "number" ? `${value >= 0 ? "+" : ""}${value.toFixed(1)}cm` : "—";
}

function fmtScore(value) {
  return typeof value === "number" ? value.toFixed(3) : "—";
}

function renderAxis(root, payload) {
  const note = root.querySelector(".note");
  if (note) note.textContent = payload.note ?? "";

  const num = (value) => (typeof value === "number" ? value.toFixed(3) : "—");

  const gate = payload.gate ?? "—";
  const gateWhy = payload.gate ? "" : GATE_REASONS[payload.gate_reason] ?? "";

  root.querySelector("dl").innerHTML = `
    <dt>유사도</dt><dd>${num(payload.score)}</dd>
    <dt>교정 후</dt><dd>${num(payload.score_after)}</dd>
    <dt>개선율(분리)</dt><dd>${num(payload.improvement?.split)} <span class="split-note"></span></dd>
    <dt>개선율(동일)</dt><dd>${num(payload.improvement?.same)}</dd>
    <dt>REQ-19 게이트</dt><dd>${gate}${gateWhy}</dd>`;

  const splitNote = root.querySelector(".split-note");
  if (splitNote) {
    const a = payload.split_n_a;
    const b = payload.split_n_b;
    if (!Object.hasOwn(payload, "split_reason")) {
      splitNote.textContent = "";
    } else {
      const reason = payload.split_reason ?? null;
      splitNote.textContent =
        reason !== null
          ? `산출 안 됨: ${SPLIT_REASONS[reason] ?? reason}`
          : `A n=${typeof a === "number" ? a : "—"} / B n=${typeof b === "number" ? b : "—"}`;
    }
  }

  const normEl = root.querySelector(".normalization");
  if (normEl) {
    const norm = payload.normalization ?? null;
    normEl.textContent = `정규화: ${NORMALIZATION_LABELS[norm] ?? "—"}`;
    normEl.className = `normalization ${NORMALIZATION_CLASS[norm] ?? "unknown"}`;
  }

  const granEl = root.querySelector(".granularity");
  if (granEl) {
    const granularity = payload.granularity ?? null;
    const referenceN = payload.reference_n;
    granEl.textContent = granularity && typeof referenceN === "number"
      ? `입도 · 표본: ${granularity} · n=${referenceN}`
      : "입도 · 표본: —";
  }

  const datesEl = root.querySelector(".reference-dates");
  if (datesEl) {
    const dates = payload.reference_dates ?? [];
    datesEl.textContent = dates.length > 0
      ? `레퍼런스 날짜 (${dates.length}개): ${dates.join(", ")}`
      : "레퍼런스 날짜: —";
  }

  const provisional = root.querySelector(".provisional");
  if (provisional) {
    provisional.textContent =
      payload.normalization === "provisional"
        ? "σ와 점수는 잰 분포가 아니라 임시 산포(0.05 m)로 정규화한 값이다."
        : "";
  }

  const axisSigma = (delta, axis) => {
    const value = delta.axis_sigmas?.[axis];
    return typeof value === "number" ? `${value.toFixed(2)}σ` : "—";
  };

  const bars = root.querySelector(".bars");
  bars.replaceChildren();
  for (const delta of payload.deltas ?? []) {
    const item = document.createElement("li");
    item.className = "bar";
    const width = Math.min(100, (delta.sigma / 6) * 100);
    const backing = typeof delta.samples === "number" ? ` (n=${delta.samples})` : "";
    item.innerHTML = `<div class="bar-row">
        <span>${delta.point}${backing}</span>
        <span class="track" style="background:${severityColor(delta.sigma)};width:${width}%"></span>
        <span>${delta.sigma.toFixed(2)}σ</span>
      </div>
      <span class="bar-axes">벽 ${axisSigma(delta, "wall")} · 수직 ${axisSigma(delta, "vertical")} · 진행축 ${axisSigma(delta, "lateral")}</span>`;
    bars.append(item);
  }

  const stanceEl = root.querySelector(".stance");
  if (stanceEl) {
    const stance = payload.stance ?? null;
    stanceEl.hidden = stance === null;
    stanceEl.textContent = stance ? stance.text : "";
  }

  const axesBody = root.querySelector(".issued-axes tbody");
  if (axesBody) {
    axesBody.replaceChildren();
    const axes = payload.axes ?? [];
    if (axes.length === 0) {
      const row = document.createElement("tr");
      const cell = document.createElement("td");
      cell.colSpan = 4;
      cell.textContent = "발부된 교정 없음";
      row.append(cell);
      axesBody.append(row);
    }
    for (const entry of axes) {
      const row = document.createElement("tr");
      for (const text of [
        POINT_LABELS[entry.point] ?? entry.point,
        AXIS_LABELS[entry.axis] ?? entry.axis,
        typeof entry.magnitude_cm === "number" ? fmtCm(entry.magnitude_cm) : "—",
        typeof entry.sigma === "number" ? `${entry.sigma.toFixed(2)}σ` : "—",
      ]) {
        const cell = document.createElement("td");
        cell.textContent = text;
        row.append(cell);
      }
      axesBody.append(row);
    }
  }

  const list = root.querySelector(".corrections");
  list.replaceChildren();
  for (const line of payload.corrections ?? []) {
    const item = document.createElement("li");
    item.textContent = line;
    list.append(item);
  }

  const findingEl = root.querySelector(".finding");
  if (findingEl) {
    const finding = payload.finding ?? null;
    findingEl.textContent = finding
      ? `근거: ${AXIS_LABELS[finding.axis] ?? finding.axis} 축 `
        + `${Math.abs(finding.magnitude_cm).toFixed(1)}cm${finding.direction ? ` ${finding.direction} 방향` : ""} `
        + `· σ=${finding.sigma.toFixed(2)} · 표본 n=${finding.samples}`
      : "";
  }
}

// ---- ② 얼마나 다른가 ------------------------------------------------------

function renderHero(measured, keys) {
  $("heroScore").textContent = fmtScore(measured.score);
  $("heroMotionKey").textContent = keys ?? "—";
  const n = measured.reference_n;
  const granularity = measured.granularity;
  $("heroSamples").textContent = typeof n === "number"
    ? `n=${n}${granularity ? ` (${granularity})` : ""}`
    : "—";

  const list = $("heroDeviations");
  list.replaceChildren();
  for (const delta of measured.deltas ?? []) {
    const item = document.createElement("li");
    const name = document.createElement("span");
    name.className = "deviation-name";
    name.textContent = POINT_LABELS[delta.point] ?? delta.point;
    const value = document.createElement("span");
    value.className = "deviation-value";
    // 0cm도 값이다 — `??`(typeof 검사)를 쓴다.
    value.textContent = typeof delta.deviation_cm === "number" ? `${delta.deviation_cm.toFixed(1)}cm` : "—";
    item.append(name, value);
    list.append(item);
  }
}

// ---- ③ 발부된 교정 목록 ---------------------------------------------------

const NUMERALS = ["①", "②", "③", "④", "⑤", "⑥", "⑦", "⑧", "⑨"];

function renderCorrections(view) {
  const corrections = view?.corrections ?? [];
  $("correctionsCount").textContent = corrections.length > 0 ? `${corrections.length}건` : "";
  const list = $("correctionList");
  list.replaceChildren();
  const state = $("correctionsState");

  if (corrections.length === 0) {
    setState(state, "empty", "발부된 교정이 없습니다 — 임계를 넘는 이탈이 없습니다.");
    return;
  }
  setState(state, "ok");

  const maxSigma = Math.max(1, ...corrections.map((c) => (typeof c.sigma === "number" ? c.sigma : 0)));
  corrections.forEach((entry, index) => {
    const item = document.createElement("li");
    item.className = "correction-item";
    const tier = severityTier(entry.sigma);
    const width = typeof entry.sigma === "number" ? Math.min(100, (entry.sigma / maxSigma) * 100) : 0;

    const badge = document.createElement("span");
    badge.className = "correction-badge";
    badge.textContent = NUMERALS[index] ?? `#${index + 1}`;

    const text = document.createElement("span");
    text.className = "correction-text";
    const point = POINT_LABELS[entry.point] ?? entry.point;
    const axis = AXIS_LABELS[entry.axis] ?? entry.axis;
    text.textContent = `${point} · ${axis}${entry.direction ? ` (${entry.direction})` : ""}`;

    const figures = document.createElement("span");
    figures.className = "correction-figures";
    const cm = document.createElement("span");
    cm.className = "correction-cm";
    cm.textContent = fmtCm(entry.magnitude_cm);
    const sigma = document.createElement("span");
    sigma.className = "correction-sigma";
    sigma.textContent = typeof entry.sigma === "number" ? `σ${entry.sigma.toFixed(2)}` : "σ—";
    figures.append(cm, sigma);

    const barTrack = document.createElement("span");
    barTrack.className = "correction-bar-track";
    const barFill = document.createElement("span");
    barFill.className = "correction-bar-fill";
    barFill.dataset.tier = tier;
    barFill.style.width = `${width}%`;
    barTrack.append(barFill);

    item.append(badge, text, figures, barTrack);
    list.append(item);
  });
}

// ---- ④ 개선율 --------------------------------------------------------------

function renderImprovement(measured) {
  const el = $("improvementValues");
  const before = measured.score;
  const after = measured.score_after;
  if (typeof before !== "number" || typeof after !== "number") {
    el.textContent = "—";
    return;
  }
  const delta = after - before;
  const tone = delta >= 0 ? "ok" : "bad";
  el.innerHTML = `${fmtScore(before)}<span class="arrow">→</span>${fmtScore(after)}` +
    `<span class="improvement-delta" data-tone="${tone}">${delta >= 0 ? "+" : ""}${delta.toFixed(3)}</span>`;
}

// ---- ① 고르기: 패스 목록 --------------------------------------------------

const listed = { query: [] };
// 기준(레퍼런스) 후보 상태. 기준은 고를 수 없다(브리프) — 이 목록에서 자동으로
// 고른 것을 [다른 기준 ▸]으로 순환할 뿐, 다른 작업자·레벨로 보낼 길이 없다.
const referenceState = { phase: "idle", candidates: [], index: 0, message: "" };

function params() {
  return {
    uv_grid: Number($("uvGrid").value),
    lateral_deg: Number($("lateralDeg").value),
    reach_m: Number($("reachM").value),
    samples: Number($("samples").value),
  };
}

function selectedQuery() {
  const select = $("queryPass");
  const entry = listed.query[Number(select.value)];
  return entry ?? null;
}

function passRef(entry) {
  return {
    worker: entry.worker,
    date: entry.date,
    level: entry.level,
    session: entry.session,
    pass_index: entry.pass_index,
  };
}

function optionLabel(entry) {
  const length = Math.round(entry.length_mm);
  return `${entry.date} L${entry.level} ${entry.session} #${entry.pass_index}`
    + `  (${entry.frames}프레임 ${length}mm)  ${entry.motion_key}`;
}

function fillQueryPasses() {
  const select = $("queryPass");
  select.replaceChildren();
  for (const [index, entry] of listed.query.entries()) {
    const option = document.createElement("option");
    option.value = String(index);
    option.textContent = optionLabel(entry);
    select.append(option);
  }
  select.disabled = listed.query.length === 0;
}

async function loadQueryPasses() {
  const worker = $("queryWorker").value;
  const status = $("queryStatus");
  if (!worker) return;
  const query = new URLSearchParams({
    worker,
    limit: $("sessionLimit").value,
    uv_grid: String(params().uv_grid),
    lateral_deg: String(params().lateral_deg),
    reach_m: String(params().reach_m),
  });
  const level = $("queryLevel").value;
  if (level) query.set("level", level);

  setState(status, "loading", "패스 목록을 불러오는 중…");
  $("run").disabled = true;
  let payload;
  try {
    const response = await fetch(`/api/passes?${query}`);
    payload = await response.json();
  } catch (err) {
    listed.query = [];
    fillQueryPasses();
    setState(status, "error", `패스 목록을 불러오지 못했습니다 — ${err}`);
    return;
  }
  if (payload.status !== "ok") {
    listed.query = [];
    fillQueryPasses();
    setState(status, "empty", UNAVAILABLE_REASONS[payload.reason] ?? "목록을 불러올 수 없습니다.");
    return;
  }

  const levelSelect = $("queryLevel");
  if (levelSelect.dataset.worker !== worker) {
    levelSelect.dataset.worker = worker;
    levelSelect.replaceChildren(new Option("전체", ""));
    for (const value of payload.levels) levelSelect.append(new Option(`레벨 ${value}`, value));
    levelSelect.value = level;
  }

  listed.query = payload.passes;
  fillQueryPasses();

  if (payload.passes.length === 0) {
    setState(status, "empty", "이 조건에 맞는 패스가 없습니다.");
  } else {
    // "ok"지만 정보 줄로 계속 보인다(4상태의 ok는 "숨김"이 기본이나, 이 줄은 몇 개를
    // 찾았는지 알려 주는 요약이라 남겨 둔다) — 톤은 중립(idle)으로 낮춘다.
    const scanned = `세션 ${payload.sessions_scanned}/${payload.sessions_total}`
      + (payload.truncated ? " (잘림 — 한도를 올리면 더 훑는다)" : "");
    setState(status, "idle", `${scanned} · 패스 ${payload.passes.length}개`);
  }
  await fetchReference();
}

async function loadWorkers() {
  const payload = await (await fetch("/api/workers")).json();
  const select = $("queryWorker");
  select.replaceChildren();
  for (const worker of payload.workers) select.append(new Option(worker, worker));
  if (payload.workers.length === 0) {
    setState($("queryStatus"), "empty", UNAVAILABLE_REASONS.no_source);
    return;
  }
  await loadQueryPasses();
}

// ---- ① 고르기: 기준(전형기) 자동 매칭 -------------------------------------
//
// `referenceState.phase`는 4상태 그대로다: "idle"(아직 안 함) · "loading"(로딩 중)
// · "ready"(있음 — candidates.length일 수도 있고 0일 수도 있다) · "error"(오류).
// "없음"은 phase "ready"이면서 candidates가 빈 배열인 경우로 표현한다 — 정상적으로
// 찾아봤는데 없는 것이지 실패가 아니다.

function renderReferenceLine() {
  const line = $("referenceLine");
  const valueEl = $("referenceValue");
  const cycleBtn = $("cycleReference");
  const cycleNote = $("referenceCycleNote");
  const { phase, candidates, index, message } = referenceState;

  cycleBtn.hidden = true;
  cycleNote.textContent = "";
  $("run").disabled = true;

  if (phase === "idle") {
    line.dataset.tone = "idle";
    valueEl.textContent = "패스를 고르면 전형기 기준을 자동으로 찾습니다.";
    return;
  }
  if (phase === "loading") {
    line.dataset.tone = "loading";
    valueEl.textContent = "전형기 기준을 찾는 중…";
    return;
  }
  if (phase === "error") {
    line.dataset.tone = "error";
    valueEl.textContent = message || "기준을 찾지 못했습니다.";
    return;
  }
  // phase === "ready"
  if (candidates.length === 0) {
    line.dataset.tone = "empty";
    valueEl.textContent = message || "이 동작 키와 일치하는 전형기 패스가 없습니다.";
    return;
  }
  const chosen = candidates[index];
  line.dataset.tone = "ok";
  valueEl.textContent = `전형기 · L${chosen.level} · ${chosen.date} ${chosen.session} #${chosen.pass_index} (자동)`;
  cycleBtn.hidden = candidates.length <= 1;
  cycleNote.textContent = candidates.length > 1 ? `${candidates.length}개 중 ${index + 1}번째` : "";
  $("run").disabled = false;
}

async function fetchReference() {
  const query = selectedQuery();
  if (query === null) {
    Object.assign(referenceState, { phase: "idle", candidates: [], index: 0, message: "" });
    renderReferenceLine();
    return;
  }
  referenceState.phase = "loading";
  renderReferenceLine();

  const q = passRef(query);
  const search = new URLSearchParams({
    worker: q.worker,
    date: q.date,
    level: q.level,
    session: q.session,
    pass_index: String(q.pass_index),
    uv_grid: String(params().uv_grid),
    lateral_deg: String(params().lateral_deg),
    reach_m: String(params().reach_m),
    samples: String(params().samples),
  });
  let payload;
  try {
    const response = await fetch(`/api/reference?${search}`);
    payload = await response.json();
  } catch (err) {
    Object.assign(referenceState, {
      phase: "error", candidates: [], index: 0,
      message: `기준 후보를 불러오지 못했습니다 — ${err}`,
    });
    renderReferenceLine();
    return;
  }
  if (payload.status !== "ok") {
    // "없음"(동작 키 불일치·전형기 없음 — 정상)과 "오류"(패스 자체를 못 읽음 등)를
    // 구별한다: 전자는 phase "ready"에 빈 candidates로, 후자는 phase "error"로 낸다.
    const isEmptyReason = EMPTY_REASONS.has(payload.reason);
    Object.assign(referenceState, {
      phase: isEmptyReason ? "ready" : "error",
      candidates: [],
      index: 0,
      message: UNAVAILABLE_REASONS[payload.reason] ?? "기준을 찾지 못했습니다.",
    });
    renderReferenceLine();
    return;
  }
  Object.assign(referenceState, {
    phase: "ready",
    candidates: payload.candidates ?? [],
    index: 0,
    message: "",
  });
  renderReferenceLine();
}

function cycleReference() {
  if (referenceState.candidates.length <= 1) return;
  referenceState.index = (referenceState.index + 1) % referenceState.candidates.length;
  renderReferenceLine();
  // 표시된 기준이 곧 계산에 쓰일 기준이어야 한다 — 이미 결과가 떠 있다면 새
  // 기준으로 다시 계산해 화면과 실제로 쓰인 기준이 갈리지 않게 한다.
  if (!$("heroBody").hidden) void run();
}

// ---- ②③④ 비교 실행 --------------------------------------------------------

function showHeroState(tone, text) {
  $("heroEmpty").hidden = true;
  $("heroBody").hidden = true;
  setState($("heroState"), tone, text);
}

function showHeroBody() {
  $("heroEmpty").hidden = true;
  $("heroBody").hidden = false;
  setState($("heroState"), "ok");
}

async function run() {
  const query = selectedQuery();
  const reference = referenceState.candidates[referenceState.index];
  if (query === null || reference === null || reference === undefined) {
    showHeroState("empty", "먼저 대상 패스를 고르고, 전형기 기준이 자동으로 잡히길 기다리세요.");
    return;
  }

  $("run").disabled = true;
  $("run").textContent = "평가 중…";
  showHeroState("loading", "비교를 계산하는 중…");
  setState($("correctionsState"), "loading", "비교를 계산하는 중…");
  $("correctionList").replaceChildren();
  $("improvementValues").textContent = "—";
  skeletonView.loading();

  let payload;
  try {
    const response = await fetch("/api/compare", {
      method: "POST",
      headers: { "content-type": "application/json" },
      body: JSON.stringify({
        query: passRef(query),
        reference: passRef(reference),
        ...params(),
      }),
    });
    payload = await response.json();
  } catch (err) {
    showHeroState("error", `비교 요청이 실패했습니다 — ${err}`);
    setState($("correctionsState"), "error", "비교 요청이 실패했습니다.");
    skeletonView.clear("비교 요청이 실패했습니다.", "error");
    $("run").disabled = false;
    $("run").textContent = "평가";
    return;
  }
  $("run").disabled = false;
  $("run").textContent = "평가";

  // 몸 기준 축(진행 방향·몸 기준 착탄 위치)이 선 요. HMD 회전에서 뽑은 실측값이면
  // 그 값을 그대로 적고, 회전을 못 읽어 대체값으로 떨어졌으면 그렇다고 적는다.
  const yaw =
    typeof payload.yaw_assumed === "number"
      ? payload.yaw_source === "measured"
        ? `  — 몸 기준 축은 실측 요 ${((payload.yaw_assumed * 180) / Math.PI).toFixed(1)}°`
        : `  — 요를 못 읽어 정면 고정(yaw=${payload.yaw_assumed}) 대체값`
      : "";
  $("keys").textContent =
    `질의 ${payload.query_key ?? "-"} / 레퍼런스 ${payload.reference_key ?? "-"}${yaw}`;

  if (payload.status !== "ok") {
    const isEmptyReason = EMPTY_REASONS.has(payload.reason);
    const text = UNAVAILABLE_REASONS[payload.reason] ?? "비교하지 않습니다.";
    showHeroState(isEmptyReason ? "empty" : "error", text);
    setState($("correctionsState"), isEmptyReason ? "empty" : "error", text);
    $("correctionsCount").textContent = "";
    $("improvementValues").textContent = "—";
    renderAxis($("measured"), {});
    renderAxis($("joint"), {});
    skeletonView.clear(text, isEmptyReason ? "empty" : "error");
    return;
  }

  showHeroBody();
  renderHero(payload.measured, payload.query_key);
  renderCorrections(payload.view);
  renderImprovement(payload.measured);
  renderAxis($("measured"), payload.measured);
  renderAxis($("joint"), payload.joint);
  // 3D 뷰 실패가 숫자 패널을 끌고 내려가면 안 된다(요구사항 3) — 위 렌더 호출들은
  // 이미 끝났고, 그 결과와 무관하게 이 호출만 실패할 수 있게 따로 감싼다.
  try {
    skeletonView.render(payload);
  } catch (err) {
    console.error("[skeleton_view] 렌더링 실패 — 숫자 패널은 그대로 유지된다.", err);
  }
}

// ---- 배선 -------------------------------------------------------------------

$("queryWorker").addEventListener("change", () => {
  $("queryLevel").dataset.worker = "";
  void loadQueryPasses();
});
$("queryLevel").addEventListener("change", () => void loadQueryPasses());
$("queryPass").addEventListener("change", () => void fetchReference());
$("cycleReference").addEventListener("click", () => cycleReference());
$("reload").addEventListener("click", () => void loadQueryPasses());
$("run").addEventListener("click", () => void run());

void loadWorkers();
