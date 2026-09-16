// 서버 응답만으로 그린다. 지표 계산을 브라우저에서 다시 하지 않는다.
// 두 곳에서 계산하면 값이 갈라지고 어느 쪽이 맞는지 알 수 없게 된다.
//
// 숫자 표시 규칙: 화면에 보이는 숫자는 모두 서버가 Python 포맷으로 지은 문자열
// (`*_display`, `figure`, `text`)이다. JS는 숫자 원값을 막대 길이·좌표 계산에만 쓴다.
// Python과 JS는 반올림 동률 처리가 달라, 두 규칙이 한 화면에 섞이면 같은 교정이
// 목록과 3D에서 다른 숫자로 보일 수 있다.
//
// 화면 구성:
//   ① 고르기(대상 패스만 — 기준은 고를 수 없다, 전형기로 자동 고정)
//   ② 얼마나 다른가 (이 화면의 답 — 가장 크게)
//   ③ 교정 — 사람이 읽는 문장. ② 바로 아래, 3D 패널 바깥
//   ④ 개선율 · ④b 건 각도(참고값)
//   3D 뷰(넓은 창에서는 오른쪽 열)
//   ⑤ 접이식 세부 수치

import { createSkeletonView } from "./skeleton_view.js";

const $ = (id) => document.getElementById(id);

// 3D 뷰. three.js가 못 실렸거나(오프라인 등) 초기화가 실패해도 createSkeletonView는
// 절대 던지지 않고 조용한 안내로만 떨어진다 — 숫자 패널은 그 성패와 무관하게 돈다.
const skeletonView = createSkeletonView($("view3d"));

// 게이트가 비어 있는 이유. 판정 자리에 이유를 적어 "아직 재지 않았다"를 드러낸다.
const GATE_REASONS = {
  no_reference_distribution: " (레퍼런스 분포 없음 — 판정하지 않음)",
  not_wired: " (배선 전)",
};

// 부위 이름. 서버는 영문 리터럴("right"/"head"/"left")을 낸다. 이름표로만 쓰고
// 강조 규칙에는 쓰지 않는다(강조는 view.weights에서만 파생).
const POINT_LABELS = { right: "오른손", head: "머리", left: "왼손" };

// 개선율(분리)이 산출되지 않은 이유. `service.py::compare`의 `split_reason`과 1:1.
const SPLIT_REASONS = {
  insufficient_dates: "날짜가 둘로 갈리지 않아 산출하지 않았습니다",
  insufficient_samples: "반쪽 표본이 2개 미만이라 산출하지 않았습니다",
  no_split_reference: "실측 레퍼런스가 없어 산출하지 않았습니다",
};

const NORMALIZATION_LABELS = { measured: "측정된 분포", provisional: "임시" };
const NORMALIZATION_CLASS = { measured: "measured", provisional: "provisional-flag" };

// 비교를 만들지 않은 이유. 서버의 reason과 1:1이다.
const UNAVAILABLE_REASONS = {
  motion_key_mismatch: "동작 키가 달라 비교하지 않습니다.",
  insufficient_frames: "착탄점이 있는 분사 프레임이 모자라 지표를 만들지 못했습니다.",
  no_source: "실데이터 소스가 배선되지 않았습니다.",
  pass_not_found: "고른 패스를 원본에서 찾지 못했습니다.",
  no_expert_worker: "이 데이터셋에 전형기 작업자가 없습니다.",
};

// "정상적인 없음"(empty 톤)과 "시스템이 못 답했다"(error 톤)를 구별한다.
const EMPTY_REASONS = new Set([
  "motion_key_mismatch",
  "insufficient_frames",
  "no_expert_worker",
]);

/** `note` 엘리먼트에 4상태(아직 안 함/로딩 중/없음/오류) 중 하나를 건다. */
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

/** 서버 표시 문자열을 그대로 쓴다. 없으면 "—". 0이나 빈 문자열을 없음으로 보지 않도록 `??`. */
function shown(value) {
  return typeof value === "string" ? value : "—";
}

function el(tag, className, text) {
  const node = document.createElement(tag);
  if (className) node.className = className;
  if (text !== undefined && text !== null) node.textContent = text;
  return node;
}

/** 숫자 문자열 + 작은 단위. 단위는 .unit(14px dim)로 따로 둔다. */
function figureWithUnit(className, figure, unit) {
  const wrap = el("span", className);
  wrap.append(document.createTextNode(shown(figure)));
  if (typeof figure === "string" && unit) wrap.append(el("span", "unit", unit));
  return wrap;
}

// ---- ⑤ 상세: 실측 축 ----------------------------------------------------------

function renderMeasuredDetails(root, measured, view) {
  const display = measured.display ?? {};
  const gate = measured.gate ?? "—";
  const gateWhy = measured.gate ? "" : GATE_REASONS[measured.gate_reason] ?? "";

  const dl = root.querySelector("dl");
  dl.replaceChildren();
  const splitNote = (() => {
    if (!Object.hasOwn(measured, "split_reason")) return "";
    const reason = measured.split_reason ?? null;
    if (reason !== null) return SPLIT_REASONS[reason] ?? reason;
    const a = measured.split_n_a;
    const b = measured.split_n_b;
    return `A n=${typeof a === "number" ? a : "—"} / B n=${typeof b === "number" ? b : "—"}`;
  })();
  for (const [term, value] of [
    ["유사도", shown(display.score)],
    ["교정 후", shown(display.score_after)],
    ["개선율(분리)", `${shown(display.improvement_split)}  ${splitNote}`],
    ["개선율(동일)", shown(display.improvement_same)],
    ["REQ-19 게이트", `${gate}${gateWhy}`],
  ]) {
    dl.append(el("dt", null, term), el("dd", null, value));
  }

  const normEl = root.querySelector(".normalization");
  const norm = measured.normalization ?? null;
  normEl.textContent = `정규화: ${NORMALIZATION_LABELS[norm] ?? "—"}`;
  normEl.className = `normalization ${NORMALIZATION_CLASS[norm] ?? "unknown"}`;

  const granEl = root.querySelector(".granularity");
  const granularity = measured.granularity ?? null;
  const referenceN = measured.reference_n;
  granEl.textContent = granularity && typeof referenceN === "number"
    ? `입도 · 표본: ${granularity} · n=${referenceN}`
    : "입도 · 표본: —";

  const datesEl = root.querySelector(".reference-dates");
  const dates = measured.reference_dates ?? [];
  datesEl.textContent = dates.length > 0
    ? `레퍼런스 날짜 (${dates.length}개): ${dates.join(", ")}`
    : "레퍼런스 날짜: —";

  root.querySelector(".provisional").textContent =
    measured.normalization === "provisional"
      ? "σ와 점수는 잰 분포가 아니라 임시 산포(0.05 m)로 정규화한 값이다."
      : "";

  const bars = root.querySelector(".bars");
  bars.replaceChildren();
  const deltasDisplay = measured.deltas_display ?? [];
  (measured.deltas ?? []).forEach((delta, index) => {
    const shownDelta = deltasDisplay[index] ?? {};
    const item = el("li", "bar");
    const row = el("div", "bar-row");
    const backing = typeof delta.samples === "number" ? ` (n=${delta.samples})` : "";
    row.append(el("span", null, `${POINT_LABELS[delta.point] ?? delta.point}${backing}`));
    const track = el("span", "track");
    track.dataset.tier = severityTier(delta.sigma);
    track.style.width = `${typeof delta.sigma === "number" ? Math.min(100, (delta.sigma / 6) * 100) : 0}%`;
    row.append(track, el("span", "bar-sigma", shown(shownDelta.sigma_display)));
    item.append(row);
    bars.append(item);
  });

  const axesBody = root.querySelector(".issued-axes tbody");
  axesBody.replaceChildren();
  const axes = measured.axes ?? [];
  if (axes.length === 0) {
    const row = el("tr");
    const cell = el("td", null, "발부된 교정 없음");
    cell.colSpan = 4;
    row.append(cell);
    axesBody.append(row);
  }
  for (const entry of axes) {
    const row = el("tr");
    row.append(
      el("td", null, typeof entry.n === "number" ? String(entry.n) : "—"),
      el("td", null, shown(entry.trace)),
    );
    const figureCell = el("td");
    figureCell.append(figureWithUnit("issued-figure", entry.figure, "cm"));
    row.append(figureCell, el("td", null, shown(entry.sigma_display)));
    axesBody.append(row);
  }

  // 3D 아래에도 fit.notes 전부가 보인다. 여기에는 잔차 라벨까지 함께 싣는다.
  const notesList = root.querySelector(".fit-notes-all");
  notesList.replaceChildren();
  const fit = view?.fit ?? null;
  const lines = fit
    ? [...(fit.notes ?? []), fit.residual_label, fit.rotation_residual_label].filter((t) => typeof t === "string")
    : [view?.fit_note].filter((t) => typeof t === "string");
  for (const line of lines) notesList.append(el("li", null, line));
}

function renderJointDetails(root, joint) {
  root.querySelector(".note").textContent = joint?.note ?? "";
  const dl = root.querySelector("dl");
  dl.replaceChildren();
  const gateWhy = joint?.gate ? "" : GATE_REASONS[joint?.gate_reason] ?? "";
  dl.append(
    el("dt", null, "상태"), el("dd", null, joint?.status ?? "—"),
    el("dt", null, "REQ-19 게이트"), el("dd", null, `${joint?.gate ?? "—"}${gateWhy}`),
  );
}

function clearDetails() {
  for (const root of [$("measured"), $("joint")]) {
    root.querySelector("dl").replaceChildren();
    for (const sel of [".bars", ".issued-axes tbody", ".fit-notes-all"]) {
      const node = root.querySelector(sel);
      if (node) node.replaceChildren();
    }
  }
}

// ---- ② 얼마나 다른가 ------------------------------------------------------

function renderHero(measured, keys) {
  $("heroScore").textContent = shown(measured.display?.score);
  $("heroMotionKey").textContent = keys ?? "—";
  const n = measured.reference_n;
  const granularity = measured.granularity;
  $("heroSamples").textContent = typeof n === "number"
    ? `n=${n}${granularity ? ` (${granularity})` : ""}`
    : "—";

  const list = $("heroDeviations");
  list.replaceChildren();
  const deltasDisplay = measured.deltas_display ?? [];
  (measured.deltas ?? []).forEach((delta, index) => {
    const shownDelta = deltasDisplay[index] ?? {};
    const item = el("li");
    item.append(
      el("span", "deviation-name", POINT_LABELS[delta.point] ?? delta.point),
      figureWithUnit("deviation-value", shownDelta.deviation_display, "cm"),
      el("span", "deviation-sigma", shown(shownDelta.sigma_display)),
    );
    list.append(item);
  });
}

// ---- ③ 교정 문장 -----------------------------------------------------------

function renderCorrections(measured) {
  const axes = measured?.axes ?? [];
  // 건수의 출처는 measured.axes다 — view.corrections는 원점이 없으면 항목이 빠질 수 있다.
  $("correctionsCount").textContent = `${axes.length}건`;
  const list = $("correctionList");
  list.replaceChildren();
  const state = $("correctionsState");

  if (axes.length === 0) {
    setState(state, "empty", "발부할 교정이 없습니다.");
    return;
  }
  setState(state, "ok");

  const maxSigma = Math.max(1, ...axes.map((c) => (typeof c.sigma === "number" ? c.sigma : 0)));
  for (const entry of axes) {
    const item = el("li", "correction-item");
    item.dataset.n = String(entry.n);
    item.dataset.textMode = entry.text_mode ?? "";

    const badge = el("span", "num-badge", typeof entry.n === "number" ? String(entry.n) : "—");

    const sentence = el("span", "correction-text");
    if (typeof entry.text === "string") {
      sentence.textContent = entry.text;
    } else {
      // 문장을 짓지 못했으면 추적용 라벨을 문장 자리에 크게 쓰고 이유를 작게 붙인다.
      sentence.textContent = shown(entry.trace);
      if (entry.text_reason) {
        sentence.append(el("span", "correction-text-reason", ` 문장 없음(${entry.text_reason})`));
      }
    }

    const figure = figureWithUnit("correction-cm", entry.figure, "cm");
    const sigma = el("span", "correction-sigma", shown(entry.sigma_display));

    const traceLine = el("span", "correction-trace");
    const traceText = typeof entry.surface_angle_display === "string"
      ? `${shown(entry.trace)} · ${entry.surface_angle_display}`
      : shown(entry.trace);
    traceLine.append(el("span", "correction-trace-text", traceText));
    const barTrack = el("span", "correction-bar-track");
    const barFill = el("span", "correction-bar-fill");
    barFill.dataset.tier = severityTier(entry.sigma);
    barFill.style.width = `${typeof entry.sigma === "number" ? Math.min(100, (entry.sigma / maxSigma) * 100) : 0}%`;
    barTrack.append(barFill);
    traceLine.append(barTrack);

    item.append(badge, sentence, figure, sigma, traceLine);
    list.append(item);
  }
}

// ---- ④ 개선율 --------------------------------------------------------------

function renderImprovement(measured) {
  const display = measured.display ?? {};
  const labels = measured.improvement_labels ?? {};
  const score = $("improvementScore");
  score.replaceChildren(
    el("span", "improvement-caption", "유사도"),
    el("span", "improvement-value", shown(display.score)),
    el("span", "improvement-arrow", "→"),
    el("span", "improvement-caption", "교정 후"),
    el("span", "improvement-value", shown(display.score_after)),
  );

  const rows = $("improvementRows");
  rows.replaceChildren();
  const splitReason = measured.split_reason ?? null;
  for (const [key, value, label] of [
    ["same", display.improvement_same, labels.same],
    ["split", display.improvement_split, labels.split],
  ]) {
    const row = el("div", "improvement-row");
    row.dataset.key = key;
    const dt = el("dt", "improvement-value", shown(value));
    const dd = el("dd", "improvement-row-label", typeof label === "string" ? label : "");
    if (key === "split" && splitReason !== null) {
      dd.append(el("span", "improvement-reason", ` — ${SPLIT_REASONS[splitReason] ?? splitReason}`));
    }
    row.append(dt, dd);
    rows.append(row);
  }
}

function clearImprovement() {
  $("improvementScore").textContent = "—";
  $("improvementRows").replaceChildren();
}

// ---- ④b 건 각도(참고값) ----------------------------------------------------

function renderOrientation(orientation) {
  const section = $("orientationSection");
  const body = $("orientationTable").querySelector("tbody");
  body.replaceChildren();
  const notes = $("orientationNotes");
  notes.replaceChildren();
  if (!orientation) {
    $("orientationTitle").textContent = "건 각도";
    $("orientationHeaderNote").textContent = "이 비교에는 각도 참고값이 없습니다.";
    section.dataset.state = "empty";
    return;
  }
  section.dataset.state = "ok";
  $("orientationTitle").textContent = orientation.title ?? "건 각도";
  $("orientationHeaderNote").textContent = orientation.header_note ?? "";
  for (const item of orientation.items ?? []) {
    const row = el("tr");
    row.dataset.key = item.key ?? "";
    row.append(el("th", null, item.label ?? item.key ?? "—"));
    row.lastChild.scope = "row";
    if (item.reason) {
      const cell = el("td", "orientation-reason", item.reason_text ?? item.reason);
      cell.colSpan = 2;
      row.append(cell);
    } else {
      row.append(
        el("td", "orientation-value", shown(item.display?.query)),
        el("td", "orientation-value", shown(item.display?.reference)),
      );
    }
    body.append(row);
  }
  // 관절 각도를 왜 안 싣는지(joint_angles_note)는 화면에 실제로 떠야 한다 — 서버만 알고 있으면
  // 표에서 빠진 이유를 아무도 못 읽는다.
  for (const text of [orientation.weight_note, orientation.joint_angles_note, orientation.excluded_note]) {
    if (typeof text === "string" && text) notes.append(el("li", null, text));
  }
}

/** 첫 화면에 남는 관절 축 평가 **요약**. 문장·상태·값 문자열은 전부 서버가 준 것이다. */
function renderJointAxesSummary(box, summary) {
  if (!summary) {
    box.hidden = true;
    return;
  }
  box.hidden = false;
  $("jointAxesSummaryTitle").textContent = summary.title ?? "관절 축 평가";
  $("jointAxesSummaryText").textContent = summary.counts_text ?? "";
  const chips = $("jointAxesChips");
  for (const chip of summary.chips ?? []) {
    const item = el("li", "joint-axis-chip");
    item.dataset.key = chip.key ?? "";
    item.dataset.status = chip.status ?? "";
    item.append(
      el("span", "joint-axis-chip-joint", chip.joint ?? "—"),
      el("span", "joint-axis-chip-status", chip.status_text ?? "—"),
    );
    const detail = chip.value ?? chip.reason_text;
    if (detail) item.append(el("span", "joint-axis-chip-value", detail));
    chips.append(item);
  }
}

/** 관절 축 평가표. 값은 서버 display 문자열만, 측정 없는 관절은 서버 사유 문장만 싣는다. */
function renderJointAxes(block) {
  const section = $("jointAxes");
  const summaryBox = $("jointAxesSummary");
  const body = $("jointAxesTable").querySelector("tbody");
  body.replaceChildren();
  $("jointAxesChips").replaceChildren();
  if (!block) {
    section.hidden = true;
    summaryBox.hidden = true;
    return;
  }
  section.hidden = false;
  renderJointAxesSummary(summaryBox, block.summary ?? null);
  $("jointAxesTitle").textContent = block.title ?? "관절 축 평가";
  $("jointAxesNote").textContent = block.note ?? "";
  for (const row of block.rows ?? []) {
    const tr = el("tr");
    tr.dataset.key = row.key ?? "";
    tr.dataset.status = row.status ?? "";
    const head = el("th", null, row.joint ?? "—");
    head.scope = "row";
    tr.append(head, el("td", "joint-axis-status", row.status_text ?? "—"));
    if (row.display && !row.reason) {
      const label = el("td", "joint-axis-label", row.label ?? "");
      tr.append(label, el("td", "orientation-value", shown(row.display.query)), el("td", "orientation-value", shown(row.display.reference)));
    } else {
      const cell = el("td", "orientation-reason", row.reason_text ?? "—");
      cell.colSpan = 3;
      tr.append(cell);
    }
    body.append(tr);
  }
}

// ---- ① 고르기: 패스 목록 --------------------------------------------------

const listed = { query: [] };
// 기준(레퍼런스) 후보 상태. 기준은 고를 수 없다 — 이 목록에서 자동으로 고른 것을
// [다른 기준 ▸]으로 순환할 뿐이다.
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
  // 표시된 기준이 곧 계산에 쓰일 기준이어야 한다.
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
  clearImprovement();
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

  // 몸 기준 축의 요 출처. 값 자체는 서버가 표시 문자열로 주지 않아 출처만 적는다.
  const yaw = typeof payload.yaw_assumed === "number"
    ? payload.yaw_source === "measured"
      ? "  — 몸 기준 축은 HMD 실측 요"
      : "  — 요를 못 읽어 정면 고정 대체값"
    : "";
  $("keys").textContent =
    `질의 ${payload.query_key ?? "-"} / 레퍼런스 ${payload.reference_key ?? "-"}${yaw}`;

  if (payload.status !== "ok") {
    const isEmptyReason = EMPTY_REASONS.has(payload.reason);
    const text = UNAVAILABLE_REASONS[payload.reason] ?? "비교하지 않습니다.";
    showHeroState(isEmptyReason ? "empty" : "error", text);
    setState($("correctionsState"), isEmptyReason ? "empty" : "error", text);
    $("correctionsCount").textContent = "";
    clearImprovement();
    renderOrientation(null);
    clearDetails();
    skeletonView.clear(text, isEmptyReason ? "empty" : "error");
    return;
  }

  showHeroBody();
  renderHero(payload.measured, payload.query_key);
  renderCorrections(payload.measured);
  renderImprovement(payload.measured);
  renderOrientation(payload.measured.orientation ?? null);
  renderJointAxes(payload.measured.joint_axes ?? null);
  renderMeasuredDetails($("measured"), payload.measured, payload.view);
  renderJointDetails($("joint"), payload.joint);
  // 3D 뷰 실패가 숫자 패널을 끌고 내려가면 안 된다 — 위 렌더는 이미 끝났다.
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
