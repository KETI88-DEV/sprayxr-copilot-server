// 서버 응답만으로 그린다. 지표 계산을 브라우저에서 다시 하지 않는다.
// 두 곳에서 계산하면 값이 갈라지고 어느 쪽이 맞는지 알 수 없게 된다.

const $ = (id) => document.getElementById(id);

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

// 비교를 만들지 않은 이유. 서버의 reason과 1:1이다.
const UNAVAILABLE_REASONS = {
  motion_key_mismatch: "동작 키가 달라 비교하지 않습니다",
  insufficient_frames: "착탄점이 있는 분사 프레임이 모자라 지표를 만들지 못했습니다",
  no_source: "실데이터 소스가 배선되지 않았습니다",
  pass_not_found: "고른 패스를 원본에서 찾지 못했습니다",
};

// 양쪽 패널의 현재 목록. 옵션 값은 인덱스라 여기서 원본 항목을 되찾는다.
const listed = { query: [], reference: [] };

function params() {
  return {
    uv_grid: Number($("uvGrid").value),
    lateral_deg: Number($("lateralDeg").value),
    reach_m: Number($("reachM").value),
    samples: Number($("samples").value),
  };
}

function severityColor(sigma) {
  if (sigma <= 1) return "var(--ok)";
  if (sigma <= 2) return "var(--warn)";
  return "var(--bad)";
}

function renderAxis(root, payload) {
  const note = root.querySelector(".note");
  if (note) note.textContent = payload.note ?? "";

  // 값이 없으면 0이 아니라 대시다. 0.000은 "못 쟀다"가 아니라 "완전히 다르다"로 읽힌다.
  const num = (value) => (typeof value === "number" ? value.toFixed(3) : "—");

  // 게이트가 없으면 판정을 만들어 내걸지 않는다. "각도 지시가 이제 정당하다"는
  // 뜻이라 상수에서 나온 pass를 적으면 화면이 보지 않은 데이터로 판정을 주장한다.
  const gate = payload.gate ?? "—";
  const gateWhy = payload.gate ? "" : GATE_REASONS[payload.gate_reason] ?? "";

  root.querySelector("dl").innerHTML = `
    <dt>유사도</dt><dd>${num(payload.score)}</dd>
    <dt>교정 후</dt><dd>${num(payload.score_after)}</dd>
    <dt>개선율(분리)</dt><dd>${num(payload.improvement?.split)} <span class="split-note"></span></dd>
    <dt>개선율(동일)</dt><dd>${num(payload.improvement?.same)}</dd>
    <dt>REQ-19 게이트</dt><dd>${gate}${gateWhy}</dd>`;

  // 개선율(분리)이 왜 산출되지 않았는지, 산출됐다면 A/B 반쪽이 몇 표본으로 섰는지.
  // 이 줄이 없으면 위의 레퍼런스 날짜(F, 전체 레퍼런스) 줄이 마치 split의 출처인
  // 것처럼 읽힌다 — split은 F가 아니라 별도로 나뉜 A/B 위에서 난다.
  const splitNote = root.querySelector(".split-note");
  if (splitNote) {
    const reason = payload.split_reason ?? null;
    const a = payload.split_n_a;
    const b = payload.split_n_b;
    splitNote.textContent =
      reason !== null
        ? `산출 안 됨: ${SPLIT_REASONS[reason] ?? reason}`
        : `A n=${typeof a === "number" ? a : "—"} / B n=${typeof b === "number" ? b : "—"}`;
  }

  // 정규화가 어느 값을 재정의하는지 눈에 크게 띄어야 한다 — 이 값이 틀리면 아래
  // σ와 점수 전부가 다른 뜻이 된다. 값이 measured/provisional 둘 다 아니면(관절
  // 축처럼 애초에 배선되지 않았으면) 지어내지 않고 "—"를 낸다.
  const normEl = root.querySelector(".normalization");
  if (normEl) {
    const norm = payload.normalization ?? null;
    normEl.textContent = `정규화: ${NORMALIZATION_LABELS[norm] ?? "—"}`;
    normEl.className = `normalization ${NORMALIZATION_CLASS[norm] ?? "unknown"}`;
  }

  // 입도·표본. `resolved`가 답을 낸 층이 없으면(레퍼런스가 없거나 사다리 전부
  // 실패) 값을 지어내지 않고 대시다.
  const granEl = root.querySelector(".granularity");
  if (granEl) {
    const granularity = payload.granularity ?? null;
    const referenceN = payload.reference_n;
    granEl.textContent = granularity && typeof referenceN === "number"
      ? `입도 · 표본: ${granularity} · n=${referenceN}`
      : "입도 · 표본: —";
  }

  // 레퍼런스 날짜. 개수와 목록을 함께 적어 이 σ가 어느 날짜들의 평균인지 보인다.
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

  // 축별 σ 하나를 다시 값으로. 서버가 이 점의 이 축을 못 냈으면 0으로 지어내지
  // 않고 대시다 — 없는 값과 0인 값은 다른 말이다.
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
    // 표본 수를 함께 적는다. 왼손은 프레임 단위로 빠지므로 더 적은 격자점 위에 선다.
    const backing = typeof delta.samples === "number" ? ` (n=${delta.samples})` : "";
    item.innerHTML = `<div class="bar-row">
        <span>${delta.point}${backing}</span>
        <span class="track" style="background:${severityColor(delta.sigma)};width:${width}%"></span>
        <span>${delta.sigma.toFixed(2)}σ</span>
      </div>
      <span class="bar-axes">벽 ${axisSigma(delta, "wall")} · 수직 ${axisSigma(delta, "vertical")} · 진행축 ${axisSigma(delta, "lateral")}</span>`;
    bars.append(item);
  }

  // 자세 문장 하나(머리). 머리가 조용하면(응답이 stance를 null로 낸다) 배너를 숨긴다
  // — 0이 아니라 "낼 문장이 없다"이므로 빈 문장을 그리지 않는다.
  const stanceEl = root.querySelector(".stance");
  if (stanceEl) {
    const stance = payload.stance ?? null;
    stanceEl.hidden = stance === null;
    stanceEl.textContent = stance ? stance.text : "";
  }

  // 발부한 축 전부. `axes`가 곧 `apply_measured`가 옮기는 것이라 이 표의 숫자와
  // 개선율이 같은 이동을 잰다.
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
        typeof entry.magnitude_cm === "number"
          ? `${entry.magnitude_cm >= 0 ? "+" : ""}${entry.magnitude_cm.toFixed(1)}cm`
          : "—",
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

  // 교정문 옆에 근거를 함께 보인다. 문장만으로는 표본 3개짜리 주장과 24개짜리
  // 주장을 구별할 수 없다 — 그 구별이 이 줄의 존재 이유다.
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

function selected(side) {
  const select = $(`${side}Pass`);
  const entry = listed[side][Number(select.value)];
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

/** 레퍼런스 목록을 질의 키로 좁힌다. 4축 키가 엄격해 사람이 눈으로 찾기 어렵다. */
function matchFilter(side) {
  if (side !== "reference" || !$("sameKeyOnly").checked) return null;
  const query = selected("query");
  return query ? query.motion_key : null;
}

function fillPasses(side) {
  const select = $(`${side}Pass`);
  const wanted = matchFilter(side);
  const entries = listed[side].map((entry, index) => ({ entry, index }));
  const shown = wanted === null ? entries : entries.filter((i) => i.entry.motion_key === wanted);

  select.replaceChildren();
  for (const { entry, index } of shown) {
    const option = document.createElement("option");
    option.value = String(index);
    option.textContent = optionLabel(entry);
    select.append(option);
  }
  select.disabled = shown.length === 0;
  $(`${side}Count`).textContent = wanted === null
    ? `${shown.length}개`
    : `${shown.length}개 (질의 키와 같은 것만)`;
}

/** 가장 큰 동일 키 묶음. 4축 키가 표본을 얼마나 쪼개는지 그대로 보인다(설계 D3). */
function bucketSummary(passes) {
  const counts = new Map();
  for (const entry of passes) {
    counts.set(entry.motion_key, (counts.get(entry.motion_key) ?? 0) + 1);
  }
  let top = null;
  for (const [key, count] of counts) {
    if (top === null || count > top[1]) top = [key, count];
  }
  if (top === null) return "패스 없음";
  return `패스 ${passes.length}개 / 서로 다른 키 ${counts.size}개 / 최대 묶음 n=${top[1]} (${top[0]})`;
}

async function loadPasses(side) {
  const worker = $(`${side}Worker`).value;
  if (!worker) return;
  const query = new URLSearchParams({
    worker,
    limit: $("sessionLimit").value,
    uv_grid: String(params().uv_grid),
    lateral_deg: String(params().lateral_deg),
    reach_m: String(params().reach_m),
  });
  const level = $(`${side}Level`).value;
  if (level) query.set("level", level);

  $(`${side}Count`).textContent = "읽는 중…";
  const payload = await (await fetch(`/api/passes?${query}`)).json();
  if (payload.status !== "ok") {
    listed[side] = [];
    fillPasses(side);
    $(`${side}Count`).textContent = UNAVAILABLE_REASONS[payload.reason] ?? "목록 없음";
    return;
  }

  const levelSelect = $(`${side}Level`);
  if (levelSelect.dataset.worker !== worker) {
    levelSelect.dataset.worker = worker;
    levelSelect.replaceChildren(new Option("전체", ""));
    for (const value of payload.levels) levelSelect.append(new Option(`레벨 ${value}`, value));
    levelSelect.value = level;
  }

  listed[side] = payload.passes;
  fillPasses(side);
  // 훑은 세션 수를 함께 적는다. 자른 사실을 숨기면 "이 작업자에게 이 패스뿐"으로 읽힌다.
  const scanned = `세션 ${payload.sessions_scanned}/${payload.sessions_total}`
    + (payload.truncated ? " (잘림 — 한도를 올리면 더 훑는다)" : "");
  $(`${side}Summary`).textContent = `${scanned} · ${bucketSummary(payload.passes)}`;
}

async function loadWorkers() {
  const payload = await (await fetch("/api/workers")).json();
  for (const side of ["query", "reference"]) {
    const select = $(`${side}Worker`);
    select.replaceChildren();
    for (const worker of payload.workers) select.append(new Option(worker, worker));
    if (payload.workers.length === 0) {
      $(`${side}Summary`).textContent = UNAVAILABLE_REASONS.no_source;
    }
  }
  if (payload.workers.length > 1) $("referenceWorker").value = payload.workers[1];
  if (payload.workers.length > 0) {
    await loadPasses("query");
    await loadPasses("reference");
  }
}

async function run() {
  const query = selected("query");
  const reference = selected("reference");
  if (query === null || reference === null) {
    $("keys").textContent = "양쪽에서 패스를 하나씩 고르세요.";
    return;
  }
  const response = await fetch("/api/compare", {
    method: "POST",
    headers: { "content-type": "application/json" },
    body: JSON.stringify({
      query: passRef(query),
      reference: passRef(reference),
      ...params(),
    }),
  });
  const payload = await response.json();

  // 몸 기준 축(진행 방향·몸 기준 착탄 위치)이 선 요. HMD 회전에서 뽑은 실측값이면
  // 그 값을 그대로 적고, 회전을 못 읽어 대체값으로 떨어졌으면 그렇다고 적는다 —
  // 그 패스에서는 몸 기준 이름을 단 축이 다시 월드 방위각이기 때문이다.
  const yaw =
    typeof payload.yaw_assumed === "number"
      ? payload.yaw_source === "measured"
        ? `  — 몸 기준 축은 실측 요 ${((payload.yaw_assumed * 180) / Math.PI).toFixed(1)}°`
        : `  — 요를 못 읽어 정면 고정(yaw=${payload.yaw_assumed}) 대체값`
      : "";
  $("keys").textContent =
    `질의 ${payload.query_key ?? "-"} / 레퍼런스 ${payload.reference_key ?? "-"}${yaw}`;

  if (payload.status !== "ok") {
    // 이유를 서버가 준 대로 적는다. 재지 못한 패스까지 "동작 키가 다르다"로 적으면
    // 재지 못한 것을 재서 다르다고 말한 것이 된다.
    $("keys").textContent += `  — ${UNAVAILABLE_REASONS[payload.reason] ?? "비교하지 않습니다"}`;
    renderAxis($("measured"), {});
    renderAxis($("joint"), {});
    return;
  }
  renderAxis($("measured"), payload.measured);
  renderAxis($("joint"), payload.joint);
}

for (const side of ["query", "reference"]) {
  $(`${side}Worker`).addEventListener("change", () => {
    $(`${side}Level`).dataset.worker = "";
    void loadPasses(side);
  });
  $(`${side}Level`).addEventListener("change", () => void loadPasses(side));
}
$("queryPass").addEventListener("change", () => fillPasses("reference"));
$("sameKeyOnly").addEventListener("change", () => fillPasses("reference"));
$("reload").addEventListener("click", () => {
  void loadPasses("query").then(() => loadPasses("reference"));
});
$("run").addEventListener("click", () => void run());

void loadWorkers();
