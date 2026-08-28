// 서버 응답만으로 그린다. 지표 계산을 브라우저에서 다시 하지 않는다.
// 두 곳에서 계산하면 값이 갈라지고 어느 쪽이 맞는지 알 수 없게 된다.

const $ = (id) => document.getElementById(id);

// 화면에서 고를 패스. 실데이터 소스를 붙이기 전까지 고정값이다.
const QUERY = { worker: "A", date: "20250901", level: "10", session: "s1", pass_index: 0 };
const REFERENCE = { worker: "B", date: "20250902", level: "10", session: "s2", pass_index: 0 };

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

  root.querySelector("dl").innerHTML = `
    <dt>유사도</dt><dd>${num(payload.score)}</dd>
    <dt>교정 후</dt><dd>${num(payload.score_after)}</dd>
    <dt>개선율(분리)</dt><dd>${num(payload.improvement?.split)}</dd>
    <dt>개선율(동일)</dt><dd>${num(payload.improvement?.same)}</dd>
    <dt>REQ-19 게이트</dt><dd>${payload.gate ?? "n/a"}</dd>`;

  const bars = root.querySelector(".bars");
  bars.replaceChildren();
  for (const delta of payload.deltas ?? []) {
    const item = document.createElement("li");
    const width = Math.min(100, (delta.sigma / 6) * 100);
    item.innerHTML = `<span>${delta.point}</span>
      <span class="track" style="background:${severityColor(delta.sigma)};width:${width}%"></span>
      <span>${delta.sigma.toFixed(2)}σ</span>`;
    bars.append(item);
  }

  const list = root.querySelector(".corrections");
  list.replaceChildren();
  for (const line of payload.corrections ?? []) {
    const item = document.createElement("li");
    item.textContent = line;
    list.append(item);
  }
}

async function run() {
  const response = await fetch("/api/compare", {
    method: "POST",
    headers: { "content-type": "application/json" },
    body: JSON.stringify({
      query: QUERY,
      reference: REFERENCE,
      uv_grid: Number($("uvGrid").value),
      lateral_deg: Number($("lateralDeg").value),
      reach_m: Number($("reachM").value),
      samples: Number($("samples").value),
    }),
  });
  const payload = await response.json();

  $("keys").textContent =
    `질의 ${payload.query_key ?? "-"} / 레퍼런스 ${payload.reference_key ?? "-"}`;

  if (payload.status !== "ok") {
    $("keys").textContent += "  — 동작 키가 달라 비교하지 않습니다";
    return;
  }
  renderAxis($("measured"), payload.measured);
  renderAxis($("joint"), payload.joint);
}

$("run").addEventListener("click", run);
