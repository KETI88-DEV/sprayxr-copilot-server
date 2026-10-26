// 모션 뷰어 위에 뜨는 Co-Pilot 질문 창. 현재 재생 중인 기록을 기준으로 질의한다.
// 유형은 칩으로 고른다: 일반 = knowledge_search, 자세 = pose_recomm, 순서 = trajectory_recomm.
// 자세·순서는 complete 리포트의 구조화 결과를 사용한다. 답은 textContent 로만 넣는다.
(() => {
  const TASKS = [
    {
      id: 'knowledge_search', label: '일반', hint: '예: 저도막이 뭐야?',
      presets: ['저도막이 뭐야?', '적정 이격거리는?', '적정 분사 속도는?', '과도막의 원인은?'],
    },
    {
      id: 'pose_recomm', label: '자세', hint: '예: 가장 문제가 큰 Zone은?',
      presets: ['전체 자세 유사도는?', '가장 문제가 큰 Zone은?', '오른손을 어떻게 고쳐?', '왼손을 어떻게 고쳐?'],
    },
    {
      id: 'trajectory_recomm', label: '순서', hint: '예: 권장 Zone 순서는?',
      presets: ['권장 Zone 순서는?', '실제 수행 순서는?', '누락한 Zone은?', '순서 준수율은?', '도포 방향은?'],
    },
  ];
  const style = document.createElement('style');
  style.textContent = `
.mvc-toggle{position:fixed;right:20px;bottom:20px;left:auto;width:auto;margin:0;z-index:50;border:0;border-radius:24px;padding:12px 18px;
  background:#2563eb;color:#fff;font-weight:700;box-shadow:0 6px 20px rgba(0,0,0,.35);cursor:pointer}
.mvc-panel{position:fixed;right:20px;bottom:76px;z-index:50;width:min(480px,calc(100vw - 40px));height:min(640px,78vh);
  display:none;flex-direction:column;background:#111827;color:#e5e7eb;border:1px solid #374151;border-radius:12px;
  box-shadow:0 10px 30px rgba(0,0,0,.45);font-size:14px}
.mvc-panel.open{display:flex}
.mvc-head{padding:10px 12px;border-bottom:1px solid #374151;display:flex;flex-direction:column;gap:8px}
.mvc-head select{width:100%;padding:6px;background:#1f2937;color:#e5e7eb;border:1px solid #4b5563;border-radius:6px}
.mvc-chips{display:flex;gap:6px}
.mvc-chip{flex:1;padding:6px 0;border:1px solid #4b5563;border-radius:16px;background:#1f2937;color:#e5e7eb;cursor:pointer}
.mvc-chip[aria-pressed=true]{background:#2563eb;border-color:#2563eb;color:#fff;font-weight:700}
.mvc-mode-row{display:grid;grid-template-columns:1fr 1fr;gap:6px;padding:4px;background:#0f172a;border:1px solid #374151;border-radius:9px}
.mvc-mode{padding:8px;border:0;border-radius:6px;background:transparent;color:#94a3b8;cursor:pointer;text-align:left}
.mvc-mode strong,.mvc-mode small{display:block}.mvc-mode small{margin-top:2px;font-size:11px;font-weight:400}
.mvc-mode[aria-pressed=true]{background:#1d4ed8;color:#fff}
.mvc-presets{display:flex;gap:6px;flex-wrap:wrap}
.mvc-preset{width:auto;flex:none;padding:5px 9px;border:1px solid #475569;border-radius:14px;background:#0f172a;color:#cbd5e1;
  font-size:12px;cursor:pointer}
.mvc-preset:hover,.mvc-preset:focus-visible{border-color:#60a5fa;background:#1e3a5f;color:#fff;outline:none}
.mvc-log{flex:1;overflow-y:auto;padding:10px 12px;display:flex;flex-direction:column;gap:8px}
.mvc-msg{padding:8px 10px;border-radius:8px;white-space:pre-wrap;overflow-wrap:anywhere;line-height:1.45}
.mvc-me{align-self:flex-end;background:#1d4ed8;color:#fff;max-width:85%}
.mvc-bot{align-self:flex-start;background:#1f2937;max-width:95%}
.mvc-meta{font-size:11px;color:#9ca3af;margin-top:4px}
.mvc-form{display:flex;gap:6px;padding:10px 12px;border-top:1px solid #374151}
.mvc-form input{flex:1;padding:11px 12px;background:#1f2937;color:#e5e7eb;border:1px solid #4b5563;border-radius:7px;font-size:14px}
.mvc-form button{width:auto;flex:none;padding:10px 16px;border:0;border-radius:7px;background:#2563eb;color:#fff;cursor:pointer}
.mvc-form button:disabled{opacity:.5;cursor:wait}`;
  document.head.append(style);

  const el = (tag, cls, text) => {
    const node = document.createElement(tag);
    if (cls) node.className = cls;
    if (text) node.textContent = text;
    return node;
  };
  const toggle = el('button', 'mvc-toggle', '질문하기');
  toggle.type = 'button';
  const panel = el('section', 'mvc-panel');
  panel.setAttribute('aria-label', 'Co-Pilot 질문');
  const head = el('div', 'mvc-head');
  const sessionSelect = el('select');
  sessionSelect.setAttribute('aria-label', '완료된 세션');
  const chips = el('div', 'mvc-chips');
  const modes = el('div', 'mvc-mode-row');
  modes.setAttribute('aria-label', '답변 수준');
  const presets = el('div', 'mvc-presets');
  presets.setAttribute('aria-label', '질문 프리셋');
  const log = el('div', 'mvc-log');
  log.setAttribute('aria-live', 'polite');
  const form = el('form', 'mvc-form');
  const input = el('input');
  input.setAttribute('aria-label', '질문');
  const send = el('button', null, '보내기');
  send.type = 'submit';
  form.append(input, send);
  head.append(sessionSelect, chips, modes, presets);
  panel.append(head, log, form);
  document.body.append(toggle, panel);

  let task = TASKS[1];
  let responseMode = 'detailed';
  let viewerData = null;

  const MODE_OPTIONS = [
    { id: 'quick', label: '빠른 안내', help: '핵심 판단과 조치만 간결하게' },
    { id: 'detailed', label: '상세 분석', help: '현재 상황·근거·원인까지 자세히' },
  ];
  for (const item of MODE_OPTIONS) {
    const button = el('button', 'mvc-mode');
    button.type = 'button';
    button.append(el('strong', null, item.label), el('small', null, item.help));
    button.setAttribute('aria-pressed', String(item.id === responseMode));
    button.addEventListener('click', () => {
      responseMode = item.id;
      for (const other of modes.children) other.setAttribute('aria-pressed', String(other === button));
    });
    modes.append(button);
  }

  const percent = (value) => `${Math.round(Number(value) || 0)}%`;
  const zoneNumber = (prompt) => prompt.match(/(?:zone|구역)\s*([0-9]+)/i)?.[1] || null;
  const currentRecordLabel = (data) => data
    ? `${data.worker?.display_name || '작업자'} · ${data.process_label || data.process} · 학습모드 ${data.mode_id} · ${data.selected_session_id}`
    : '현재 재생 기록을 불러오는 중입니다';
  const syncViewerContext = () => {
    sessionSelect.replaceChildren(new Option(currentRecordLabel(viewerData), viewerData?.selected_session_id || ''));
  };

  function poseAnswer(prompt, data) {
    const evaluation = data.pose_evaluation || {};
    const zones = Object.values(evaluation.zones || {});
    if (evaluation.status !== 'available' || !zones.length) return '현재 재생 기록에서는 자세 비교값을 계산할 수 없습니다.';
    const requested = zoneNumber(prompt);
    const target = requested && zones.find((zone) => String(zone.zone_id) === requested);
    if (target) {
      const guidance = target.corrections?.length ? ` ${target.corrections.join(' ')}` : ' 별도의 자세 개선 항목은 없습니다.';
      return `Zone ${target.zone_id} 자세 유사도는 ${percent(target.similarity_percent)}입니다.${guidance}`;
    }
    if (/오른|right/i.test(prompt) || /왼|left/i.test(prompt)) {
      const side = /오른|right/i.test(prompt) ? '오른손' : '왼손';
      const matching = (evaluation.corrections || []).map((item) => item.message || item).filter((item) => String(item).includes(side));
      return matching.length
        ? `${side} 개선 안내: ${matching.join(' ')}`
        : `현재 기록에서 ${side}에 특정된 개선 항목은 없습니다. 전체 자세 유사도는 ${percent(evaluation.similarity_percent)}입니다.`;
    }
    const worst = zones.reduce((left, right) => Number(left.similarity_percent) <= Number(right.similarity_percent) ? left : right);
    if (/문제|최악|낮|어느|어디/i.test(prompt)) {
      const guidance = worst.corrections?.length ? ` ${worst.corrections.join(' ')}` : '';
      return `현재 기록에서 자세 유사도가 가장 낮은 구역은 Zone ${worst.zone_id}(${percent(worst.similarity_percent)})입니다.${guidance}`;
    }
    return `현재 재생 기록의 전체 자세 유사도는 ${percent(evaluation.similarity_percent)}입니다. 가장 낮은 구역은 Zone ${worst.zone_id}(${percent(worst.similarity_percent)})입니다.`;
  }

  function orderAnswer(prompt, data) {
    const evaluation = data.reference_order_evaluation || {};
    const recommended = evaluation.recommended_flow || [];
    const selected = evaluation.selected_flow || [];
    if (!recommended.length) return '현재 재생 기록에서는 전문가 Zone 순서 기준을 계산할 수 없습니다.';
    if (/실제|수행 순서|칠한 순서/i.test(prompt)) return `현재 기록의 실제 수행 순서는 ${selected.join(' → ') || '없음'}입니다.`;
    if (/누락|안 칠|빠진|미수행/i.test(prompt)) {
      const missing = recommended.filter((zone) => !selected.includes(zone));
      return missing.length ? `현재 기록에서 누락된 Zone은 ${missing.join('·')}번입니다.` : '현재 기록은 전문가 기준의 모든 Zone을 방문했습니다.';
    }
    if (/준수|유사|몇 퍼센트|점수/i.test(prompt)) {
      return `현재 기록의 전문가 순서 유사도는 ${percent(Number(evaluation.adherence_rate) * 100)}입니다. 가장 가까운 전문가 수행과 ${percent(Number(evaluation.best_reference_similarity) * 100)}, 순번별 통계와 ${percent(Number(evaluation.position_similarity) * 100)} 유사합니다.`;
    }
    if (/방향/i.test(prompt)) {
      const direction = data.direction_summary?.classification;
      return direction ? `현재 기록의 주요 도포 방향은 ${direction}입니다.` : '현재 기록에서는 주요 도포 방향을 계산할 수 없습니다.';
    }
    const distributions = (evaluation.position_distribution || []).map((row, index) => {
      const choices = row.slice(0, 2).map((item) => `Z${item.zone_id} ${percent(Number(item.probability) * 100)}`);
      return `${index + 1}순위 ${choices.join('·')}`;
    });
    if (recommended.length === 1) return `현재 학습모드는 Zone ${recommended[0]} 한 곳으로 구성되어 있습니다.`;
    return `현재 학습모드의 전문가 대표 흐름은 ${recommended.join(' → ')}입니다.${distributions.length ? ` 순번별 주요 분포는 ${distributions.join(', ')}입니다.` : ''}`;
  }

  const structuredAnswer = (taskId, prompt, data) => taskId === 'pose_recomm'
    ? poseAnswer(prompt, data)
    : orderAnswer(prompt, data);
  const renderPresets = (item) => {
    presets.replaceChildren();
    for (const question of item.presets) {
      const button = el('button', 'mvc-preset', question);
      button.type = 'button';
      button.addEventListener('click', () => {
        input.value = question;
        input.focus();
      });
      presets.append(button);
    }
  };
  for (const item of TASKS) {
    const chip = el('button', 'mvc-chip', item.label);
    chip.type = 'button';
    chip.setAttribute('aria-pressed', String(item === task));
    chip.addEventListener('click', () => {
      task = item;
      for (const other of chips.children) other.setAttribute('aria-pressed', String(other === chip));
      input.placeholder = item.hint;
      renderPresets(item);
      input.focus();
    });
    chips.append(chip);
  }
  input.placeholder = task.hint;
  renderPresets(task);

  const say = (text, cls, meta) => {
    const bubble = el('div', `mvc-msg ${cls}`, text);
    if (meta) bubble.append(el('div', 'mvc-meta', meta));
    log.append(bubble);
    log.scrollTop = log.scrollHeight;
    return bubble;
  };

  addEventListener('motion-viewer-data', (event) => {
    viewerData = event.detail;
    syncViewerContext();
  });
  syncViewerContext();

  toggle.addEventListener('click', () => {
    const open = panel.classList.toggle('open');
    if (open) {
      syncViewerContext();
      input.focus();
    }
  });

  form.addEventListener('submit', async (event) => {
    event.preventDefault();
    const prompt = input.value.trim();
    if (!prompt) return;
    if (!viewerData && task.id !== 'knowledge_search') {
      say('재생할 기록을 먼저 불러와 주세요.', 'mvc-bot');
      return;
    }
    say(prompt, 'mvc-me', task.label);
    input.value = '';
    send.disabled = true;
    const pending = say('답을 찾는 중입니다…', 'mvc-bot');
    const started = performance.now();
    try {
      const response = await fetch('/api/v1/copilot/query', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({
          user_id: viewerData?.worker?.worker_id || 'viewer',
          session_id: viewerData?.selected_session_id ?? null,
          scenario_id: viewerData?.mode_id != null ? String(viewerData.mode_id) : null,
          task_type: task.id,
          prompt,
          response_mode: responseMode,
          use_rag: responseMode === 'detailed' ? 'Y' : 'N',
        }),
      });
      const payload = await response.json();
      pending.remove();
      if (!response.ok || payload.status !== 'SUCCESS') {
        say(`질문을 처리하지 못했습니다 (${payload.message || response.status}).`, 'mvc-bot');
        return;
      }
      const data = payload.data;
      const seconds = ((performance.now() - started) / 1000).toFixed(1);
      const answer = data.answer;
      const modeLabel = responseMode === 'detailed' ? '상세 분석' : '빠른 안내';
      say(answer, 'mvc-bot', `${task.label} · ${modeLabel} · ${seconds}초${data.citations?.length ? ` · 인용 ${data.citations.length}` : ''}`);
    } catch {
      pending.remove();
      say('서버에 연결하지 못했습니다.', 'mvc-bot');
    } finally {
      send.disabled = false;
      input.focus();
    }
  });
})();
