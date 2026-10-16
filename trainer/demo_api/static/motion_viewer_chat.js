// 모션 뷰어 위에 뜨는 Co-Pilot 질문 창. complete 가 끝난 세션을 골라 질의 API(/api/v1/copilot/query)에 묻는다.
// 유형은 칩으로 고른다: 일반 = knowledge_search, 자세 = pose_recomm, 순서 = trajectory_recomm.
// 자세·순서는 complete 리포트로 바로 답한다(LLM 없음). 답은 textContent 로만 넣는다.
(() => {
  const TASKS = [
    { id: 'knowledge_search', label: '일반', hint: '예: 저도막이 뭐야?' },
    { id: 'pose_recomm', label: '자세', hint: '예: 가장 문제 있는 구역이 어디야?' },
    { id: 'trajectory_recomm', label: '순서', hint: '예: 순서는 잘 지켰어?' },
  ];
  const style = document.createElement('style');
  style.textContent = `
.mvc-toggle{position:fixed;right:20px;bottom:20px;left:auto;width:auto;margin:0;z-index:50;border:0;border-radius:24px;padding:12px 18px;
  background:#2563eb;color:#fff;font-weight:700;box-shadow:0 6px 20px rgba(0,0,0,.35);cursor:pointer}
.mvc-panel{position:fixed;right:20px;bottom:76px;z-index:50;width:min(380px,calc(100vw - 40px));height:min(520px,70vh);
  display:none;flex-direction:column;background:#111827;color:#e5e7eb;border:1px solid #374151;border-radius:12px;
  box-shadow:0 10px 30px rgba(0,0,0,.45);font-size:14px}
.mvc-panel.open{display:flex}
.mvc-head{padding:10px 12px;border-bottom:1px solid #374151;display:flex;flex-direction:column;gap:8px}
.mvc-head select{width:100%;padding:6px;background:#1f2937;color:#e5e7eb;border:1px solid #4b5563;border-radius:6px}
.mvc-chips{display:flex;gap:6px}
.mvc-chip{flex:1;padding:6px 0;border:1px solid #4b5563;border-radius:16px;background:#1f2937;color:#e5e7eb;cursor:pointer}
.mvc-chip[aria-pressed=true]{background:#2563eb;border-color:#2563eb;color:#fff;font-weight:700}
.mvc-log{flex:1;overflow-y:auto;padding:10px 12px;display:flex;flex-direction:column;gap:8px}
.mvc-msg{padding:8px 10px;border-radius:8px;white-space:pre-wrap;overflow-wrap:anywhere;line-height:1.45}
.mvc-me{align-self:flex-end;background:#1d4ed8;color:#fff;max-width:85%}
.mvc-bot{align-self:flex-start;background:#1f2937;max-width:95%}
.mvc-meta{font-size:11px;color:#9ca3af;margin-top:4px}
.mvc-form{display:flex;gap:6px;padding:10px 12px;border-top:1px solid #374151}
.mvc-form input{flex:1;padding:8px;background:#1f2937;color:#e5e7eb;border:1px solid #4b5563;border-radius:6px}
.mvc-form button{width:auto;flex:none;padding:8px 12px;border:0;border-radius:6px;background:#2563eb;color:#fff;cursor:pointer}
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
  const log = el('div', 'mvc-log');
  log.setAttribute('aria-live', 'polite');
  const form = el('form', 'mvc-form');
  const input = el('input');
  input.setAttribute('aria-label', '질문');
  const send = el('button', null, '보내기');
  send.type = 'submit';
  form.append(input, send);
  head.append(sessionSelect, chips);
  panel.append(head, log, form);
  document.body.append(toggle, panel);

  let task = TASKS[1];
  let sessions = [];
  for (const item of TASKS) {
    const chip = el('button', 'mvc-chip', item.label);
    chip.type = 'button';
    chip.setAttribute('aria-pressed', String(item === task));
    chip.addEventListener('click', () => {
      task = item;
      for (const other of chips.children) other.setAttribute('aria-pressed', String(other === chip));
      input.placeholder = item.hint;
      input.focus();
    });
    chips.append(chip);
  }
  input.placeholder = task.hint;

  const say = (text, cls, meta) => {
    const bubble = el('div', `mvc-msg ${cls}`, text);
    if (meta) bubble.append(el('div', 'mvc-meta', meta));
    log.append(bubble);
    log.scrollTop = log.scrollHeight;
    return bubble;
  };

  async function loadSessions() {
    sessionSelect.replaceChildren();
    try {
      const response = await fetch('/motion-viewer/api/completed');
      sessions = response.ok ? (await response.json()).sessions || [] : [];
    } catch {
      sessions = [];
    }
    if (!sessions.length) {
      sessionSelect.append(new Option('complete 된 세션이 없습니다', ''));
      return;
    }
    for (const item of sessions) {
      sessionSelect.append(new Option(`${item.session_id} · 레벨 ${item.scenario_id ?? '-'} · ${item.requested_at ?? ''}`, item.session_id));
    }
  }

  toggle.addEventListener('click', () => {
    const open = panel.classList.toggle('open');
    if (open) {
      loadSessions();
      input.focus();
    }
  });

  form.addEventListener('submit', async (event) => {
    event.preventDefault();
    const prompt = input.value.trim();
    const session = sessions.find((item) => item.session_id === sessionSelect.value);
    if (!prompt) return;
    if (!session && task.id !== 'knowledge_search') {
      say('완료된 세션을 먼저 골라 주세요.', 'mvc-bot');
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
          user_id: session?.user_id || 'viewer',
          session_id: session?.session_id ?? null,
          scenario_id: session?.scenario_id ?? null,
          task_type: task.id,
          prompt,
          use_rag: 'Y',
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
      say(data.answer, 'mvc-bot', `${task.label} · ${seconds}초${data.citations?.length ? ` · 인용 ${data.citations.length}` : ''}`);
    } catch {
      pending.remove();
      say('서버에 연결하지 못했습니다.', 'mvc-bot');
    } finally {
      send.disabled = false;
      input.focus();
    }
  });
})();
