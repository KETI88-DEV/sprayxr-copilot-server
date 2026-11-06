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
      id: 'pose_recomm', label: '자세', hint: '예: 지금 자세는 어때?',
      presets: ['지금 자세는 어때?', '이 Zone에서 어떻게 고쳐?', '전체 작업 유사도는?', '가장 문제가 큰 Zone은?', '오른손을 어떻게 고쳐?', '왼손을 어떻게 고쳐?'],
    },
    {
      id: 'trajectory_recomm', label: '순서', hint: '예: 권장 Zone 순서는?',
      presets: ['권장 Zone 순서는?', '실제 수행 순서는?', '누락한 Zone은?', '순서 준수율은?', '도포 방향은?'],
    },
  ];
  const style = document.createElement('style');
  style.textContent = `
.mvc-toggle{position:fixed;right:20px;bottom:20px;left:auto;width:auto;margin:0;z-index:10020;border:0;border-radius:24px;padding:12px 18px;
  background:#2563eb;color:#fff;font-weight:700;box-shadow:0 6px 20px rgba(0,0,0,.35);cursor:pointer}
.mvc-panel{position:fixed;right:16px;bottom:76px;z-index:10020;width:min(760px,calc(100vw - 32px));height:min(900px,calc(100vh - 96px));height:min(900px,calc(100dvh - 96px));
  display:none;flex-direction:column;background:#111827;color:#e5e7eb;border:1px solid #374151;border-radius:12px;
  box-shadow:0 10px 30px rgba(0,0,0,.45);font-size:14px}
.mvc-panel.open{display:flex}
.mvc-head{padding:12px 16px;border-bottom:1px solid #374151;display:flex;flex:0 0 auto;flex-direction:column;gap:8px;max-height:38%;overflow-y:auto}
.mvc-head select{width:100%;padding:6px;background:#1f2937;color:#e5e7eb;border:1px solid #4b5563;border-radius:6px}
.mvc-language-picker{flex:0 0 auto;min-width:0;padding:10px 16px 0;border-top:1px solid #374151}
.mvc-language-label{display:flex;justify-content:space-between;gap:12px;margin-bottom:6px;font-size:12px;color:#cbd5e1}
.mvc-language-label small{font-size:11px;color:#94a3b8;white-space:nowrap;align-self:center}
.mvc-language-flags{display:flex;gap:7px;overflow-x:auto;overscroll-behavior-x:contain;scrollbar-width:thin;scrollbar-color:#475569 transparent;padding:2px 0 5px}
.mvc-language-flag{width:42px;height:34px;flex:0 0 42px;display:grid;place-items:center;padding:5px;border:1px solid #475569;
  border-radius:7px;background:#1f2937;cursor:pointer}
.mvc-language-flag img{display:block;width:26px;height:20px;object-fit:contain}
.mvc-language-flag[aria-pressed=true]{border-color:#60a5fa;background:#1e3a5f;box-shadow:inset 0 0 0 1px #60a5fa}
.mvc-language-flag:hover,.mvc-language-flag:focus-visible{outline:2px solid #93c5fd;outline-offset:-2px}
.mvc-chips{display:flex;gap:6px}
.mvc-chip{flex:1;padding:6px 0;border:1px solid #4b5563;border-radius:16px;background:#1f2937;color:#e5e7eb;cursor:pointer}
.mvc-chip[aria-pressed=true]{background:#2563eb;border-color:#2563eb;color:#fff;font-weight:700}
.mvc-mode-row{display:grid;grid-template-columns:1fr 1fr;gap:6px;padding:4px;background:#0f172a;border:1px solid #374151;border-radius:9px}
.mvc-mode{padding:8px;border:0;border-radius:6px;background:transparent;color:#94a3b8;cursor:pointer;text-align:left}
.mvc-mode strong{display:flex;align-items:center;gap:8px;flex-wrap:wrap}.mvc-mode small{display:block;margin-top:2px;font-size:11px;font-weight:400}
.mvc-model{white-space:nowrap;padding:1px 6px;border:1px solid currentColor;border-radius:5px;font-size:11px;line-height:1.5;opacity:.9}
.mvc-mode[aria-pressed=true]{background:#1d4ed8;color:#fff}
.mvc-presets{display:flex;gap:6px;flex-wrap:wrap}
.mvc-preset{width:auto;flex:none;padding:5px 9px;border:1px solid #475569;border-radius:14px;background:#0f172a;color:#cbd5e1;
  font-size:12px;cursor:pointer}
.mvc-preset:hover,.mvc-preset:focus-visible{border-color:#60a5fa;background:#1e3a5f;color:#fff;outline:none}
.mvc-log{flex:1;min-height:0;overflow-y:auto;padding:16px;display:flex;flex-direction:column;gap:12px;scrollbar-width:thin;scrollbar-color:#475569 transparent}
.mvc-msg{padding:11px 14px;border-radius:9px;white-space:pre-wrap;overflow-wrap:anywhere;line-height:1.65;font-size:15px}
.mvc-me{align-self:flex-end;background:#1d4ed8;color:#fff;max-width:85%}
.mvc-bot{align-self:flex-start;background:#1f2937;max-width:95%}
.mvc-meta{font-size:11px;color:#9ca3af;margin-top:4px}
.mvc-context{font-size:12px;color:#bfdbfe;line-height:1.5}
.mvc-form{display:flex;flex:0 0 auto;gap:8px;padding:10px 16px 14px}
.mvc-form input{flex:1;min-width:0;padding:13px 12px;background:#1f2937;color:#e5e7eb;border:1px solid #4b5563;border-radius:7px;font-size:15px}
.mvc-form button{width:auto;flex:none;padding:10px 16px;border:0;border-radius:7px;background:#2563eb;color:#fff;cursor:pointer}
.mvc-form button:disabled{opacity:.5;cursor:wait}
@media(max-height:650px){.mvc-head{padding:8px 12px;gap:6px}.mvc-presets{flex-wrap:nowrap;overflow-x:auto}.mvc-preset{white-space:nowrap}.mvc-mode small{display:none}}
@media(max-width:480px){.mvc-panel{right:8px;width:calc(100vw - 16px)}.mvc-head,.mvc-log{padding:10px 12px}.mvc-language-picker{padding:8px 12px 0}.mvc-form{padding:8px 12px 12px}.mvc-msg{padding:9px 11px}.mvc-language-label{gap:6px}}`;
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
  const LANGUAGES = [
    ['ko', 'kr', '한국어'], ['en', 'us', 'English (영어)'], ['vi', 'vn', 'Tiếng Việt (베트남어)'], ['id', 'id', 'Bahasa Indonesia (인도네시아어)'],
    ['th', 'th', 'ไทย (태국어)'], ['ne', 'np', 'नेपाली (네팔어)'], ['my', 'mm', 'မြန်မာ (미얀마어)'], ['uz', 'uz', 'O‘zbekcha (우즈베크어)'],
    ['si', 'lk', 'සිංහල (싱할라어)'], ['ru', 'ru', 'Русский (러시아어)'], ['zh', 'cn', '中文 (중국어)'], ['km', 'kh', 'ខ្មែរ (크메르어)'],
  ];
  let outputLanguage = 'ko';
  const languagePicker = el('div', 'mvc-language-picker');
  const languageLabel = el('div', 'mvc-language-label');
  const selectedLanguage = el('span', null, '답변 언어 · 한국어');
  languageLabel.append(selectedLanguage, el('small', null, '질문은 한국어'));
  const languageFlags = el('div', 'mvc-language-flags');
  languageFlags.setAttribute('role', 'group');
  languageFlags.setAttribute('aria-label', '답변 언어');
  const chooseLanguage = (index) => {
    outputLanguage = LANGUAGES[index][0];
    selectedLanguage.textContent = `답변 언어 · ${LANGUAGES[index][2]}`;
    for (let i = 0; i < languageFlags.children.length; i++) {
      languageFlags.children[i].setAttribute('aria-pressed', String(i === index));
    }
  };
  LANGUAGES.forEach(([code, country, label], index) => {
    const button = el('button', 'mvc-language-flag');
    button.type = 'button';
    button.title = `${label}로 답변`;
    button.setAttribute('aria-label', `${label} 답변`);
    button.setAttribute('aria-pressed', String(code === outputLanguage));
    const flag = el('img');
    flag.src = `/motion-viewer/assets/flags/${country}.svg`;
    flag.alt = '';
    button.append(flag);
    button.addEventListener('click', () => chooseLanguage(index));
    button.addEventListener('keydown', (event) => {
      const next = event.key === 'ArrowRight' ? (index + 1) % LANGUAGES.length
        : event.key === 'ArrowLeft' ? (index + LANGUAGES.length - 1) % LANGUAGES.length
        : event.key === 'Home' ? 0 : event.key === 'End' ? LANGUAGES.length - 1 : null;
      if (next == null) return;
      event.preventDefault();
      chooseLanguage(next);
      languageFlags.children[next].focus();
      languageFlags.children[next].scrollIntoView?.({ block: 'nearest', inline: 'nearest' });
    });
    languageFlags.append(button);
  });
  languageFlags.addEventListener('wheel', (event) => {
    if (event.deltaX || languageFlags.scrollWidth <= languageFlags.clientWidth) return;
    const previous = languageFlags.scrollLeft;
    languageFlags.scrollLeft += event.deltaY;
    if (languageFlags.scrollLeft !== previous) event.preventDefault();
  }, { passive: false });
  languagePicker.append(languageLabel, languageFlags);
  const chips = el('div', 'mvc-chips');
  const modes = el('div', 'mvc-mode-row');
  modes.setAttribute('aria-label', '답변 수준');
  const presets = el('div', 'mvc-presets');
  presets.setAttribute('aria-label', '질문 프리셋');
  const contextIndicator = el('div', 'mvc-context');
  contextIndicator.setAttribute('aria-label', '질문 기준');
  const log = el('div', 'mvc-log');
  log.setAttribute('aria-live', 'polite');
  const form = el('form', 'mvc-form');
  const input = el('input');
  input.setAttribute('aria-label', '질문');
  const send = el('button', null, '보내기');
  send.type = 'submit';
  form.append(input, send);
  head.append(sessionSelect, contextIndicator, chips, modes, presets);
  panel.append(head, log, languagePicker, form);
  document.body.append(toggle, panel);

  let task = TASKS[1];
  let responseMode = 'quick';
  let viewerData = null;
  let playback = { map_num: null, zone_id: null, source_index: 0, playhead_seconds: 0 };

  const MODE_OPTIONS = [
    { id: 'quick', label: '빠른 안내', model: 'Qwen3-8B', help: '핵심 판단과 조치만 간결하게' },
    { id: 'detailed', label: '상세 분석', model: 'Qwen3.6-27B', help: '현재 상황·근거·원인까지 자세히' },
  ];
  for (const item of MODE_OPTIONS) {
    const button = el('button', 'mvc-mode');
    button.type = 'button';
    const title = el('strong');
    title.append(el('span', null, item.label), el('span', 'mvc-model', item.model));
    button.append(title, el('small', null, item.help));
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
  const recordKey = (data) => data
    ? JSON.stringify([data.worker?.worker_id, data.process, data.mode_id, data.selected_session_id]) : null;
  const knownZone = (frame) => {
    if (frame?.map_num == null || !Number.isInteger(Number(frame.zone_id)) || Number(frame.zone_id) <= 0) return null;
    const map = String(frame.map_num), zone = String(frame.zone_id);
    return viewerData?.pose_evaluation?.zones?.[`${map}:${zone}`] ? { map_num: map, zone_id: zone } : null;
  };
  const sprayedZone = (frame) => frame?.spraying ? knownZone(frame) : null;
  const playbackLabel = (state) => `${state.map_num != null ? `Map ${state.map_num} · Zone ${state.zone_id}` : 'Zone 확인 전'} · ${state.playhead_seconds.toFixed(2)}초`;
  const syncPlaybackContext = () => {
    contextIndicator.textContent = `질문 기준: ${playbackLabel(playback)} · 지금/이 Zone 질문은 해당 구간 평가, 전체 질문은 전체 기록`;
  };
  const syncViewerContext = () => {
    sessionSelect.replaceChildren(new Option(currentRecordLabel(viewerData), viewerData?.selected_session_id || ''));
    syncPlaybackContext();
  };

  function poseAnswer(prompt, data) {
    const evaluation = data.pose_evaluation || {};
    const zones = Object.values(evaluation.zones || {});
    if (evaluation.status !== 'available' || !zones.length) return '현재 재생 기록에서는 작업 비교값을 계산할 수 없습니다.';
    const requested = zoneNumber(prompt);
    const target = requested && zones.find((zone) => String(zone.zone_id) === requested);
    if (target) {
      const guidance = target.corrections?.length ? ` ${target.corrections.join(' ')}` : ' 별도의 작업 개선 항목은 없습니다.';
      return `Zone ${target.zone_id} 작업 유사도는 ${percent(target.similarity_percent)}입니다.${guidance}`;
    }
    if (/오른|right/i.test(prompt) || /왼|left/i.test(prompt)) {
      const side = /오른|right/i.test(prompt) ? '오른손' : '왼손';
      const matching = (evaluation.corrections || []).map((item) => item.message || item).filter((item) => String(item).includes(side));
      return matching.length
        ? `${side} 개선 안내: ${matching.join(' ')}`
        : `현재 기록에서 ${side}에 특정된 개선 항목은 없습니다. 전체 작업 유사도는 ${percent(evaluation.similarity_percent)}입니다.`;
    }
    const worst = zones.reduce((left, right) => Number(left.similarity_percent) <= Number(right.similarity_percent) ? left : right);
    if (/문제|최악|낮|어느|어디/i.test(prompt)) {
      const guidance = worst.corrections?.length ? ` ${worst.corrections.join(' ')}` : '';
      return `현재 기록에서 작업 유사도가 가장 낮은 구역은 Zone ${worst.zone_id}(${percent(worst.similarity_percent)})입니다.${guidance}`;
    }
    return `현재 재생 기록의 전체 작업 유사도는 ${percent(evaluation.similarity_percent)}입니다. 가장 낮은 구역은 Zone ${worst.zone_id}(${percent(worst.similarity_percent)})입니다.`;
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
    if (recommended.length === 1) return `현재 학습모드는 Zone ${recommended[0]} 한 곳으로 구성되어 있습니다.`;
    const top = (evaluation.position_distribution || []).slice(0, 2).map((row) => row?.[0]);
    const stableStart = top.length === 2
      && String(top[0]?.zone_id) === String(recommended[0])
      && String(top[1]?.zone_id) === String(recommended[1])
      && Math.min(Number(top[0]?.probability), Number(top[1]?.probability)) >= 0.6;
    const guidance = stableStart
      ? `우선 Zone ${recommended[0]}에서 시작한 뒤 Zone ${recommended[1]}로 이동하세요. 이 시작 순서는 전문가 작업의 약 ${Math.round(Math.min(Number(top[0].probability), Number(top[1].probability)) * 100)}%에서 동일하게 나타났습니다.`
      : `우선 Zone ${recommended[0]}에서 시작하세요.`;
    return `권장 작업 순서는 Zone ${recommended.join(' → ')}입니다.\n\n${guidance}`;
  }

  const structuredAnswer = (taskId, prompt, data) => taskId === 'pose_recomm'
    ? poseAnswer(prompt, data)
    : orderAnswer(prompt, data);
  const buildViewerQueryContext = (taskId, prompt, data, state) => {
    if (!data) return null;
    if (taskId === 'knowledge_search') return `motion_viewer_playback:${JSON.stringify({ record: currentRecordLabel(data), playback_context: state })}`;
    let analysisContext;
    if (taskId === 'pose_recomm') {
      const evaluation = data.pose_evaluation || {};
      analysisContext = {
        record: currentRecordLabel(data),
        status: evaluation.status ?? null,
        similarity_percent: evaluation.similarity_percent ?? null,
        zones: evaluation.zones ?? null,
        corrections: evaluation.corrections ?? null,
      };
    } else {
      const evaluation = data.reference_order_evaluation || {};
      analysisContext = {
        record: currentRecordLabel(data),
        recommended_flow: evaluation.recommended_flow ?? null,
        selected_flow: evaluation.selected_flow ?? null,
        adherence_rate: evaluation.adherence_rate ?? null,
        best_reference_similarity: evaluation.best_reference_similarity ?? null,
        position_similarity: evaluation.position_similarity ?? null,
        position_distribution: evaluation.position_distribution ?? null,
        reference_session_count: evaluation.reference_session_count ?? null,
        direction_summary: data.direction_summary || null,
        zone_keys: Object.keys(data.pose_evaluation?.zones || {}).sort(),
      };
    }
    return `motion_viewer_analysis:${JSON.stringify({
      verified_answer: structuredAnswer(taskId, prompt, data),
      analysis_context: analysisContext,
      signature: data.query_context_signatures?.[taskId],
      playback_context: state,
    })}`;
  };
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
    if (recordKey(viewerData) !== recordKey(event.detail)) {
      playback = { map_num: null, zone_id: null, source_index: 0, playhead_seconds: 0 };
    }
    viewerData = event.detail;
    if (playback.map_num != null && !knownZone(playback)) playback = { ...playback, map_num: null, zone_id: null };
    syncViewerContext();
  });
  addEventListener('motion-viewer-frame', (event) => {
    if (!viewerData) return;
    const { frame, sourceIndex, playheadSeconds } = event.detail;
    const seconds = Number(playheadSeconds);
    const nextIndex = Number.isInteger(sourceIndex) && sourceIndex >= 0 ? sourceIndex : playback.source_index;
    let zone = sprayedZone(frame);
    if (!zone && frame?.spraying) {
      zone = { map_num: null, zone_id: null };
    } else if (!zone && (nextIndex < playback.source_index || nextIndex > playback.source_index + 1 || playback.map_num == null)) {
      // Seeking into a transition resolves the latest Zone before that position.
      zone = { map_num: null, zone_id: null };
      for (let index = Math.min(nextIndex, (viewerData.frames?.length || 0) - 1); index >= 0; index--) {
        const previous = sprayedZone(viewerData.frames[index]);
        if (previous) { zone = previous; break; }
        if (viewerData.frames[index]?.spraying) break;
      }
    }
    playback = { ...playback, ...(zone || {}), source_index: nextIndex,
      playhead_seconds: Number.isFinite(seconds) && seconds >= 0 ? seconds : playback.playhead_seconds };
    syncPlaybackContext();
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
    if (!prompt || send.disabled) return;
    if (!viewerData && task.id !== 'knowledge_search') {
      say('재생할 기록을 먼저 불러와 주세요.', 'mvc-bot');
      return;
    }
    const sentTask = task;
    const sentMode = responseMode;
    const sentLanguage = outputLanguage;
    const sentLanguageLabel = LANGUAGES.find(([code]) => code === sentLanguage)[2];
    const sentData = viewerData;
    const sentPlayback = { ...playback };
    const sentLabel = `${currentRecordLabel(sentData)} · ${playbackLabel(sentPlayback)}`;
    const sentContext = buildViewerQueryContext(sentTask.id, prompt, sentData, sentPlayback);
    say(prompt, 'mvc-me', `${sentTask.label} · ${sentLabel}`);
    input.value = '';
    send.disabled = true;
    const pending = say('답을 찾는 중입니다…', 'mvc-bot');
    const started = performance.now();
    try {
      const response = await fetch('/api/v1/copilot/query', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({
          user_id: sentData?.worker?.worker_id || 'viewer',
          session_id: sentData?.selected_session_id ?? null,
          scenario_id: sentData?.mode_id != null ? String(sentData.mode_id) : null,
          task_type: sentTask.id,
          prompt,
          response_mode: sentMode,
          language: 'ko',
          response_language: sentLanguage,
          use_rag: 'Y',
          context: sentContext,
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
      const modeLabel = sentMode === 'detailed' ? '상세 분석 · Qwen3.6-27B' : '빠른 안내 · Qwen3-8B';
      say(answer, 'mvc-bot', `${sentTask.label} · ${modeLabel} · ${sentLanguageLabel} · ${sentLabel} · 응답 ${seconds}초${data.citations?.length ? ` · 인용 ${data.citations.length}` : ''}`);
    } catch {
      pending.remove();
      say('서버에 연결하지 못했습니다.', 'mvc-bot');
    } finally {
      send.disabled = false;
      input.focus();
    }
  });
})();
