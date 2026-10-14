(() => {
  'use strict';
  const byId = id => document.getElementById(id);
  const template = byId('poseUiTemplate');
  if (!template) return;
  const fragment = template.content.cloneNode(true);
  const sourceControls = fragment.querySelector('[data-pose-slot="source"]');
  const tabs = fragment.querySelector('[data-pose-slot="tabs"]');
  const poseStage = fragment.querySelector('[data-pose-slot="stage"]');
  const summary = fragment.querySelector('[data-pose-slot="summary"]');
  document.querySelector('.workspace > .side:not(.right)').prepend(sourceControls);
  document.querySelector('.workspace > .stage').prepend(tabs, poseStage);
  document.querySelector('.workspace > .right').prepend(summary);
  const poseCanvas = byId('poseCanvas');
  if (!poseCanvas) return;
  const layerLabels = ['측정 작업자', '전문가 기준', '교정 목표', '별도 기준 좌표'];
  poseCanvas.setAttribute('aria-description', layerLabels.join(' · '));
  const poseCtx = poseCanvas.getContext('2d');
  const pathCanvas = byId('canvas');
  const timelineInput = byId('timeline');
  const playButton = byId('play');
  const restartButton = byId('restart');
  const originalHandlers = {
    timeline: timelineInput.oninput,
    play: playButton.onclick,
    restart: restartButton.onclick,
  };
  let evaluationLoadGeneration = 0;
  let evaluationData = null;
  let evaluationCatalog = null;
  let activeTab = 'pose';
  let posePlaying = false;
  let poseLast = 0;
  let poseCursor = 0;
  let poseYaw = -0.45;
  let posePitch = 0.12;
  let poseZoom = 175;
  let posePanX = 0;
  let posePanY = 0;
  let poseDrag = null;

  const setState = (state, message = '') => {
    const panel = byId('poseViewState');
    panel.dataset.state = state; // loading / empty / error / ready
    panel.textContent = message;
    panel.hidden = state === 'ready';
  };
  const escapeText = value => String(value == null ? '—' : value);
  const sessionSelect = byId('evaluationSession');
  const sequenceSelect = byId('evaluationSequence');

  function fillSequences() {
    const session = (evaluationCatalog?.sessions || []).find(item => item.session_id === sessionSelect.value);
    sequenceSelect.replaceChildren(...(session?.sequences || []).map(item => {
      const option = document.createElement('option');
      option.value = item.sequence_id;
      option.textContent = `${item.sequence_id} · ${item.frames}프레임`;
      return option;
    }));
  }

  async function loadEvaluationCatalog() {
    try {
      const response = await fetch('/motion-viewer/api/evaluations');
      if (!response.ok) throw new Error('평가 목록을 읽을 수 없습니다.');
      evaluationCatalog = await response.json();
      const sessions = evaluationCatalog.sessions || [];
      sessionSelect.replaceChildren(...sessions.map(item => {
        const option = document.createElement('option');
        option.value = item.session_id;
        option.textContent = `${item.session_id} · ${item.sequences.length}개 시퀀스`;
        return option;
      }));
      fillSequences();
      if (!sessions.length) {
        setState('empty', '아직 저장된 TSB 시퀀스가 없습니다. 시뮬레이터 전송 후 다시 확인하세요.');
        return;
      }
      await loadEvaluation();
    } catch (error) {
      setState('error', error.message || '평가 목록을 불러오지 못했습니다.');
    }
  }

  async function loadEvaluation() {
    const generation = ++evaluationLoadGeneration;
    const sessionId = sessionSelect.value;
    const sequenceId = sequenceSelect.value;
    if (!sessionId || !sequenceId) {
      evaluationData = null;
      setState('empty', '평가할 세션과 시퀀스를 선택하세요.');
      return;
    }
    posePlaying = false;
    poseCursor = 0;
    playButton.textContent = '재생';
    setState('loading', '제품 자세 분석 엔진의 결과를 불러오는 중입니다.');
    try {
      const query = new URLSearchParams({session_id: sessionId, sequence_id: sequenceId});
      const response = await fetch(`/motion-viewer/api/evaluation?${query}`);
      if (!response.ok) {
        const body = await response.json().catch(() => ({}));
        throw new Error(body.detail || '자세 평가를 불러오지 못했습니다.');
      }
      const incoming = await response.json();
      if (generation !== evaluationLoadGeneration) return;
      evaluationData = incoming;
      timelineInput.min = 0;
      timelineInput.max = Math.max(0, incoming.layers.measured.frames.length - 1);
      timelineInput.value = 0;
      renderSummary(incoming);
      setState('ready');
      resizePose();
      drawPose();
    } catch (error) {
      if (generation !== evaluationLoadGeneration) return;
      evaluationData = null;
      setState('error', error.message || '자세 평가를 불러오지 못했습니다.');
    }
  }

  function renderSummary(view) {
    const e = view.evaluation;
    const identity = view.identity;
    const correction = e.top_correction;
    byId('poseVerdict').textContent = e.verdict;
    byId('poseScore').textContent = e.similarity_score == null ? '측정 불가' : `${Math.round(e.similarity_score * 100)}% / ${e.expert_percentile ?? '—'}백분위`;
    byId('poseDeviation').textContent = correction?.joint || '통상 범위';
    byId('poseAction').textContent = correction?.message || e.message;
    byId('poseConfidence').textContent = e.confidence;
    byId('poseZoneSummary').textContent = `Zone ${identity.zone_id || '—'} · ${identity.sequence_id} · ${view.strokes.length || 1} 스트로크`;
    byId('poseAdvancedJson').textContent = JSON.stringify({
      analysis_status: e.analysis_status,
      reason_code: e.reason_code,
      strokes: view.strokes,
      coordinate_spaces: Object.fromEntries(Object.entries(view.layers).map(([key, layer]) => [key, layer.coordinate_space])),
    }, null, 2);
  }

  function setViewerTab(tab) {
    activeTab = tab;
    document.querySelectorAll('.viewer-tab').forEach(button => {
      button.setAttribute('aria-selected', String(button.dataset.tab === tab));
    });
    const pose = tab === 'pose';
    byId('poseStage').hidden = !pose;
    pathCanvas.hidden = pose;
    document.querySelectorAll('.path-only').forEach(node => node.classList.toggle('path-ui-hidden', pose));
    byId('poseSummary').hidden = !pose;
    if (pose) {
      timelineInput.max = Math.max(0, (evaluationData?.layers.measured.frames.length || 1) - 1);
      timelineInput.value = Math.min(Number(timelineInput.value), Number(timelineInput.max));
      timelineInput.oninput = event => { poseCursor = Number(event.target.value); drawPose(); };
      playButton.onclick = togglePosePlayback;
      restartButton.onclick = resetPosePlayback;
      resizePose(); drawPose();
    } else {
      posePlaying = false;
      playButton.textContent = '재생';
      timelineInput.oninput = originalHandlers.timeline;
      playButton.onclick = originalHandlers.play;
      restartButton.onclick = originalHandlers.restart;
      if (typeof data !== 'undefined' && data?.frames) {
        timelineInput.max = Math.max(0, data.frames.length - 1);
        timelineInput.value = Math.min(index || 0, Number(timelineInput.max));
      }
    }
  }

  function togglePosePlayback() {
    if (!evaluationData) return;
    posePlaying = !posePlaying;
    poseLast = 0;
    playButton.textContent = posePlaying ? '일시정지' : '재생';
  }
  function resetPosePlayback() {
    posePlaying = false; poseCursor = 0; poseLast = 0;
    timelineInput.value = 0; playButton.textContent = '재생'; drawPose();
  }

  function project(point, width, height, inset = null) {
    if (!Array.isArray(point) || point.length < 3) return null;
    let [x, y, z] = point.map(Number);
    const cy = Math.cos(poseYaw), sy = Math.sin(poseYaw);
    const cp = Math.cos(posePitch), sp = Math.sin(posePitch);
    const rx = x * cy - z * sy;
    const rz0 = x * sy + z * cy;
    const ry = y * cp - rz0 * sp;
    const rz = y * sp + rz0 * cp;
    const scale = poseZoom / Math.max(.45, 2.8 + rz);
    const box = inset || {x: 0, y: 0, w: width, h: height};
    return [box.x + box.w / 2 + rx * scale + posePanX, box.y + box.h * .62 - ry * scale + posePanY];
  }

  function drawLine(a, b, color, width = 2, dashed = false, inset = null) {
    const pa = project(a, poseCanvas.clientWidth, poseCanvas.clientHeight, inset);
    const pb = project(b, poseCanvas.clientWidth, poseCanvas.clientHeight, inset);
    if (!pa || !pb) return;
    poseCtx.save(); poseCtx.strokeStyle = color; poseCtx.lineWidth = width;
    if (dashed) poseCtx.setLineDash([7, 5]);
    poseCtx.beginPath(); poseCtx.moveTo(...pa); poseCtx.lineTo(...pb); poseCtx.stroke(); poseCtx.restore();
  }

  const bonePairs = [
    ['Pelvis','Spine'],['Spine','Chest'],['Chest','Head'],
    ['Chest','Left Shoulder'],['Left Shoulder','Left Upper Arm'],['Left Upper Arm','Left Fore Arm'],['Left Fore Arm','Left Hand'],
    ['Chest','Right Shoulder'],['Right Shoulder','Right Upper Arm'],['Right Upper Arm','Right Fore Arm'],['Right Fore Arm','Right Hand'],
  ];
  function jointOf(joints, name) {
    if (!joints) return null;
    const normalized = name.replaceAll(' ', '').toLowerCase();
    const key = Object.keys(joints).find(item => item.replaceAll(' ', '').toLowerCase() === normalized);
    return key ? joints[key] : null;
  }
  function drawSkeleton(joints, color, inset = null, dashed = false) {
    for (const [a, b] of bonePairs) drawLine(jointOf(joints, a), jointOf(joints, b), color, 2.4, dashed, inset);
    for (const point of Object.values(joints || {})) {
      const p = project(point, poseCanvas.clientWidth, poseCanvas.clientHeight, inset);
      if (!p) continue;
      poseCtx.fillStyle = color; poseCtx.beginPath(); poseCtx.arc(p[0], p[1], 3.2, 0, Math.PI * 2); poseCtx.fill();
    }
  }

  function drawMesh(mesh) {
    if (!mesh?.available) return;
    const vertices = mesh.vertices || [];
    const triangles = mesh.triangles || [];
    const step = Math.max(1, Math.ceil(triangles.length / 12000));
    poseCtx.save(); poseCtx.strokeStyle = 'rgba(148,163,184,.25)'; poseCtx.fillStyle = 'rgba(71,85,105,.11)'; poseCtx.lineWidth = .6;
    for (let i = 0; i < triangles.length; i += step) {
      const points = triangles[i].map(index => project(vertices[index], poseCanvas.clientWidth, poseCanvas.clientHeight)).filter(Boolean);
      if (points.length !== 3) continue;
      poseCtx.beginPath(); poseCtx.moveTo(...points[0]); poseCtx.lineTo(...points[1]); poseCtx.lineTo(...points[2]); poseCtx.closePath(); poseCtx.fill(); poseCtx.stroke();
    }
    poseCtx.restore();
  }

  function drawExpertInset(layer) {
    if (!layer?.available) return;
    const inset = {x: poseCanvas.clientWidth - 250, y: 76, w: 220, h: 220};
    poseCtx.fillStyle = 'rgba(7,16,31,.82)'; poseCtx.fillRect(inset.x, inset.y, inset.w, inset.h);
    poseCtx.strokeStyle = 'rgba(34,197,94,.55)'; poseCtx.strokeRect(inset.x, inset.y, inset.w, inset.h);
    const joints = Object.fromEntries(Object.entries(layer.tracks || {}).map(([name, track]) => [name, track[Math.min(track.length - 1, Math.round((poseCursor / Math.max(1, Number(timelineInput.max))) * (track.length - 1)))]]));
    drawSkeleton(joints, '#22C55E', inset, false);
    poseCtx.fillStyle = '#bbf7d0'; poseCtx.font = '11px sans-serif'; poseCtx.fillText('전문가 기준 · 별도 기준 좌표', inset.x + 10, inset.y + 18);
  }

  function drawCorrections(layer, measuredFrame) {
    for (const target of layer?.targets || []) {
      if (Number(target.index) !== Number(measuredFrame?.index)) continue;
      const start = jointOf(measuredFrame.joints, String(target.joint || ''));
      const end = target.point || target.position || target.target;
      drawLine(start, end, '#FACC15', 3, true);
    }
  }

  function drawPose() {
    const rect = poseCanvas.getBoundingClientRect();
    poseCtx.clearRect(0, 0, rect.width, rect.height);
    if (!evaluationData) return;
    const frames = evaluationData.layers.measured.frames || [];
    const frame = frames[Math.max(0, Math.min(frames.length - 1, Math.round(poseCursor)))];
    drawMesh(evaluationData.object_mesh);
    for (let i = 1; i <= Math.min(Math.round(poseCursor), frames.length - 1); i++) {
      drawLine(frames[i - 1].tool_position, frames[i].tool_position, '#FB923C', 1.5);
    }
    drawSkeleton(frame?.joints || {}, '#38BDF8');
    drawCorrections(evaluationData.layers.corrected, frame);
    drawExpertInset(evaluationData.layers.expert);
  }

  function resizePose() {
    const ratio = window.devicePixelRatio || 1;
    const rect = poseCanvas.getBoundingClientRect();
    poseCanvas.width = Math.max(1, Math.round(rect.width * ratio));
    poseCanvas.height = Math.max(1, Math.round(rect.height * ratio));
    poseCtx.setTransform(ratio, 0, 0, ratio, 0, 0);
  }
  function poseTick(now) {
    if (posePlaying && evaluationData && activeTab === 'pose') {
      if (!poseLast) poseLast = now;
      const speed = Number(byId('speed').value || 1);
      poseCursor += Math.min(.1, (now - poseLast) / 1000) * 30 * speed;
      poseLast = now;
      const max = Number(timelineInput.max);
      if (poseCursor > max) poseCursor = 0;
      timelineInput.value = String(Math.round(poseCursor));
      drawPose();
    }
    requestAnimationFrame(poseTick);
  }

  poseCanvas.onpointerdown = event => {
    if (event.button !== 0 && event.button !== 1) return;
    event.preventDefault(); poseCanvas.setPointerCapture(event.pointerId);
    poseDrag = {mode: event.button === 1 ? 'pan' : 'rotate', x: event.clientX, y: event.clientY, yaw: poseYaw, pitch: posePitch, panX: posePanX, panY: posePanY};
  };
  poseCanvas.onpointermove = event => {
    if (!poseDrag) return;
    if (poseDrag.mode === 'pan') { posePanX = poseDrag.panX + event.clientX - poseDrag.x; posePanY = poseDrag.panY + event.clientY - poseDrag.y; }
    else { poseYaw = poseDrag.yaw + (event.clientX - poseDrag.x) * .01; posePitch = Math.max(-1.2, Math.min(1.2, poseDrag.pitch + (event.clientY - poseDrag.y) * .01)); }
    drawPose();
  };
  poseCanvas.onpointerup = event => { if (poseCanvas.hasPointerCapture(event.pointerId)) poseCanvas.releasePointerCapture(event.pointerId); poseDrag = null; };
  poseCanvas.onwheel = event => { event.preventDefault(); poseZoom = Math.max(70, Math.min(420, poseZoom * (event.deltaY > 0 ? .9 : 1.1))); drawPose(); };
  poseCanvas.onauxclick = event => event.preventDefault();

  document.querySelectorAll('.viewer-tab').forEach(button => button.onclick = () => setViewerTab(button.dataset.tab));
  sessionSelect.onchange = () => { ++evaluationLoadGeneration; fillSequences(); };
  byId('loadEvaluation').onclick = loadEvaluation;
  addEventListener('resize', () => { resizePose(); drawPose(); });
  setViewerTab(document.querySelector('.viewer-tabs').dataset.defaultTab || 'pose');
  loadEvaluationCatalog();
  requestAnimationFrame(poseTick);
})();
