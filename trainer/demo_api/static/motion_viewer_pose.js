(() => {
  'use strict';
  const byId = id => document.getElementById(id);
  const template = byId('poseUiTemplate');
  const pathCanvas = byId('canvas');
  if (!template || !pathCanvas || !window.MotionViewerCamera) return;
  const layerLabels = ['측정 작업자', '전문가 기준', '교정 목표', '별도 기준 좌표'];
  const sharedControls = {timeline: byId('timeline'), play: byId('play')};
  pathCanvas.setAttribute('aria-description', layerLabels.join(' · '));

  const fragment = template.content.cloneNode(true);
  const sourceControls = fragment.querySelector('[data-pose-slot="source"]');
  const gizmo = fragment.querySelector('[data-pose-slot="gizmo"]');
  const summary = fragment.querySelector('[data-pose-slot="summary"]');
  document.querySelector('.workspace > .side:not(.right)').prepend(sourceControls);
  document.querySelector('.workspace > .stage').append(gizmo);
  document.querySelector('.workspace > .right').prepend(summary);

  const poseZoneNav = document.createElement('div');
  poseZoneNav.id = 'poseZoneNav';
  poseZoneNav.className = 'zone-order';
  summary.prepend(poseZoneNav);

  let evaluationLoadGeneration = 0;
  let evaluationData = null;
  let evaluationCatalog = null;
  let selectedPoseZone = 'auto';
  let camera = window.MotionViewerCamera.create({target: [0, 0, 0], yaw: -.55, pitch: .1, distance: 175});
  let cameraDrag = null;
  const pressedKeys = new Set();
  const sessionSelect = byId('evaluationSession');
  const sequenceSelect = byId('evaluationSequence');

  const syncCameraGlobals = state => {
    viewCenter = [...state.target];
    yaw = state.yaw;
    pitch = state.pitch;
    zoom = state.distance;
    panX = 0;
    panY = 0;
  };

  const visiblePoints = payload => {
    const points = [];
    for (const item of payload?.reference_surface?.points || []) if (Array.isArray(item.point)) points.push(item.point);
    for (const surface of payload?.zone_surfaces || []) points.push(...(surface.vertices || []));
    for (const frame of payload?.frames || []) {
      for (const item of Object.values(frame.tracked || {})) if (Array.isArray(item)) points.push(item);
    }
    return points;
  };

  const originalResetView = resetViewToCamera;
  resetViewToCamera = payload => {
    originalResetView(payload);
    camera = window.MotionViewerCamera.create({target: viewCenter || [0, 0, 0], yaw, pitch, distance: zoom});
    syncCameraGlobals(camera.fit(visiblePoints(payload)));
  };

  const setState = (state, message = '') => {
    sourceControls.dataset.state = state;
    if (state !== 'ready' && message) byId('status').textContent = message;
  };

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
      if (sessions.length) await loadEvaluation();
      else setState('empty', '저장된 TSB 시퀀스가 없어 전문가 CSV 경로만 표시합니다.');
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
      draw();
      return;
    }
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
      selectedPoseZone = 'auto';
      renderSummary(incoming);
      setState('ready');
      draw();
    } catch (error) {
      if (generation !== evaluationLoadGeneration) return;
      evaluationData = null;
      setState('error', error.message || '자세 평가를 불러오지 못했습니다.');
      draw();
    }
  }

  function renderSummary(view) {
    const evaluation = view.evaluation;
    const identity = view.identity;
    const correction = evaluation.top_correction;
    byId('poseVerdict').textContent = evaluation.verdict;
    byId('poseScore').textContent = evaluation.similarity_score == null ? '측정 불가' : `${Math.round(evaluation.similarity_score * 100)}% / ${evaluation.expert_percentile ?? '—'}백분위`;
    byId('poseDeviation').textContent = correction?.joint || '통상 범위';
    byId('poseAction').textContent = correction?.message || evaluation.message;
    byId('poseConfidence').textContent = evaluation.confidence;
    byId('poseZoneSummary').textContent = `Zone ${identity.zone_id || '—'} · ${identity.sequence_id} · ${view.strokes.length} 스트로크`;
    poseZoneNav.replaceChildren();
    for (const [key, label, color] of [['all', '전체', '#67E8F9'], [identity.zone_id || 'unknown', `Zone ${identity.zone_id || '—'}`, '#22D3EE']]) {
      const button = document.createElement('button');
      button.className = 'zone-chip';
      button.textContent = label;
      button.style.setProperty('--zone-color', color);
      button.onclick = () => {
        playing = false;
        sharedControls.play.textContent = '재생';
        selectedPoseZone = key;
        poseZoneNav.querySelectorAll('.zone-chip').forEach(item => item.classList.toggle('selected', item === button));
        byId('poseZoneSummary').textContent = key === 'all' ? `전체 · ${identity.sequence_id}` : `${label} · ${identity.sequence_id}`;
        draw();
      };
      poseZoneNav.append(button);
    }
    byId('poseAdvancedJson').textContent = JSON.stringify({
      score_source: '운영 pose_recomm 엔진',
      expert_layer_source: view.layers.expert.available ? '엔진 제공 전문가 기준' : '제공되지 않음',
      analysis_status: evaluation.analysis_status,
      reason_code: evaluation.reason_code,
      strokes: view.strokes,
    }, null, 2);
  }

  const bonePairs = [
    ['Pelvis','Spine'],['Spine','Chest'],['Chest','Head'],
    ['Chest','Left Shoulder'],['Left Shoulder','Left Upper Arm'],['Left Upper Arm','Left Fore Arm'],['Left Fore Arm','Left Hand'],
    ['Chest','Right Shoulder'],['Right Shoulder','Right Upper Arm'],['Right Upper Arm','Right Fore Arm'],['Right Fore Arm','Right Hand'],
  ];
  const jointOf = (joints, name) => {
    const normalized = name.replaceAll(' ', '').toLowerCase();
    const key = Object.keys(joints || {}).find(item => item.replaceAll(' ', '').toLowerCase() === normalized);
    return key ? joints[key] : null;
  };
  const drawSkeleton = (joints, color, dashed = false) => {
    ctx.save();
    if (dashed) ctx.setLineDash([7, 5]);
    for (const [a, b] of bonePairs) {
      const start = jointOf(joints, a), end = jointOf(joints, b);
      if (start && end) line(start, end, color, 2.4, viewCenter, canvas.clientWidth, canvas.clientHeight);
    }
    for (const item of Object.values(joints || {})) point(item, 3.2, color, viewCenter, canvas.clientWidth, canvas.clientHeight);
    ctx.restore();
  };
  const frameForProgress = (frames, progress) => frames[Math.max(0, Math.min(frames.length - 1, Math.round(progress * (frames.length - 1))))];

  function drawPoseOverlay({durationSeconds}) {
    if (!evaluationData) return;
    const progress = Math.max(0, Math.min(1, playheadSeconds / Math.max(.001, durationSeconds || 1)));
    const measured = evaluationData.layers.measured;
    const expert = evaluationData.layers.expert;
    const corrected = evaluationData.layers.corrected;
    const measuredFrame = frameForProgress(measured.frames || [], progress);
    if (byId('poseLayerMeasured').checked && measuredFrame?.joints) drawSkeleton(measuredFrame.joints, '#38BDF8');
    if (byId('poseLayerExpert').checked && expert?.available) {
      const joints = Object.fromEntries(Object.entries(expert.tracks || {}).map(([name, track]) => [name, frameForProgress(track, progress)]));
      drawSkeleton(joints, '#22C55E', true);
    }
    if (byId('poseLayerCorrection').checked && measuredFrame) {
      for (const target of corrected?.targets || []) {
        if (Number(target.index) !== Number(measuredFrame.index)) continue;
        const start = jointOf(measuredFrame.joints, String(target.joint || ''));
        const end = target.point || target.position || target.target;
        if (start && end) arrow(start, end, '#FACC15', viewCenter, canvas.clientWidth, canvas.clientHeight);
      }
    }
    if (selectedPoseZone === 'auto') {
      const strokes = evaluationData.strokes || [];
      const strokeNumber = strokes.length ? Math.min(strokes.length, Math.floor(progress * strokes.length) + 1) : 0;
      byId('poseZoneSummary').textContent = `현재 Zone ${evaluationData.identity.zone_id || '—'} · ${evaluationData.identity.sequence_id} · 스트로크 ${strokeNumber}/${strokes.length}`;
    }
  }
  window.motionViewerPoseOverlay = {draw: drawPoseOverlay};

  const applyCamera = state => { syncCameraGlobals(state); draw(); };
  const refit = () => applyCamera(camera.focus(visiblePoints(data)));
  byId('cameraHome').onclick = () => applyCamera(camera.home());
  byId('cameraAxisX').onclick = () => applyCamera(camera.axisView('x'));
  byId('cameraAxisY').onclick = () => applyCamera(camera.axisView('y'));
  byId('cameraAxisZ').onclick = () => applyCamera(camera.axisView('z'));

  pathCanvas.onpointerdown = event => {
    if (![0, 1, 2].includes(event.button)) return;
    event.preventDefault();
    pathCanvas.setPointerCapture(event.pointerId);
    cameraDrag = {mode: event.button === 1 ? 'pan' : event.button === 2 ? 'look' : 'orbit', x: event.clientX, y: event.clientY};
    pathCanvas.style.cursor = cameraDrag.mode === 'pan' ? 'move' : 'grabbing';
  };
  pathCanvas.onpointermove = event => {
    if (!cameraDrag) return;
    const dx = event.clientX - cameraDrag.x, dy = event.clientY - cameraDrag.y;
    cameraDrag.x = event.clientX; cameraDrag.y = event.clientY;
    if (cameraDrag.mode === 'pan') applyCamera(camera.pan(dx, dy, pathCanvas.clientHeight));
    else if (cameraDrag.mode === 'look') applyCamera(camera.look(dx, dy));
    else applyCamera(camera.orbit(dx, dy));
  };
  pathCanvas.onpointerup = event => {
    if (pathCanvas.hasPointerCapture(event.pointerId)) pathCanvas.releasePointerCapture(event.pointerId);
    cameraDrag = null; pathCanvas.style.cursor = 'grab';
  };
  pathCanvas.onpointercancel = () => { cameraDrag = null; pathCanvas.style.cursor = 'grab'; };
  pathCanvas.oncontextmenu = event => event.preventDefault();
  pathCanvas.onauxclick = event => event.preventDefault();
  pathCanvas.onwheel = event => { event.preventDefault(); applyCamera(camera.dolly(event.deltaY < 0 ? 1 : -1)); };
  addEventListener('keydown', event => {
    if (event.target instanceof HTMLInputElement || event.target instanceof HTMLSelectElement) return;
    pressedKeys.add(event.code);
    if (event.code === 'KeyF') { event.preventDefault(); refit(); }
  });
  addEventListener('keyup', event => pressedKeys.delete(event.code));
  function flyTick() {
    if (cameraDrag?.mode === 'look') {
      const forward = Number(pressedKeys.has('KeyW')) - Number(pressedKeys.has('KeyS'));
      const right = Number(pressedKeys.has('KeyD')) - Number(pressedKeys.has('KeyA'));
      const up = Number(pressedKeys.has('KeyE')) - Number(pressedKeys.has('KeyQ'));
      if (forward || right || up) applyCamera(camera.fly(forward, right, up));
    }
    requestAnimationFrame(flyTick);
  }

  sessionSelect.onchange = () => { ++evaluationLoadGeneration; fillSequences(); };
  byId('loadEvaluation').onclick = loadEvaluation;
  for (const id of ['poseLayerMeasured', 'poseLayerExpert', 'poseLayerCorrection']) byId(id).onchange = draw;
  resize();
  if (typeof data !== 'undefined' && data?.frames?.length) resetViewToCamera(data);
  loadEvaluationCatalog();
  requestAnimationFrame(flyTick);
})();
