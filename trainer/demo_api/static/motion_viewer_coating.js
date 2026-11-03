(() => {
  'use strict';
  const canvas = document.getElementById('coatingCanvas');
  const zoneLabel = document.getElementById('coatingZone');
  const status = document.getElementById('coatingStatus');
  const view2d = document.getElementById('coatingView2d');
  const view3d = document.getElementById('coatingView3d');
  const scaleRelative = document.getElementById('coatingScaleRelative');
  const scaleAbsolute = document.getElementById('coatingScaleAbsolute');
  const legend = document.getElementById('coatingLegend');
  const scaleNote = document.getElementById('coatingScaleNote');
  const geometryNote = document.getElementById('coatingGeometryNote');
  if (!canvas || !zoneLabel || !status || !view2d || !view3d || !scaleRelative || !scaleAbsolute || !legend || !scaleNote || !geometryNote) return;
  const context = canvas.getContext('2d');
  let payload = null;
  let zoneKey = null;
  let mode = '2d';
  let colorMode = 'relative';
  let yaw = -.7;
  let pitch = .72;
  let zoom = 1;
  let drag = null;

  function fitCanvas() {
    const ratio = window.devicePixelRatio || 1;
    const bounds = canvas.getBoundingClientRect();
    const width = Math.max(1, Math.round(bounds.width * ratio));
    const height = Math.max(1, Math.round(bounds.height * ratio));
    if (canvas.width !== width || canvas.height !== height) {
      canvas.width = width;
      canvas.height = height;
    }
    context.setTransform(ratio, 0, 0, ratio, 0, 0);
    return {width: bounds.width, height: bounds.height};
  }

  function heatColor(value) {
    const stops = [
      [25, 70, 160], [28, 178, 193], [49, 203, 122], [255, 218, 92], [255, 104, 72],
    ];
    const scaled = Math.max(0, Math.min(1, value)) * (stops.length - 1);
    const index = Math.floor(scaled), mix = scaled - index;
    const left = stops[index], right = stops[Math.min(stops.length - 1, index + 1)];
    return `rgb(${left.map((item, axis) => Math.round(item + (right[axis] - item) * mix)).join(',')})`;
  }

  function currentReference() {
    return zoneKey ? payload?.coating_reference?.[zoneKey] || null : null;
  }

  function validCells(reference) {
    if (!Array.isArray(reference?.cells)) return [];
    return reference.cells.filter(cell => Array.isArray(cell) && cell.length >= 3
      && cell.slice(0, 3).every(value => typeof value === 'number' && Number.isFinite(value))
      && cell[2] >= 0);
  }

  function absoluteScale(reference) {
    const scale = reference?.color_scale_um;
    return reference?.cell_unit === 'um' && reference.physical_calibration === 'available'
      && Number.isFinite(scale?.minimum) && Number.isFinite(scale?.maximum)
      && scale.minimum >= 0 && scale.maximum > scale.minimum ? scale : null;
  }

  function colorRange(reference, cells) {
    const physical = absoluteScale(reference);
    if (colorMode === 'absolute' && physical) return {...physical, absolute: true};
    const heights = cells.map(cell => cell[2]);
    return {minimum: Math.min(...heights), maximum: Math.max(...heights), absolute: false};
  }

  function normalizedValue(value, range) {
    const span = range.maximum - range.minimum;
    return span > 0 ? Math.max(0, Math.min(1, (value - range.minimum) / span)) : .5;
  }

  function formatNumber(value) {
    if (value > 0 && value < .01) return value.toPrecision(3);
    return String(Number(value.toFixed(value < 10 ? 2 : 1)));
  }

  function renderLegend(labels) {
    legend.replaceChildren(...labels.map(label => {
      const tick = document.createElement('span');
      tick.textContent = label;
      return tick;
    }));
  }

  function renderScale(reference, cells) {
    const physical = absoluteScale(reference);
    const absolute = colorMode === 'absolute' && !!physical && cells.length > 0;
    scaleAbsolute.disabled = !physical || !cells.length;
    for (const [button, active] of [[scaleRelative, !absolute], [scaleAbsolute, absolute]]) {
      button.classList.toggle('active', active);
      button.setAttribute('aria-pressed', String(active));
    }
    geometryNote.textContent = mode === '3d'
      ? '훈련자 진행 도막 아님 · 3D 높이는 시각적 강조'
      : '훈련자의 진행 도막이 아닌 전문가 최종 기준입니다.';
    geometryNote.title = '3D 높이는 분포를 읽기 위한 시각적 강조이며 실제 표면 형상·축척이 아닙니다.';
    if (!cells.length) {
      legend.textContent = '색상 척도 자료 없음';
      scaleNote.textContent = '표시할 도막 셀이 없습니다.';
      return;
    }
    if (absolute) {
      renderLegend([0, .25, .5, .75, 1].map(fraction =>
        formatNumber(physical.minimum + (physical.maximum - physical.minimum) * fraction)
          + (fraction === 1 ? ' µm' : '')
      ));
      const clipped = cells.filter(cell => cell[2] < physical.minimum || cell[2] > physical.maximum).length;
      scaleNote.textContent = `모든 Zone 공통 0–${formatNumber(physical.maximum)} µm · 범위 밖 ${clipped}셀`;
      scaleNote.title = '로드한 프로필의 물리 두께 셀 최댓값 기준입니다. 목표 두께가 아닙니다. 범위를 벗어난 색상은 양 끝으로 제한하며 원래 두께 값은 유지합니다.';
    } else {
      renderLegend(['0%', '25%', '50%', '75%', '100%']);
      const range = colorRange(reference, cells);
      const units = physical ? ' µm' : ' (보정 전 값)';
      scaleNote.textContent = physical
        ? `Zone 내 상대 분포 · ${formatNumber(range.minimum)}–${formatNumber(range.maximum)}${units}`
        : 'Zone 내 상대 분포 · 물리 두께 보정 정보 없음';
      scaleNote.title = 'Zone의 최솟값과 최댓값을 각각 0%, 100%에 대응합니다. 균일한 값은 중간색으로 표시합니다. '
        + (physical ? 'Zone 간 색상 비교는 절대 두께를 선택하세요.' : '물리 보정이 없어 절대 두께를 표시할 수 없습니다.');
    }
  }

  function drawEmpty(width, height, message) {
    context.fillStyle = '#07101f';
    context.fillRect(0, 0, width, height);
    context.fillStyle = '#91a0b8';
    context.font = '12px system-ui';
    context.textAlign = 'center';
    context.fillText(message, width / 2, height / 2);
  }

  function drawDistribution(reference, cells, range, width, height) {
    const bounds = reference.bounds;
    const columns = Math.max(1, bounds.max_x - bounds.min_x + 1);
    const rows = Math.max(1, bounds.max_y - bounds.min_y + 1);
    const padding = 18;
    const scale = Math.min((width - padding * 2) / columns, (height - padding * 2) / rows);
    const drawWidth = columns * scale, drawHeight = rows * scale;
    const startX = (width - drawWidth) / 2, startY = (height - drawHeight) / 2;
    context.fillStyle = '#07101f';
    context.fillRect(0, 0, width, height);
    for (const [x, y, heightValue] of cells) {
      const normalized = normalizedValue(heightValue, range);
      context.fillStyle = heatColor(normalized);
      context.fillRect(
        startX + (x - bounds.min_x) * scale,
        startY + (bounds.max_y - y) * scale,
        Math.max(1.2, scale + .35), Math.max(1.2, scale + .35),
      );
    }
    context.strokeStyle = 'rgba(211,236,255,.65)';
    context.lineWidth = 1;
    context.strokeRect(startX - 1, startY - 1, drawWidth + 2, drawHeight + 2);
  }

  function drawTerrain(reference, cells, range, width, height) {
    const bounds = reference.bounds;
    const columns = Math.max(1, bounds.max_x - bounds.min_x + 1);
    const rows = Math.max(1, bounds.max_y - bounds.min_y + 1);
    const cellScale = Math.max(2.5, Math.min(10, Math.min(width / (columns + rows), height / (columns + rows)) * 1.8)) * zoom;
    const verticalScale = Math.max(22, Math.min(68, height * .28)) * zoom;
    const cos = Math.cos(yaw), sin = Math.sin(yaw);
    const centerX = width * .5, centerY = height * .67;
    const polygons = cells.map(([x, y, heightValue]) => {
      const gx = x - (bounds.min_x + bounds.max_x) / 2;
      const gy = y - (bounds.min_y + bounds.max_y) / 2;
      const rotatedX = gx * cos - gy * sin;
      const rotatedY = gx * sin + gy * cos;
      const normalized = normalizedValue(heightValue, range);
      const project = (dx, dy, z) => [
        centerX + (rotatedX + dx * cos - dy * sin) * cellScale,
        centerY + (rotatedY + dx * sin + dy * cos) * cellScale * Math.sin(pitch) - z * verticalScale,
      ];
      return {
        depth: rotatedY,
        normalized,
        points: [project(-.48, -.48, normalized), project(.48, -.48, normalized), project(.48, .48, normalized), project(-.48, .48, normalized)],
      };
    }).sort((left, right) => left.depth - right.depth);
    context.fillStyle = '#07101f';
    context.fillRect(0, 0, width, height);
    context.strokeStyle = 'rgba(113,174,207,.18)';
    for (let row = -4; row <= 4; row += 1) {
      context.beginPath(); context.moveTo(12, centerY + row * 14); context.lineTo(width - 12, centerY + row * 14); context.stroke();
    }
    for (const polygon of polygons) {
      context.beginPath();
      polygon.points.forEach((point, index) => index ? context.lineTo(...point) : context.moveTo(...point));
      context.closePath();
      context.fillStyle = heatColor(polygon.normalized);
      context.fill();
      context.strokeStyle = 'rgba(6,18,31,.32)';
      context.lineWidth = .7;
      context.stroke();
    }
  }

  function renderCoatingZone() {
    const {width, height} = fitCanvas();
    const reference = currentReference();
    const cells = validCells(reference);
    renderScale(reference, cells);
    if (!payload) {
      zoneLabel.textContent = 'Zone 대기 중';
      status.textContent = '도막 기준 준비 중';
      drawEmpty(width, height, '기록을 선택하세요.');
      return;
    }
    if (!zoneKey) {
      zoneLabel.textContent = '이동 구간';
      status.textContent = '분사 Zone 없음';
      drawEmpty(width, height, '현재 프레임에는 도장 Zone이 없습니다.');
      return;
    }
    const zoneId = zoneKey.split(':')[1];
    zoneLabel.textContent = `Zone ${String(zoneId).padStart(2, '0')}`;
    if (!reference || reference.source !== 'expert_final_heightmap') {
      status.textContent = '높이맵 기준 없음';
      drawEmpty(width, height, '이 Zone의 최종 높이맵 기준이 없습니다.');
      return;
    }
    if (!cells.length || !reference.bounds || !Object.values(reference.bounds).every(Number.isFinite)) {
      status.textContent = '이 Zone에 표시할 도막 셀이 없습니다.';
      drawEmpty(width, height, '이 Zone의 유효한 도막 셀이 없습니다.');
      return;
    }
    const thickness = reference.thickness_um;
    const summary = absoluteScale(reference) && thickness && ['median', 'minimum', 'maximum'].every(key => Number.isFinite(thickness[key]))
      ? ` · 중앙 ${Number(thickness.median).toFixed(1)} µm · 범위 ${Number(thickness.minimum).toFixed(1)}–${Number(thickness.maximum).toFixed(1)} µm`
      : '';
    const mask = reference.mask_source === 'tsb_color_label_map' ? ' · TSB 공식 Zone' : '';
    status.textContent = `전문가 최종 도막 기준 · ${reference.session_count || 0}회${mask}${summary}`;
    const range = colorRange(reference, cells);
    if (mode === '3d') drawTerrain(reference, cells, range, width, height);
    else drawDistribution(reference, cells, range, width, height);
  }

  function selectMode(next) {
    mode = next;
    view2d.classList.toggle('active', mode === '2d');
    view3d.classList.toggle('active', mode === '3d');
    renderCoatingZone();
  }

  view2d.onclick = () => selectMode('2d');
  view3d.onclick = () => selectMode('3d');
  scaleRelative.onclick = () => { colorMode = 'relative'; renderCoatingZone(); };
  scaleAbsolute.onclick = () => {
    if (scaleAbsolute.disabled) return;
    colorMode = 'absolute';
    renderCoatingZone();
  };
  canvas.onpointerdown = event => {
    if (mode !== '3d' || event.button !== 0) return;
    canvas.setPointerCapture(event.pointerId);
    drag = {x: event.clientX, y: event.clientY, yaw, pitch};
  };
  canvas.onpointermove = event => {
    if (!drag) return;
    yaw = drag.yaw + (event.clientX - drag.x) * .012;
    pitch = Math.max(.25, Math.min(1.25, drag.pitch + (event.clientY - drag.y) * .008));
    renderCoatingZone();
  };
  canvas.onpointerup = event => {
    if (canvas.hasPointerCapture(event.pointerId)) canvas.releasePointerCapture(event.pointerId);
    drag = null;
  };
  canvas.onwheel = event => {
    if (mode !== '3d') return;
    event.preventDefault();
    zoom = Math.max(.55, Math.min(2.2, zoom * (event.deltaY > 0 ? .9 : 1.1)));
    renderCoatingZone();
  };
  addEventListener('motion-viewer-data', event => {
    payload = event.detail;
    zoneKey = null;
    renderCoatingZone();
  });
  addEventListener('motion-viewer-frame', event => {
    const frame = event.detail?.frame;
    const next = frame && frame.spraying && Number(frame.zone_id) > 0 && frame.map_num != null
      ? `${frame.map_num}:${frame.zone_id}` : null;
    if (!next || next === zoneKey) return;
    zoneKey = next;
    renderCoatingZone();
  });
  new ResizeObserver(renderCoatingZone).observe(canvas.parentElement);
  renderCoatingZone();
})();
