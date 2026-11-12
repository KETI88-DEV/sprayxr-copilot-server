(() => {
  'use strict';

  const clamp = (value, low, high) => Math.max(low, Math.min(high, value));
  const add = (a, b) => a.map((value, index) => value + b[index]);
  const scale = (value, amount) => value.map(item => item * amount);
  const copyState = state => ({...state, target: [...state.target]});

  function create(initial = {}) {
    let state = {
      target: [...(initial.target || [0, 0, 0])],
      yaw: Number(initial.yaw ?? -0.55),
      pitch: Number(initial.pitch ?? 0.1),
      distance: Number(initial.distance ?? 175),
    };
    let homeState = copyState(state);

    const axes = () => {
      const cy = Math.cos(state.yaw), sy = Math.sin(state.yaw);
      const cp = Math.cos(state.pitch), sp = Math.sin(state.pitch);
      return {
        right: [cy, 0, -sy],
        up: [-sy * sp, cp, -cy * sp],
        forward: [sy * cp, sp, cy * cp],
      };
    };

    const fit = points => {
      const valid = (points || []).filter(point => Array.isArray(point) && point.length >= 3 && point.every(Number.isFinite));
      if (!valid.length) return copyState(state);
      const min = [0, 1, 2].map(axis => Math.min(...valid.map(point => point[axis])));
      const max = [0, 1, 2].map(axis => Math.max(...valid.map(point => point[axis])));
      state.target = min.map((value, axis) => (value + max[axis]) / 2);
      const radius = Math.max(.25, ...max.map((value, axis) => value - min[axis]));
      state.distance = clamp(290 / radius, 55, 520);
      homeState = copyState(state);
      return copyState(state);
    };

    const api = {
      state: () => copyState(state),
      fit,
      orbit(dx, dy) {
        state.yaw += dx * .01;
        state.pitch = clamp(state.pitch + dy * .01, -1.45, 1.45);
        return copyState(state);
      },
      pan(dx, dy, viewportHeight = 600) {
        const basis = axes();
        const worldPerPixel = 2.2 / Math.max(80, state.distance) * (600 / Math.max(1, viewportHeight));
        state.target = add(state.target, add(scale(basis.right, -dx * worldPerPixel), scale(basis.up, dy * worldPerPixel)));
        return copyState(state);
      },
      dolly(delta) {
        state.distance = clamp(state.distance * (delta > 0 ? .9 : 1.1), 45, 620);
        return copyState(state);
      },
      look(dx, dy) { return api.orbit(dx, dy); },
      fly(forward, right, up, amount = .035) {
        const basis = axes();
        state.target = add(state.target, add(add(scale(basis.forward, forward * amount), scale(basis.right, right * amount)), [0, up * amount, 0]));
        return copyState(state);
      },
      focus(points) { return fit(points); },
      axisView(axis) {
        if (axis === 'x') { state.yaw = Math.PI / 2; state.pitch = 0; }
        if (axis === 'y') { state.yaw = 0; state.pitch = -Math.PI / 2 + .001; }
        if (axis === 'z') { state.yaw = 0; state.pitch = 0; }
        return copyState(state);
      },
      home() { state = copyState(homeState); return copyState(state); },
    };
    return api;
  }

  // 우하단 축 표시와 X·Y·Z 버튼이 함께 쓰는 색. 3D 공간(#2e3646~#1c2230)에서 3:1 이상이고 팔 청록·Zone 원색·강조 파랑과 겹치지 않게 채도를 낮췄다
  const AXIS_COLORS = {x: '#e8767f', y: '#9bc77a', z: '#8aa6e8'};

  // 장면과 같은 회전(transform: 원점 기준 점 -> [x, y(위), 깊이])으로 세 단위축이 화면에서 가리키는 쪽을 낸다. dx 는 오른쪽, dy 는 아래가 양수, 먼 축부터
  function axisGizmo(transform) {
    return Object.keys(AXIS_COLORS).map((axis, index) => {
      const unit = [0, 0, 0]; unit[index] = 1;
      const [x, y, depth] = transform(unit, [0, 0, 0]);
      return {axis, dx: x, dy: -y, depth, color: AXIS_COLORS[axis]};
    }).sort((a, b) => b.depth - a.depth);
  }

  window.MotionViewerCamera = {create, AXIS_COLORS, axisGizmo};
})();
