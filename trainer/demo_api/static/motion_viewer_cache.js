/* Real, bounded payload cache for this page. Nothing is persisted to storage. */
(function (root, factory) {
  const api = factory();
  if (typeof module === 'object' && module.exports) module.exports = api;
  else root.MotionViewerCache = api;
})(typeof globalThis !== 'undefined' ? globalThis : this, function () {
  'use strict';
  function create({ fetchJson, now = Date.now, ttlMs = 300000, maxEntries = 3,
                    maxBytes = 33554432, onChange = () => {} }) {
    const entries = new Map(), pending = new Map();
    let bytes = 0;
    function remove(url) {
      const entry = entries.get(url);
      if (entry) { bytes -= entry.bytes; entries.delete(url); onChange(); }
    }
    function has(url) {
      const entry = entries.get(url);
      if (entry && now() - entry.created < ttlMs) return true;
      if (entry) remove(url);
      return false;
    }
    async function validated(url, entry, signal) {
      try {
        const status = await fetchJson('/motion-viewer/api/cache-status?token=' + encodeURIComponent(entry.token), { signal });
        if (entries.get(url) !== entry) return false;
        if (status.valid === true && now() - entry.created < ttlMs) return true;
      } catch { /* Unverifiable sources must be fetched again. */ }
      remove(url);
      return false;
    }
    function emit(state, value) {
      for (const callback of state.listeners) {
        try { callback(value); } catch { /* A view callback cannot break a shared request. */ }
      }
    }
    function startPolling(state) {
      if (!state.loadId || state.timer || !state.listeners.size || state.finished) return;
      let busy = false, seen = false;
      const poll = async () => {
        if (busy || state.finished || state.controller.signal.aborted) return;
        busy = true;
        try {
          const value = await fetchJson('/motion-viewer/api/load-progress?load_id=' + encodeURIComponent(state.loadId),
            { signal: state.pollController.signal });
          if (state.finished || state.controller.signal.aborted) return;
          if (value.status !== 'unavailable') { seen = true; emit(state, value); }
          else if (seen) emit(state, { status: 'running', stage: 'progress_unavailable', stage_label: '진행도 상태가 만료되었거나 더 이상 제공되지 않습니다.', completed: 0, total: null });
        } catch { /* Loading still has its own failure and cancellation result. */ }
        finally { busy = false; }
      };
      state.timer = setInterval(poll, 350);
      poll();
    }
    async function read(url, state) {
      if (has(url)) {
        const entry = entries.get(url);
        if (await validated(url, entry, state.controller.signal)) {
          entries.delete(url); entries.set(url, entry);
          emit(state, { status: 'running', stage: 'cache_restore', stage_label: '이 창의 캐시에서 복원', completed: 1, total: 1, unit: 'records' });
          return entry.json;
        }
      }
      if (state.controller.signal.aborted) throw new DOMException('Load cancelled', 'AbortError');
      const random = new Uint8Array(16);
      globalThis.crypto.getRandomValues(random);
      state.loadId = [...random].map(value => value.toString(16).padStart(2, '0')).join('');
      emit(state, { status: 'running', stage: 'request', stage_label: '서버 작업 시작 대기', completed: 0, total: null });
      startPolling(state);
      const payload = await fetchJson(url, { headers: { 'X-Motion-Load-ID': state.loadId }, signal: state.controller.signal });
      if (state.controller.signal.aborted) throw new DOMException('Load cancelled', 'AbortError');
      const json = JSON.stringify(payload);
      emit(state, { status: 'running', stage: 'received', stage_label: '재생 자료 수신 완료', completed: 0, total: null });
      const token = payload.viewer_cache?.token, size = json.length * 2;
      if (typeof token === 'string' && size <= maxBytes) {
        remove(url);
        while (entries.size >= maxEntries || bytes + size > maxBytes) remove(entries.keys().next().value);
        entries.set(url, { json, token, bytes: size, created: now() }); bytes += size;
        onChange();
      }
      return json;
    }
    function load(url, { signal, onProgress } = {}) {
      if (signal?.aborted) return Promise.reject(new DOMException('Load cancelled', 'AbortError'));
      let state = pending.get(url);
      if (!state || state.controller.signal.aborted) {
        state = { controller: new AbortController(), pollController: new AbortController(), listeners: new Set(),
          consumers: new Set(), loadId: null, timer: null, finished: false };
        const active = state;
        active.promise = Promise.resolve().then(() => read(url, active)).finally(() => {
          active.finished = true;
          clearInterval(active.timer);
          active.pollController.abort();
          if (pending.get(url) === active) pending.delete(url);
        });
        pending.set(url, active);
      }
      const active = state, consumer = {}, listener = onProgress ? value => onProgress(value) : null;
      active.consumers.add(consumer);
      if (listener) { active.listeners.add(listener); startPolling(active); }
      return new Promise((resolve, reject) => {
        let detached = false;
        const detach = () => {
          if (detached) return;
          detached = true;
          active.consumers.delete(consumer);
          if (listener) active.listeners.delete(listener);
          signal?.removeEventListener('abort', cancel);
          if (!active.consumers.size && !active.finished) {
            active.controller.abort(); active.pollController.abort(); clearInterval(active.timer);
            if (pending.get(url) === active) pending.delete(url);
          }
        };
        const cancel = () => { detach(); reject(new DOMException('Load cancelled', 'AbortError')); };
        signal?.addEventListener('abort', cancel, { once: true });
        active.promise.then(json => {
          if (detached) return;
          detach();
          try { resolve(JSON.parse(json)); } catch (error) { reject(error); }
        }, error => { if (!detached) { detach(); reject(error); } });
      });
    }
    async function refresh() {
      await Promise.all([...entries].map(async ([url, entry]) => {
        if (has(url)) await validated(url, entry);
      }));
      onChange();
    }
    return { load, has, refresh };
  }
  return { create };
});
