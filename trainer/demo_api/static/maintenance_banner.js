// 점검 배너. /api/maintenance 를 60초마다, 그리고 화면을 다시 볼 때 확인한다.
// 요청이 실패하면 배너를 그대로 둔다(agent-api 가 내려가면 화면도 안 뜨니, 그때 알림은 Slack 이 맡는다).
// 시험(호스트): node --test trainer/demo_api/tests/test_maintenance_banner.js
(() => {
  'use strict';
  const KST_MS = 9 * 3600 * 1000; // 한국은 서머타임이 없다
  const pad = (n) => String(n).padStart(2, '0');

  /** 한국 시각. 오늘이면 "10:30", 다른 날이면 "11월 18일 10:30". */
  const kstTime = (end, now) => {
    const e = new Date(end.getTime() + KST_MS);
    const n = new Date(now.getTime() + KST_MS);
    const time = `${pad(e.getUTCHours())}:${pad(e.getUTCMinutes())}`;
    const sameDay = e.getUTCFullYear() === n.getUTCFullYear() && e.getUTCMonth() === n.getUTCMonth()
      && e.getUTCDate() === n.getUTCDate();
    return sameDay ? time : `${e.getUTCMonth() + 1}월 ${e.getUTCDate()}일 ${time}`;
  };

  /** 보일 문구. 점검이 아니거나 기한이 지났으면 빈 문자열. */
  const bannerText = (state, now = new Date()) => {
    if (!state || state.active !== true) return '';
    const end = new Date(state.until);
    if (!(end.getTime() > now.getTime())) return '';
    return ['점검 중', state.impact, `${kstTime(end, now)}까지(한국 시각)`].filter(Boolean).join(' · ');
  };

  function mount(doc, fetchJson, { intervalMs = 60000, setIntervalFn = setInterval } = {}) {
    const banner = doc.getElementById('copilot-maintenance');
    if (!banner) return null;
    const label = banner.querySelector('[data-maintenance-text]');
    let state = null;
    // 아래 화면들의 높이 계산(100dvh - 상단 바 - 배너)이 이 값을 쓴다.
    const sync = () => doc.documentElement.style.setProperty('--copilot-banner-height',
      banner.hidden ? '0px' : `${banner.offsetHeight}px`);
    const render = () => {
      const text = bannerText(state);
      if (text) label.textContent = text;
      banner.hidden = !text;
      sync();
    };
    const check = () => fetchJson('/api/maintenance').then((next) => { state = next; render(); }, () => {});
    if (typeof ResizeObserver === 'function') new ResizeObserver(sync).observe(banner); // 휴대폰에서 줄이 바뀔 때
    setIntervalFn(check, intervalMs);
    doc.addEventListener('visibilitychange', () => { if (doc.visibilityState === 'visible') check(); });
    return check();
  }

  const api = { bannerText, kstTime, mount };
  globalThis.copilotMaintenance = api;
  if (typeof module === 'object' && module.exports) module.exports = api;
  if (typeof document === 'object' && typeof fetch === 'function') {
    mount(document, (url) => fetch(url, { cache: 'no-store' })
      .then((response) => (response.ok ? response.json() : Promise.reject(response.status))));
  }
})();
