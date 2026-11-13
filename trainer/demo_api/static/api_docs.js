(() => {
  'use strict';
  const input = document.getElementById('api-docs-search');
  const host = document.getElementById('api-docs-results');
  const count = document.getElementById('api-docs-count');
  const clear = document.getElementById('api-docs-clear');
  const tools = document.getElementById('api-docs-tools');
  const metadata = document.getElementById('api-docs-index');
  if (!input || !host || !metadata || !count || !clear || !tools) return;
  let entries;
  try { entries = JSON.parse(metadata.textContent); } catch { return; }
  const normalize = value => String(value ?? '').normalize('NFKC').toLocaleLowerCase().replace(/[\u200b-\u200d\ufeff]/g, '');
  const indexed = entries.map(entry => ({...entry, search: normalize(`${entry.method} ${entry.path} ${entry.summary} ${entry.tag}`)}));
  const hide = () => { host.hidden = true; input.setAttribute('aria-expanded', 'false'); };
  const reset = () => {
    input.value = '';
    clear.hidden = true;
    count.textContent = `전체 ${entries.length}개 API`;
    host.replaceChildren();
    hide();
  };
  const sectionFor = tag => [...document.querySelectorAll('.opblock-tag-section')]
    .find(section => section.querySelector('.opblock-tag a')?.textContent.trim() === tag);
  const scrollTo = element => requestAnimationFrame(() => requestAnimationFrame(() => {
    element.scrollIntoView({block: 'start'});
  }));
  const nativeClick = event => event.ctrlKey || event.metaKey || event.shiftKey || event.altKey
    || (event.button != null && event.button !== 0);
  function reveal(entry, event) {
    if (nativeClick(event)) return;
    event.preventDefault();
    reset();
    const section = sectionFor(entry.tag);
    if (!section) return;
    let observer = null, timer = null;
    const select = () => {
      const operation = [...section.querySelectorAll('.opblock')].find(node =>
        node.querySelector('.opblock-summary-path')?.getAttribute('data-path') === entry.path
        && node.querySelector('.opblock-summary-method')?.textContent.trim().toUpperCase() === entry.method);
      if (!operation) return false;
      observer?.disconnect();
      if (timer) clearTimeout(timer);
      const control = operation.querySelector('.opblock-summary-control');
      if (control?.getAttribute('aria-expanded') === 'false') control.click();
      location.hash = entry.href.slice(1);
      control?.focus({preventScroll: true});
      scrollTo(operation);
      return true;
    };
    if (select()) return;
    observer = new MutationObserver(select);
    observer.observe(section, {childList: true, subtree: true});
    timer = setTimeout(() => observer.disconnect(), 3000);
    const control = section.querySelector('.opblock-tag button');
    if (control?.getAttribute('aria-expanded') === 'false') control.click();
    select();
  }
  const render = () => {
    const tokens = normalize(input.value).trim().split(/\s+/).filter(Boolean);
    if (!tokens.length) { reset(); return; }
    const matches = indexed.filter(entry => tokens.every(token => entry.search.includes(token)));
    clear.hidden = false;
    count.textContent = `${matches.length}개 검색 · 전체 ${entries.length}개`;
    host.replaceChildren();
    for (const entry of matches) {
      const link = document.createElement('a');
      link.className = 'api-docs-result';
      link.href = entry.href;
      for (const [tag, cls, text] of [['span', 'api-docs-method', entry.method], ['span', 'api-docs-path', entry.path], ['small', '', entry.summary]]) {
        const node = document.createElement(tag);
        node.className = cls;
        node.textContent = text;
        link.append(node);
      }
      link.addEventListener('click', event => reveal(entry, event));
      host.append(link);
    }
    if (!matches.length) {
      const empty = document.createElement('p');
      empty.className = 'api-docs-empty';
      empty.textContent = '일치하는 API가 없습니다. 경로나 기능 이름으로 검색해 주세요.';
      host.append(empty);
    }
    host.hidden = false;
    input.setAttribute('aria-expanded', 'true');
  };
  input.addEventListener('input', render);
  input.addEventListener('focus', () => { if (input.value.trim()) render(); });
  input.addEventListener('keydown', event => {
    if (event.key === 'Escape') { reset(); return; }
    if (event.key === 'ArrowDown') {
      const first = host.querySelector('a');
      if (first) { event.preventDefault(); first.focus(); }
    }
    if (event.key === 'Enter') {
      const links = host.querySelectorAll('a');
      if (links.length === 1) { event.preventDefault(); links[0].click(); }
    }
  });
  clear.addEventListener('click', () => { reset(); input.focus(); });
  tools.querySelectorAll('.api-docs-sections a').forEach(link => link.addEventListener('click', event => {
    if (nativeClick(event)) return;
    event.preventDefault();
    reset();
    const tag = decodeURIComponent(link.hash.slice(2));
    const section = sectionFor(tag);
    if (!section) return;
    const control = section.querySelector('.opblock-tag button');
    if (control?.getAttribute('aria-expanded') === 'false') control.click();
    location.hash = link.hash.slice(1);
    scrollTo(section.querySelector('.opblock-tag') || section);
  }));
  document.addEventListener('click', event => { if (!tools.contains(event.target)) hide(); });
  const measure = () => document.documentElement.style.setProperty('--api-docs-tools-height', `${tools.getBoundingClientRect().height}px`);
  new ResizeObserver(measure).observe(tools);
  measure();
})();
