window.copilotApiDocsPlugin = ({React}) => {
  const h = React.createElement;
  const headings = /^### (입력 파라미터|출력 파라미터|실행 전 확인: 자동 사용 입력|응답 예시)[ \t]*$/gm;
  const fieldLine = /^- \x60([^\x60]+)\(([^()\x60]+)\)\x60 : (.+)$/;
  const fieldPresence = / \*\*\[(필수|선택|필수 키|생략 가능)\]\*\*$/;
  const defaultLine = /^ +- 기본값: \x60([^\x60]+)\x60$/;
  const descriptionLine = /^ {2}> ?(.*)$/;
  const exampleBlock = /^(.*?)\n\x60{3}(json|text)\n(.*)\n\x60{3}$/s;

  function splitFields(source) {
    const notes = [];
    const fields = [];
    for (const line of source.trim().split('\n')) {
      const match = line.match(fieldLine);
      if (match) {
        const [, name, type, description] = match;
        const presence = description.match(fieldPresence)?.[1];
        fields.push({name, type, presence, description: [description.replace(fieldPresence, '')], details: []});
        continue;
      }
      const field = fields.at(-1);
      const defaultValue = line.match(defaultLine)?.[1];
      const continuation = line.match(descriptionLine)?.[1];
      if (!field) {
        notes.push(line);
      } else if (continuation !== undefined) {
        field.description.push(continuation);
      } else if (defaultValue !== undefined) {
        field.defaultValue = defaultValue;
      } else if (/^ {2,}- /.test(line)) {
        field.details.push(line.trimStart());
      } else if (line.startsWith('      >')) {
        field.details.push(line.slice(4));
      } else if (line.trim()) {
        return null;
      }
    }
    return fields.length ? {fields, notes: notes.join('\n').trim()} : null;
  }

  function overviewParts(source) {
    const paragraphBreak = source.search(/\n\s*\n/);
    const paragraph = paragraphBreak === -1 ? source : source.slice(0, paragraphBreak);
    const firstSentence = paragraph.length > 240 ? paragraph.match(/^.{0,239}?[가-힣]\.(?=\s)/u) : null;
    const end = firstSentence ? firstSentence[0].length : paragraph.length;
    return [source.slice(0, end), source.slice(end).trim()];
  }

  function Panel({title, children}) {
    const [opened, setOpened] = React.useState(false);
    return h('details', {
      className: 'api-docs-panel',
      onToggle: event => {
        if (event.target === event.currentTarget && event.currentTarget.open) setOpened(true);
      },
    }, h('summary', null, title), opened && h('div', {className: 'api-docs-panel-body'}, children));
  }

  return {
    wrapComponents: {
      Markdown: (Markdown, {getComponent}) => {
        function ResponseExample({source, render}) {
          const match = source.match(exampleBlock);
          if (!match) return render(source);
          const [, description, format, example] = match;
          const HighlightCode = getComponent('HighlightCode', true);
          return h('div', {className: 'api-docs-response-example'},
            description.trim() && render(description.trim()),
            h(HighlightCode, {className: 'example', language: format === 'json' ? 'json' : undefined}, example));
        }

        function FieldTable({source, render}) {
          const parsed = splitFields(source);
          if (!parsed) return render(source);
          return h(React.Fragment, null,
            parsed.notes && h('div', {className: 'api-docs-field-notes'}, render(parsed.notes)),
            h('table', {className: 'api-docs-field-table', role: 'table'},
              h('thead', {role: 'rowgroup'}, h('tr', {role: 'row'},
                ...['필드', '타입', '필수 여부', '설명'].map(label =>
                  h('th', {key: label, scope: 'col', role: 'columnheader'}, label)))),
              h('tbody', {role: 'rowgroup'}, ...parsed.fields.map(field =>
                h('tr', {key: field.name, role: 'row'},
                  h('th', {scope: 'row', role: 'rowheader'}, h('code', null, field.name)),
                  h('td', {role: 'cell'}, h('code', null, field.type)),
                  h('td', {role: 'cell', className: 'api-docs-field-presence'},
                    field.presence && h('span', {
                      className: field.presence.startsWith('필수') ? 'api-docs-required' : 'api-docs-optional',
                    }, field.presence)),
                  h('td', {role: 'cell'},
                    render(field.description.join('\n')),
                    field.defaultValue !== undefined && h('p', {className: 'api-docs-default'},
                      '기본값 ', h('code', null, field.defaultValue)),
                    field.details.length > 0 && h('details', {className: 'api-docs-field-details'},
                      h('summary', null, '허용값·하위 항목'),
                      h('div', null, render(field.details.join('\n')))),
                  ),
                ),
              )),
            ),
          );
        }

        return function OperationDocumentation(props) {
          const parts = typeof props.source === 'string' ? props.source.split(headings) : [];
          if (parts.length < 3) return h(Markdown, props);
          const render = source => h(Markdown, {...props, source});
          const [lead, extra] = overviewParts(parts[0].trim());
          const panels = [];
          for (let index = 1; index < parts.length; index += 2) {
            const title = parts[index];
            const source = parts[index + 1].trim();
            const Component = title === '응답 예시' ? ResponseExample
              : title === '입력 파라미터' || title === '출력 파라미터' ? FieldTable : null;
            panels.push(h(Panel, {key: title, title}, Component ? h(Component, {source, render}) : render(source)));
          }
          return h('div', {className: 'api-docs-operation'},
            lead && h('div', {className: 'api-docs-operation-lead'}, render(lead)),
            extra && h(Panel, {title: '연동 설명'}, render(extra)),
            ...panels);
        };
      },
    },
  };
};

document.addEventListener('DOMContentLoaded', () => {
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
    const section = sectionFor(entry.tag);
    if (!section) return;
    event.preventDefault();
    reset();
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
  for (const link of document.querySelectorAll('.api-docs-workflows a')) {
    const entry = indexed.find(candidate => candidate.href === link.getAttribute('href'));
    if (entry) link.addEventListener('click', event => reveal(entry, event));
  }
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
});
