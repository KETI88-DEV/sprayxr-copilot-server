// Run with Node. Exercises the real chat script with only its DOM/network boundary replaced.
const assert = require('node:assert/strict');
const fs = require('node:fs');
const path = require('node:path');
const vm = require('node:vm');

class Element {
  constructor(tag, text = '') { this.tagName = tag; this.children = []; this.attributes = {}; this.events = {}; this._text = text; this.value = ''; }
  set textContent(value) { this._text = value; this.children = []; }
  get textContent() { return this._text + this.children.map(child => child.textContent).join(''); }
  append(...nodes) { for (const node of nodes) { node.parent = this; this.children.push(node); } }
  replaceChildren(...nodes) { this.children = []; this.append(...nodes); }
  setAttribute(name, value) { this.attributes[name] = value; }
  addEventListener(name, handler) { (this.events[name] ||= []).push(handler); }
  async fire(name) { for (const handler of this.events[name] || []) await handler({ preventDefault() {} }); }
  focus() {}
  remove() { this.parent.children = this.parent.children.filter(child => child !== this); }
  get classList() { return { toggle: name => { const names = new Set(this.className.split(' ')); const added = !names.has(name); added ? names.add(name) : names.delete(name); this.className = [...names].join(' '); return added; } }; }
}

function setup() {
  const document = { head: new Element('head'), body: new Element('body'), createElement: tag => new Element(tag) };
  const events = {};
  const requests = [];
  const context = vm.createContext({ document, Option: class extends Element { constructor(text, value) { super('option', text); this.value = value; } },
    addEventListener: (name, handler) => (events[name] ||= []).push(handler), performance: { now: () => 1000 },
    fetch: (url, options) => new Promise(resolve => requests.push({ url, headers: options.headers, body: JSON.parse(options.body),
      finish: () => resolve({ ok: true, json: async () => ({ status: 'SUCCESS', data: { answer: '서버 평가 답변', citations: [] } }) }) })) });
  vm.runInContext(fs.readFileSync(path.join(__dirname, '../static/motion_viewer_chat.js'), 'utf8'), context);
  function all(node = document.body) { return [node, ...node.children.flatMap(all)]; }
  const find = cls => all().find(node => (node.className || '').split(' ').includes(cls));
  return { emit: (name, detail) => { for (const handler of events[name] || []) handler({ detail }); },
    requests, find, all, form: find('mvc-form'), input: all().find(node => node.tagName === 'input') };
}

const record = (id = 'RECORD-A') => ({ worker: { worker_id: 'W', display_name: '작업자 A' }, process: 'spray',
  process_label: '도장', mode_id: 3, selected_session_id: id, pose_evaluation: { status: 'available', similarity_percent: 70,
    zones: { '1:2': { map_num: '1', zone_id: '2', similarity_percent: 88, corrections: [] },
             '1:3': { map_num: '1', zone_id: '3', similarity_percent: 40, corrections: [] } }, corrections: [] },
  coating_evaluation: { analysis_status: 'available', scope: 'final_record', zones: [{ zone_id: '2', actual_um: 180, reference_um: 120 }] },
  reference_order_evaluation: { recommended_flow: ['2', '3'] }, query_context_signatures: { pose_recomm: 'signed', trajectory_recomm: 'signed-order', coating_status: 'signed-coating' },
  frames: [{ t: 0, map_num: null, zone_id: null }, { t: 1, map_num: '1', zone_id: '2', spraying: true },
           { t: 2, map_num: null, zone_id: null, spraying: false }, { t: 3, map_num: '1', zone_id: '3', spraying: true },
           { t: 4, map_num: null, zone_id: null, spraying: false }] });
const frame = (app, sample, index, seconds = sample.t) => app.emit('motion-viewer-frame', { frame: sample, sourceIndex: index, playheadSeconds: seconds });
const decode = request => JSON.parse(request.body.context.slice('motion_viewer_analysis:'.length));

async function submit(app, question = '지금 자세는 어때?') {
  app.input.value = question;
  const completion = app.form.fire('submit');
  const request = app.requests.at(-1);
  return { request, completion };
}

(async () => {
  for (const index of [0, 1, 2]) {
    const app = setup(), data = record();
    app.emit('motion-viewer-data', data);
    await app.find('mvc-chips').children[index].fire('click');
    const { request, completion } = await submit(app, '도막이 과하거나 모자란 영역이 있어?');
    const value = JSON.parse(request.body.context.slice(request.body.context.indexOf(':') + 1));
    assert.equal(value.coating_signature, 'signed-coating');
    assert.equal(value.coating_context.zones[0].actual_um, 180);
    request.finish(); await completion;
  }
  {
    const app = setup(), data = record();
    app.emit('motion-viewer-data', data);
    const flags = app.find('mvc-language-flags');
    assert.ok(flags, 'Answer languages must use a flag strip above the composer');
    assert.equal(flags.children.length, 12);
    assert.equal(flags.children[0].attributes['aria-pressed'], 'true');
    assert.match(app.find('mvc-mode-row').children[0].textContent, /8B/);
    assert.match(app.find('mvc-mode-row').children[1].textContent, /27B/);
    assert.equal(app.find('mvc-mode-row').children[0].attributes['aria-pressed'], 'true');
    // Vietnamese output never changes the Korean input contract.
    await flags.children[2].fire('click');
    const sent = await submit(app);
    assert.equal(sent.request.body.language, 'ko');
    assert.equal(sent.request.body.response_language, 'vi');
    assert.equal(sent.request.body.response_mode, 'quick');
    await flags.children[0].fire('click');
    sent.request.finish(); await sent.completion;
    assert.match(app.find('mvc-log').children.at(-1).textContent, /Tiếng Việt/);
    const next = await submit(app);
    assert.equal(next.request.body.language, 'ko');
    assert.equal(next.request.body.response_language, 'ko');
    next.request.finish(); await next.completion;
    assert.equal(flags.parent.parent.children.at(-1), app.form);
  }
  {
    const app = setup(), data = record();
    // Explicitly choose detailed mode to verify in-flight request snapshots.
    await app.find('mvc-mode-row').children[1].fire('click');
    app.emit('motion-viewer-data', data);
    frame(app, data.frames[1], 1, 1.25);
    frame(app, data.frames[2], 2, 2.5);
    const { request, completion } = await submit(app);
    assert.equal(request.body.use_rag, 'Y');
    assert.equal(request.body.response_mode, 'detailed');
    assert.deepEqual(decode(request).playback_context, { map_num: '1', zone_id: '2', source_index: 2, playhead_seconds: 2.5 });
    assert.match(app.find('mvc-context').textContent, /Map 1.*Zone 2.*2\.50/);
    // Changes after fetch starts must not relabel the eventual answer.
    await app.find('mvc-mode-row').children[0].fire('click');
    await app.find('mvc-chips').children[2].fire('click');
    app.emit('motion-viewer-data', record('RECORD-B'));
    frame(app, data.frames[3], 3, 3.5);
    request.finish(); await completion;
    const answer = app.find('mvc-log').children.at(-1).textContent;
    assert.match(answer, /자세.*상세 분석.*RECORD-A.*Map 1.*Zone 2.*2\.50/);
    assert.doesNotMatch(answer, /RECORD-B|빠른 안내|순서/);
    const next = await submit(app, '권장 Zone 순서는?');
    assert.equal(next.request.body.use_rag, 'Y');
    assert.equal(next.request.body.response_mode, 'quick');
    assert.deepEqual(decode(next.request).analysis_context.zone_keys, ['1:2', '1:3']);
    next.request.finish(); await next.completion;
  }
  {
    const app = setup(), data = record();
    app.emit('motion-viewer-data', data);
    frame(app, data.frames[1], 1);
    // A forward seek over another Zone must resolve the latest preceding Zone.
    frame(app, data.frames[4], 4);
    let sent = await submit(app);
    assert.equal(decode(sent.request).playback_context.zone_id, '3');
    sent.request.finish(); await sent.completion;
    // Seeking backwards into a non-spray transition uses the earlier Zone.
    frame(app, data.frames[2], 2);
    sent = await submit(app);
    assert.equal(decode(sent.request).playback_context.zone_id, '2');
    sent.request.finish(); await sent.completion;
    app.emit('motion-viewer-data', record('RECORD-B'));
    sent = await submit(app);
    assert.equal(decode(sent.request).playback_context.zone_id, null);
    assert.equal(decode(sent.request).playback_context.playhead_seconds, 0);
    sent.request.finish(); await sent.completion;
    // An active but unevaluated Zone must not retain a different Zone's score.
    frame(app, { map_num: '9', zone_id: '9', spraying: true, t: 4 }, 4);
    sent = await submit(app);
    assert.equal(decode(sent.request).playback_context.zone_id, null);
    sent.request.finish(); await sent.completion;
  }
  {
    const app = setup(), data = record();
    data.frames = [
      { t: 0, map_num: '1', zone_id: '3', spraying: false },
      { t: 1, map_num: '1', zone_id: '2', spraying: true },
      { t: 2, map_num: '1', zone_id: '3', spraying: false },
      { t: 3, map_num: null, zone_id: null, spraying: false },
      { t: 4, map_num: '1', zone_id: '3', spraying: true },
    ];
    app.emit('motion-viewer-data', data);
    frame(app, data.frames[1], 1);
    // A valid ray hit while moving is not a new sprayed Zone.
    frame(app, data.frames[2], 2);
    let sent = await submit(app);
    assert.equal(decode(sent.request).playback_context.zone_id, '2');
    sent.request.finish(); await sent.completion;
    // Revalidating the held selector does not require a spraying property.
    app.emit('motion-viewer-data', data);
    sent = await submit(app);
    assert.equal(decode(sent.request).playback_context.zone_id, '2');
    sent.request.finish(); await sent.completion;
    frame(app, data.frames[4], 4);
    frame(app, data.frames[3], 3);
    sent = await submit(app);
    assert.equal(decode(sent.request).playback_context.zone_id, '2');
    sent.request.finish(); await sent.completion;
    frame(app, data.frames[0], 0);
    sent = await submit(app);
    assert.equal(decode(sent.request).playback_context.zone_id, null);
    sent.request.finish(); await sent.completion;
  }
  {
    const app=setup(), data=record(); app.emit('motion-viewer-data', data);
    const first=await submit(app, '안녕');
    assert.equal(first.request.headers['X-Copilot-Source'], 'motion-viewer');
    assert.ok(first.request.headers['X-Copilot-Conversation'].startsWith('mv-'));
    assert.equal(first.request.body.user_id, 'W');
    first.request.finish(); await first.completion;
    await app.find('mvc-chips').children[2].fire('click');
    const second=await submit(app, '야');
    assert.equal(second.request.headers['X-Copilot-Conversation'], first.request.headers['X-Copilot-Conversation']);
    second.request.finish(); await second.completion;
    const another=setup(); another.emit('motion-viewer-data', {...record(), worker: {}});
    const third=await submit(another, '안녕');
    assert.notEqual(third.request.headers['X-Copilot-Conversation'], first.request.headers['X-Copilot-Conversation']);
    assert.equal(third.request.body.user_id, 'COPILOT-DEMO-USER');
    third.request.finish(); await third.completion;
  }
  console.log('motion viewer chat behavior: 5 scenarios passed');
})().catch(error => { console.error(error); process.exitCode = 1; });
