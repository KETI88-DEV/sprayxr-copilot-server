#!/usr/bin/env node
'use strict';
// Operator sample verification; never starts/replaces a frozen audit or restarts a service.
const fs = require('node:fs');
const path = require('node:path');
const {createRequire} = require('node:module');
const ID = /^[0-9a-f]{32}$/, SHA = /^[0-9a-f]{64}$/;
const SAFE_FAILURES = new Set(`invalid_arguments missing_required_arguments invalid_base_url invalid_max_records
invalid_dataset_id explicit_retest_requires_one_record credentials_file_must_not_be_dotenv
current_completed_audit_required explicit_record_not_unique_or_eligible duplicate_dataset_identity
selection_identity_mismatch rendered_identity_mismatch private_files_must_be_outside_git invalid_credentials_file
playwright_module_must_be_absolute playwright_dependency_unavailable invalid_timeline renderer_not_ready
sample_position_unavailable playback_did_not_advance invalid_admin_session invalid_renderer browser_script_error
verification_run_or_files_changed proof_not_recorded`.split(/\s+/));
const HELP = `Usage: node trainer/scripts/verify_training_playback.cjs
  --base-url URL --credentials-file FILE --output-dir DIR
  [--dataset-id ID] [--max-records 1..25]

Defaults: one pending eligible record, sequential playback, one page at a time.
--dataset-id explicitly retests one eligible record, including failed/passed records.
Credentials: protected JSON {"username":"...","password":"..."} outside Git.
Output must also be outside Git; a new playback-* evidence directory is created.
Uses demo-ui's @playwright/test or PLAYBACK_PLAYWRIGHT_MODULE (absolute module path).
Requires an existing current completed JSON/source audit; never POSTs /start.
Writes actual playback proofs using administrator login and same-origin CSRF.
Load budget: 45s navigation + 120s ready; each playback advance: 10s.
Three samples prove start/middle/end advancement, not full playback or AI accuracy.
Full bulk operation needs supervised server-worker recycling; not implemented here.
Exit: 0 accepted passes/no pending records; 1 stopped/invalid input; 2 playback failure.
`;
function check(condition, code) {if (!condition) throw new Error(code);}
function parseArgs(argv) {
  const values = {}, allowed = new Set(['base-url','credentials-file','output-dir','dataset-id','max-records']);
  if (argv.length === 1 && ['--help','-h'].includes(argv[0])) return {help:true};
  for (let i = 0; i < argv.length; i += 2) {
    const key = argv[i]?.replace(/^--/, '');
    check(argv[i]?.startsWith('--') && allowed.has(key) && !Object.hasOwn(values,key)
      && argv[i+1] && !argv[i+1].startsWith('--'), 'invalid_arguments');
    values[key] = argv[i+1];
  }
  check(values['base-url'] && values['credentials-file'] && values['output-dir'], 'missing_required_arguments');
  let url; try {url = new URL(values['base-url']);} catch (_) {throw new Error('invalid_base_url');}
  check(['http:','https:'].includes(url.protocol) && !url.username && !url.password
    && url.pathname === '/' && !url.search && !url.hash, 'invalid_base_url');
  const maximum = values['max-records'] || '1';
  check(/^\d+$/.test(maximum) && Number(maximum) >= 1 && Number(maximum) <= 25, 'invalid_max_records');
  check(!values['dataset-id'] || ID.test(values['dataset-id']), 'invalid_dataset_id');
  check(!values['dataset-id'] || Number(maximum) === 1, 'explicit_retest_requires_one_record');
  check(!/^\.env(?:\.|$)/i.test(path.basename(values['credentials-file'])), 'credentials_file_must_not_be_dotenv');
  return {base:url.origin, credentialsFile:path.resolve(values['credentials-file']),
    outputDir:path.resolve(values['output-dir']), datasetId:values['dataset-id'], maxRecords:Number(maximum)};
}
function selectQueue(report, options) {
  check(report && ID.test(report.run_id) && SHA.test(report.catalog_sha256) && !report.stale
    && ['awaiting_playback','finished'].includes(report.phase) && Array.isArray(report.items), 'current_completed_audit_required');
  const eligible = report.items.filter(item => item && !item.duplicate && item.json_status === 'passed'
    && item.source_status === 'passed' && ID.test(item.dataset_id) && SHA.test(item.artifact_sha256)
    && SHA.test(item.source_sha256) && typeof item.session_id === 'string');
  if (options.datasetId) {
    const matches = eligible.filter(item => item.dataset_id === options.datasetId);
    check(matches.length === 1, 'explicit_record_not_unique_or_eligible');
    return matches;
  }
  const pending = eligible.filter(item => item.browser_status === 'pending');
  check(new Set(pending.map(item => item.dataset_id)).size === pending.length, 'duplicate_dataset_identity');
  return pending.slice(0, options.maxRecords);
}
function assertIdentity(item, selected, state) {
  check(selected?.dataset_id === item.dataset_id && selected.process === item.process
    && Number(selected.mode_id) === Number(item.scenario_id) && typeof selected.session_id === 'string'
    && selected.session_id && typeof selected.worker_id === 'string' && selected.worker_id, 'selection_identity_mismatch');
  // Selection API owns worker/session aliases; raw catalog identities can differ legitimately.
  check(state?.datasetId === selected.dataset_id && state.sessionId === selected.session_id
    && state.workerId === selected.worker_id && state.process === selected.process
    && Number(state.modeId) === Number(selected.mode_id), 'rendered_identity_mismatch');
}
function outsideGit(target) {
  let directory = target;
  while (!fs.existsSync(directory)) directory = path.dirname(directory);
  directory = fs.realpathSync(directory);
  if (!fs.statSync(directory).isDirectory()) directory = path.dirname(directory);
  for (;;) {
    check(!fs.existsSync(path.join(directory,'.git')), 'private_files_must_be_outside_git');
    const parent = path.dirname(directory); if (parent === directory) return;
    directory = parent;
  }
}
function readCredentials(filename) {
  check(!/^\.env(?:\.|$)/i.test(path.basename(fs.realpathSync(filename))), 'credentials_file_must_not_be_dotenv');
  outsideGit(filename);
  check(fs.statSync(filename).isFile() && fs.statSync(filename).size <= 65536, 'invalid_credentials_file');
  const data = JSON.parse(fs.readFileSync(filename,'utf8').replace(/^\uFEFF/,''));
  check(data && typeof data.username === 'string' && data.username && typeof data.password === 'string'
    && data.password, 'invalid_credentials_file');
  return {username:data.username,password:data.password};
}
// Never serialize external error messages, request bodies, cookie jars, or auth responses.
function failureCode(error) {
  const code = String(error?.message || '');
  return SAFE_FAILURES.has(code) || /^verification_http_\d{3}$/.test(code)
    ? code : error?.name === 'TimeoutError' ? 'browser_timeout' : 'operation_failed';
}
const probe = page => page.evaluate(() => window.__motionPlaybackProbe?.() || null);
async function pause(page) {if ((await probe(page))?.playing) await page.locator('#v2Play').click();}
async function sample(page, position, fraction) {
  await pause(page);
  const maximum = Number(await page.locator('#timeline').getAttribute('max'));
  check(Number.isSafeInteger(maximum) && maximum >= 1, 'invalid_timeline');
  let low = 0, high = maximum, best = 0;
  const setIndex = index => page.locator('#timeline').evaluate((node, value) => {
    node.value = String(value); node.dispatchEvent(new Event('input',{bubbles:true}));
  },index);
  // Slider indices are sampled frames, not linear timestamps.
  for (let attempts = 0; low <= high && attempts < 24; attempts++) {
    const index = Math.floor((low + high)/2); await setIndex(index);
    const current = await probe(page);
    check(current && Number.isFinite(current.currentTime) && current.durationSeconds > 0, 'invalid_timeline');
    if (current.currentTime <= current.durationSeconds * fraction) {best = index; low = index + 1;} else high = index - 1;
  }
  await setIndex(best);
  const before = await probe(page), ranges = {start:[0,.15],middle:[.35,.65],end:[.8,1]};
  check(before?.rendererReady && !before.contextLost, 'renderer_not_ready');
  const ratio = before.currentTime / before.durationSeconds;
  check(ratio >= ranges[position][0] && ratio <= ranges[position][1], 'sample_position_unavailable');
  await page.locator('#v2Play').click();
  await page.waitForFunction(({before,minimum}) => {
    const current = window.__motionPlaybackProbe?.();
    return current?.playing && current.rendererReady && !current.contextLost
      && current.currentTime >= before.currentTime + minimum && current.renderedFrameIndex > before.renderedFrameIndex
      && current.rendererFrameCount > before.rendererFrameCount;
  },{before,minimum:Math.min(.3,before.durationSeconds*.03)},{timeout:10000,polling:'raf'});
  const after = await probe(page); await pause(page);
  check(after.currentTime > before.currentTime && after.renderedFrameIndex > before.renderedFrameIndex
    && after.rendererFrameCount > before.rendererFrameCount, 'playback_did_not_advance');
  return {position,time_before:before.currentTime,time_after:after.currentTime,
    frame_before:before.renderedFrameIndex,frame_after:after.renderedFrameIndex,
    draw_before:before.rendererFrameCount,draw_after:after.rendererFrameCount,
    renderer_ready:after.rendererReady,context_lost:after.contextLost};
}
async function main(options) {
  outsideGit(options.outputDir);
  const credentials = readCredentials(options.credentialsFile);
  const requireUI = createRequire(path.resolve(__dirname,'../../demo-ui/package.json'));
  const supplied = process.env.PLAYBACK_PLAYWRIGHT_MODULE;
  check(!supplied || path.isAbsolute(supplied), 'playwright_module_must_be_absolute');
  let chromium;
  try {({chromium} = supplied ? require(supplied) : requireUI('@playwright/test'));}
  catch (_) {throw new Error('playwright_dependency_unavailable');}
  check(chromium && typeof chromium.launch === 'function','playwright_dependency_unavailable');
  fs.mkdirSync(options.outputDir,{recursive:true});
  const output = fs.mkdtempSync(path.join(options.outputDir,'playback-'));
  const write = (filename, data) => fs.writeFileSync(path.join(output,filename),JSON.stringify(data,null,2));
  const log = data => console.log(JSON.stringify({at:new Date().toISOString(),...data}));
  const browser = await chromium.launch({headless:true});
  let context, csrf = '', expires = 0;
  const results = [];
  try {
    context = await browser.newContext({viewport:{width:1600,height:1000}});
    const fetchJSON = async (endpoint, method = 'GET', data) => {
      const response = await context.request.fetch(options.base+endpoint,{method,data,
        headers:{Origin:options.base,...(method === 'POST' && csrf ? {'X-DB-CSRF':csrf} : {})},
        timeout:120000,maxRedirects:0});
      try {check(response.ok(),'verification_http_'+response.status()); return await response.json();}
      finally {await response.dispose();}
    };
    const login = async () => {
      const session = await fetchJSON('/admin/api/manage/login','POST',credentials);
      check(typeof session.csrf_token === 'string' && session.csrf_token
        && Number(session.expires_at)*1000 > Date.now(), 'invalid_admin_session');
      csrf = session.csrf_token; expires = Number(session.expires_at)*1000;
    };
    await login();
    const report = await fetchJSON('/admin/api/training-verification');
    const queue = selectQueue(report,options), run = report.run_id;
    write('report-before.json',report);
    log({phase:'selected',run_id:run,records:queue.length,concurrency:1});
    for (const item of queue) {
      const started = Date.now();
      const evidence = {run_id:run,dataset_id:item.dataset_id,artifact_sha256:item.artifact_sha256,
        source_sha256:item.source_sha256,catalog_sha256:report.catalog_sha256,
        application_version:report.application_version,viewer_sha256:report.viewer_sha256,
        viewer_canonical_sha256:report.viewer_canonical_sha256,verifier_sha256:report.verifier_sha256,
        started_at:new Date().toISOString(),stage:'opening_page'};
      const proof = {run_id:run,artifact_sha256:item.artifact_sha256,status:'failed',dataset_id:item.dataset_id,
        session_id:item.session_id,renderer:'canvas-2d',frames_loaded:0,duration_seconds:0,samples:[]};
      const save = () => write(item.dataset_id+'.json',{...evidence,proof});
      let page, scriptError = false;
      save();
      try {
        evidence.selection = await fetchJSON('/motion-viewer/api/selection?dataset_id='+item.dataset_id);
        page = await context.newPage(); page.setDefaultTimeout(10000);
        page.on('pageerror',() => {scriptError = true;});
        await page.route('**/motion-viewer/api/**',route => {
          const request = route.request();
          return request.method() === 'GET' && new URL(request.url()).origin === options.base
            ? route.continue() : route.abort('blockedbyclient');
        });
        await page.goto(options.base+'/motion-viewer?dataset_id='+item.dataset_id,
          {waitUntil:'domcontentloaded',timeout:45000});
        await page.waitForFunction(() => {
          const value = window.__motionPlaybackProbe?.();
          return value?.loadState === 'ready' && value.rendererReady && value.framesLoaded >= 2;
        },null,{timeout:120000});
        const state = await probe(page);
        evidence.observed_state = state;
        assertIdentity(item,evidence.selection,state);
        check(state.renderer === 'canvas-2d' && state.durationSeconds > 0 && !state.contextLost, 'invalid_renderer');
        proof.selected_session_id = state.sessionId;
        proof.frames_loaded = state.framesLoaded; proof.duration_seconds = state.durationSeconds;
        for (const [position,fraction] of [['start',.02],['middle',.5],['end',.85]]) {
          proof.samples.push(await sample(page,position,fraction)); save();
        }
        assertIdentity(item,evidence.selection,await probe(page));
        check(!scriptError,'browser_script_error'); proof.status = 'passed';
      } catch (error) {
        evidence.error = failureCode(error);
        if (page) {
          evidence.last_state = await probe(page).catch(() => null);
          await page.screenshot({path:path.join(output,item.dataset_id+'-failed.png')}).catch(() => {});
        }
      } finally {if (page) await page.close().catch(() => {});}
      evidence.stage = 'submitting_proof'; evidence.elapsed_ms = Date.now() - started; save();
      // Never retry a write automatically: uncertain delivery must be inspected on the dashboard.
      if (expires - Date.now() < 60000) await login();
      try {
        const applied = await fetchJSON('/admin/api/training-verification/playback/'+item.dataset_id,'POST',proof);
        check(applied.run_id === run && !applied.stale,'verification_run_or_files_changed');
        const accepted = applied.items?.filter(row => row.dataset_id === item.dataset_id);
        check(accepted?.length === 1 && accepted[0].browser_status === proof.status,'proof_not_recorded');
        evidence.stage = 'accepted'; save();
      } catch (error) {
        evidence.stage = 'submission_unconfirmed'; evidence.submission_error = failureCode(error); save(); throw error;
      }
      results.push({dataset_id:item.dataset_id,status:proof.status,error:evidence.error});
      log({phase:'accepted',run_id:run,dataset_id:item.dataset_id,status:proof.status});
      if (proof.status === 'failed') {process.exitCode = 2; break;} // Stop before further server-cache pressure.
    }
    const final = await fetchJSON('/admin/api/training-verification');
    check(final.run_id === run && !final.stale,'verification_run_or_files_changed');
    write('report-after.json',final); write('summary.json',{run_id:run,results,summary:final.summary});
    log({phase:'finished',run_id:run,checked:results.length,summary:final.summary});
  } catch (error) {
    write('stopped.json',{error:failureCode(error),results,at:new Date().toISOString()}); throw error;
  } finally {
    if (context && csrf) await context.request.post(options.base+'/admin/api/manage/logout',
      {headers:{Origin:options.base,'X-DB-CSRF':csrf},timeout:10000,maxRedirects:0}).catch(() => {});
    csrf = ''; await browser.close();
  }
}
module.exports = {parseArgs,selectQueue,assertIdentity};
if (require.main === module) {
  try {
    const options = parseArgs(process.argv.slice(2));
    if (options.help) console.log(HELP);
    else main(options).catch(error => {console.error(failureCode(error)); process.exitCode = 1;});
  } catch (error) {console.error(failureCode(error)); process.exitCode = 1;}
}
