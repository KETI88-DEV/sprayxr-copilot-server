# Motion Viewer Pose Evaluation Integration Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Integrate the production posture-comparison engine and the useful visualization pieces from `trainer/pose_lab/` into `/motion-viewer`, so a reviewer can inspect one TSB sequence on the real training object, understand the expert comparison, and see one concrete corrective action without leaving the Co-Pilot server.

**Architecture:** Keep `/pose_estimation/sequence` as the only authoritative scoring path. The motion viewer receives the same stored TSB request body from `SequenceStore`, asks the MCP engine for the raw product result, and converts that result plus the request frames into a viewer-specific read model. Reuse `trainer/pose_lab/` mesh placement, reference playback, correction-vector, and Korean explanation helpers; do not run the lab's pairwise `/api/compare` score as a second product score. The page stays a single route with two center tabs (`자세 비교`, `도장 경로·Zone`) sharing one timeline and camera.

**Tech Stack:** Python 3.11, FastAPI, SQLite `SequenceStore`, httpx, existing pose engine and `trainer/pose_lab` modules, vanilla HTML/CSS/Canvas JavaScript, pytest, Docker Compose on DGX Spark.

**Spec:** [`docs/superpowers/specs/2026-10-14-motion-viewer-pose-integration-design.md`](../specs/2026-10-14-motion-viewer-pose-integration-design.md)

## Global Constraints

- Do not change the TSB request JSON or add a required simulator parameter.
- Keep the product score, percentile, verdict, and corrections sourced from `POST /pose_estimation/sequence` only.
- Keep lab-only pairwise comparison values clearly separated from product values; preferably do not expose them on the integrated page.
- Use `sequence_id` as the external evaluation unit and show internal stroke/pass numbers only as secondary detail.
- Render an aligned FBX mesh only when alignment status is `verified`; mode 2 and missing assets must fall back to the existing Ray Point/Zone surface without inventing placement.
- Preserve current expert motion/Zone behavior and all existing `/motion-viewer/api/*` response fields.
- Preserve the untracked user file `docs/SprayXR_CoPilot_샘플_2026-09-21.json`.
- Keep browser labels in Korean and explain unavailable states rather than substituting fake scores.

## Review Focus

- Confirm there is no duplicated scoring implementation and no divergence between Query API and viewer results.
- Confirm sequence/session selection cannot read another session accidentally and SQL remains parameterized.
- Confirm large frame, point-cloud, and mesh payloads are bounded and do not freeze long sessions.
- Confirm missing FBX files, unavailable mode-2 alignment, MCP timeout, and insufficient skeleton frames all have useful fallbacks.
- Confirm the UI distinguishes measured worker pose, expert reference pose, and corrected target pose by both label and color.

---

## File Structure

- Modify `trainer/demo_api/sequence_store.py` — add bounded read-only catalog metadata for viewer session/sequence selection.
- Modify `trainer/demo_api/pose_query.py` — expose one internal raw MCP scoring primitive and keep existing public projection behavior unchanged.
- Create `trainer/demo_api/motion_viewer_pose.py` — build the viewer read model from a stored TSB sequence, raw product result, pose-lab reference layers, and mesh availability.
- Modify `trainer/demo_api/motion_viewer.py` — inject dependencies, add evaluation endpoints, include mesh data in the existing motion response, and load the new UI assets.
- Create `trainer/demo_api/static/motion_viewer_pose.js` — pose tab, shared timeline, 3D layers, correction arrows, state handling, and accessibility labels.
- Create `trainer/demo_api/static/motion_viewer_pose.css` — integrated layout, posture summary cards, responsive behavior, and measured/reference/corrected visual language.
- Modify `trainer/demo_api/app.py` — share the existing `SequenceStore` and MCP URL with the motion viewer router.
- Modify `deploy/spark/compose.yaml` and `.env.example` — mount the separately managed FBX model directory read-only and declare its environment path.
- Modify `deploy/spark/scripts/bundle_manifest.py` and related tests only if model assets are placed in the verified bundle; otherwise verify the external read-only model directory at runtime without making API startup fail.
- Modify `trainer/demo_api/tests/test_sequence_store.py` — catalog ordering and limits.
- Modify `trainer/demo_api/tests/test_pose_recomm.py` — raw/product projection parity.
- Create `trainer/demo_api/tests/test_motion_viewer_pose.py` — adapter, engine failures, reference layers, and mesh fallbacks.
- Modify `trainer/demo_api/tests/test_motion_viewer.py` — routes, HTML hooks, backward compatibility, and static asset contract.
- Modify `deploy/spark/scripts/smoke_v2_5_api.py` — live viewer route and evaluation unavailability/success smoke assertions.

## Task 1: Add a bounded sequence catalog without changing ingestion

**Files:**
- Modify: `trainer/demo_api/sequence_store.py`
- Test: `trainer/demo_api/tests/test_sequence_store.py`

- [ ] Add `test_recent_sequence_catalog_orders_sessions_and_sequences_by_receive_time` using an in-memory store with two sessions and non-lexical sequence IDs; assert newest sessions first and sequence receive order inside each session.
- [ ] Add `test_recent_sequence_catalog_respects_session_and_sequence_limits` and assert the method never returns unbounded rows.
- [ ] Run `pytest -q trainer/demo_api/tests/test_sequence_store.py -k recent_sequence_catalog` and confirm the new tests fail because the method is absent.
- [ ] Implement `SequenceStore.recent_sequence_catalog(*, session_limit: int = 50, sequence_limit: int = 200)` with one parameterized aggregate query plus a bounded sequence query; return only IDs, receive timestamps, and frame counts, never raw request bodies.
- [ ] Run `pytest -q trainer/demo_api/tests/test_sequence_store.py` and confirm all store tests pass.
- [ ] Commit with `git add trainer/demo_api/sequence_store.py trainer/demo_api/tests/test_sequence_store.py && git commit -m "feat: expose bounded sequence catalog for viewer"`.

## Task 2: Make the MCP product result reusable without creating a second scorer

**Files:**
- Modify: `trainer/demo_api/pose_query.py`
- Test: `trainer/demo_api/tests/test_pose_recomm.py`

- [ ] Add `test_score_body_raw_and_projected_result_share_same_engine_payload` with `httpx.MockTransport`; assert one raw result produces the existing `PoseQueryResult` score, sequence ID, corrections, and summary unchanged.
- [ ] Add `test_score_body_raw_preserves_engine_unavailable_reason_without_fake_values` for timeout, non-200, and MCP `analysis_status != available`.
- [ ] Run `pytest -q trainer/demo_api/tests/test_pose_recomm.py -k 'raw_and_projected or raw_preserves'` and confirm failure.
- [ ] Extract the current POST/filter/validation logic into `score_body_raw(...) -> PoseEngineResult`, a typed internal dataclass containing `analysis_status`, `reason_code`, `sequence_id`, and the validated raw `result` mapping.
- [ ] Refactor `score_body(...)` to project `score_body_raw(...)`; preserve every existing Query API field and reason code.
- [ ] Run `pytest -q trainer/demo_api/tests/test_pose_recomm.py` and confirm all pose-query tests pass.
- [ ] Commit with `git add trainer/demo_api/pose_query.py trainer/demo_api/tests/test_pose_recomm.py && git commit -m "refactor: share raw posture engine result"`.

## Task 3: Build a viewer-specific pose read model

**Files:**
- Create: `trainer/demo_api/motion_viewer_pose.py`
- Create: `trainer/demo_api/tests/test_motion_viewer_pose.py`
- Reuse without copying: `trainer/pose_lab/service.py`, `trainer/scripts/pose_fk.py`, `trainer/scripts/level_mesh.py`

- [ ] Add `test_build_pose_view_reports_sequence_identity_and_product_score` with a minimal TSB body and fake raw MCP result; assert `session_id`, `sequence_id`, scenario/mode, Zone, score, percentile, verdict, confidence, and the highest-priority correction.
- [ ] Add `test_build_pose_view_labels_strokes_as_secondary_units` for a burst containing multiple strokes; assert sequence remains the headline unit and strokes are numbered details.
- [ ] Add `test_build_pose_view_downsamples_tracks_but_keeps_first_last_and_correction_frames`; assert long sequences stay under the configured payload cap without losing correction evidence.
- [ ] Add `test_build_pose_view_keeps_measured_reference_and_corrected_layers_separate`; assert blue measured world coordinates, green expert normalized/reference coordinates, and yellow corrected targets are distinct fields with explicit coordinate-space metadata.
- [ ] Add parametrized unavailability tests for missing body, insufficient skeleton, missing expert reference, and engine timeout; assert no score is synthesized and the Korean explanation identifies the missing dependency.
- [ ] Add mesh tests for `verified`, `unverifiable`, and missing FBX cases; assert only verified data contains triangles and all others declare the Ray Point fallback reason.
- [ ] Run `pytest -q trainer/demo_api/tests/test_motion_viewer_pose.py` and confirm import/behavior failures.
- [ ] Implement small pure helpers for frame extraction, Zone inference, bounded track sampling, correction prioritization, confidence labeling, and result-to-Korean verdict mapping.
- [ ] Wrap `trainer.pose_lab.service.mesh_block(str(mode_id))` behind an injectable mesh provider; do not duplicate FBX parsing or alignment matrices.
- [ ] Reuse pose-lab/reference helpers to produce display layers; annotate `world`, `body_normalized`, or `object_aligned` coordinate spaces so the client cannot overlay incompatible coordinates silently.
- [ ] Run `pytest -q trainer/demo_api/tests/test_motion_viewer_pose.py` and confirm all adapter tests pass.
- [ ] Commit with `git add trainer/demo_api/motion_viewer_pose.py trainer/demo_api/tests/test_motion_viewer_pose.py && git commit -m "feat: build posture evaluation viewer model"`.

## Task 4: Add motion-viewer evaluation and mesh APIs

**Files:**
- Modify: `trainer/demo_api/motion_viewer.py`
- Modify: `trainer/demo_api/app.py`
- Modify: `trainer/demo_api/tests/test_motion_viewer.py`

- [ ] Add `test_evaluation_catalog_lists_stored_tsb_sequences_without_bodies` and assert the JSON contains no raw skeleton/body payload.
- [ ] Add `test_evaluation_endpoint_uses_injected_store_and_product_scorer`; assert exact `session_id` and `sequence_id` selection and product score parity.
- [ ] Add `test_evaluation_endpoint_returns_404_for_unknown_sequence_and_200_unavailable_for_engine_failure` to distinguish user selection errors from analysis availability.
- [ ] Add `test_motion_endpoint_adds_verified_object_mesh_and_preserves_existing_fields`; assert old response keys remain unchanged.
- [ ] Run `pytest -q trainer/demo_api/tests/test_motion_viewer.py -k 'evaluation or object_mesh'` and confirm failure.
- [ ] Extend `create_motion_viewer_router(...)` with injected `SequenceStore`, MCP base URL, transport/scorer seam, and mesh provider while preserving defaults used by existing tests.
- [ ] Add `GET /motion-viewer/api/evaluations` for the bounded catalog and `GET /motion-viewer/api/evaluation?session_id=...&sequence_id=...` for one read model.
- [ ] Add `object_mesh` and explicit `object_mesh_status` to `/motion-viewer/api/motion`; keep the existing inferred Zone surfaces as fallback/evidence.
- [ ] Store the existing `sequence_store` on `app.state` and pass that exact object into the viewer router; do not instantiate a second database connection for the page.
- [ ] Run `pytest -q trainer/demo_api/tests/test_motion_viewer.py trainer/demo_api/tests/test_openapi_workflow.py` and confirm pass.
- [ ] Commit with `git add trainer/demo_api/motion_viewer.py trainer/demo_api/app.py trainer/demo_api/tests/test_motion_viewer.py && git commit -m "feat: expose posture evaluation viewer APIs"`.

## Task 5: Integrate the pose-comparison interface into the motion viewer

**Files:**
- Modify: `trainer/demo_api/motion_viewer.py`
- Create: `trainer/demo_api/static/motion_viewer_pose.js`
- Create: `trainer/demo_api/static/motion_viewer_pose.css`
- Modify: `trainer/demo_api/tests/test_motion_viewer.py`

- [ ] Add page contract assertions for title `작업 자세 비교 평가`, default `자세 비교` tab, `도장 경로·Zone` tab, TSB session/sequence selectors, one shared timeline, verdict, similarity/percentile, biggest deviation, `지금 고칠 한 가지`, confidence, and Zone summary.
- [ ] Add static source assertions that measured/expert/corrected layers use labelled blue/green/yellow styles and that no pairwise-lab score label appears.
- [ ] Add interaction-source assertions for tab switching, evaluation loading/empty/error states, play/pause, timeline seek, and selection cancellation via generation token.
- [ ] Run `pytest -q trainer/demo_api/tests/test_motion_viewer.py -k page` and confirm failure.
- [ ] Add no-cache asset routes or a static mount scoped under `/motion-viewer/assets`; include the CSS and deferred JS from the existing page.
- [ ] Add the two center tabs while retaining the existing Zone/paint viewer as the second tab and defaulting to the pose tab.
- [ ] Render the verified object mesh first, observed Ray Point cloud as optional evidence, worker tracked skeleton/head/hands in blue, expert reference in green, corrected target in yellow outline/dashes, paint trace in orange, and correction arrows in yellow.
- [ ] If coordinate spaces are incompatible, render expert normalized pose in the existing separate reference panel and display a clear `별도 기준 좌표` badge rather than forcing a false overlay.
- [ ] Bind pose and path views to the same timestamp/index and camera controls; preserve left-drag rotate, middle-drag pan, wheel zoom, playback speed 0.25–3×, loading overlay, and current camera-centered reset.
- [ ] Move advanced joint sigma/UV/resampling/internal stroke tables into collapsed details; keep the primary card limited to the action a trainee should take now.
- [ ] Run `pytest -q trainer/demo_api/tests/test_motion_viewer.py trainer/pose_lab/tests/test_static_assets.py` and fix only regressions caused by the integration.
- [ ] Commit with `git add trainer/demo_api/motion_viewer.py trainer/demo_api/static/motion_viewer_pose.js trainer/demo_api/static/motion_viewer_pose.css trainer/demo_api/tests/test_motion_viewer.py && git commit -m "feat: integrate posture comparison into motion viewer"`.

## Task 6: Add Zone and sequence-level evaluation summaries

**Files:**
- Modify: `trainer/demo_api/motion_viewer_pose.py`
- Modify: `trainer/demo_api/static/motion_viewer_pose.js`
- Modify: `trainer/demo_api/tests/test_motion_viewer_pose.py`

- [ ] Add `test_zone_summary_aggregates_completed_sequences_without_rescoring_zone` with three product results across two Zones; assert counts, available ratio, median score, worst correction, and confidence are derived from sequence results only.
- [ ] Add `test_zone_summary_does_not_mix_unavailable_results_into_score` and assert unavailable counts remain visible.
- [ ] Run the two tests and confirm failure.
- [ ] Implement a pure sequence-result aggregation function; never submit an invented concatenated Zone body to MCP.
- [ ] Add right-panel navigation for `전체` and each Zone, using the design-document colors; selecting one pauses playback and shows the matching expert standard pattern and aggregate.
- [ ] Show `현재 Zone · SEQ-xxx · 스트로크 n/m` during playback, and restore sequential playback when the user presses play.
- [ ] Run `pytest -q trainer/demo_api/tests/test_motion_viewer_pose.py trainer/demo_api/tests/test_motion_viewer.py` and confirm pass.
- [ ] Commit with `git add trainer/demo_api/motion_viewer_pose.py trainer/demo_api/static/motion_viewer_pose.js trainer/demo_api/tests/test_motion_viewer_pose.py && git commit -m "feat: summarize posture evaluation by zone"`.

## Task 7: Wire FBX assets into Spark deployment with an honest fallback

**Files:**
- Modify: `deploy/spark/compose.yaml`
- Modify: `deploy/spark/.env.example`
- Modify: `deploy/spark/README.md`
- Test: add or modify the closest existing Compose/deployment contract test discovered with `find deploy trainer -name 'test*compose*' -o -name 'test*deploy*'`.

- [ ] Add a deployment contract test asserting `agent-api` receives a read-only model directory and an explicit `POSE_MODELINGS_ROOT` path.
- [ ] Add a startup/runtime test asserting absent models do not fail `/api/ready` and the viewer reports `ray_point_fallback`.
- [ ] Run the focused deployment tests and confirm failure.
- [ ] Add `${POSE_MODELINGS_DIR:-/home/keti/shipcoat/data/modelings}:/data/sources/modelings:ro` and `POSE_MODELINGS_ROOT=/data/sources/modelings` to `agent-api`.
- [ ] Update pose-lab mesh source resolution to honor the environment path only if the current constant does not already do so; retain the existing default for development.
- [ ] Document required filenames from `trainer/data/level_alignment.json`, verified modes 1 and 3–10, mode 2 fallback, ownership/permissions, and a checksum/list command.
- [ ] On Spark, inventory `/home/keti/shipcoat/data/modelings` without deleting or moving anything. If files are absent, leave the deployed viewer functional in Ray Point fallback and report the exact missing filenames to the user; do not fabricate or download models.
- [ ] Run focused deploy tests and `docker compose -f deploy/spark/compose.yaml config` with the existing `.env`; confirm the rendered mount is read-only.
- [ ] Commit with `git add deploy/spark/compose.yaml deploy/spark/.env.example deploy/spark/README.md <test-files> && git commit -m "deploy: mount aligned training object models"`.

## Task 8: Verify end-to-end behavior and deploy safely

**Files:**
- Modify only if failures expose integration defects.
- Test: `trainer/demo_api/tests/`, `trainer/pose_lab/tests/`, `mcp_servers/test_pose_sequence_route.py`, Spark smoke scripts.

- [ ] Run `pytest -q trainer/demo_api/tests/test_sequence_store.py trainer/demo_api/tests/test_pose_recomm.py trainer/demo_api/tests/test_motion_viewer_pose.py trainer/demo_api/tests/test_motion_viewer.py trainer/demo_api/tests/test_openapi_workflow.py`.
- [ ] Run `pytest -q mcp_servers/test_pose_sequence_route.py trainer/pose_lab/tests/test_app.py trainer/pose_lab/tests/test_static_assets.py`; classify any fixture-path failures separately from code failures.
- [ ] Run `python deploy/spark/scripts/smoke_v2_5_api.py --help` (or its documented dry-run/unit-test mode) before pointing it at the live service.
- [ ] Build the `agent-api` image and start only through the repository's documented Spark Compose procedure; do not interrupt the current public service until the new container passes `/api/ready`.
- [ ] Against the local Spark endpoint, verify `/motion-viewer`, `/motion-viewer/api/catalog`, `/motion-viewer/api/evaluations`, one known evaluation, Query API `pose_recomm`, and `POST /pose_estimation/sequence` return the same score/corrections for the same stored sequence.
- [ ] In a browser, verify long-mode playback stays smooth, tab changes preserve the timeline, missing mode-2 FBX falls back visibly, Korean labels are not clipped, and the right panel remains usable at the current display width.
- [ ] Deploy to the public service only after local checks; verify `http://keti-xr-lab.iptime.org:58000/motion-viewer` and retain the previous image tag for rollback.
- [ ] Run `git status --short` and ensure only the user-owned untracked JSON remains; push the feature branch and record the commit SHA used by the container.
- [ ] Commit any final test-only corrections with `git commit -m "test: verify integrated posture evaluation viewer"`.

## Completion Criteria

- The motion viewer's default experience is posture comparison, while the current Zone/path viewer remains intact in a second tab.
- A TSB `session_id` + `sequence_id` can be selected and evaluated without changing the simulator JSON.
- The viewer and Query API agree exactly on product score and corrections.
- The UI communicates one actionable correction and confidence without exposing raw diagnostics by default.
- Verified FBX assets render with their stored alignment; missing/unverified assets fall back honestly to Ray Point geometry.
- Focused tests, MCP route tests, pose-lab tests, Compose rendering, local smoke tests, and public route checks pass before completion is claimed.
