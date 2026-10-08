# Vertical-axis blast radius (measurement spike)

Measures the effect of correcting the codebase's assumption that world **Z** is
vertical to the evidence-backed reality that world **Y** is vertical
(Unity's left-handed, Y-up convention — confirmed directly against the CSV
head/hand geometry and the HMD's own up vector). **No production code was
changed.** All numbers below come from a throwaway analysis script that
monkeypatches the convention-encoding module globals at runtime (restored
immediately after each use) and measures both the original and corrected
geometry through the real `service.compare()` / `measured_score.score_track`
/ `motion_key.motion_key` code paths.

## Sample and method

- **Sample**: the project's pre-registered 200-pair sample —
  `trainer/scripts/improvement_study.py`'s `select_pairs(candidates,
  expert_worker="전형기", pairs=200, seed=20260828)` — drawn from the same
  on-disk candidate index the project uses
  (`improvement_study_index_cache.json`, 64,098 candidates, loaded directly
  rather than through `index_corpus_cached()`'s root-string match, since that
  cache records the container path `/data/sources/professionals` and this
  host mounts the same tree at `/data/ship-paint-ai/data/sources/professionals`
  — using it directly avoided the ~4 minute full re-parse without changing
  which 200 pairs get selected). Reference artifact:
  `output/spread_reference_expert.json`.
- Of the 200 selected query passes, **157** had a same-bucket expert
  reference under the current (Z-up) motion key and so produced a real
  `measured` comparison (`status="ok"`); the other 43 are
  `no_expert_reference` under the current convention and are only usable for
  the motion-key measurement (§5), which needs no reference.
- **Corrected variant**: built as a single `up_index` parameter (0/1/2)
  threaded through generalized reimplementations of
  `measured_score._axis_components`, `stroke_frame.stroke_basis`'s
  `WORLD_UP`/`FALLBACK_TANGENT`, `motion_key.signed_band` /
  `_elevation_deg` / `heading_word` / `body_placement` (including its
  `height_band(hit[...])` call), and `measured_pose.facing_yaw` — not a
  copy of each function with `2` replaced by `1`. **Verified bit-for-bit**
  against every real original function at `up_index=2` over 20,000 random
  trials each before any pair was measured (see §0). The corrected run uses
  `up_index=1` (Y), with `FALLBACK_TANGENT` and the horizontal-plane index
  pair re-derived from the same `up_index` rule rather than hand-picked.
- For each of the 157 "ok" pairs, geometry was computed **both ways** by
  re-running the actual production functions
  (`service.pass_frames`→`service._local_tracks`→`service.representative_basis`→
  `measured_score.score_track`→`measured_score.corrections_text`/
  `service.axes_block`/`service.stance_block`/`measured_score.right_hand_finding`)
  once unpatched and once inside the `up_index=1` patch context, reusing the
  **same** per-axis spread numbers captured from the real `service.compare()`
  call for that pair (so only the geometry convention varies, not which
  reference distribution was resolved).
- For all 200 query passes, `motion_key.motion_key` was recomputed both ways
  via `service.key_inputs` → `service.key_from`, re-parsing the pass's frames
  fresh under each convention (so the patched `measured_pose.facing_yaw` is
  in effect during frame construction, not just during key assembly).

## 0. Correctness gate (run before any pair was measured)

`generic_axis_components`/`generic_signed_band`/`generic_heading_word`/
`generic_body_placement`/`generic_facing_yaw` at `up_index=2` reproduced
`measured_score._axis_components`, `motion_key.signed_band`,
`motion_key.heading_word`, `motion_key.body_placement`, and
`measured_pose.facing_yaw` **exactly** (`==` on the returned tuples/floats,
not "close") over 20,000 random trials each. No mismatches. This gate passed
before the 200-pair run started.

## 1. Score / combined-σ invariance — investigated, not ~0, and here is why

**Headline number the task asked for: max |Δscore| across the 157 "ok" pairs
= 0.799 (median 0.091); max |Δcombined σ| per point = 5.37 (median 0.455).
This is not ~0.** Per the pre-registered check, that means something needed
investigating before trusting the rest — so before reporting §2–§6, here is
what was checked and what the non-zero number actually is.

**What is confirmed exact (the code is not buggy):**

- The manual reproduction of the *current* (Z-up) pipeline matches the real
  `service.compare()` score bit-for-bit for all 157 pairs (max diff =
  `0.000e+00`) — the harness reproduces production exactly before any
  convention is touched.
- Given one fixed 3D vector and two **independently, correctly** built
  orthonormal frames (one from the real `stroke_basis` under Z-up, one under
  Y-up, both derived from the same hit/tangent/gun inputs), decomposing that
  *single* vector into wall/vertical/lateral and summing the squared
  magnitudes reproduces `|vector|²` to floating-point precision under
  **either** convention, and the two conventions' sums agree with each other
  to a relative error of `3.6e-14`. This is the literal identity the task
  described ("wall² + vertical² + lateral² == |world_delta|² for random
  bases") — it holds, unconditionally, for any `up_index`.

**Why the real 200-pair score is not invariant anyway:** `score_track`'s
input is not "one fixed world vector, decomposed twice." `service._local_tracks`
builds a **fresh orthonormal `stroke_basis` for every one of the 20 resampled
grid points** along a pass (using that point's own local tangent to the next
point), converts query and reference into that point's own local coordinates,
and only *then* does `score_track` average the 20 per-point local
differences and hand the average to `_axis_components` together with **one**
representative basis for the whole pass. Averaging vectors that were each
expressed in their own (slightly different, tangent-dependent) local frame,
then reprojecting the average through a different single frame, is not a
change of basis on a fixed vector — swapping `WORLD_UP` rotates all 20
per-point frames (differently, since each has a different tangent), so the
*local coordinates being averaged* change, not just their later axis labels.
Confirmed directly: reconstructing `_world_delta(mean_delta, representative_basis)`
under each convention for the same pair gives two different 3D vectors
(differences up to ~1 m on deltas that are themselves ~0.1–0.9 m), not the
same vector re-expressed. Two things compound this further: real per-axis
spreads are anisotropic (e.g. one pair's right-hand σ inputs were wall=0.93,
vertical=0.12, lateral=1.83 — not close to uniform), and `score_track` clips
each axis's σ at `MAX_SIGMA=6.0` **before** squaring and summing, which is
also not norm-preserving when energy shifts between axes near the clip.

**Conclusion for the reader:** the axis-decomposition code itself is
correct and norm-preserving (§0 and the two exact checks above). The
combined σ / score genuinely moves when the convention is corrected, because
correcting `WORLD_UP` doesn't just relabel three numbers per pair — it
changes what `_local_tracks` measures at every one of the 20 grid points of
every pass. That is a real property of this codebase's per-point local-frame
design, not an artifact of this measurement. Numbers below (§2–§6) use the
real, faithfully-reproduced pipeline as it actually runs.

## 2–6. The six measurements

| # | Measurement | Result |
|---|---|---|
| 1 | Score/σ delta | **Not ~0** (see §1): max Δscore = 0.799, median 0.091 (n=157). Root cause identified and verified, not a bug in the harness. |
| 2 | Chosen correction axis (right hand) | Fires under both conventions for 43/157 pairs; **same axis chosen only 55.8%** of the time (24/43). Fires only-original 19, only-corrected 26, neither 69. |
| 3 | Correction phrase | Of 88 pairs with a sentence on either side, **text changes in 85/88 (96.6%)**. Of the 43 pairs firing both sides, **category** (up/down ↔ toward/away-from-wall ↔ along-the-wall) changes for **19 (44.2%)**. Concrete harm (up/down ↔ wall-distance swap specifically): **6 pairs** (1 vertical→wall, 5 wall→vertical). |
| 4 | Per-axis \|magnitude_cm\| (right hand) | wall: median 24.5→23.9cm (n 32→33). vertical: median 42.4→35.4cm (n 21→23). lateral: median 31.1→30.9cm (n 24→27). Magnitudes are broadly similar in scale; it's *which axis carries the message* that reshuffles, not the typical size of the number. |
| 5 | Motion key bucket | **193/200 (96.5%)** of query passes land in a different `motion_key.motion_key` bucket. Component breakdown: elevation band 158/200 (79%), body-relative height 155/200 (77.5%), heading word 150/200 (75%), reach 51/200 (25.5%), lateral (left/center/right) 27/200 (13.5%). |
| 6 | Head stance | Present under both conventions for 14/157 pairs; of those, **8/14 (57%) flip** between a height instruction and a wall-distance instruction. Overall, stance text differs (presence or axis) for **39/157 (24.8%)** of pairs. |

### §3 concrete examples (the harm case named in the brief)

```
orig[wall]:     "오른손을 벽에서 31cm 더 띄우세요."   ->  corrected[vertical]: "오른손을 16cm 올리세요."
orig[wall]:     "오른손을 벽에서 37cm 더 띄우세요."   ->  corrected[vertical]: "오른손을 41cm 내리세요."
orig[wall]:     "오른손을 벽에서 19cm 더 띄우세요."   ->  corrected[vertical]: "오른손을 42cm 올리세요."
orig[wall]:     "오른손을 벽에서 30cm 더 띄우세요."   ->  corrected[vertical]: "오른손을 35cm 내리세요."
orig[vertical]: "오른손을 23cm 올리세요."             ->  corrected[wall]:     "오른손을 벽에서 16cm 더 붙이세요."
orig[wall]:     "오른손을 벽에서 30cm 더 띄우세요."   ->  corrected[vertical]: "오른손을 37cm 올리세요."
```

Each of these is the same measurement, same trainee, same instant — with the
current code telling the worker to move toward/away from the wall, and the
corrected geometry saying the actual outlier is height (or vice versa). A
worker who dutifully steps toward the wall by 31cm when the real problem was
16cm too low has followed the instruction and made no progress on the thing
that was actually wrong.

### §5 detail — why this is "the deepest consequence"

`select_pairs` matches a query to its expert reference only when they share
the same `motion_key` bucket. With 96.5% of the 200 query passes changing
bucket, correcting the axis doesn't just relabel existing comparisons — it
would silently reassign nearly the entire query→reference pairing, and by
extension redefine what the `spread_reference_expert.json` per-bucket σ
values are even being asked to represent for any given query. The 43 pairs
that currently have `status="no_expert_reference"` (excluded from §2–§4,
§6) would very likely not be the same 43 under the corrected convention
either — some currently-unreferenced queries would probably gain a match,
and some currently-matched ones would probably lose theirs, though
confirming that precisely requires rebuilding the reference-candidate
buckets under the corrected convention, which is out of scope for this
spike.

## Plain-language summary

The axis mislabeling is not a cosmetic issue: on the same 200-pair sample
the project already uses, correcting it would change which of the three
correction axes gets reported for **44%** of the passes that currently fire
a right-hand correction, would change the actual sentence text for **97%**
of passes with any correction sentence, would flip the head-stance
instruction between "stand up/down" and "stand toward/away from the wall"
for **57%** of passes where a stance instruction fires under both
conventions, and — most consequentially — would move **96.5%** of the 200
query passes into a different motion-key bucket, which is the mechanism
that decides which expert pass a trainee is even being compared against;
the axis is wrong and it changes essentially everything downstream of it,
including several concrete cases where a worker would currently be told to
step toward or away from the wall when the actual outlier was height, or
told to raise or lower their hand when the actual outlier was wall distance.
