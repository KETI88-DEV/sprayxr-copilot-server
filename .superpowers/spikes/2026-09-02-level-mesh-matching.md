# Level-mesh hit matching (measurement spike)

**Question**: can each spray hit point be matched to a point on its level's FBX
mesh, giving REQ-19's angle work the surface normal it's been blocked on?

**Answer, up front**: yes for level 1 (essentially exact — millimetre
residual, confirmed three independent ways), and yes-but-noisier for level 10
(centimetre residual, confirmed by position but not by the two independent
checks). In both cases the FBX file's own declared node transform does **not**
place the mesh correctly — it's short one rotation that isn't recorded
anywhere in the FBX. A solved-for rigid + uniform-scale transform closes the
gap. **No production code was changed or committed** — all of this lives in a
scratch directory and nothing here is fitted-then-reported-as-declared; every
number below is labelled either "declared" (read straight off the file) or
"solved" (fit against real hit points).

## Data and method

- Meshes: `/data/ship-paint-ai/data/sources/modelings/learning_mode_lv.1_rev.1.1.fbx`
  (Model `Box003`, 16 vertices, 28 triangles after fan-triangulating its 84
  `PolygonVertexIndex` entries) and `learning_mode_lv.10_rev.1.2.fbx` (Model
  `Box001`, 220 vertices, 436 triangles from 1308 indices). Both binary FBX
  7400, parsed with a from-scratch stdlib-only reader (`struct` + `zlib`) per
  the header/record layout already established for this spike — no mesh
  library was installed or used.
- Hits: real capture CSVs under `/data/ship-paint-ai/data/sources/professionals/**`,
  matched by the plain `<level>_*.csv` naming (45 files / 75,326 spraying
  frames for level 1, 48 files / 389,060 for level 10 — `1-A`/`1-B`/`1-C`
  practice-mode files excluded, matching `trainer/scripts/csv_source.py`'s own
  "실전모드는 범위 밖이다" convention). Hit points, UV, active-gun position and
  head position were read via the repo's own `trainer/scripts/measured_pose.py`
  / `simulator_csv.py` (imported read-only, not modified) rather than
  re-deriving gun selection and the `" Ray Point Y"` leading-space column
  spelling that module already solved. All distance tests below sample 15,000
  hits per level (of the totals above), satisfying the ≥10,000-hit floor.
- Distance metric: exact closest point on any mesh triangle (Ericson's
  point-in-triangle-region algorithm), not closest vertex.
- Fitting: a from-scratch Umeyama similarity-transform solver (rotation +
  uniform scale + translation), because no numpy/scipy is installed in this
  container (confirmed) and none was installed. **Validated against synthetic
  data with a known transform before use** (exact recovery, reconstruction
  error ~1e-15) and hardened against one real degenerate case hit during
  fitting (see "A bug worth flagging" below). Point correspondence for the
  fit comes from point-to-mesh-surface ICP: each hit's nearest point on any
  triangle, converted back into raw model space via that triangle's
  barycentric weights, re-solved each iteration — not point-to-fixed-sample
  ICP, which was tried first and turned out to be biased by partial surface
  coverage (see level 10).

## 1. What the FBX files actually declare

Both files declare, identically:

| Property | Value |
|---|---|
| `GlobalSettings.UpAxis` / `UpAxisSign` | 1 (Y) / +1 |
| `GlobalSettings.FrontAxis` / `CoordAxis` | 2 (Z) / 0 (X) — i.e. the file states the standard X-right/Y-up/Z-front convention, not a permuted one |
| `GlobalSettings.UnitScaleFactor` | 2.54 (cm per file-unit — the file's native unit is effectively "1 inch") |
| Model `PostRotation`, `GeometricTranslation/Rotation/Scaling` | all identity (no override in either file) |
| Model `RotationOrder` | not overridden → FBX SDK default `eEulerXYZ` |
| Model `Lcl Translation` | (0, 0, 0) in both |
| Model `Lcl Scaling` | (1, 1, 1) in both (no override) |

They differ in exactly one place:

| Property | Level 1 (`Box003`) | Level 10 (`Box001`) |
|---|---|---|
| `PreRotation` | (−90°, 0°, 0°) | (−90°, 0°, 0°) |
| `Lcl Rotation` | not overridden (0°,0°,0°) | (−90.00000933°, 0°, 0°) — **overridden**, effectively another −90° about X |

`PreRotation` alone is the textbook 3ds-Max→FBX up-axis bake (the mesh is
authored lying flat, Z-up in Max, and this rotation stands it up to Y-up on
export) — this matches the given fact that the raw mesh is thin in one axis.
Since both non-identity rotations present in either file are pure rotations
about the same single axis (X), their composition is unambiguous regardless
of Euler order (same-axis rotations commute), so the `RotationOrder`
ambiguity does not actually matter for these two files, even though it would
for a general FBX. Level 10's extra `Lcl Rotation` composes with `PreRotation`
into a combined −180° about X — i.e. level 10's declared transform only
sign-flips two axes, it does **not** perform the same "stand it up" swap that
level 1's does.

**No `PostRotation`, no `GeometricTranslation/Rotation/Scaling`** in either
file — good, because silently ignoring those is the standard way to
mis-place an FBX mesh, and here there was nothing to ignore.

## 2. Naive placement (declared transform only) — fails, by metres

Placing each mesh with nothing but the declared transform above
(`PreRotation`/`Lcl Rotation` → rotation, `UnitScaleFactor/100` → scale to
metres, zero translation) and measuring nearest-triangle distance for 15,000
real hits:

| Level | median | p25 | p75 | p95 | max | mean |
|---|---|---|---|---|---|---|
| 1 | 3.68 m | 3.22 m | 4.18 m | 4.56 m | 4.82 m | 3.70 m |
| 10 | 4.47 m | 4.06 m | 4.90 m | 5.20 m | 5.41 m | 4.47 m |

Metres, not millimetres — the naive placement fails outright, as flagged as
the expected failure mode if the transform is wrong.

The diagnostic that explains *why*: level 1's hit-point X coordinate is
**exactly** 1.4365 (bit-identical) across all 75,326 spraying frames, from 11
different session files across 3 different workers on 3 different days. Real
hits lie in a perfectly flat world plane. The declared transform, however,
puts the mesh's thin axis on world **Z** (extent 0.025 m), not **X**
(declared-transform X extent is 1.42 m, matching the *real* hit spread on
world **Z**, 1.42 m — the wide and thin axes are swapped end for end). The
same swap-pattern shows up for level 10 (declared Y-extent 0.47 m ≈ real
X-spread 0.469 m; declared Z-extent 1.92 m ≈ real Y-spread 1.92 m). This is
strong, non-fitted evidence — matching bounding-box magnitudes on individual
axes, before any optimisation — that the shapes are the same and only the
*orientation* is wrong, which is exactly what step 3 exists to distinguish.

## 3. Solved-for rigid + uniform-scale fit — the two outcomes, kept separate

**Level 1: excellent fit.** Point-to-surface ICP (seeded from the FBX's own
declared rotation composed with the best of the 24 axis-aligned "cube"
rotations, then refined) converges to an ICP correspondence RMS of **2.7 mm**.
Measuring the *same* nearest-triangle-distance test as §2, now with the fitted
transform, over the same 15,000 held-out hits:

| | median | p25 | p75 | p95 | max | mean |
|---|---|---|---|---|---|---|
| Level 1, fitted | 0.0 mm | 0.0 mm | 0.0 mm | 0.0 mm | 46.2 mm | 0.5 mm |

Fitted scale = 0.025684, vs. the declared `UnitScaleFactor/100` = 0.025400 —
**1.1% apart**. This is the "good fit, transform under-specified" outcome:
the mesh is the same geometry as the hits; the FBX's per-asset transform
alone just doesn't carry the scene-level placement (specifically, a ~90°
extra yaw around the up axis) that Unity must be applying when it instantiates
this asset — and that placement lives in the Unity scene, not in this FBX
file, so it cannot be recovered from the FBX alone.

**Level 10: real but weaker fit.** Same method (24-candidate prefilter,
top 6 refined with point-to-surface ICP, 2,500-point correspondence sample):
best ICP RMS = **4.17 cm**. Nearest-triangle distance over 15,000 held-out
hits with the fitted transform:

| | median | p25 | p75 | p95 | max | mean |
|---|---|---|---|---|---|---|
| Level 10, fitted | 14.4 mm | 8.6 mm | 38.9 mm | 97.1 mm | 219.4 mm | 28.7 mm |

~100–300x better than the naive 4.47 m median, and centimetre-scale, not
metre-scale — so this is still the "matches, transform under-specified"
outcome, not "not the same geometry." But it's a real, un-tuned-away
residual, not measurement noise: fitted scale = 0.028476 vs. declared 0.0254
(12% off, versus level 1's 1.1%), and — as §5/§6 show — the fit isn't precise
enough to reliably pick the *correct* one of level 10's 436 triangles per
hit, only the right neighbourhood. Plausible cause: level 10's mesh is far
more geometrically complex (220 vertices vs. 16) and real hit coverage is
very unlikely to reach every face equally, and unlike level 1 (a flat two-
sided panel) a partially-covered complex mesh gives point-to-surface ICP
multiple similar-looking local optima to get stuck near. This was checked,
not assumed: switching from point-to-fixed-sample ICP (which spreads the
*whole* mesh sample cloud to match the hit cloud, and is provably biased
toward shrinking the fit when coverage is partial: median 5.3 cm, fitted
scale ratio 0.76–0.79) to point-to-continuous-surface ICP (median 1.4 cm,
scale ratio 1.12) recovered a large chunk of this — exactly the partial-
coverage signature — but not all of it.

## 4. Up-axis cross-check — agrees

The FBX's own `GlobalSettings.UpAxis=1 (Y)`, confirmed identical in both
files, is one independent witness. The CSV world's Y-up finding (established
separately, and independently reconfirmed by a sibling spike in this same
`.superpowers/spikes/` directory that traced and fixed a codebase assumption
that Z was vertical) is the other. In the final **solved** placement for both
levels, the mesh's physically tall raw dimension — unambiguous from its own
model-space extents (83.5 raw units for level 1, 75.6 for level 10, both far
larger than the other two axes) — lands on world **Y** in both cases. The two
witnesses agree. Note this is *not* a trivial/circular check: level 10's
declared transform alone (−180° about X) leaves the tall raw axis on
declared-Z, not declared-Y; it's only after adding the solved-for extra
rotation that it lands on world-Y. So the up-axis agreement survives even
though the file's own node transform doesn't correctly place the object.

## 5. Payoff: spray angle vs. surface normal

Using the fitted (not declared) transform, for 500 sampled spraying hits per
level: nearest triangle → barycentric-interpolated `LayerElementNormal`
(rotated to world by the fitted rotation only, no scale/translation) →
compared to the spray direction `hit − gun` (gun = the CSV's own
active-gun-selection logic, reused from `measured_pose.active_gun`).

**Level 1** (the level with a trustworthy, millimetre-precision fit):

> incidence-from-perpendicular: median **29.8°**, p25 22.2°, p75 35.2°, p95
> 42.7°, mean 28.6° (n=500)

This is the number the design doc has been missing: it clusters well away
from 90°, consistent with painters spraying roughly perpendicular to the
surface — exactly the sanity pattern the task predicted for a real match, and
distinct from what a wrong correspondence would produce (see level 10 below).

**Level 10**: median 84.4°, IQR 49°–140°, essentially uninformative. Given
§3's diagnosis (right neighbourhood, not reliably the right *triangle*), this
is expected, not a separate finding — a few-cm position error is enough to
pick an adjacent triangle with a different normal on a 220-vertex mesh, even
though the aggregate position match is good. **Level 10's normal is not
usable at this fit precision**; level 1's is.

## 6. UV cross-check — confirms level 1 independently, flags level 10

Interpolating the mesh's own `UV`/`UVIndex` at the same barycentric point and
regressing it against the CSV's `Ray Texture UV X/Y` (which turn out to be
raw texel coordinates on a texture atlas, not normalised 0–1 — the mesh's own
UV occupies only [0.012, 0.372] × [0.016, 0.427] of that atlas):

| Level | axis | slope | R² | median \|residual\| | p95 \|residual\| |
|---|---|---|---|---|---|
| 1 | U | −2050.4 | **0.999955** | 0.10 px | 0.31 px |
| 1 | V | +2048.8 | **0.999999** | 0.05 px | 0.10 px |
| 10 | U | +888.6 | 0.230 | 292 px | 998 px |
| 10 | V | −909.2 | 0.320 | 232 px | 788 px |

Level 1's fitted slopes land almost exactly on ±2048 — a clean 2048×2048
texture atlas falls straight out of the regression, with sub-pixel residual
and R² indistinguishable from 1. This is the strongest evidence in this
report: it validates the 3D fit, the triangle/barycentric correspondence, and
the UV data *simultaneously*, via a completely different measurement (2D
texel space, not 3D distance) that could easily have disagreed if the 3D fit
were merely "close enough."

Level 10's UV regression is poor (R² 0.23–0.32) — consistent with, not
contradicting, §5: the fit places hits near the right patch of surface but
not reliably on the right triangle, and UV (like the normal) is a
per-triangle attribute, so it inherits that imprecision.

## A bug worth flagging (fixed, not hidden)

The hand-rolled 3x3 SVD's degenerate-singular-value fallback initially used
an absolute `1e-12` threshold to detect a zero singular value. Level 1's hit
cloud is *exactly* planar (bit-identical X, see §2), which makes its target
covariance matrix exactly rank-deficient; floating-point noise in that matrix
made the "is this singular value zero" test flip unpredictably across ICP
iterations, occasionally producing a matrix with a rotation "row" of literally
`(0, 0, 0)` — not a valid rotation. Root cause understood analytically (a
matrix row of exact zeros forces the corresponding output-space basis vector
into the null-space column deterministically, not arbitrarily) and fixed by
computing that column via cross product instead of a threshold-gated
Gram-Schmidt search, plus switching the threshold to be relative to the
largest singular value. Re-validated against both the synthetic self-test and
a synthetic degenerate case (planar target, known transform) before being
used on real data again. Flagging this because it's exactly the kind of
"looks like a great fit" trap the task's discipline note warns about — the
raw ICP RMS number would have looked fine even with the invalid rotation,
and only the row-norm/determinant check caught it.

## Bottom line for REQ-19 / angle-based coaching

- **Level 1**: hit points match the mesh, essentially exactly, once the
  correct — *solved-for*, not declared — placement is used. The FBX's own
  per-node transform is short one rotation (roughly a 90° yaw around the up
  axis) that isn't recorded in the file; it most likely lives in the Unity
  scene that instantiates this asset. Surface normal at each hit is directly
  usable: spray angle from perpendicular clusters around 30°, a sane,
  usable signal for angle-based correction.
- **Level 10**: same geometry, same conclusion in kind, but the achievable
  fit here is centimetre- not millimetre-precision, and that's not tight
  enough to trust the *per-triangle* normal or UV yet (both come back
  effectively random). Positionally it's a clear win over the naive
  placement.
- **Neither level's FBX file is usable as-is** for a "read the declared
  transform and go" pipeline — both need a solved-for correction, and the
  correction differs per level (not a single global fudge factor), which
  points at a per-level scene placement rather than a bug in one file.
- **What would unblock level 10 specifically**: the Unity scene/prefab file
  itself (if available) would remove the need to solve for placement at all;
  short of that, more ICP restarts / a proper global-optimisation seed search
  beyond the 24 axis-aligned candidates, or hand-inspecting which of level
  10's faces are actually covered by real hits (to check the partial-coverage
  hypothesis directly rather than inferring it), would be the next steps —
  not attempted here to keep this a bounded spike.
