# 자세 비교 실험대 — 표시 필드 계약 (2026-09-15 피드백 라운드 3)

`POST /api/compare`의 **표시 전용** 필드만 적는다. 채점 필드(score·improvement·gate·axes의
magnitude_cm/direction/sigma)는 `compare(include_view=False)`가 HEAD b5f7341 기준선과 JSON 문자열까지
같다(`test_scoring_isolation.py`). 이 문서의 enum·키 목록은 `test_instructions.py::
test_contract_doc_lists_every_text_mode_and_note_key`가 코드 상수와 대조한다.

## measured.axes[] 표시 필드

- `text_mode`: `"directive"` | `"stroke_relative"`.
  - `directive`: 방향 동사 문장("오른손(건)을 벽 쪽으로 15.4cm 붙이세요.").
  - `stroke_relative`: wall 축이 검증 메시 착탄면 법선과 30° 넘게 어긋날 때. 스트로크 기저 법선은 정의상
    수평이고 작업자 쪽이 +라서 "착탄 궤적 쪽으로 수평으로 … 붙이세요/다가서세요"로 방향을 지킨다.
  - 옛 계약의 `"axis_only"`(방향 없이 수치만)는 라운드 2에서 `stroke_relative`로 바뀌었다.
- `figure`: U+2212 부호 + 소수 1자리. `sigma_display`, `trace`, `surface_angle_display`.

## measured.orientation (건 각도 참고표)

- `items[].display`: 정수 도. **음수 부호는 U+2212**(교정 `figure`와 같다). ASCII `-`를 쓰지 않는다.

## measured.joint_axes (관절 축 평가표, 라운드 3 신설)

`{title, note, rows[]}`. `rows[]`는 `{key, joint, status, status_text, label, query, reference, samples,
display, reason, reason_text}`.

- `status`: `"measured"` | `"measured_above"` | `"not_evaluated"` | `"not_measured"`.
- 값(`display`)이 있는 행은 실측 강체 축뿐이다: `head_hmd_heading`(HMD 시선 수평 방위와 머리→착탄 수평
  방위 사이 각, 부호 없음). 나머지 관절 행은 값 없이 `reason_text`만 싣는다.

## view.fit

- `notes[]`와 `note_keys[]`는 같은 길이·같은 순서다. **화면은 위치가 아니라 키로 짝짓는다.**
  키: `avatar_estimate`, `gravity`, `legs`, `torso`, `grip_limit`(파지 상한 초과일 때만).
  라운드 1 계약의 "[2]=몸통"은 다리 고지가 끼어 `[3]`으로 밀렸다 — 그래서 키를 신설했다.
- `notes_collapsed[]`: 캔버스 아래에서 접어 둘 키(현재 `grip_limit`). 접힌 고지도 누르면 그 자리에서 펼쳐진다.
- `legs`: `{plausible, joints, checks, limits, frame_offset, searched_frames, display_default,
  placeholder_joints, placeholder_bones}`.
  - `display_default`: `"placeholder"`(곧게 선 자리표시 다리, 측정 아님) | `"ik"` | `"hidden"`.
  - `placeholder_joints`: 배치된 엉덩이 관절에서 중력 아래로 리그 허벅지·종아리 길이, 발끝은 대표 프레임
    착탄 방향(수평)으로 리그 발 길이. 수평 방향이 없으면 발끝 관절은 없다.
- `legs_visible_default`: IK 다리를 기본으로 켜는가. 자리표시 다리가 있으면 `false`.

## view

- `surface_hits`: 대표 패스의 **원본** 분사 착탄점(최대 200, 균등 추출). `layers.query.hit`은 호길이
  리샘플 격자라 모서리를 가로지르면 벽에서 뜬다(L6 최대 62.9mm) — 화면은 착탄 궤적·벽 포함 맞춤에 이
  필드만 쓴다.
- `composite.spray_ray_angle_deg`: 건 분사축과 (착탄점 − 건 마커) 사이 각. `spray_ray_hit_miss_mm`는
  건–벽 거리에 비례하므로 레벨마다 같은 문턱으로 읽지 않는다.

## 화면 합격 기준 문구 수정

- "교정 숫자 한 벌": 화면 innerText의 `(\d+\.\d)cm` 숫자 집합은 `figure` 절댓값 ∪ `miss_display` 숫자 ∪
  ② 부위별 편차 `deviation_display`(패스 평균 거리, 교정 아님)로 **모두 설명**돼야 하고, `figure` 절댓값은
  전부 화면에 있어야 한다. 옛 문구 "figure ∪ miss_display와 같다"는 ② 편차 숫자를 빠뜨렸다.
