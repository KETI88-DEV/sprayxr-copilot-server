# 팔 자세 레퍼런스 아티팩트 계약

`POST /pose_estimation/{reference,align,score,verify,sequence}`가 읽는 파일의 형태다.
직접 만들어 넣을 수 있도록 **무엇이 판정을 만드는지**까지 적는다.

## 어디서 찾나

```
ARM_REFERENCE_PATH (환경변수)
   ↓ 없으면
/output/expert_reference/expert_arm_reference.json
```

JSON **한 파일**이다. 파일 mtime으로 캐시하므로 **통째로 교체하면 다음 요청부터 반영**된다
(`_read_arm_reference`의 `lru_cache` 키가 `(path, mtime)`이다).

`/output`은 마운트 산출물이지 저장소 데이터가 아니다(AGENTS.md) — 이 파일은 커밋되지 않는다.

## 버킷을 고르는 키

```
reach = median(착탄 수직) − median(건 수직)      ← 둘 다 월드, 바닥 상수가 지워진다
band  = f"{floor(reach / 0.10) * 0.10:+.2f}"     ← "+0.00", "-0.10" 꼴. 부호가 항상 붙는다
```

**건 기준인 이유**: CSV의 머리는 HMD 강체이고 런타임의 머리는 골격 관절이라 같은 점이
아니다. 건은 양쪽 모두 같은 물건이다(`measured_pose.right` / `tool_pos_rot.pos`).
0.10 m 어긋나면 버킷의 44%가 바뀐다 —
`docs/measurements/2026-09-29-two-rigs-agree-only-below-the-shoulders.md` §5.

## 형태

```jsonc
{
  "version": 1,
  "scoring_set_id": "v3-arm-6",   // 채점 집합 딱지. 바뀌면 점수의 뜻이 바뀐다
  "reach_band_m": 0.1,            // 밴드 폭. `arm_reference.REACH_BAND_M`과 같아야 한다
  "built_at": "2026-09-29T09:38:02Z",
  "sessions": 45, "passes": 1217, "skipped": 0,
  "workers": ["김경호", …],

  "buckets": {
    "-0.10": {
      "n": 322,        // ← 품질 사다리를 만든다
      "workers": 6,    // ← 품질 사다리를 만든다

      // ── 접기 채점(시퀀스를 한 점으로) ──
      "joints": {
        "RightHand": {
          "mean": [-0.0162, 0.2241, 0.2265],   // **몸 프레임** 좌표(m). 월드가 아니다
          "sd":   [ 0.2427, 0.1665, 0.2545],
          "n": 322,
          "radius":  { "mean": 0.4960, "sd": 0.0816 },
          "heading": { "mean": [-0.1465, 0.6875, 0.7113], "angular_sd": 0.8788 }
        }
        // RightUpperArm · RightForeArm · RightHand · LeftUpperArm · LeftForeArm · LeftHand
      },
      "expert_scores": {          // 접기 점수의 백분위 사다리
        "p10":0.305, "p25":0.656, "p50":0.839, "p75":0.920, "p90":0.951,
        "reference_pass_count": 322, "in_sample": true
      },

      // ── 위상 채점(착탄 경로 DTW). 없으면 세 단계 경로가 phase_reference_missing으로 멈춘다 ──
      "prototype": [[0.0,0.2332,-0.0323], …12점],   // 착탄 경로 무게중심, **시작점 기준 변위**
      "prototype_support": { "to_prototype":0.4629, "between":1.3091, "n":322 },
      "phases": [ { "n":730, "joints": { …위 `joints`와 같은 모양… } }, …12개 ],
      "expert_scores_phase": { "p10":0.277, …, "reference_pass_count":322, "in_sample":true }
    }
  }
}
```

## 값이 만드는 판정

| 조건 | `reference_quality` | 결과 |
|---|---|---|
| `n` < 20 또는 버킷 없음 | `unavailable` | **점수 없음**(`reference_data_missing`) |
| `workers` ≤ 1, `n` ≥ 20 | `single_worker` | `degraded` |
| `workers` ≥ 3 **그리고** `n` ≥ 100 | `multi_worker` | `available` |
| 그 밖 | `coarse_bucket` | `degraded` |

`workers`를 1로 적으면 **`available`로 올라가지 않는다.** 1인 산포는 "숙련자 통상 범위"가
아니라 "그 사람의 습관"이기 때문이다.

위상 블록에서는 `phases[i].n` < 20인 위상이 **채점에서 빠진다** — 얕은 위상의 SD는 우연이다.

## 직접 만들 때 지켜야 할 것

1. **`mean`/`sd`는 몸 프레임 좌표다.** 원점은 어깨 중점, x̂ = 왼어깨→오른어깨,
   ŷ = Chest→어깨 중점(직교화), ẑ = x̂ × ŷ. 반드시 `arm_reference.normalize()`로 만든다 —
   손으로 적으면 두 구현이 갈린다.
2. **`radius`/`heading`이 없으면 축 RMS로 내려간다.** 옛 아티팩트 호환 경로이고, 그때는
   손의 뻗음이 희석돼 45 cm를 옮겨도 교정이 안 나간다.
3. **`reference_pass_count`가 버킷과 대조된다.** 평균·산포만 다시 굽고 사다리를 그대로
   두면 백분위가 `null`이 된다. 틀린 값이 조용히 나가는 것보다 낫다.
4. **두 사다리를 섞지 않는다.** `expert_scores`(접기)와 `expert_scores_phase`(위상)는 다른
   분포다(레벨1 밴드 −0.10에서 0.839 vs 0.804). 섞으면 전원이 낮게 나온다.
5. **`prototype`은 시작점 기준 변위다.** 월드 절대좌표를 넣으면 벽 위치가 정렬에 들어온다.
6. **`in_sample`을 정직하게 적는다.** 빌더가 만든 사다리는 각 패스가 자기 버킷 평균에
   기여한 표본내 값이다. n이 수백이면 낙관이 작지만(LOO 0.809 vs 표본내 0.81) 사실은 남는다.

## 만드는 법

```bash
docker exec -w /workspace -e PYTHONPATH=/workspace ship-paint-trainer \
  python3 trainer/scripts/build_arm_reference.py \
    --root /data/sources/professionals \
    --levels 1,2,3 \
    --out /output/expert_reference/expert_arm_reference.json
```

`--levels`를 비우면 전부(479세션·64,098패스·약 30분). 위상 표본은 밴드당 400패스까지만
쓰지만 **점 레퍼런스는 전량으로 굽는다.**
