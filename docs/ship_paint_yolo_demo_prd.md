# Ship Paint Defect Detection — YOLO Demo PRD (Claude Code 실행용)

> **목적**: AI Hub 데이터셋 71447(선박 도장 품질 데이터)을 활용하여 YOLO 기반 도장 결함 탐지 모델을 **빠르게 파인튜닝하여 데모만** 수행한다.
> **우선순위**: 빠른 실행 > 정확도. 정확도 튜닝은 본 데모 범위 외.
> **타겟 환경**: RTX 4080 16GB VRAM (Meejin 로컬 워크스테이션)
> **본 문서 작성 원칙**: 확인된 사실은 본문에, 다운로드 후 검증이 필요한 항목은 §9에 별도 표기.

---

## 0. Scope

### 0.1 In-scope (이번 데모에 포함)
- AI Hub aihubshell로 데이터셋 71447 다운로드
- 라벨(JSON) 구조 확인 및 COCO → YOLO 변환
- YOLO11 모델 파인튜닝 (bbox 탐지)
- 추론 및 시각화 데모 (이미지 1~5장에 bbox overlay)

### 0.2 Out-of-scope (이번 데모에서 제외)
- Segmentation 학습 (bbox만 진행. 데이터셋이 segmentation polygon을 포함하므로 향후 확장 가능)
- 모델 정확도 최적화 / 하이퍼파라미터 튜닝
- 클래스 불균형 정밀 처리 (단순 정상 클래스 제외만 수행)
- ROS2 연동 (별도 후속 작업으로 분리)
- 다국어 라벨 처리 / 도메인 특화 후처리

### 0.3 성공 기준 (Definition of Done)
1. YOLO 학습이 epoch 끝까지 정상 종료되어 `best.pt`가 생성된다.
2. `best.pt`로 검증셋 이미지 1장 이상에서 결함 bbox가 그려진 추론 결과 이미지가 저장된다.
3. 학습 중/후 mAP, precision, recall 등 ultralytics 기본 지표가 출력된다.

> **사실**: 본 데모는 **모델 품질 검증이 아닌 파이프라인 동작 검증**이 목적이다.

---

## 1. 환경 준비

### 1.1 사전 조건 (사용자 수동 확인 필요)
- AI Hub 회원가입 및 로그인 완료
- 데이터셋 71447 페이지에서 **다운로드 신청 승인 완료** (Claude Code가 대신 할 수 없음)
- API Key 발급 완료 (사용자 보유: 환경변수로 주입)

### 1.2 디렉토리 구조 (Claude Code가 생성)
```
ship_paint_demo/
├── data/
│   ├── raw/                 # aihubshell 다운로드 원본
│   ├── coco/                # 추출된 이미지 + COCO JSON
│   └── yolo/                # YOLO 형식으로 변환된 데이터
│       ├── images/{train,val}/
│       └── labels/{train,val}/
├── configs/
│   └── ship_paint.yaml      # YOLO 데이터 설정
├── scripts/
│   ├── 01_download.sh
│   ├── 02_inspect_labels.py
│   ├── 03_coco_to_yolo.py
│   ├── 04_train.sh
│   └── 05_predict.py
├── runs/                    # ultralytics 학습 결과
└── README.md
```

### 1.3 Python 환경
- Python 3.10+ (RTX 4080 + CUDA 12.x 드라이버 가정)
- 필요 패키지:
  ```bash
  pip install ultralytics opencv-python pillow tqdm
  ```
- ultralytics 설치 시 PyTorch도 함께 설치됨

> **추론**: 기존 sLMM 학습 환경(`.venv-train`)과 분리하여 새 venv를 만드는 것을 권장. 의존성 충돌 회피 목적.

---

## 2. 데이터 다운로드

### 2.1 aihubshell 설치 (정정됨)

> ⚠️ **첨부 원본 PRD의 `pip install aihubshell`은 잘못된 정보**.
> AI Hub 공식 방식은 curl로 셸 스크립트를 다운로드하는 방식.

```bash
# 작업 디렉토리에서
curl -o aihubshell https://api.aihub.or.kr/api/aihubshell.do
chmod +x aihubshell

# (선택) 전역 등록
sudo mv aihubshell /usr/local/bin/aihubshell
```

### 2.2 API Key 환경변수 주입

> ⚠️ **보안 주의**: API Key를 코드/스크립트에 하드코딩하지 말 것. 환경변수 사용.

```bash
# 사용자 키는 .env 또는 셸 환경에서 주입
export AIHUB_API_KEY="<USER_API_KEY>"   # 실제 값은 사용자 본인 보유
```

### 2.3 데이터셋 정보 확인 (다운로드 전)

```bash
# 데이터셋 71447의 파일 트리 조회 (다운로드 X, 목록만)
aihubshell -mode l -datasetkey 71447 -aihubapikey "$AIHUB_API_KEY"
```

**확인 포인트**:
- 전체 파일 수 및 총 용량
- Training / Validation 분할이 이미 제공되는지
- 라벨 폴더 명칭(`라벨링데이터` / `Label` 등)
- 원천 폴더 명칭(`원천데이터` / `Source` 등)

### 2.4 전체 다운로드

```bash
# 압축해제 포함 원본 대비 2~3배 저장공간 필요
mkdir -p data/raw && cd data/raw

aihubshell -mode d \
  -datasetkey 71447 \
  -filekey all \
  -aihubapikey "$AIHUB_API_KEY"
```

> **사실**: 다운로드 완료 시 aihubshell이 자동으로 분할 파일 병합 + 압축 해제 + 압축파일 제거를 수행함.

### 2.5 부분 다운로드 옵션 (데모 목적)

> **추론**: 71447이 대용량일 경우 `-filekey` 옵션으로 일부 파일만 받아 데모를 진행할 수 있음. `-mode l` 결과로 확인된 filekey 중 가장 작은 단위(예: 단일 zip 1개)만 받아 빠르게 검증.

```bash
# 예시 (filekey는 -mode l 결과에서 확인 후 치환)
aihubshell -mode d -datasetkey 71447 -filekey <SMALL_FILEKEY> -aihubapikey "$AIHUB_API_KEY"
```

---

## 3. 라벨 구조 검증 (다운로드 후 필수)

> ⚠️ **첨부 원본 PRD에 명시된 클래스 ID(101=정상, 201=워터스포팅, …)는 사실 검증 전 정보**.
> 실제 다운로드된 JSON에서 직접 확인해야 함.

### 3.1 검증 스크립트 (`scripts/02_inspect_labels.py`)

```python
"""라벨 JSON 구조 자동 검증 스크립트.
- COCO 형식인지 확인
- categories 실제 ID와 이름 출력
- annotations에 bbox, segmentation 존재 여부 확인
- 이미지 수 / 어노테이션 수 통계
"""
import json
import sys
from pathlib import Path
from collections import Counter

def inspect(label_json_path: Path):
    with open(label_json_path, "r", encoding="utf-8") as f:
        data = json.load(f)

    print(f"== {label_json_path} ==")
    print("Top-level keys:", list(data.keys()))

    if "images" in data:
        print(f"  images: {len(data['images'])}")
    if "categories" in data:
        print(f"  categories ({len(data['categories'])}):")
        for c in data["categories"]:
            print(f"    id={c.get('id')}, name={c.get('name')}")
    if "annotations" in data:
        anns = data["annotations"]
        print(f"  annotations: {len(anns)}")
        cat_counter = Counter(a["category_id"] for a in anns)
        print(f"  per-category counts: {dict(cat_counter)}")
        if anns:
            sample = anns[0]
            print(f"  sample annotation keys: {list(sample.keys())}")
            print(f"    has bbox: {'bbox' in sample}")
            print(f"    has segmentation: {'segmentation' in sample}")

if __name__ == "__main__":
    # 인자로 받은 단일 JSON 또는 디렉토리 내 모든 JSON 검사
    target = Path(sys.argv[1])
    if target.is_dir():
        for p in sorted(target.rglob("*.json")):
            inspect(p)
            print()
    else:
        inspect(target)
```

### 3.2 검증 후 확정해야 할 정보
1. **실제 categories 구조** (id ↔ name 매핑)
2. **bbox 형식** (COCO 기본은 `[x, y, w, h]` 절대좌표 — 정정 변환 필요)
3. **이미지 경로 규약** (file_name 필드의 상대/절대 여부)
4. **train/val 분할 제공 여부**

→ 위 정보가 확인되면 §4의 변환 스크립트의 매핑 테이블/경로 처리를 그에 맞게 수정한다.

---

## 4. COCO → YOLO 변환

### 4.1 변환 규칙 (사실)
- COCO bbox: `[x_min, y_min, width, height]` (절대 픽셀 좌표)
- YOLO bbox: `class_id x_center y_center width height` (이미지 폭/높이 대비 0~1 정규화)
- 변환 공식:
  ```
  x_center = (x_min + width/2) / image_width
  y_center = (y_min + height/2) / image_height
  w_norm   = width / image_width
  h_norm   = height / image_height
  ```

### 4.2 변환 정책 (이번 데모용 결정)
- **정상 클래스(=결함 없음) 제외** — 어노테이션이 있는 결함 클래스만 학습
- **클래스 ID는 0부터 연속 재할당** (YOLO 요구사항)
  - 원본 ID → 새 ID 매핑은 §3.1 결과로 확정
- **train/val 분할**:
  - 데이터셋이 이미 분할을 제공하면 그대로 사용
  - 제공하지 않으면 `8:2` 무작위 분할 (seed 고정)

### 4.3 변환 스크립트 (`scripts/03_coco_to_yolo.py`)

```python
"""COCO → YOLO 변환 스크립트.
사용 예:
  python scripts/03_coco_to_yolo.py \
    --coco data/coco/annotations.json \
    --images data/coco/images \
    --out data/yolo \
    --val-ratio 0.2 \
    --seed 42

NOTE: 정상 클래스는 --exclude-cat-ids 로 제외.
"""
import argparse
import json
import random
import shutil
from pathlib import Path

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--coco", type=Path, required=True)
    ap.add_argument("--images", type=Path, required=True)
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--val-ratio", type=float, default=0.2)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--exclude-cat-ids", type=int, nargs="*", default=[],
                    help="제외할 원본 카테고리 ID (예: 정상)")
    ap.add_argument("--predefined-split", type=Path, default=None,
                    help="(선택) val 이미지 ID 리스트 JSON. 제공 시 무작위 분할 무시.")
    args = ap.parse_args()

    with open(args.coco, "r", encoding="utf-8") as f:
        data = json.load(f)

    # 1) 사용할 카테고리 추리고 0-base 재매핑
    used_cats = [c for c in data["categories"] if c["id"] not in args.exclude_cat_ids]
    id_map = {c["id"]: i for i, c in enumerate(used_cats)}
    print("[INFO] class mapping (orig_id -> new_id):")
    for c in used_cats:
        print(f"  {c['id']} ({c['name']}) -> {id_map[c['id']]}")

    # 2) 이미지 인덱싱
    images = {img["id"]: img for img in data["images"]}

    # 3) 어노테이션을 이미지 단위로 묶기 (제외 카테고리 필터링)
    img_to_anns = {}
    for ann in data["annotations"]:
        if ann["category_id"] in args.exclude_cat_ids:
            continue
        img_to_anns.setdefault(ann["image_id"], []).append(ann)

    # 4) train/val 분할
    img_ids = list(img_to_anns.keys())
    if args.predefined_split and args.predefined_split.exists():
        with open(args.predefined_split, "r") as f:
            val_ids = set(json.load(f))
    else:
        random.Random(args.seed).shuffle(img_ids)
        n_val = int(len(img_ids) * args.val_ratio)
        val_ids = set(img_ids[:n_val])

    # 5) 출력 디렉토리
    for split in ("train", "val"):
        (args.out / "images" / split).mkdir(parents=True, exist_ok=True)
        (args.out / "labels" / split).mkdir(parents=True, exist_ok=True)

    # 6) 변환 실행
    n_train = n_val_out = 0
    for img_id, anns in img_to_anns.items():
        img = images[img_id]
        split = "val" if img_id in val_ids else "train"
        src = args.images / img["file_name"]
        if not src.exists():
            print(f"[WARN] missing image: {src}")
            continue

        dst_img = args.out / "images" / split / Path(img["file_name"]).name
        shutil.copy2(src, dst_img)

        label_lines = []
        W, H = img["width"], img["height"]
        for ann in anns:
            x, y, w, h = ann["bbox"]
            xc = (x + w / 2) / W
            yc = (y + h / 2) / H
            wn = w / W
            hn = h / H
            cls = id_map[ann["category_id"]]
            label_lines.append(f"{cls} {xc:.6f} {yc:.6f} {wn:.6f} {hn:.6f}")

        dst_label = args.out / "labels" / split / (Path(img["file_name"]).stem + ".txt")
        dst_label.write_text("\n".join(label_lines), encoding="utf-8")

        if split == "val":
            n_val_out += 1
        else:
            n_train += 1

    print(f"[DONE] train={n_train}, val={n_val_out}")
    print(f"[NEXT] Generate ship_paint.yaml using class names: "
          f"{[c['name'] for c in used_cats]}")

if __name__ == "__main__":
    main()
```

### 4.4 데이터 설정 파일 (`configs/ship_paint.yaml`)

> ⚠️ `names` 필드는 §3.1 검증 결과로 확정된 클래스명/순서로 수정해야 함.
> 아래는 **첨부 원본 PRD에 적힌 클래스명을 기준으로 한 임시 템플릿**이며, 실제 라벨 검증 후 갱신 필수.

```yaml
# 절대 경로 또는 프로젝트 루트 기준 상대 경로
path: /absolute/path/to/ship_paint_demo/data/yolo
train: images/train
val: images/val

# 검증 후 확정 — 임시값
names:
  0: 워터스포팅
  1: 흐름
  2: 도막분리
  3: 핀홀
  4: 균열
  5: 부풀음
  6: 이물질포함
  7: 용접손상
  8: 스크래치
  9: 도막떨어짐
```

---

## 5. YOLO 학습

### 5.1 모델 선택 근거 (RTX 4080 16GB 기준)

| 옵션 | 파라미터 | 이번 데모 적합성 |
|------|---------|------------------|
| `yolo11n.pt` | 가장 작음 | ✅ **가장 빠름. 데모 1순위.** |
| `yolo11s.pt` | small | ⚠️ 가능하나 학습 시간 증가 |
| `yolo11m.pt` | medium | ❌ imgsz 1024에서 OOM 위험 |

> **추론**: 이번 목적이 "파이프라인 동작 검증 데모"이므로 `yolo11n.pt`가 합리적.
> `yolo11s.pt`도 문제는 없지만 학습 시간이 길어진다.

### 5.2 학습 명령 (`scripts/04_train.sh`)

```bash
#!/usr/bin/env bash
set -e

# RTX 4080 16GB 기준 보수적 설정
yolo detect train \
  model=yolo11n.pt \
  data=configs/ship_paint.yaml \
  imgsz=640 \
  epochs=30 \
  batch=16 \
  device=0 \
  project=runs/ship_paint \
  name=demo_v1 \
  patience=10 \
  save=True \
  plots=True
```

> **사실 vs 추론 구분**:
> - imgsz=640은 ultralytics 기본값 (사실).
> - epochs=30은 데모 시간 단축을 위한 선택 (추론). 실제 PoC 권장은 50~100.
> - batch=16은 yolo11n + imgsz 640 + RTX 4080에서 안정적이다 (추론). OOM이면 8로 낮춘다.

### 5.3 결과물 위치
- 가중치: `runs/ship_paint/demo_v1/weights/best.pt`, `last.pt`
- 메트릭/그래프: `runs/ship_paint/demo_v1/`

---

## 6. 추론 데모

### 6.1 추론 스크립트 (`scripts/05_predict.py`)

```python
"""학습된 best.pt로 검증셋 이미지에 추론 후 시각화 저장."""
import argparse
from pathlib import Path
from ultralytics import YOLO

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--weights", type=Path, default=Path("runs/ship_paint/demo_v1/weights/best.pt"))
    ap.add_argument("--source", type=Path, default=Path("data/yolo/images/val"))
    ap.add_argument("--n", type=int, default=5, help="추론할 이미지 수")
    ap.add_argument("--conf", type=float, default=0.25)
    args = ap.parse_args()

    model = YOLO(str(args.weights))
    images = sorted(args.source.glob("*.jpg")) + sorted(args.source.glob("*.png"))
    if not images:
        print(f"[ERROR] no images in {args.source}")
        return
    images = images[: args.n]

    results = model.predict(
        source=[str(p) for p in images],
        conf=args.conf,
        save=True,
        project="runs/ship_paint",
        name="demo_predict",
        exist_ok=True,
    )

    for r in results:
        print(f"{Path(r.path).name}: detections={len(r.boxes)}")
        if len(r.boxes):
            for b in r.boxes:
                cls = int(b.cls[0])
                conf = float(b.conf[0])
                xyxy = b.xyxy[0].tolist()
                print(f"  cls={cls} ({model.names[cls]}) conf={conf:.2f} xyxy={[round(v,1) for v in xyxy]}")

if __name__ == "__main__":
    main()
```

### 6.2 데모 산출물
- 시각화 이미지: `runs/ship_paint/demo_predict/*.jpg`
- 콘솔 출력: 이미지별 클래스/신뢰도/좌표 리스트

---

## 7. Claude Code 실행 순서 (요약)

> 아래 순서대로 진행. 각 단계 완료 후 다음 단계로.

1. **사용자 확인**: AI Hub 71447 다운로드 승인 완료 여부 (Claude Code가 대신 못 함)
2. **환경 변수 설정**: `export AIHUB_API_KEY="..."`
3. **디렉토리 생성**: `mkdir -p ship_paint_demo/{data/{raw,coco,yolo},configs,scripts,runs}`
4. **aihubshell 설치**: §2.1
5. **데이터셋 목록 조회**: `aihubshell -mode l -datasetkey 71447 ...` (구조 확인)
6. **다운로드**: §2.4 또는 §2.5
7. **라벨 검증**: `python scripts/02_inspect_labels.py <JSON 경로>` → **결과를 사용자에게 보고하고 클래스 매핑 확정**
8. **`configs/ship_paint.yaml` 갱신**: §3.1 결과로 names 필드 확정
9. **변환**: `python scripts/03_coco_to_yolo.py ...`
10. **학습**: `bash scripts/04_train.sh`
11. **추론**: `python scripts/05_predict.py`
12. **결과 보고**: `runs/ship_paint/demo_predict/` 의 이미지 및 콘솔 출력

---

## 8. 디렉토리 정리 (Claude Code가 생성할 README.md 항목)

```markdown
# Ship Paint Defect Detection — Demo

## Quick Start
1. `export AIHUB_API_KEY="..."`
2. `bash scripts/01_download.sh`
3. `python scripts/02_inspect_labels.py data/raw/<labels_dir>` → categories 확인
4. `configs/ship_paint.yaml` 의 `names`/`path` 갱신
5. `python scripts/03_coco_to_yolo.py --coco ... --images ... --out data/yolo --exclude-cat-ids <정상 ID>`
6. `bash scripts/04_train.sh`
7. `python scripts/05_predict.py`

## Outputs
- 모델: `runs/ship_paint/demo_v1/weights/best.pt`
- 추론: `runs/ship_paint/demo_predict/*.jpg`
```

---

## 9. 불확실성 / 검증 필요 항목

이번 PRD는 다운로드 전 정보의 한계가 있다. 아래 항목은 **다운로드 후 반드시 확인하고 본 PRD를 갱신**해야 한다.

| 항목 | 첨부 원본 PRD의 가정 | 검증 방법 | 영향도 |
|------|---------------------|-----------|-------|
| 클래스 ID 매핑 (101=정상, 201~303=결함) | 정확하다고 가정됨 | `02_inspect_labels.py` 실행 | **High** — 학습 라벨 자체가 어긋남 |
| COCO 형식 준수 여부 | 그렇다고 가정됨 | JSON top-level keys 확인 | **High** — 변환 스크립트 자체가 동작 안 함 |
| segmentation polygon 품질 | 다양함 가능성 | bbox 학습엔 영향 없음 | Low (이번 데모 범위 외) |
| train/val 사전 분할 제공 여부 | 명시 안 됨 | 폴더 구조 확인 | Medium — 분할 정책 결정 |
| 전체 데이터 용량 | 미상 | `-mode l` 출력 | Medium — 디스크/시간 계획 |
| 클래스별 샘플 수 분포 | 미상 | `02_inspect_labels.py` 출력 | Medium — 클래스 불균형 시 결과 해석 주의 |

> **원칙**: §9의 항목이 모두 검증되기 전까지는 §3.1 결과를 §4.4 yaml에 정확히 반영하지 않은 상태로 학습을 시작하지 않는다.

---

## 10. 후속 작업 (이번 데모 이후, 본 PRD 범위 외)

본 데모 성공 시 고려할 다음 단계 — **본 데모 진행 중에는 손대지 않는다**:

1. **Segmentation 학습** — `yolo11n-seg.pt`로 polygon 학습 (변환 시 segmentation 필드 활용)
2. **클래스 불균형 처리** — class weights, 오버샘플링, focal loss 등
3. **모델 크기 확대** — `yolo11s.pt` / `yolo11m.pt`로 정확도 향상
4. **ROS2 연동** — `vision_msgs/Detection2DArray` publish 노드 별도 구현
5. **Co-Pilot RAG 연동** — 결함 클래스 탐지 결과를 §RAG 검색 키워드로 사용하여 결함 원인/대처 가이드 자동 생성

---

## 11. 변경 이력 (이전 PRD 대비)

| 항목 | 이전 PRD | 현재 PRD | 사유 |
|------|---------|---------|------|
| aihubshell 설치 | `pip install aihubshell` | `curl -o aihubshell https://api.aihub.or.kr/api/aihubshell.do` | 사실 검증 결과 정정 |
| API Key 처리 | 명령줄에 직접 노출 | 환경변수 `$AIHUB_API_KEY` | 보안 |
| 학습 모델 | yolo11s.pt, imgsz=1024, epochs=50 | yolo11n.pt, imgsz=640, epochs=30 | 빠른 데모 우선 |
| Segmentation | 포함 (혼합) | 제외 (bbox만) | 데모 범위 축소 |
| ROS2 연동 | 포함 | 제외 (후속) | 데모 범위 축소 |
| 클래스 매핑 | 단정적 | "다운로드 후 검증 필수"로 명시 | 사실 미확인 |
| 디렉토리 구조 | 부분 명시 | 전체 구조 + 스크립트 파일명까지 명시 | Claude Code 실행성 |
| 실행 순서 | 분산 | §7에 단일 체크리스트로 통합 | Claude Code 실행성 |
