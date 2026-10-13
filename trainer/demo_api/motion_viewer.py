"""Read-only browser viewer for expert skeleton motion references."""

from __future__ import annotations

import csv
import gzip
import json
import math
import statistics
from functools import lru_cache
from pathlib import Path
from typing import Any, TextIO

from fastapi import APIRouter, HTTPException, Query
from fastapi.responses import HTMLResponse

_JOINT_PARENTS: tuple[tuple[str, str | None], ...] = (
    ("Pelvis", None),
    ("Spine", "Pelvis"),
    ("Chest", "Spine"),
    ("Head", "Chest"),
    ("Left Shoulder", "Chest"),
    ("Left Upper Arm", "Left Shoulder"),
    ("Left Fore Arm", "Left Upper Arm"),
    ("Left Hand", "Left Fore Arm"),
    ("Right Shoulder", "Chest"),
    ("Right Upper Arm", "Right Shoulder"),
    ("Right Fore Arm", "Right Upper Arm"),
    ("Right Hand", "Right Fore Arm"),
    ("Left Thigh", "Pelvis"),
    ("Left Calf", "Left Thigh"),
    ("Left Foot", "Left Calf"),
    ("Left Toe", "Left Foot"),
    ("Right Thigh", "Pelvis"),
    ("Right Calf", "Right Thigh"),
    ("Right Foot", "Right Calf"),
    ("Right Toe", "Right Foot"),
)

_BONES = [[parent, joint] for joint, parent in _JOINT_PARENTS if parent is not None]
_PROCESS_LABELS = {"spray": "스프레이", "blasting": "블라스팅"}
_EXPERT_NAME = "전형기 반장"
_DIRECTION_LABELS = {
    "horizontal": "수평 중심",
    "vertical": "수직 중심",
    "diagonal": "대각선 중심",
    "curved_irregular": "곡선·불규칙",
    "unavailable": "분석 자료 없음",
}


def _number(value: Any) -> float | None:
    try:
        parsed = float(value)
    except (TypeError, ValueError):
        return None
    return parsed if math.isfinite(parsed) else None


def _normalized_row(row: dict[str, str]) -> dict[str, str]:
    return {str(key).strip().casefold(): value for key, value in row.items()}


def _vector(row: dict[str, str], prefix: str) -> list[float] | None:
    normalized = _normalized_row(row)
    values = [_number(normalized.get(f"{prefix} position {axis}".casefold())) for axis in "XYZ"]
    if any(value is None for value in values):
        return None
    return [round(float(value), 6) for value in values]


def _ray_point(row: dict[str, str]) -> list[float] | None:
    normalized = _normalized_row(row)
    values = [_number(normalized.get(f"ray point {axis}")) for axis in "xyz"]
    if any(value is None for value in values):
        return None
    return [round(float(value), 6) for value in values]


def _spraying(value: Any) -> bool:
    return str(value).strip().casefold() in {"on", "true", "1", "yes"}


def _multiply_quaternion(
    left: tuple[float, float, float, float],
    right: tuple[float, float, float, float],
) -> tuple[float, float, float, float]:
    lx, ly, lz, lw = left
    rx, ry, rz, rw = right
    return (
        lw * rx + lx * rw + ly * rz - lz * ry,
        lw * ry - lx * rz + ly * rw + lz * rx,
        lw * rz + lx * ry - ly * rx + lz * rw,
        lw * rw - lx * rx - ly * ry - lz * rz,
    )


def _axis_quaternion(axis: str, radians: float) -> tuple[float, float, float, float]:
    sine = math.sin(radians / 2.0)
    cosine = math.cos(radians / 2.0)
    if axis == "x":
        return sine, 0.0, 0.0, cosine
    if axis == "y":
        return 0.0, sine, 0.0, cosine
    return 0.0, 0.0, sine, cosine


def _unity_euler_quaternion(x: float, y: float, z: float) -> tuple[float, float, float, float]:
    """Match Unity Quaternion.Euler's Z-X-Y application order."""

    qx = _axis_quaternion("x", math.radians(x))
    qy = _axis_quaternion("y", math.radians(y))
    qz = _axis_quaternion("z", math.radians(z))
    return _multiply_quaternion(_multiply_quaternion(qy, qx), qz)


def _rotate(
    quaternion: tuple[float, float, float, float],
    vector: tuple[float, float, float],
) -> tuple[float, float, float]:
    qx, qy, qz, qw = quaternion
    vx, vy, vz = vector
    tx = 2.0 * (qy * vz - qz * vy)
    ty = 2.0 * (qz * vx - qx * vz)
    tz = 2.0 * (qx * vy - qy * vx)
    return (
        vx + qw * tx + qy * tz - qz * ty,
        vy + qw * ty + qz * tx - qx * tz,
        vz + qw * tz + qx * ty - qy * tx,
    )


def _world_joints(row: dict[str, str]) -> dict[str, list[float]]:
    positions: dict[str, tuple[float, float, float]] = {}
    rotations: dict[str, tuple[float, float, float, float]] = {}
    for joint, parent in _JOINT_PARENTS:
        position_values = tuple(_number(row.get(f"{joint} Position {axis}")) for axis in "XYZ")
        rotation_values = tuple(_number(row.get(f"{joint} Rotation {axis}")) for axis in "XYZ")
        if any(value is None for value in (*position_values, *rotation_values)):
            continue
        local_position = position_values  # type: ignore[assignment]
        local_rotation = _unity_euler_quaternion(*rotation_values)  # type: ignore[arg-type]
        if parent is None:
            world_position = local_position
            world_rotation = local_rotation
        else:
            if parent not in positions:
                continue
            rotated = _rotate(rotations[parent], local_position)
            parent_position = positions[parent]
            world_position = tuple(parent_position[index] + rotated[index] for index in range(3))
            world_rotation = _multiply_quaternion(rotations[parent], local_rotation)
        positions[joint] = world_position  # type: ignore[assignment]
        rotations[joint] = world_rotation
    return {joint: [round(value, 6) for value in position] for joint, position in positions.items()}


def _open_csv(path: Path) -> TextIO:
    if path.name.lower().endswith(".gz"):
        return gzip.open(path, "rt", encoding="utf-8-sig", newline="")
    return path.open("r", encoding="utf-8-sig", newline="")


@lru_cache(maxsize=64)
def _sample_csv(
    path_text: str, stamp: tuple[int, int], frame_count: int
) -> tuple[dict[str, Any], ...]:
    del stamp
    path = Path(path_text)
    with _open_csv(path) as handle:
        row_count = max(0, sum(1 for _ in handle) - 1)
    if row_count == 0:
        return ()
    count = min(frame_count, row_count)
    targets = {
        round(index * (row_count - 1) / max(1, count - 1))
        for index in range(count)
    }
    frames: list[dict[str, Any]] = []
    tool_prefix = "블라스트 건" if "blasting" in {part.casefold() for part in path.parts} else "한길숏건"
    with _open_csv(path) as handle:
        for index, row in enumerate(csv.DictReader(handle)):
            if index not in targets:
                continue
            joints = _world_joints(row)
            tracked = {
                "head": _vector(row, "카메라"),
                "right_hand": _vector(row, tool_prefix),
                "left_hand": _vector(row, "플래쉬"),
            }
            tracked = {key: value for key, value in tracked.items() if value is not None}
            ray_point = _ray_point(row)
            if not joints and not tracked and ray_point is None:
                continue
            raw_time = _number(row.get("시간(초)"))
            frames.append({
                "t": raw_time if raw_time is not None else float(index),
                "joints": joints,
                "tracked": tracked,
                "ray_point": ray_point,
                "spraying": _spraying(row.get("분사")),
                "zone_id": str(row.get("Zone ID", "")).strip() or None,
            })
    if frames:
        origin = float(frames[0]["t"])
        for frame in frames:
            frame["t"] = round(float(frame["t"]) - origin, 6)
    return tuple(frames)


def _read_profile(root: Path, process: str, mode_id: int) -> dict[str, Any]:
    if process not in _PROCESS_LABELS or not 1 <= mode_id <= 10:
        raise HTTPException(status_code=404, detail="motion profile not found")
    path = root / "generated" / "profiles" / process / "learning" / f"mode_{mode_id:02d}.json"
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (FileNotFoundError, OSError, json.JSONDecodeError) as error:
        raise HTTPException(status_code=404, detail="motion profile not found") from error
    if not isinstance(payload.get("sessions"), list):
        raise HTTPException(status_code=422, detail="motion profile is invalid")
    return payload


def _representative_session(profile: dict[str, Any]) -> str:
    sessions = [item for item in profile["sessions"] if isinstance(item, dict)]
    if not sessions:
        raise HTTPException(status_code=404, detail="expert session not found")
    orders = profile.get("aggregate", {}).get("macro_orders", [])
    dominant_order = orders[0].get("order") if orders and isinstance(orders[0], dict) else None
    aggregate_zones = profile.get("aggregate", {}).get("zones", {})

    def score(session: dict[str, Any]) -> tuple[float, str]:
        analysis = session.get("analysis", {})
        penalty = 0.0 if dominant_order is None or analysis.get("macro_order") == dominant_order else 1000.0
        session_zones = analysis.get("zones", {})
        for key, reference in aggregate_zones.items():
            observed = session_zones.get(key, {}) if isinstance(session_zones, dict) else {}
            for metric in ("speed_mms", "distance_mm"):
                median = _number(reference.get(metric, {}).get("median"))
                value = _number(observed.get(metric)) if isinstance(observed, dict) else None
                if median is not None and value is not None:
                    penalty += abs(value - median) / max(abs(median), 1.0)
        return penalty, str(session.get("session_id", ""))

    return str(min(sessions, key=score)["session_id"])


def _session_path(
    root: Path, process: str, mode_id: int, profile: dict[str, Any], session_id: str
) -> Path:
    session = next(
        (item for item in profile["sessions"] if str(item.get("session_id")) == session_id),
        None,
    )
    if session is None:
        raise HTTPException(status_code=404, detail="expert session not found")
    source = session.get("sources", {}).get("csv_path")
    if not isinstance(source, str) or not source:
        raise HTTPException(status_code=422, detail="expert session source is invalid")
    mode_root = (root / "raw" / process / "learning" / f"mode_{mode_id:02d}").resolve()
    candidate = (mode_root / source).resolve()
    if not candidate.is_relative_to(mode_root) or not candidate.is_file():
        raise HTTPException(status_code=404, detail="expert session file not found")
    return candidate


def _frames(path: Path, frame_count: int) -> list[dict[str, Any]]:
    stat = path.stat()
    return [dict(frame) for frame in _sample_csv(str(path), (stat.st_mtime_ns, stat.st_size), frame_count)]


def _spread(
    selected_frames: list[dict[str, Any]], all_frames: list[list[dict[str, Any]]]
) -> list[dict[str, Any]]:
    output: list[dict[str, Any]] = []
    for frame_index in range(len(selected_frames)):
        progress = frame_index / max(1, len(selected_frames) - 1)
        aligned = [frames[round(progress * (len(frames) - 1))] for frames in all_frames if frames]
        joint_output: dict[str, Any] = {}
        for joint, _ in _JOINT_PARENTS:
            points = [frame["joints"].get(joint) for frame in aligned]
            points = [point for point in points if point is not None]
            if not points:
                continue
            mean = [statistics.fmean(point[axis] for point in points) for axis in range(3)]
            distances = sorted(math.dist(point, mean) for point in points)
            percentile_index = round(0.75 * (len(distances) - 1))
            joint_output[joint] = {
                "mean": [round(value, 6) for value in mean],
                "spread": round(distances[percentile_index], 6),
            }
        tracked_output: dict[str, Any] = {}
        for marker in ("head", "right_hand", "left_hand"):
            points = [frame.get("tracked", {}).get(marker) for frame in aligned]
            points = [point for point in points if point is not None]
            if not points:
                continue
            mean = [statistics.fmean(point[axis] for point in points) for axis in range(3)]
            distances = sorted(math.dist(point, mean) for point in points)
            tracked_output[marker] = {
                "mean": [round(value, 6) for value in mean],
                "spread": round(distances[round(0.75 * (len(distances) - 1))], 6),
            }
        ray_points = [frame.get("ray_point") for frame in aligned if frame.get("spraying")]
        ray_points = [point for point in ray_points if point is not None]
        ray_mean = None
        ray_spread = None
        if ray_points:
            ray_mean = [statistics.fmean(point[axis] for point in ray_points) for axis in range(3)]
            ray_distances = sorted(math.dist(point, ray_mean) for point in ray_points)
            ray_mean = [round(value, 6) for value in ray_mean]
            ray_spread = round(ray_distances[round(0.75 * (len(ray_distances) - 1))], 6)
        output.append({
            "mean": {joint: value["mean"] for joint, value in joint_output.items()},
            "spread": {joint: value["spread"] for joint, value in joint_output.items()},
            "tracked_mean": {marker: value["mean"] for marker, value in tracked_output.items()},
            "tracked_spread": {marker: value["spread"] for marker, value in tracked_output.items()},
            "ray_mean": ray_mean,
            "ray_spread": ray_spread,
        })
    return output


def _paint_trace(frames: list[dict[str, Any]]) -> list[dict[str, Any]]:
    points = [
        (frame_index, frame)
        for frame_index, frame in enumerate(frames)
        if frame.get("spraying") and frame.get("ray_point") is not None
    ]
    return [
        {
            "frame_index": frame_index,
            "t": frame["t"],
            "point": frame["ray_point"],
            "zone_id": frame.get("zone_id"),
            "progress": round(index / max(1, len(points) - 1), 6),
        }
        for index, (frame_index, frame) in enumerate(points)
    ]


def _direction_summary(trace: list[dict[str, Any]]) -> dict[str, Any]:
    points = [item["point"] for item in trace]
    if len(points) < 2:
        return {"classification": "unavailable", "label": _DIRECTION_LABELS["unavailable"], "arrow_count": 0}
    deltas = [[b[axis] - a[axis] for axis in range(3)] for a, b in zip(points, points[1:])]
    movement = [sum(abs(delta[axis]) for delta in deltas) for axis in range(3)]
    total = sum(movement)
    path_length = sum(math.dist(a, b) for a, b in zip(points, points[1:]))
    vertical_ratio = movement[1] / total if total else 0.0
    if total == 0:
        classification = "unavailable"
    elif vertical_ratio >= 0.7:
        classification = "vertical"
    elif vertical_ratio <= 0.3:
        classification = "horizontal"
    else:
        classification = "diagonal"
    vector = [points[-1][axis] - points[0][axis] for axis in range(3)]
    magnitude = math.sqrt(sum(value * value for value in vector))
    normalized = [round(value / magnitude, 6) for value in vector] if magnitude else [0.0, 0.0, 0.0]
    return {
        "classification": classification,
        "label": _DIRECTION_LABELS[classification],
        "arrow_count": min(12, max(1, len(points) // 12)),
        "direction_vector": normalized,
        "path_length_m": round(path_length, 3),
    }


_VIEWER_HTML = r"""<!doctype html>
<html lang="ko"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<title>전문가 모션 뷰어</title><style>
:root{--bg:#07101f;--panel:#101b2e;--panel2:#16243a;--ink:#eef5ff;--muted:#91a0b8;--line:#293b56;--cyan:#22d3ee;--green:#34d399}
*{box-sizing:border-box}body{margin:0;background:var(--bg);color:var(--ink);font-family:Inter,Pretendard,"Noto Sans KR",system-ui,sans-serif}
.shell{min-height:100vh;display:grid;grid-template-rows:auto 1fr}.top{display:flex;justify-content:space-between;align-items:center;padding:16px 22px;border-bottom:1px solid var(--line);background:#0a1424}
.title{font-weight:800;font-size:18px}.title small{display:block;color:var(--muted);font-size:11px;font-weight:500;margin-top:4px}.top a{color:#dce8ff;text-decoration:none;font-size:13px;margin-left:14px}
.workspace{display:grid;grid-template-columns:275px minmax(420px,1fr) 280px;min-height:0}.side{padding:19px;border-right:1px solid var(--line);background:#0c1728;overflow:auto}.right{border-right:0;border-left:1px solid var(--line)}
label{display:block;color:var(--muted);font-size:12px;margin:16px 0 7px}.select,button,input[type=range]{width:100%}.select,button{border:1px solid var(--line);border-radius:8px;background:var(--panel2);color:var(--ink);padding:10px 11px}
button{cursor:pointer;font-weight:700}.primary{background:#3157d5;border-color:#3157d5}.controls{display:grid;grid-template-columns:1fr 1fr;gap:8px;margin-top:16px}.stage{position:relative;min-height:590px;background:radial-gradient(circle at 50% 42%,#132845 0,#07101f 65%)}canvas{width:100%;height:100%;display:block;min-height:590px}
.overlay{position:absolute;left:18px;top:16px;background:rgba(7,16,31,.72);border:1px solid rgba(89,116,150,.45);border-radius:10px;padding:10px 12px;font-size:12px;color:var(--muted);pointer-events:none}.overlay strong{display:block;color:var(--ink);font-size:14px;margin-bottom:4px}
.card{background:var(--panel);border:1px solid var(--line);border-radius:11px;padding:14px;margin-bottom:11px}.card span{display:block;color:var(--muted);font-size:11px}.card strong{display:block;font-size:20px;margin-top:5px}.legend{font-size:12px;color:var(--muted);line-height:2}.dot{display:inline-block;width:9px;height:9px;border-radius:50%;margin-right:7px}.help{font-size:12px;color:var(--muted);line-height:1.65;margin-top:18px}.status{min-height:18px;color:var(--cyan);font-size:12px;margin-top:12px}
.toggle{display:flex;align-items:center;gap:8px;margin-top:14px;color:var(--muted);font-size:12px}.toggle input{width:auto}.badge{display:inline-flex;padding:5px 9px;border:1px solid #3157d5;border-radius:999px;color:#b9caff;font-size:11px;margin-top:12px}
@media(max-width:1000px){.workspace{grid-template-columns:240px 1fr}.right{display:none}}@media(max-width:720px){.workspace{display:block}.side{border:0}.stage,canvas{min-height:480px}}
</style></head><body><div class="shell">
<header class="top"><div class="title">전문가 모션 뷰어<small>기준 전문가: 전형기 반장 · 반복 수행 기반</small></div><nav><a href="/docs">Swagger</a><a href="/admin">DB 뷰어</a></nav></header>
<main class="workspace"><aside class="side">
<label for="process">훈련 종류</label><select id="process" class="select"></select>
<label for="mode">학습모드</label><select id="mode" class="select"></select>
<label for="session">재생할 기록</label><select id="session" class="select"></select>
<div class="controls"><button id="play" class="primary">재생</button><button id="restart">처음으로</button></div>
<label for="timeline">진행 위치</label><input id="timeline" type="range" min="0" max="0" value="0">
<label for="speed">재생 속도</label><select id="speed" class="select"><option value="0.25">0.25배</option><option value="0.5">0.5배</option><option value="1" selected>1배</option><option value="1.5">1.5배</option><option value="2">2배</option></select>
<label class="toggle"><input id="skeleton" type="checkbox">참고용 스켈레톤 표시</label>
<div class="badge">관절 좌표는 참고용 · 평가 근거 아님</div>
<div id="status" class="status">데이터를 불러오는 중입니다.</div>
<p class="help">마우스로 드래그하면 시점을 돌리고, 휠로 확대·축소할 수 있습니다. 페인트 누적 궤적과 머리·양손의 실측 위치가 기본 표시됩니다.</p>
</aside>
<section class="stage"><canvas id="canvas"></canvas><div class="overlay"><strong id="overlayTitle">학습모드</strong><span id="overlayFrame">프레임 준비 중</span></div></section>
<aside class="side right"><div class="card"><span>통계에 포함된 수행</span><strong id="sessionCount">-</strong></div><div class="card"><span>현재 기록 길이</span><strong id="duration">-</strong></div><div class="card"><span>주요 도장 방향</span><strong id="direction">-</strong></div><div class="card"><span>현재 양손 위치 편차</span><strong id="handSpread">-</strong></div>
<div class="legend"><div><i class="dot" style="background:#ff8a3d"></i>선택 기록의 페인트 누적 궤적</div><div><i class="dot" style="background:var(--green)"></i>반복 수행 75% 통계 범위</div><div><i class="dot" style="background:var(--cyan)"></i>전형기 반장의 머리·양손 실측 위치</div></div>
<p class="help">밝은 중심 경로와 반투명 범위는 같은 학습모드의 반복 수행을 진행률 0~100%로 맞춘 통계입니다. 페인트 표현은 실제 도막 두께가 아니라 Ray Point가 기록한 착탄 위치와 진행 방향입니다.</p></aside>
</main></div><script>
const $=id=>document.getElementById(id); const canvas=$('canvas'),ctx=canvas.getContext('2d');
let catalog=null,data=null,index=0,playheadSeconds=0,playing=false,last=0,yaw=-0.55,pitch=0.1,zoom=175,drag=null;
async function json(url){const r=await fetch(url);if(!r.ok)throw new Error((await r.json()).detail||'데이터를 읽을 수 없습니다.');return r.json()}
function option(value,text){const o=document.createElement('option');o.value=value;o.textContent=text;return o}
function currentProcess(){return catalog.processes.find(p=>p.process===$('process').value)}
function currentMode(){return currentProcess().modes.find(m=>String(m.mode_id)===$('mode').value)}
function fillModes(){const p=currentProcess();$('mode').replaceChildren(...p.modes.map(m=>option(m.mode_id,`학습모드 ${m.mode_id} · ${m.session_count}회`)));const four=p.modes.find(m=>m.mode_id===4);if(four)$('mode').value='4';fillSessions()}
function fillSessions(){const m=currentMode();$('session').replaceChildren(option('representative',`전형기 대표 실측 모션 · ${m.representative_session_id}`),...m.sessions.map(s=>option(s,`전형기 · ${s}`)));loadMotion()}
async function loadMotion(){playing=false;last=0;$('play').textContent='재생';$('status').textContent='모션과 통계 범위를 계산하는 중입니다.';try{const q=new URLSearchParams({process:$('process').value,mode_id:$('mode').value,session_id:$('session').value});data=await json('/motion-viewer/api/motion?'+q);index=0;playheadSeconds=0;$('timeline').max=Math.max(0,data.frames.length-1);$('timeline').value=0;$('sessionCount').textContent=`${data.sampled_session_count}회`;$('duration').textContent=`${data.duration_seconds.toFixed(1)}초`;$('direction').textContent=data.direction_summary.label;$('overlayTitle').textContent=`전형기 반장 · ${data.process_label} · 학습모드 ${data.mode_id}`;$('status').textContent=`${data.selected_session_id} 준비 완료`;draw()}catch(e){data=null;$('status').textContent=e.message;draw()}}
function resize(){const dpr=devicePixelRatio||1,rect=canvas.getBoundingClientRect();canvas.width=Math.round(rect.width*dpr);canvas.height=Math.round(rect.height*dpr);ctx.setTransform(dpr,0,0,dpr,0,0);draw()}
function transform(point,center){let x=point[0]-center[0],y=point[1]-center[1],z=point[2]-center[2];let cy=Math.cos(yaw),sy=Math.sin(yaw),x1=x*cy-z*sy,z1=x*sy+z*cy;let cp=Math.cos(pitch),sp=Math.sin(pitch),y1=y*cp-z1*sp,z2=y*sp+z1*cp;return [x1,y1,z2]}
function project(point,center,w,h){const [x,y,z]=transform(point,center),pers=4.5/(4.5-z);return [w/2+x*zoom*pers,h*.59-y*zoom*pers,pers]}
function line(a,b,color,width,center,w,h){const p=project(a,center,w,h),q=project(b,center,w,h);ctx.strokeStyle=color;ctx.lineWidth=width;ctx.lineCap='round';ctx.beginPath();ctx.moveTo(p[0],p[1]);ctx.lineTo(q[0],q[1]);ctx.stroke()}
function point(p,r,color,center,w,h){const q=project(p,center,w,h);ctx.fillStyle=color;ctx.beginPath();ctx.arc(q[0],q[1],r*q[2],0,Math.PI*2);ctx.fill()}
function arrow(a,b,color,center,w,h){const p=project(a,center,w,h),q=project(b,center,w,h),angle=Math.atan2(q[1]-p[1],q[0]-p[0]);ctx.strokeStyle=color;ctx.fillStyle=color;ctx.lineWidth=2;ctx.beginPath();ctx.moveTo(p[0],p[1]);ctx.lineTo(q[0],q[1]);ctx.stroke();ctx.beginPath();ctx.moveTo(q[0],q[1]);ctx.lineTo(q[0]-10*Math.cos(angle-.48),q[1]-10*Math.sin(angle-.48));ctx.lineTo(q[0]-10*Math.cos(angle+.48),q[1]-10*Math.sin(angle+.48));ctx.closePath();ctx.fill()}
function statisticalBand(samples,center,w,h){let previous=null,previousSpread=0;for(const sample of samples){if(!sample.ray_mean){previous=null;previousSpread=0;continue}if(previous){const width=Math.max(8,Math.min(36,((previousSpread+(sample.ray_spread||0))/2)*zoom));line(previous,sample.ray_mean,'rgba(52,211,153,.12)',width,center,w,h);line(previous,sample.ray_mean,'rgba(52,211,153,.48)',2,center,w,h)}previous=sample.ray_mean;previousSpread=sample.ray_spread||0}}
function mixPoint(a,b,ratio){if(!a)return b||null;if(!b)return a;return a.map((value,axis)=>value+(b[axis]-value)*ratio)}
function interpolatedFrame(seconds){let current=Math.max(0,Math.min(index,data.frames.length-1));while(current+1<data.frames.length&&data.frames[current+1].t<=seconds)current++;while(current>0&&data.frames[current].t>seconds)current--;const a=data.frames[current],b=data.frames[Math.min(current+1,data.frames.length-1)],span=Math.max(.000001,b.t-a.t),ratio=Math.max(0,Math.min(1,(seconds-a.t)/span));const tracked={},joints={};for(const key of new Set([...Object.keys(a.tracked||{}),...Object.keys(b.tracked||{})]))tracked[key]=mixPoint((a.tracked||{})[key],(b.tracked||{})[key],ratio);for(const key of new Set([...Object.keys(a.joints||{}),...Object.keys(b.joints||{})]))joints[key]=mixPoint((a.joints||{})[key],(b.joints||{})[key],ratio);const continuousPaint=a.spraying&&b.spraying&&a.ray_point&&b.ray_point&&a.zone_id===b.zone_id;return{index:current,frame:{...a,t:seconds,tracked,joints,ray_point:continuousPaint?mixPoint(a.ray_point,b.ray_point,ratio):a.ray_point}}}
function draw(){const rect=canvas.getBoundingClientRect(),w=rect.width,h=rect.height;ctx.clearRect(0,0,w,h);ctx.strokeStyle='rgba(100,130,165,.16)';ctx.lineWidth=1;for(let i=-5;i<=5;i++){ctx.beginPath();ctx.moveTo(w/2+i*55,h*.78);ctx.lineTo(w/2+i*90,h*.95);ctx.stroke()}
if(!data||!data.frames.length)return;const rendered=interpolatedFrame(playheadSeconds),frame=rendered.frame,sourceIndex=rendered.index,stats=data.statistics[sourceIndex];const allPaint=data.frames.map(f=>f.ray_point).filter(Boolean),center=allPaint.length?allPaint[Math.floor(allPaint.length/2)]:((frame.tracked||{}).head||(frame.joints||{}).Pelvis||[0,0,0]);
statisticalBand(data.statistics,center,w,h);
let previous=null,paintCount=0;for(let i=0;i<=sourceIndex;i++){const f=data.frames[i];if(!f.spraying||!f.ray_point){previous=null;continue}const progress=i/Math.max(1,data.frames.length-1),color=`hsla(${24+progress*28},100%,62%,${.35+progress*.5})`;point(f.ray_point,3.5,color,center,w,h);if(previous){line(previous,f.ray_point,color,7,center,w,h);if(paintCount%12===0)arrow(previous,f.ray_point,'rgba(255,232,190,.9)',center,w,h)}previous=f.ray_point;paintCount++}if(previous&&frame.spraying&&frame.ray_point){line(previous,frame.ray_point,'rgba(255,184,92,.78)',7,center,w,h);point(frame.ray_point,3.5,'rgba(255,205,125,.9)',center,w,h)}
const tracked=frame.tracked||{},trackedStats=stats.tracked_spread||{};for(const marker of ['head','right_hand','left_hand']){const p=tracked[marker],radius=trackedStats[marker]||0;if(p&&radius>0){const q=project(p,center,w,h);ctx.fillStyle='rgba(52,211,153,.12)';ctx.beginPath();ctx.arc(q[0],q[1],Math.max(3,radius*zoom*q[2]),0,Math.PI*2);ctx.fill()}}
if(tracked.head&&tracked.left_hand)line(tracked.head,tracked.left_hand,'rgba(34,211,238,.65)',3,center,w,h);if(tracked.head&&tracked.right_hand)line(tracked.head,tracked.right_hand,'rgba(34,211,238,.65)',3,center,w,h);if(tracked.right_hand&&frame.ray_point&&frame.spraying)line(tracked.right_hand,frame.ray_point,'rgba(255,244,200,.45)',2,center,w,h);
if(tracked.head)point(tracked.head,13,'#f5f8ff',center,w,h);if(tracked.right_hand)point(tracked.right_hand,7,'#22d3ee',center,w,h);if(tracked.left_hand)point(tracked.left_hand,6,'#8be9fd',center,w,h);
if($('skeleton').checked){for(const [a,b] of data.bones){if(frame.joints[a]&&frame.joints[b])line(frame.joints[a],frame.joints[b],'rgba(245,248,255,.38)',3,center,w,h)}for(const p of Object.values(frame.joints))point(p,3,'rgba(245,248,255,.55)',center,w,h)}
$('overlayFrame').textContent=`${sourceIndex+1} / ${data.frames.length} · ${frame.t.toFixed(2)}초`;
const spreads=['left_hand','right_hand'].map(j=>trackedStats[j]).filter(v=>v!==undefined);$('handSpread').textContent=spreads.length?`${(spreads.reduce((a,b)=>a+b,0)/spreads.length*100).toFixed(1)} cm`:'자료 없음'}
function tick(now){if(playing&&data&&data.frames.length){if(!last)last=now;const elapsed=Math.min(.1,(now-last)/1000)*Number($('speed').value);last=now;playheadSeconds+=elapsed;if(playheadSeconds>=data.duration_seconds){playheadSeconds=0;index=0}else index=interpolatedFrame(playheadSeconds).index;$('timeline').value=index;draw()}requestAnimationFrame(tick)}
$('process').onchange=fillModes;$('mode').onchange=fillSessions;$('session').onchange=loadMotion;$('timeline').oninput=e=>{index=Number(e.target.value);playheadSeconds=data.frames[index].t;last=0;draw()};$('skeleton').onchange=draw;$('play').onclick=()=>{playing=!playing;last=0;$('play').textContent=playing?'일시정지':'재생'};$('restart').onclick=()=>{index=0;playheadSeconds=0;last=0;$('timeline').value=0;draw()};
canvas.onpointerdown=e=>drag=[e.clientX,e.clientY,yaw,pitch];canvas.onpointermove=e=>{if(!drag)return;yaw=drag[2]+(e.clientX-drag[0])*.01;pitch=Math.max(-1.2,Math.min(1.2,drag[3]+(e.clientY-drag[1])*.01));draw()};canvas.onpointerup=()=>drag=null;canvas.onpointerleave=()=>drag=null;canvas.onwheel=e=>{e.preventDefault();zoom=Math.max(70,Math.min(420,zoom*(e.deltaY>0?.9:1.1)));draw()};
addEventListener('resize',resize);(async()=>{try{catalog=await json('/motion-viewer/api/catalog');$('process').replaceChildren(...catalog.processes.map(p=>option(p.process,p.label)));fillModes();resize();requestAnimationFrame(tick)}catch(e){$('status').textContent=e.message}})();
</script></body></html>"""


def create_motion_viewer_router(
    reference_root: Path, *, frame_count: int = 240
) -> APIRouter:
    root = Path(reference_root)
    router = APIRouter()

    @router.get("/motion-viewer", include_in_schema=False, response_class=HTMLResponse)
    def motion_viewer() -> HTMLResponse:
        return HTMLResponse(_VIEWER_HTML)

    @router.get("/motion-viewer/api/catalog", include_in_schema=False)
    def motion_catalog() -> dict[str, Any]:
        processes: list[dict[str, Any]] = []
        for process, label in _PROCESS_LABELS.items():
            modes: list[dict[str, Any]] = []
            for mode_id in range(1, 11):
                try:
                    profile = _read_profile(root, process, mode_id)
                    representative = _representative_session(profile)
                except HTTPException:
                    continue
                sessions = sorted(str(item.get("session_id")) for item in profile["sessions"] if item.get("session_id"))
                modes.append({
                    "mode_id": mode_id,
                    "session_count": len(sessions),
                    "representative_session_id": representative,
                    "sessions": sessions,
                })
            if modes:
                processes.append({"process": process, "label": label, "modes": modes})
        return {"processes": processes}

    @router.get("/motion-viewer/api/motion", include_in_schema=False)
    def motion_data(
        process: str = Query(pattern="^(spray|blasting)$"),
        mode_id: int = Query(ge=1, le=10),
        session_id: str = Query(min_length=1, max_length=100),
    ) -> dict[str, Any]:
        profile = _read_profile(root, process, mode_id)
        representative = _representative_session(profile)
        selected = representative if session_id == "representative" else session_id
        selected_frames = _frames(_session_path(root, process, mode_id, profile, selected), frame_count)
        if not selected_frames:
            raise HTTPException(status_code=422, detail="expert session has no usable skeleton frames")
        sampled_sessions: list[list[dict[str, Any]]] = []
        for session in profile["sessions"]:
            current_id = str(session.get("session_id", ""))
            if not current_id:
                continue
            try:
                sampled = _frames(_session_path(root, process, mode_id, profile, current_id), frame_count)
            except HTTPException:
                continue
            if sampled:
                sampled_sessions.append(sampled)
        duration = float(selected_frames[-1]["t"]) if selected_frames else 0.0
        paint_trace = _paint_trace(selected_frames)
        return {
            "expert_name": _EXPERT_NAME,
            "process": process,
            "process_label": _PROCESS_LABELS[process],
            "mode_id": mode_id,
            "selected_session_id": selected,
            "representative_session_id": representative,
            "sampled_session_count": len(sampled_sessions),
            "duration_seconds": round(duration, 3),
            "bones": _BONES,
            "frames": selected_frames,
            "statistics": _spread(selected_frames, sampled_sessions),
            "paint_trace": paint_trace,
            "direction_summary": _direction_summary(paint_trace),
        }

    return router
