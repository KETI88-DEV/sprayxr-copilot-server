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


def _number(value: Any) -> float | None:
    try:
        parsed = float(value)
    except (TypeError, ValueError):
        return None
    return parsed if math.isfinite(parsed) else None


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
    with _open_csv(path) as handle:
        for index, row in enumerate(csv.DictReader(handle)):
            if index not in targets:
                continue
            joints = _world_joints(row)
            if not joints:
                continue
            raw_time = _number(row.get("시간(초)"))
            frames.append({"t": raw_time if raw_time is not None else float(index), "joints": joints})
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
        output.append({
            "mean": {joint: value["mean"] for joint, value in joint_output.items()},
            "spread": {joint: value["spread"] for joint, value in joint_output.items()},
        })
    return output


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
@media(max-width:1000px){.workspace{grid-template-columns:240px 1fr}.right{display:none}}@media(max-width:720px){.workspace{display:block}.side{border:0}.stage,canvas{min-height:480px}}
</style></head><body><div class="shell">
<header class="top"><div class="title">전문가 모션 뷰어<small>전형기 반장님 반복 수행 기반</small></div><nav><a href="/docs">Swagger</a><a href="/admin">DB 뷰어</a></nav></header>
<main class="workspace"><aside class="side">
<label for="process">훈련 종류</label><select id="process" class="select"></select>
<label for="mode">학습모드</label><select id="mode" class="select"></select>
<label for="session">재생할 기록</label><select id="session" class="select"></select>
<div class="controls"><button id="play" class="primary">재생</button><button id="restart">처음으로</button></div>
<label for="timeline">진행 위치</label><input id="timeline" type="range" min="0" max="0" value="0">
<label for="speed">재생 속도</label><select id="speed" class="select"><option value="0.25">0.25배</option><option value="0.5">0.5배</option><option value="1" selected>1배</option><option value="1.5">1.5배</option><option value="2">2배</option></select>
<div id="status" class="status">데이터를 불러오는 중입니다.</div>
<p class="help">마우스로 드래그하면 시점을 돌리고, 휠로 확대·축소할 수 있습니다. 밝은 뼈대는 실제 대표 수행이며 초록 원은 같은 시점에서 반복 수행이 퍼진 범위입니다.</p>
</aside>
<section class="stage"><canvas id="canvas"></canvas><div class="overlay"><strong id="overlayTitle">학습모드</strong><span id="overlayFrame">프레임 준비 중</span></div></section>
<aside class="side right"><div class="card"><span>통계에 포함된 수행</span><strong id="sessionCount">-</strong></div><div class="card"><span>현재 기록 길이</span><strong id="duration">-</strong></div><div class="card"><span>현재 손 위치 편차</span><strong id="handSpread">-</strong></div>
<div class="legend"><div><i class="dot" style="background:#f5f8ff"></i>대표 또는 선택한 실측 뼈대</div><div><i class="dot" style="background:var(--green)"></i>반복 수행의 75% 관절 편차</div><div><i class="dot" style="background:var(--cyan)"></i>오른손 평균 이동 경로</div></div>
<p class="help">통계 범위는 길이가 다른 수행을 시작 0%부터 종료 100%까지 정규화해 같은 진행 시점끼리 비교합니다. 평균 관절로 가상 사람을 만들지 않고 실제 CSV 모션을 중심으로 표시합니다.</p></aside>
</main></div><script>
const $=id=>document.getElementById(id); const canvas=$('canvas'),ctx=canvas.getContext('2d');
let catalog=null,data=null,index=0,playing=false,last=0,yaw=-0.55,pitch=0.1,zoom=175,drag=null;
async function json(url){const r=await fetch(url);if(!r.ok)throw new Error((await r.json()).detail||'데이터를 읽을 수 없습니다.');return r.json()}
function option(value,text){const o=document.createElement('option');o.value=value;o.textContent=text;return o}
function currentProcess(){return catalog.processes.find(p=>p.process===$('process').value)}
function currentMode(){return currentProcess().modes.find(m=>String(m.mode_id)===$('mode').value)}
function fillModes(){const p=currentProcess();$('mode').replaceChildren(...p.modes.map(m=>option(m.mode_id,`학습모드 ${m.mode_id} · ${m.session_count}회`)));const four=p.modes.find(m=>m.mode_id===4);if(four)$('mode').value='4';fillSessions()}
function fillSessions(){const m=currentMode();$('session').replaceChildren(option('representative',`대표 실측 모션 · ${m.representative_session_id}`),...m.sessions.map(s=>option(s,s)));loadMotion()}
async function loadMotion(){playing=false;$('play').textContent='재생';$('status').textContent='모션과 통계 범위를 계산하는 중입니다.';try{const q=new URLSearchParams({process:$('process').value,mode_id:$('mode').value,session_id:$('session').value});data=await json('/motion-viewer/api/motion?'+q);index=0;$('timeline').max=Math.max(0,data.frames.length-1);$('timeline').value=0;$('sessionCount').textContent=`${data.sampled_session_count}회`;$('duration').textContent=`${data.duration_seconds.toFixed(1)}초`;$('overlayTitle').textContent=`${data.process_label} · 학습모드 ${data.mode_id}`;$('status').textContent=`${data.selected_session_id} 준비 완료`;draw()}catch(e){data=null;$('status').textContent=e.message;draw()}}
function resize(){const dpr=devicePixelRatio||1,rect=canvas.getBoundingClientRect();canvas.width=Math.round(rect.width*dpr);canvas.height=Math.round(rect.height*dpr);ctx.setTransform(dpr,0,0,dpr,0,0);draw()}
function transform(point,center){let x=point[0]-center[0],y=point[1]-center[1],z=point[2]-center[2];let cy=Math.cos(yaw),sy=Math.sin(yaw),x1=x*cy-z*sy,z1=x*sy+z*cy;let cp=Math.cos(pitch),sp=Math.sin(pitch),y1=y*cp-z1*sp,z2=y*sp+z1*cp;return [x1,y1,z2]}
function project(point,center,w,h){const [x,y,z]=transform(point,center),pers=4.5/(4.5-z);return [w/2+x*zoom*pers,h*.59-y*zoom*pers,pers]}
function line(a,b,color,width,center,w,h){const p=project(a,center,w,h),q=project(b,center,w,h);ctx.strokeStyle=color;ctx.lineWidth=width;ctx.lineCap='round';ctx.beginPath();ctx.moveTo(p[0],p[1]);ctx.lineTo(q[0],q[1]);ctx.stroke()}
function draw(){const rect=canvas.getBoundingClientRect(),w=rect.width,h=rect.height;ctx.clearRect(0,0,w,h);ctx.strokeStyle='rgba(100,130,165,.16)';ctx.lineWidth=1;for(let i=-5;i<=5;i++){ctx.beginPath();ctx.moveTo(w/2+i*55,h*.78);ctx.lineTo(w/2+i*90,h*.95);ctx.stroke()}
if(!data||!data.frames.length)return;const frame=data.frames[index],stats=data.statistics[index],center=frame.joints.Pelvis||[0,0,0];
const meanPath=data.statistics.map(s=>s.mean['Right Hand']).filter(Boolean);for(let i=1;i<meanPath.length;i++)line(meanPath[i-1],meanPath[i],'rgba(34,211,238,.38)',3,center,w,h);
for(const [joint,p] of Object.entries(frame.joints)){const radius=stats.spread[joint]||0;if(radius>0){const q=project(p,center,w,h);ctx.fillStyle='rgba(52,211,153,.13)';ctx.strokeStyle='rgba(52,211,153,.35)';ctx.lineWidth=1;ctx.beginPath();ctx.arc(q[0],q[1],Math.max(3,radius*zoom*q[2]),0,Math.PI*2);ctx.fill();ctx.stroke()}}
for(const [a,b] of data.bones){if(frame.joints[a]&&frame.joints[b])line(frame.joints[a],frame.joints[b],'#f5f8ff',5,center,w,h)}
for(const [joint,p] of Object.entries(frame.joints)){const q=project(p,center,w,h);ctx.fillStyle=joint.includes('Hand')?'#22d3ee':'#f5f8ff';ctx.beginPath();ctx.arc(q[0],q[1],joint==='Head'?7:4,0,Math.PI*2);ctx.fill()}
$('overlayFrame').textContent=`${index+1} / ${data.frames.length} · ${frame.t.toFixed(2)}초`;
const spreads=['Left Hand','Right Hand'].map(j=>stats.spread[j]).filter(v=>v!==undefined);$('handSpread').textContent=spreads.length?`${(spreads.reduce((a,b)=>a+b,0)/spreads.length*100).toFixed(1)} cm`:'자료 없음'}
}
function tick(now){if(playing&&data){if(!last)last=now;const fps=(data.frames.length-1)/Math.max(data.duration_seconds,1);if(now-last>1000/(fps*Number($('speed').value))){index=(index+1)%data.frames.length;$('timeline').value=index;last=now;draw()}}requestAnimationFrame(tick)}
$('process').onchange=fillModes;$('mode').onchange=fillSessions;$('session').onchange=loadMotion;$('timeline').oninput=e=>{index=Number(e.target.value);draw()};$('play').onclick=()=>{playing=!playing;last=0;$('play').textContent=playing?'일시정지':'재생'};$('restart').onclick=()=>{index=0;$('timeline').value=0;draw()};
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
        return {
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
        }

    return router
