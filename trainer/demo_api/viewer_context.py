"""Authenticate analysis snapshots returned by this process to the motion viewer."""
import hashlib
import hmac
import json
import secrets
from typing import Any

_KEY = secrets.token_bytes(32)


def _normalized(value: Any) -> Any:
    if isinstance(value, dict):
        return {key: _normalized(item) for key, item in value.items()}
    if isinstance(value, list):
        return [_normalized(item) for item in value]
    if isinstance(value, (int, float)) and not isinstance(value, bool):
        # JSON.parse/stringify does not preserve 1.0 versus 1 representation, and turns -0.0 into 0.
        return format(value + 0, ".12g")
    return value


def sign_context(task_type: str, context: dict[str, Any]) -> str:
    claims = {key: value for key, value in context.items() if key != "record"}
    raw = json.dumps(_normalized([task_type, claims]), sort_keys=True, ensure_ascii=False, separators=(",", ":")).encode()
    return hmac.new(_KEY, raw, hashlib.sha256).hexdigest()


def verify_context(task_type: str, context: dict[str, Any], signature: object) -> bool:
    return isinstance(signature, str) and len(signature) == 64 and hmac.compare_digest(sign_context(task_type, context), signature)


def viewer_signatures(payload: dict[str, Any]) -> dict[str, str]:
    pose = payload.get("pose_evaluation") or {}
    order = payload.get("reference_order_evaluation") or {}
    pose_context = {key: pose.get(key) for key in ("status", "similarity_percent", "zones", "corrections")}
    order_context = {key: order.get(key) for key in (
        "recommended_flow", "selected_flow", "adherence_rate", "best_reference_similarity",
        "position_similarity", "position_distribution", "reference_session_count",
    )}
    order_context["direction_summary"] = payload.get("direction_summary")
    order_context["zone_keys"] = sorted((pose.get("zones") or {}).keys())
    return {"pose_recomm": sign_context("pose_recomm", pose_context), "trajectory_recomm": sign_context("trajectory_recomm", order_context),
            "dialogue": sign_context("dialogue", payload.get("dialogue_evaluation") or {}),
            "coating_status": sign_context("coating_status", payload.get("coating_evaluation") or {})}
