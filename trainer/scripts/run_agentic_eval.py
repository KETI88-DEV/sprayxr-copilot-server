#!/usr/bin/env python3

from __future__ import annotations

import argparse
import csv
import json
import logging
import re
import statistics
from collections import Counter
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Tuple

logger = logging.getLogger(__name__)


RUBRIC_DIMENSIONS = [
    "tool_selection_accuracy",
    "parameter_correctness",
    "reasoning_depth",
    "context_integration",
    "response_structure",
]


@dataclass
class EvaluationResult:
    q_id: str
    category: str
    difficulty: str
    tool_selection_accuracy: int
    parameter_correctness: int
    reasoning_depth: int
    context_integration: int
    response_structure: int
    total: int
    expected_tools: str
    actual_tools: str
    note: str


class AgenticEvaluator:
    def __init__(self, eval_set_path: Path, rubric_path: Path, tool_schemas_dir: Path):
        self.eval_set = self.load_jsonl(eval_set_path)
        self.rubric = self.load_rubric(rubric_path)
        self.tool_names, self.tool_schemas = self.load_tool_schemas(tool_schemas_dir)

    @staticmethod
    def load_jsonl(path: Path) -> List[Dict[str, Any]]:
        records: List[Dict[str, Any]] = []
        with path.open("r", encoding="utf-8") as fh:
            for line_no, line in enumerate(fh, start=1):
                payload = line.strip()
                if not payload:
                    continue
                try:
                    records.append(json.loads(payload))
                except json.JSONDecodeError:
                    logger.warning("Skipping invalid JSONL line %d: %s", line_no, path)
        logger.info("Loaded %d items from %s", len(records), path)
        return records

    @staticmethod
    def load_rubric(path: Path) -> Dict[str, str]:
        text = path.read_text(encoding="utf-8")
        sections = re.findall(r"###\s+\d+\.\s+([a-z_]+)", text)
        return {"path": str(path), "dimensions": ",".join(sections), "raw": text}

    @staticmethod
    def load_tool_schemas(tool_schemas_dir: Path) -> Tuple[List[str], Dict[str, Dict[str, Any]]]:
        schemas: Dict[str, Dict[str, Any]] = {}
        for schema_file in sorted(tool_schemas_dir.glob("*.json")):
            try:
                payload = json.loads(schema_file.read_text(encoding="utf-8"))
            except json.JSONDecodeError:
                logger.warning("Invalid schema file: %s", schema_file)
                continue
            fn = payload.get("function", {})
            name = fn.get("name")
            if not name:
                continue
            schemas[name] = payload

        names = sorted(schemas.keys())
        logger.info("Loaded %d tool schemas from %s", len(names), tool_schemas_dir)
        return names, schemas

    @staticmethod
    def _normalize_actual_tool_calls(response: Dict[str, Any]) -> List[Dict[str, Any]]:
        candidates = response.get("tool_calls")
        if candidates is None:
            candidates = response.get("tools")
        if candidates is None:
            candidates = response.get("tool_trace")

        normalized: List[Dict[str, Any]] = []
        if isinstance(candidates, list):
            for item in candidates:
                if isinstance(item, str):
                    normalized.append({"name": item, "arguments": {}})
                elif isinstance(item, dict):
                    name = (
                        item.get("name")
                        or item.get("tool_name")
                        or item.get("function", {}).get("name")
                    )
                    arguments = (
                        item.get("arguments")
                        or item.get("args")
                        or item.get("function", {}).get("arguments")
                        or {}
                    )
                    if isinstance(arguments, str):
                        try:
                            arguments = json.loads(arguments)
                        except json.JSONDecodeError:
                            arguments = {}
                    if name:
                        normalized.append({"name": name, "arguments": arguments})
        return normalized

    @staticmethod
    def _extract_answer_text(response: Dict[str, Any]) -> str:
        answer = response.get("answer")
        if isinstance(answer, str):
            return answer
        text_guide = response.get("text_guide")
        if isinstance(text_guide, dict):
            body = text_guide.get("body")
            if isinstance(body, str):
                return body
        content = response.get("response")
        if isinstance(content, str):
            return content
        return ""

    @staticmethod
    def _expected_params_for_tool(tool_name: str, question: Dict[str, Any]) -> Dict[str, Any]:
        context = question.get("context_data", {})
        if not isinstance(context, dict):
            return {}

        if tool_name == "dft_judge":
            if "measurements" in context:
                return {"measurements": context.get("measurements"), "spec": context.get("spec")}
            return {"measurements": context.get("dft_measurements"), "spec": context.get("spec")}

        if tool_name == "anomaly_detection":
            return {
                "image_context": context.get("image_context"),
                "dft_measurements": context.get("dft_measurements") or context.get("measurements"),
                "env": context.get("env"),
            }

        if tool_name == "pose_estimation":
            return {
                "current_pose": context.get("current_pose"),
                "reference_pose": context.get("reference_pose"),
            }

        if tool_name == "sequence_validation":
            return {
                "trajectory": context.get("trajectory"),
                "zone_id": context.get("zone_id"),
                "zone_type": context.get("zone_type"),
                "reference_pattern": context.get("reference_pattern"),
            }

        if tool_name == "timeseries":
            base: Dict[str, Any] = {"session_id": context.get("session_id")}
            if "timeseries_data" in context:
                base["action"] = "update"
                base["data"] = context.get("timeseries_data")
            else:
                base["action"] = "query"
            return base

        if tool_name == "location_tracking":
            return {
                "coordinates": context.get("coordinates"),
                "ship_model_id": context.get("ship_model_id"),
            }

        return {}

    @staticmethod
    def _is_type_match(value: Any, expected_type: str) -> bool:
        if expected_type == "number":
            return isinstance(value, (int, float)) and not isinstance(value, bool)
        if expected_type == "string":
            return isinstance(value, str)
        if expected_type == "array":
            return isinstance(value, list)
        if expected_type == "object":
            return isinstance(value, dict)
        if expected_type == "boolean":
            return isinstance(value, bool)
        return True

    def score_tool_selection(self, expected_tools: List[str], actual_tools: List[str]) -> int:
        if not expected_tools:
            return 2
        if not actual_tools:
            return 0
        if actual_tools == expected_tools:
            return 2

        overlap = len([tool for tool in expected_tools if tool in actual_tools])
        overlap_ratio = overlap / len(expected_tools)

        # Order-aware partial score
        lcs = self._longest_common_subsequence(expected_tools, actual_tools)
        order_ratio = lcs / len(expected_tools)

        if overlap_ratio >= 0.6 and order_ratio >= 0.5:
            return 1
        return 0

    @staticmethod
    def _longest_common_subsequence(seq_a: List[str], seq_b: List[str]) -> int:
        rows = len(seq_a) + 1
        cols = len(seq_b) + 1
        dp = [[0 for _ in range(cols)] for _ in range(rows)]
        for i in range(1, rows):
            for j in range(1, cols):
                if seq_a[i - 1] == seq_b[j - 1]:
                    dp[i][j] = dp[i - 1][j - 1] + 1
                else:
                    dp[i][j] = max(dp[i - 1][j], dp[i][j - 1])
        return dp[-1][-1]

    def score_parameter_correctness(
        self,
        expected_params: Dict[str, Any],
        actual_params: Dict[str, Any],
        schema: Dict[str, Any],
    ) -> int:
        param_schema = schema.get("function", {}).get("parameters", {})
        required = param_schema.get("required", [])
        properties = param_schema.get("properties", {})

        if not actual_params:
            return 0

        missing_required = [name for name in required if name not in actual_params]
        if missing_required:
            return 0

        type_errors = 0
        value_mismatch = 0

        for key, expected_type_info in properties.items():
            if key not in actual_params:
                continue
            expected_type = expected_type_info.get("type")
            if expected_type and not self._is_type_match(actual_params.get(key), expected_type):
                type_errors += 1
                continue

            if key in expected_params and expected_params.get(key) is not None:
                if actual_params.get(key) != expected_params.get(key):
                    value_mismatch += 1

        if type_errors > 0:
            return 0
        if value_mismatch > 0:
            return 1
        return 2

    def score_reasoning_depth(self, response: Dict[str, Any], expected_steps: int) -> int:
        text = self._extract_answer_text(response)
        if not text:
            return 0

        numbered_steps = len(re.findall(r"(?:^|\n)\s*(?:\d+[\).]|[-*])\s+", text))
        connectors = len(re.findall(r"먼저|다음|이후|그리고|마지막으로|우선", text))
        inferred_steps = max(numbered_steps, connectors + 1)
        has_evidence = bool(re.search(r"DOC-[\w-]+|근거|citation|chunk", text, flags=re.IGNORECASE))

        if inferred_steps >= expected_steps and has_evidence:
            return 2
        if inferred_steps >= max(2, expected_steps - 1):
            return 1
        return 0

    def score_context_integration(self, response: Dict[str, Any], context_data: Dict[str, Any]) -> int:
        text = self._extract_answer_text(response)
        citations = response.get("citations", [])
        if not citations:
            metadata = response.get("metadata", {})
            if isinstance(metadata, dict):
                citations = metadata.get("citations", [])

        if not text:
            return 0

        used_chunk = bool(re.search(r"DOC-[\w-]+", text)) or bool(citations)

        context_hits = 0
        for key in ["zone_id", "reference_pattern", "session_id", "image_context"]:
            value = context_data.get(key)
            if isinstance(value, str) and value and value in text:
                context_hits += 1

        numbers = re.findall(r"\d+(?:\.\d+)?", text)
        if numbers and any(k in context_data for k in ["dft_measurements", "measurements", "env"]):
            context_hits += 1

        if used_chunk and context_hits >= 1:
            return 2
        if used_chunk or context_hits >= 1:
            return 1
        return 0

    def score_response_structure(self, response: Dict[str, Any], category: str) -> int:
        text = self._extract_answer_text(response).strip()
        if not text:
            return 0

        if category == "xr_content":
            parsed = self._extract_json_from_text(text)
            if not isinstance(parsed, dict):
                return 0
            required = ["xr_type", "steps", "highlight_zone", "tool_angle", "safety_notes"]
            present = sum(1 for key in required if key in parsed)
            if present == len(required):
                return 2
            if present >= 3:
                return 1
            return 0

        if category == "cot_reasoning":
            has_citation = bool(re.search(r"DOC-[\w-]+", text)) or bool(response.get("citations"))
            has_structure = bool(re.search(r"\n\s*\d+[\).]", text)) or "결론" in text
            if has_citation and has_structure:
                return 2
            if has_citation or has_structure:
                return 1
            return 0

        has_verdict = bool(re.search(r"합격|불합격|진행|중지|재작업", text))
        has_list = bool(re.search(r"\n\s*(?:\d+[\).]|[-*])\s+", text))
        if has_verdict and has_list:
            return 2
        if has_verdict or has_list:
            return 1
        return 0

    @staticmethod
    def _extract_json_from_text(text: str) -> Any:
        text = text.strip()
        if text.startswith("{") and text.endswith("}"):
            try:
                return json.loads(text)
            except json.JSONDecodeError:
                return None

        match = re.search(r"\{[\s\S]*\}", text)
        if not match:
            return None
        try:
            return json.loads(match.group(0))
        except json.JSONDecodeError:
            return None

    def evaluate_single(self, question: Dict[str, Any], response: Dict[str, Any]) -> Dict[str, Any]:
        expected_tools = question.get("expected_tool_sequence", [])
        actual_calls = self._normalize_actual_tool_calls(response)
        actual_tools = [call.get("name", "") for call in actual_calls if call.get("name")]

        tool_selection = self.score_tool_selection(expected_tools, actual_tools)

        param_scores: List[int] = []
        for idx, tool_name in enumerate(expected_tools):
            schema = self.tool_schemas.get(tool_name, {})
            expected_params = self._expected_params_for_tool(tool_name, question)

            actual_params: Dict[str, Any] = {}
            if idx < len(actual_calls) and actual_calls[idx].get("name") == tool_name:
                actual_params = actual_calls[idx].get("arguments", {})
            else:
                for call in actual_calls:
                    if call.get("name") == tool_name:
                        actual_params = call.get("arguments", {})
                        break

            param_scores.append(self.score_parameter_correctness(expected_params, actual_params, schema))

        parameter_correctness = round(sum(param_scores) / len(param_scores)) if param_scores else 0

        reasoning_depth = self.score_reasoning_depth(response, int(question.get("reasoning_steps", 3)))
        context_integration = self.score_context_integration(response, question.get("context_data", {}))
        response_structure = self.score_response_structure(response, str(question.get("category", "")))

        total = tool_selection + parameter_correctness + reasoning_depth + context_integration + response_structure

        return {
            "q_id": question.get("q_id", ""),
            "category": question.get("category", ""),
            "difficulty": question.get("difficulty", ""),
            "scores": {
                "tool_selection_accuracy": int(tool_selection),
                "parameter_correctness": int(parameter_correctness),
                "reasoning_depth": int(reasoning_depth),
                "context_integration": int(context_integration),
                "response_structure": int(response_structure),
            },
            "total": int(total),
            "details": {
                "expected_tools": expected_tools,
                "actual_tools": actual_tools,
                "missing_response": not bool(response),
            },
        }

    @staticmethod
    def _load_responses(responses_path: Path) -> Dict[str, Dict[str, Any]]:
        by_qid: Dict[str, Dict[str, Any]] = {}
        with responses_path.open("r", encoding="utf-8") as fh:
            for line_no, line in enumerate(fh, start=1):
                payload = line.strip()
                if not payload:
                    continue
                try:
                    item = json.loads(payload)
                except json.JSONDecodeError:
                    logger.warning("Skipping invalid response JSON line %d", line_no)
                    continue
                q_id = item.get("q_id")
                if not q_id:
                    logger.warning("Response line %d missing q_id", line_no)
                    continue
                by_qid[q_id] = item
        logger.info("Loaded %d responses from %s", len(by_qid), responses_path)
        return by_qid

    def evaluate_all(self, responses_path: Path, output_dir: Path) -> Dict[str, Any]:
        output_dir.mkdir(parents=True, exist_ok=True)
        responses_by_qid = self._load_responses(responses_path)

        results: List[Dict[str, Any]] = []
        for question in self.eval_set:
            q_id = question.get("q_id", "")
            response = responses_by_qid.get(q_id, {})
            results.append(self.evaluate_single(question, response))

        csv_path = output_dir / "agentic_eval_results.csv"
        self._write_csv(results, csv_path)

        report_path = output_dir / "agentic_eval_report.md"
        summary = self._write_report(results, report_path)

        summary["csv_path"] = str(csv_path)
        summary["report_path"] = str(report_path)
        return summary

    @staticmethod
    def _write_csv(results: List[Dict[str, Any]], output_path: Path) -> None:
        rows: List[EvaluationResult] = []
        for item in results:
            scores = item.get("scores", {})
            details = item.get("details", {})
            rows.append(
                EvaluationResult(
                    q_id=item.get("q_id", ""),
                    category=item.get("category", ""),
                    difficulty=item.get("difficulty", ""),
                    tool_selection_accuracy=int(scores.get("tool_selection_accuracy", 0)),
                    parameter_correctness=int(scores.get("parameter_correctness", 0)),
                    reasoning_depth=int(scores.get("reasoning_depth", 0)),
                    context_integration=int(scores.get("context_integration", 0)),
                    response_structure=int(scores.get("response_structure", 0)),
                    total=int(item.get("total", 0)),
                    expected_tools=",".join(details.get("expected_tools", [])),
                    actual_tools=",".join(details.get("actual_tools", [])),
                    note="missing_response" if details.get("missing_response") else "",
                )
            )

        with output_path.open("w", newline="", encoding="utf-8") as fh:
            writer = csv.DictWriter(
                fh,
                fieldnames=list(EvaluationResult.__dataclass_fields__.keys()),
            )
            writer.writeheader()
            for row in rows:
                writer.writerow(row.__dict__)

        logger.info("Wrote CSV results to %s", output_path)

    @staticmethod
    def _write_report(results: List[Dict[str, Any]], output_path: Path) -> Dict[str, Any]:
        totals = [item.get("total", 0) for item in results]
        avg_total = statistics.mean(totals) if totals else 0.0

        dimension_avg: Dict[str, float] = {}
        for dim in RUBRIC_DIMENSIONS:
            dim_scores = [item.get("scores", {}).get(dim, 0) for item in results]
            dimension_avg[dim] = statistics.mean(dim_scores) if dim_scores else 0.0

        by_category: Dict[str, List[int]] = {}
        for item in results:
            by_category.setdefault(item.get("category", "unknown"), []).append(item.get("total", 0))

        by_difficulty: Dict[str, List[int]] = {}
        for item in results:
            by_difficulty.setdefault(item.get("difficulty", "unknown"), []).append(item.get("total", 0))

        tool_missing = 0
        for item in results:
            expected = item.get("details", {}).get("expected_tools", [])
            actual = item.get("details", {}).get("actual_tools", [])
            if expected and not actual:
                tool_missing += 1

        score_distribution = Counter(totals)

        lines: List[str] = []
        lines.append("# Agentic Evaluation Report")
        lines.append("")
        lines.append(f"- total_questions: {len(results)}")
        lines.append(f"- average_total_score: {avg_total:.2f} / 10")
        lines.append(f"- tool_not_called_count: {tool_missing}")
        lines.append("")
        lines.append("## Dimension Averages")
        lines.append("")
        for dim in RUBRIC_DIMENSIONS:
            lines.append(f"- {dim}: {dimension_avg[dim]:.2f} / 2")

        lines.append("")
        lines.append("## Category Breakdown")
        lines.append("")
        for category, values in sorted(by_category.items()):
            lines.append(f"- {category}: {statistics.mean(values):.2f} / 10 ({len(values)} items)")

        lines.append("")
        lines.append("## Difficulty Breakdown")
        lines.append("")
        for difficulty, values in sorted(by_difficulty.items()):
            lines.append(f"- {difficulty}: {statistics.mean(values):.2f} / 10 ({len(values)} items)")

        lines.append("")
        lines.append("## Total Score Distribution")
        lines.append("")
        for score in sorted(score_distribution.keys()):
            lines.append(f"- score_{score}: {score_distribution[score]}")

        output_path.write_text("\n".join(lines), encoding="utf-8")
        logger.info("Wrote analysis report to %s", output_path)

        return {
            "total_questions": len(results),
            "average_total_score": round(avg_total, 4),
            "dimension_average": {k: round(v, 4) for k, v in dimension_avg.items()},
            "tool_not_called_count": tool_missing,
        }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate agentic responses against tool and CoT rubric.")
    parser.add_argument(
        "--eval-set",
        type=Path,
        default=Path("/data/eval/eval_set_agentic.jsonl"),
        help="Path to agentic evaluation set JSONL.",
    )
    parser.add_argument(
        "--rubric",
        type=Path,
        default=Path("/workspace/trainer/configs/eval_rubric_agentic.md"),
        help="Path to agentic rubric markdown.",
    )
    parser.add_argument(
        "--tool-schemas-dir",
        type=Path,
        default=Path("/workspace/trainer/configs/tool_schemas"),
        help="Path to tool schema directory.",
    )
    parser.add_argument(
        "--responses",
        type=Path,
        required=True,
        help="Path to model responses JSONL.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        required=True,
        help="Directory to write evaluation outputs.",
    )
    parser.add_argument(
        "--log-level",
        default="INFO",
        choices=["DEBUG", "INFO", "WARNING", "ERROR"],
        help="Logging verbosity.",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    logging.basicConfig(level=getattr(logging, args.log_level), format="%(asctime)s %(levelname)s %(message)s")

    if not args.eval_set.is_file():
        logger.error("Eval set not found: %s", args.eval_set)
        return 1
    if not args.rubric.is_file():
        logger.error("Rubric not found: %s", args.rubric)
        return 1
    if not args.tool_schemas_dir.is_dir():
        logger.error("Tool schema directory not found: %s", args.tool_schemas_dir)
        return 1
    if not args.responses.is_file():
        logger.error("Responses file not found: %s", args.responses)
        return 1

    evaluator = AgenticEvaluator(args.eval_set, args.rubric, args.tool_schemas_dir)
    summary = evaluator.evaluate_all(args.responses, args.output_dir)
    logger.info("Evaluation completed: %s", json.dumps(summary, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
