"""Convert and compare user-supplied film thickness without session lookups."""

import re
from decimal import Decimal

_NUMBER = r"(?<![\d.+-])\d{1,8}(?:\.\d{1,6})?"
_MM = r"mm(?![a-z/])|밀리미터"
_UM = r"[µμu]m(?![a-z/])|마이크론|마이크로미터"
_UNIT = rf"{_MM}|{_UM}"
_RANGE = re.compile(
    rf"(?P<low>{_NUMBER})\s*(?P<low_unit>{_UNIT})?\s*[~～∼–-]\s*"
    rf"(?P<high>\d{{1,8}}(?:\.\d{{1,6}})?)\s*(?P<high_unit>{_UNIT})?",
    re.IGNORECASE,
)
_READING = re.compile(rf"(?P<value>{_NUMBER})\s*(?P<unit>{_UNIT})", re.IGNORECASE)
_BARE_READING = re.compile(rf"(?P<value>{_NUMBER})\s*(?:은|는|이면|이라면|일\s*때)")
_OUTPUT_UNIT = re.compile(rf"(?P<unit>{_UNIT})\s*(?:으로|로|단위)", re.IGNORECASE)


def _unit_name(unit: str) -> str:
    return "mm" if re.fullmatch(_MM, unit, re.IGNORECASE) else "µm"


def _to_um(value: Decimal, unit: str) -> Decimal:
    return value * 1000 if unit == "mm" else value


def _format(value: Decimal) -> str:
    number = format(value, "f")
    return number.rstrip("0").rstrip(".") if "." in number else number


def supplied_thickness_answer(prompt: str) -> str | None:
    """Handle self-contained comparisons only when the user requests a unit."""
    if not re.search(r"도막|두께|(?:정상|목표)\s*범위", prompt):
        return None
    if not re.search(r"범위|상한|하한|비교|어느\s*쪽", prompt):
        return None
    stated_range = _RANGE.search(prompt)
    output = _OUTPUT_UNIT.search(prompt)
    if stated_range is None or output is None:
        return None
    target_unit = _unit_name(output["unit"])
    remaining = prompt[:stated_range.start()] + " " + prompt[stated_range.end():]
    readings = list(_READING.finditer(remaining))
    if len(readings) > 1:
        return None
    reading = readings[0] if readings else _BARE_READING.search(remaining)
    if reading is None:
        return None
    # The existing simulator/UI thickness ranges use µm when the unit is
    # omitted. State the assumed unit when displaying the supplied range.
    range_unit = _unit_name(stated_range["high_unit"] or stated_range["low_unit"] or "µm")
    low_unit = _unit_name(stated_range["low_unit"]) if stated_range["low_unit"] else range_unit
    high = _to_um(Decimal(stated_range["high"]), range_unit)
    low = _to_um(Decimal(stated_range["low"]), low_unit)
    if low > high:
        return None
    value = Decimal(reading["value"])
    source_unit = _unit_name(reading["unit"]) if readings else range_unit
    thickness = _to_um(value, source_unit)
    scale = 1000 if target_unit == "mm" else 1
    range_scale = 1000 if range_unit == "mm" else 1
    if thickness < low:
        status = "범위의 하한에 못 미칩니다"
    elif thickness > high:
        status = "범위의 상한을 초과합니다"
    else:
        status = "범위 안에 있습니다"
    supplied = f"{_format(low / range_scale)}~{_format(high / range_scale)} {range_unit}"
    return f"{_format(value)} {source_unit}는 {_format(thickness / scale)} {target_unit}이며, 제시된 {supplied} {status}."
