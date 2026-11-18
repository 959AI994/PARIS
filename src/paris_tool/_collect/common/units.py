from math import isfinite


_LENGTH_TO_UM = {"um": 1.0, "nm": 1e-3, "mm": 1e3}
_TIME_TO_NS = {"ns": 1.0, "ps": 1e-3, "s": 1e9}
_RESISTANCE_TO_KOHM = {"kohm": 1.0, "ohm": 1e-3}
_CAPACITANCE_TO_FF = {"ff": 1.0, "pf": 1e3, "f": 1e15}


def convert(value: float, source_unit: str, quantity: str) -> float:
    tables = {
        "length": _LENGTH_TO_UM,
        "time": _TIME_TO_NS,
        "resistance": _RESISTANCE_TO_KOHM,
        "capacitance": _CAPACITANCE_TO_FF,
    }
    try:
        scale = tables[quantity][source_unit.lower()]
    except KeyError as error:
        raise ValueError(f"不支持的 {quantity} 单位: {source_unit}") from error
    result = float(value) * scale
    if not isfinite(result):
        raise ValueError(f"换算结果不是有限数值: {value} {source_unit}")
    return result


def dbu_to_um(value: int | float, dbu_per_micron: int) -> float:
    if dbu_per_micron <= 0:
        raise ValueError("dbu_per_micron 必须大于零")
    return float(value) / dbu_per_micron

