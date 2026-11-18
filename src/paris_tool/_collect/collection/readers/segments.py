from pathlib import Path

from ..models import Segment


def read_route_segments(path: str | Path) -> dict[str, list[Segment]]:
    source = Path(path)
    result: dict[str, list[Segment]] = {}
    current_net: str | None = None
    in_block = False
    with source.open(encoding="utf-8") as stream:
        for line_number, raw in enumerate(stream, start=1):
            line = raw.strip()
            if not line:
                continue
            if current_net is None:
                current_net = line
                if current_net in result:
                    raise ValueError(f"route segments 中 net 重复: {current_net}")
                result[current_net] = []
                continue
            if line == "(":
                if in_block:
                    raise ValueError(f"第 {line_number} 行出现嵌套括号")
                in_block = True
                continue
            if line == ")":
                if not in_block:
                    raise ValueError(f"第 {line_number} 行出现未配对右括号")
                in_block = False
                current_net = None
                continue
            if not in_block:
                raise ValueError(f"第 {line_number} 行不在 net 段内")
            fields = line.split()
            if len(fields) != 6:
                raise ValueError(f"第 {line_number} 行 segment 字段数量不是 6")
            result[current_net].append(
                Segment(
                    current_net,
                    int(fields[0]),
                    int(fields[1]),
                    fields[2],
                    int(fields[3]),
                    int(fields[4]),
                    fields[5],
                )
            )
    if current_net is not None or in_block:
        raise ValueError("route segments 文件在 net 段中意外结束")
    return result
