import csv
import json
import os
from pathlib import Path
from tempfile import NamedTemporaryFile
from typing import Any, Iterable, Mapping, Sequence


def atomic_write_json(path: str | Path, payload: Any) -> None:
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    with NamedTemporaryFile(
        "w", encoding="utf-8", dir=destination.parent, delete=False
    ) as stream:
        json.dump(payload, stream, ensure_ascii=False, indent=2, sort_keys=True)
        stream.write("\n")
        temporary = Path(stream.name)
    os.replace(temporary, destination)


def read_json(path: str | Path) -> Any:
    with Path(path).open(encoding="utf-8") as stream:
        return json.load(stream)


def write_csv(
    path: str | Path,
    rows: Iterable[Mapping[str, Any]],
    columns: Sequence[str],
) -> None:
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    with NamedTemporaryFile(
        "w", encoding="utf-8", newline="", dir=destination.parent, delete=False
    ) as stream:
        writer = csv.DictWriter(stream, fieldnames=list(columns), extrasaction="raise")
        writer.writeheader()
        writer.writerows(rows)
        temporary = Path(stream.name)
    os.replace(temporary, destination)


def read_tsv(path: str | Path) -> list[dict[str, str]]:
    with Path(path).open(encoding="utf-8", newline="") as stream:
        return list(csv.DictReader(stream, delimiter="\t"))


def write_parquet(path: str | Path, rows: list[dict[str, Any]], columns: Sequence[str]) -> None:
    try:
        import pyarrow as pa
        import pyarrow.parquet as pq
    except ImportError as error:
        raise RuntimeError(
            "写入 Parquet 需要 pyarrow；请安装项目依赖后重新运行"
        ) from error
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    normalized = [{column: row[column] for column in columns} for row in rows]
    table = pa.Table.from_pylist(normalized)
    pq.write_table(table, destination, compression="zstd")

