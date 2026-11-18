import hashlib
import json
import os
from pathlib import Path
import signal
import subprocess
import time
from functools import lru_cache
from contextlib import contextmanager


def digest(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as f:
        for b in iter(lambda: f.read(1024**2), b""):
            h.update(b)
    return h.hexdigest()


@lru_cache(maxsize=1024)
def _fingerprint_cached(path, device, inode, size, mtime):
    return digest(path)


def fingerprint(path):
    p = Path(path).resolve()
    s = p.stat()
    return _fingerprint_cached(str(p), s.st_dev, s.st_ino, s.st_size, s.st_mtime_ns)


def save(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + ".tmp")
    tmp.write_text(
        json.dumps(value, indent=2, ensure_ascii=False, allow_nan=False, default=str)
        + "\n"
    )
    os.replace(tmp, path)


def fresh(path):
    path = Path(path)
    path.mkdir(parents=True, exist_ok=False)
    return path


def run(command, cwd, env=None, timeout=3600, name="tool"):
    cwd = Path(cwd)
    cwd.mkdir(parents=True, exist_ok=True)
    start = time.monotonic()
    log = cwd / (name + ".log")
    if log.exists():
        raise FileExistsError(log)
    with log.open("x") as out:
        p = subprocess.Popen(
            [str(x) for x in command],
            cwd=cwd,
            env=env,
            stdin=subprocess.DEVNULL,
            stdout=out,
            stderr=subprocess.STDOUT,
            start_new_session=True,
        )
        try:
            code = p.wait(timeout=timeout)
        except subprocess.TimeoutExpired:
            os.killpg(p.pid, signal.SIGTERM)
            try:
                p.wait(timeout=5)
            except subprocess.TimeoutExpired:
                os.killpg(p.pid, signal.SIGKILL)
                p.wait()
            save(
                cwd / (name + ".process.json"),
                dict(status="timeout", seconds=time.monotonic() - start),
            )
            raise TimeoutError(f"Tool timeout; inspect {log}")
    save(
        cwd / (name + ".process.json"),
        dict(
            returncode=code,
            seconds=time.monotonic() - start,
            executable=str(command[0]),
        ),
    )
    if code:
        raise RuntimeError(f"Tool exited {code}; inspect {log}")


def seal(root, card):
    root = Path(root)
    save(root / "MODEL_CARD.json", card)
    save(
        root / "SHA256.json",
        {
            str(p.relative_to(root)): digest(p)
            for p in root.rglob("*")
            if p.is_file() and p.name != "SHA256.json"
        },
    )


def verify(root):
    root = Path(root).resolve()
    for name, h in json.loads((root / "SHA256.json").read_text()).items():
        p = (root / name).resolve()
        if not p.is_relative_to(root) or not p.is_file() or digest(p) != h:
            raise ValueError("Model checksum mismatch: " + name)
    return json.loads((root / "MODEL_CARD.json").read_text())


@contextmanager
def project_lock(root):
    import fcntl

    with (Path(root) / ".project.lock").open("a") as stream:
        try:
            fcntl.flock(stream, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError:
            raise RuntimeError("Another PARIS command is using this workspace")
        yield
