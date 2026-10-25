"""Start, index, check, and stop the local shard processes.

Each shard is a separate Python process bound to 127.0.0.1. The pid file
records only the processes this command started, and stop signals only those
pids. Database paths in the shard config are relative to the repository root.
"""

import argparse
import json
import os
import subprocess
import sys
import time
from pathlib import Path
from urllib.parse import urlparse

CODE_ROOT = Path(__file__).resolve().parent
if str(CODE_ROOT) not in sys.path:
    sys.path.insert(0, str(CODE_ROOT))

from config.settings import repo_root
from retrieval.index_coordinator import IndexCoordinator
from retrieval.shard_client import (
    ShardClient,
    ShardClientError,
    load_shard_config,
    load_shard_members,
    require_loopback,
)


def default_pid_file() -> Path:
    return repo_root() / "var" / "cluster-pids.json"


def _resolve(path_value: str, root: Path) -> Path:
    path = Path(path_value)
    if path.is_absolute():
        return path
    return root / path


def _process_command(pid: int) -> str:
    if os.name == "nt":
        completed = subprocess.run(
            [
                "powershell",
                "-NoProfile",
                "-Command",
                f"(Get-CimInstance Win32_Process -Filter \"ProcessId={int(pid)}\").CommandLine",
            ],
            capture_output=True,
            text=True,
            check=False,
        )
        return completed.stdout or ""
    try:
        raw = Path(f"/proc/{int(pid)}/cmdline").read_bytes()
    except OSError:
        return ""
    return raw.replace(b"\0", b" ").decode("utf-8", errors="replace")


def _stop_pid(pid: int) -> None:
    if pid <= 0:
        return
    if "shard_server.py" not in _process_command(pid):
        return
    if os.name == "nt":
        subprocess.run(
            ["taskkill", "/PID", str(pid), "/T", "/F"],
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
            check=False,
        )
        return
    try:
        os.kill(pid, 15)
    except OSError:
        return


def stop_cluster(pid_file: Path) -> int:
    if not pid_file.exists():
        print(f"no pid file at {pid_file}")
        return 0
    payload = json.loads(pid_file.read_text(encoding="utf-8"))
    stopped = 0
    for record in payload.get("processes", []):
        pid = int(record.get("pid", 0))
        _stop_pid(pid)
        print(f"stopped {record.get('shard_id', '?')} pid={pid}")
        stopped += 1
    pid_file.unlink(missing_ok=True)
    return stopped


def _launch_member(item, url, db_value, role, root, processes, logs) -> None:
    parsed = urlparse(url)
    host = require_loopback(parsed.hostname or "")
    if parsed.port is None:
        raise ValueError(f"{item['id']} {role} is missing a port in {url}")
    db_path = _resolve(db_value, root)
    db_path.parent.mkdir(parents=True, exist_ok=True)
    log_path = db_path.with_suffix(".log")
    log_handle = log_path.open("a", encoding="utf-8")
    logs.append(log_handle)
    command = [
        sys.executable,
        str(CODE_ROOT / "retrieval" / "shard_server.py"),
        "--shard-id",
        str(item["id"]),
        "--host",
        host,
        "--port",
        str(parsed.port),
        "--db",
        str(db_path),
    ]
    proc = subprocess.Popen(
        command,
        cwd=str(root),
        stdout=log_handle,
        stderr=subprocess.STDOUT,
    )
    processes.append(
        {
            "shard_id": str(item["id"]),
            "role": role,
            "pid": proc.pid,
            "url": url,
            "db": str(db_path),
            "proc": proc,
        }
    )


def start_cluster(config_path: Path, pid_file: Path, root: Path, timeout: float = 15.0) -> list:
    if pid_file.exists():
        stop_cluster(pid_file)
    shards = load_shard_config(config_path)
    pid_file.parent.mkdir(parents=True, exist_ok=True)
    processes = []
    logs = []
    try:
        for item in shards:
            _launch_member(
                item,
                str(item["url"]),
                str(item.get("db") or f"var/shards/{item['id']}.db"),
                "primary",
                root,
                processes,
                logs,
            )
            if item.get("replica_url"):
                _launch_member(
                    item,
                    str(item["replica_url"]),
                    str(item.get("replica_db") or f"var/shards/{item['id']}-replica.db"),
                    "replica",
                    root,
                    processes,
                    logs,
                )
        deadline = time.time() + timeout
        for record in processes:
            client = ShardClient(record["shard_id"], record["url"], timeout=2.0)
            while True:
                if record["proc"].poll() is not None:
                    raise RuntimeError(
                        f"{record['shard_id']} exited before /health "
                        f"(code {record['proc'].returncode})"
                    )
                try:
                    client.health()
                    break
                except (ShardClientError, OSError, TimeoutError):
                    if time.time() >= deadline:
                        raise RuntimeError(f"{record['shard_id']} did not answer /health")
                    time.sleep(0.1)
        for record in processes:
            proc = record["proc"]
            if proc.poll() is None:
                # The shard is supposed to outlive this command. Mark the handle
                # collected so Popen's destructor does not warn while it is running.
                proc.returncode = 0
    except Exception:
        for record in processes:
            _stop_pid(int(record["pid"]))
        raise
    finally:
        for handle in logs:
            handle.close()
    stored = [
        {key: record[key] for key in ("shard_id", "role", "pid", "url", "db")}
        for record in processes
    ]
    pid_file.write_text(
        json.dumps({"config": str(config_path), "processes": stored}, indent=2),
        encoding="utf-8",
    )
    for record in stored:
        print(
            f"started {record['shard_id']} {record['role']} "
            f"pid={record['pid']} url={record['url']}"
        )
    return stored


def index_cluster(config_path: Path, corpus: Path, timeout: float) -> dict:
    primaries, replicas = load_shard_members(config_path, timeout=timeout)
    report = IndexCoordinator(primaries, replicas=replicas).build(corpus)
    print(json.dumps(report, indent=2))
    return report


def health_cluster(config_path: Path, timeout: float) -> int:
    primaries, replicas = load_shard_members(config_path, timeout=timeout)
    clients = [client for client in primaries + replicas if client is not None]
    generations = set()
    ready = True
    for client in clients:
        try:
            row = client.health()
        except (ShardClientError, OSError, TimeoutError) as exc:
            print(f"{client.shard_id} unavailable: {exc}")
            return 1
        print(
            f"{client.base_url} {row.get('shard_id')} ready={row.get('ready')} "
            f"generation={row.get('generation')} documents={row.get('document_count')}"
        )
        if not row.get("ready"):
            ready = False
        generation = row.get("generation")
        if isinstance(generation, str) and generation:
            generations.add(generation)
    if not ready or len(generations) != 1:
        print("generation agreement: no")
        return 1
    print(f"generation agreement: yes {next(iter(generations))}")
    return 0


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description="Operate the local search shard cluster")
    parser.add_argument("command", choices=("start", "index", "health", "stop"))
    parser.add_argument("--shard-config", required=True)
    parser.add_argument("--corpus", default="")
    parser.add_argument("--pid-file", default="")
    parser.add_argument("--timeout", type=float, default=30.0)
    args = parser.parse_args(argv)
    root = repo_root()
    config_path = _resolve(args.shard_config, root)
    pid_file = _resolve(args.pid_file, root) if args.pid_file else default_pid_file()
    if args.command == "start":
        start_cluster(config_path, pid_file, root, timeout=args.timeout)
        return 0
    if args.command == "stop":
        stop_cluster(pid_file)
        return 0
    if args.command == "index":
        corpus = _resolve(args.corpus, root) if args.corpus else root / "data"
        index_cluster(config_path, corpus, args.timeout)
        return 0
    return health_cluster(config_path, args.timeout)


if __name__ == "__main__":
    try:
        sys.exit(main())
    except Exception as exc:
        print(f"cluster command failed: {exc}", file=sys.stderr)
        sys.exit(1)
