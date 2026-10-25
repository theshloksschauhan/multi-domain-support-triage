"""HTTP client for one local shard."""

import ipaddress
import json
from pathlib import Path
from typing import List, Optional
from urllib import error, request
from urllib.parse import urlparse


def require_loopback(host: str) -> str:
    """Accept only a local loopback name. The shard protocol has no authentication."""

    normalized = host.strip().strip("[]")
    if normalized.lower() == "localhost":
        return normalized
    try:
        address = ipaddress.ip_address(normalized)
    except ValueError as exc:
        raise ValueError(f"{host} is not a loopback address") from exc
    if not address.is_loopback:
        raise ValueError(f"{host} is not a loopback address")
    return normalized


class ShardClientError(RuntimeError):
    def __init__(self, message: str, status: Optional[int] = None):
        super().__init__(message)
        self.status = status


class ShardClient:
    def __init__(self, shard_id: str, base_url: str, timeout: float = 5.0):
        self.shard_id = shard_id
        self.base_url = base_url.rstrip("/")
        self.timeout = timeout

    def health(self) -> dict:
        return self._request("GET", "/health")

    def document_ids(self) -> List[str]:
        payload = self._request("GET", "/v1/ids")
        ids = payload.get("ids")
        if not isinstance(ids, list):
            raise ShardClientError(f"{self.shard_id} returned malformed ids")
        return [str(doc_id) for doc_id in ids]

    def stats(self, terms: List[str]) -> dict:
        return self._request("POST", "/v1/stats", {"terms": terms})

    def search(self, payload: dict) -> dict:
        return self._request("POST", "/v1/search", payload)

    def begin_index(self) -> dict:
        return self._request("POST", "/v1/index/begin", {})

    def index_batch(self, documents: list) -> dict:
        return self._request("POST", "/v1/index/batch", {"documents": documents})

    def commit(self, generation: str, expected_count: int) -> dict:
        return self._request(
            "POST",
            "/v1/index/commit",
            {"generation": generation, "expected_count": expected_count},
        )

    def _request(self, method: str, path: str, payload: Optional[dict] = None) -> dict:
        data = None
        headers = {}
        if payload is not None:
            data = json.dumps(payload, ensure_ascii=False).encode("utf-8")
            headers["Content-Type"] = "application/json"
        req = request.Request(self.base_url + path, data=data, headers=headers, method=method)
        try:
            with request.urlopen(req, timeout=self.timeout) as response:
                raw = response.read()
        except error.HTTPError as exc:
            detail = exc.read().decode("utf-8", errors="replace")
            raise ShardClientError(f"{self.shard_id} HTTP {exc.code}: {detail}", exc.code) from exc
        except error.URLError as exc:
            raise ShardClientError(f"{self.shard_id} connection failed: {exc.reason}") from exc
        except TimeoutError as exc:
            raise ShardClientError(f"{self.shard_id} timed out") from exc
        try:
            parsed = json.loads(raw.decode("utf-8"))
        except (UnicodeDecodeError, json.JSONDecodeError) as exc:
            raise ShardClientError(f"{self.shard_id} returned malformed JSON") from exc
        if not isinstance(parsed, dict):
            raise ShardClientError(f"{self.shard_id} returned a non-object JSON body")
        return parsed


def _check_url(shard_id: str, url: str) -> None:
    parsed = urlparse(url)
    if parsed.hostname is None or parsed.port is None:
        raise ValueError(f"{shard_id} needs a loopback URL with a port")
    require_loopback(parsed.hostname)


def load_shard_config(path: Path) -> List[dict]:
    data = json.loads(Path(path).read_text(encoding="utf-8"))
    shards = data.get("shards") if isinstance(data, dict) else None
    if not isinstance(shards, list) or not shards:
        raise ValueError(f"shard config {path} must contain a non-empty shards list")
    normalized = []
    seen = set()
    urls = set()
    for item in shards:
        if not isinstance(item, dict) or "id" not in item or "url" not in item:
            raise ValueError(f"shard config {path} has an entry without id and url")
        shard_id = str(item["id"])
        if shard_id in seen:
            raise ValueError(f"shard config {path} repeats shard id {shard_id}")
        seen.add(shard_id)
        primary_url = str(item["url"])
        _check_url(shard_id, primary_url)
        if primary_url in urls:
            raise ValueError(f"shard config {path} repeats url {primary_url}")
        urls.add(primary_url)
        replica_url = item.get("replica_url")
        if replica_url:
            replica_url = str(replica_url)
            _check_url(f"{shard_id} replica", replica_url)
            if replica_url in urls:
                raise ValueError(f"shard config {path} repeats url {replica_url}")
            urls.add(replica_url)
        normalized.append(item)
    return normalized


def load_shard_clients(path: Path, timeout: float = 5.0) -> List[ShardClient]:
    return [
        ShardClient(str(item["id"]), str(item["url"]), timeout=timeout)
        for item in load_shard_config(path)
    ]


def load_shard_members(path: Path, timeout: float = 5.0) -> tuple:
    """Return primary clients and an aligned replica list.

    A missing replica is ``None``. Replicas are not extra partitions.
    """
    primaries = []
    replicas: List[Optional[ShardClient]] = []
    for item in load_shard_config(path):
        shard_id = str(item["id"])
        primaries.append(ShardClient(shard_id, str(item["url"]), timeout=timeout))
        replica_url = item.get("replica_url")
        if replica_url:
            replicas.append(ShardClient(shard_id, str(replica_url), timeout=timeout))
        else:
            replicas.append(None)
    return primaries, replicas
