"""Coordinate capacity and retry safe requests on the shared Airbyte worker."""

from contextlib import contextmanager
import fcntl
from pathlib import Path
import time
from urllib.error import HTTPError, URLError

from elt_taskgen.runtime.airbyte import AirbyteClient, AirbyteError


LOCAL_AIRBYTE_ROOT = Path.home() / ".airbyte/abctl/data/elt-environment-duckdb"


@contextmanager
def airbyte_lease(name: str, storage_root: Path | None = None):
    base = LOCAL_AIRBYTE_ROOT.resolve()
    root = storage_root.resolve() if storage_root is not None else base
    if root.is_relative_to(base):
        root = base
    with (root / name).open("a") as lease:
        fcntl.flock(lease, fcntl.LOCK_EX)
        yield


def is_transient_airbyte_error(error: AirbyteError) -> bool:
    cause = error.__cause__
    if isinstance(cause, HTTPError):
        return cause.code in {408, 429, 500, 502, 503, 504}
    return isinstance(cause, (TimeoutError, ConnectionError, URLError))


class ReplayAirbyteClient(AirbyteClient):
    def request(self, method, path, payload=None):
        method = method.upper()
        for attempt in range(3):
            try:
                return super().request(method, path, payload)
            except AirbyteError as error:
                cause = error.__cause__
                if method == "DELETE" and isinstance(cause, HTTPError) and cause.code == 404:
                    return {}
                if method not in {"GET", "DELETE"} or attempt == 2 or not is_transient_airbyte_error(error):
                    raise
                time.sleep(2 ** attempt)
