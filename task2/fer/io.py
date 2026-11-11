"""Atomic artifacts that tolerate temporary Windows file-sharing locks."""
from contextlib import contextmanager
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import sys
import time
from uuid import uuid4


def _retry_file_access(operation, *, timeout):
    deadline = time.monotonic() + timeout
    delay = 0.025
    while True:
        try:
            return operation()
        except OSError as error:
            temporary_lock = isinstance(error, PermissionError) or getattr(error, 'winerror', None) in (5, 32, 33)
            remaining = deadline - time.monotonic()
            if not temporary_lock or remaining <= 0:
                raise
            time.sleep(min(delay, remaining))
            delay = min(delay * 2, 0.25)


def replace_with_retry(source, destination, *, timeout=5.0):
    return _retry_file_access(lambda: os.replace(source, destination), timeout=timeout)


def read_json(path, *, timeout=5.0):
    return _retry_file_access(lambda: json.loads(Path(path).read_text(encoding='utf-8')),
                              timeout=timeout)


@contextmanager
def atomic_output(path, *, timeout=5.0):
    path = Path(path)
    # Each writer gets its own file on the destination filesystem. Close the
    # creation handle before another library opens it (required on Windows).
    temp = path.with_name(f'{path.name}.{uuid4().hex}.tmp')
    try:
        with temp.open('xb'):
            pass
        yield temp
        replace_with_retry(temp, path, timeout=timeout)
    finally:
        try:
            temp.unlink(missing_ok=True)
        except OSError as error:
            print(f'Warning: could not remove temporary artifact {temp}: {error}', file=sys.stderr, flush=True)


def atomic_json(path, value, *, required=True, timeout=5.0):
    payload = json.dumps(value, ensure_ascii=False, indent=2) + '\n'
    try:
        with atomic_output(path, timeout=timeout) as temp:
            temp.write_text(payload, encoding='utf-8')
        return True
    except OSError as error:
        if required:
            raise
        print(f'Warning: could not update status {path}; retaining previous status: {error}',
              file=sys.stderr, flush=True)
        return False


def update_status(path, value, *, required=False, timeout=5.0):
    return atomic_json(path, {**value, 'updated_at': datetime.now(timezone.utc).isoformat()},
                       required=required, timeout=timeout)
