"""Retry transient Windows file sharing failures without changing training math.

This optional launcher leaves the frozen training sources and their digest intact.
Only status/checkpoint writes are retried; data, optimizer and RNG are unchanged.
"""

from functools import wraps
import time


def retry_file_operation(operation, attempts=20, sleep=time.sleep):
    @wraps(operation)
    def wrapped(*args, **kwargs):
        for attempt in range(attempts):
            try:
                return operation(*args, **kwargs)
            except OSError as error:
                if getattr(error, "winerror", None) not in (5, 32, 33) or attempt + 1 == attempts:
                    raise
                sleep(min(0.05 * 2 ** attempt, 0.5))
    return wrapped


def install():
    from vessel import runtime as base
    from vessel_lab import runtime as lab

    for name in ("write_json", "save_checkpoint"):
        original = getattr(base, name)
        if not getattr(original, "_windows_retry", False):
            operation = retry_file_operation(original)
            operation._windows_retry = True
            setattr(base, name, operation)
            setattr(lab, name, operation)


if __name__ == "__main__":
    install()
    from .train import main
    main()
