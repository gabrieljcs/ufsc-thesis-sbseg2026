from __future__ import annotations

import sys
from contextlib import contextmanager
from time import perf_counter


def info(message: str) -> None:
    print(f"[thesis-eval] {message}", file=sys.stderr, flush=True)


@contextmanager
def step(message: str):
    started = perf_counter()
    info(f"START {message}")
    try:
        yield
    except Exception:
        elapsed = perf_counter() - started
        info(f"FAIL  {message} ({elapsed:.1f}s)")
        raise
    else:
        elapsed = perf_counter() - started
        info(f"DONE  {message} ({elapsed:.1f}s)")
