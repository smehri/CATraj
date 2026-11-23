"""Running independent pieces of work on several cores.

Epochs of a cross-validation and time blocks of an annotation are independent,
so they can be spread over processes. The standard library's
``concurrent.futures.ProcessPoolExecutor`` does the work: no extra dependency,
and it behaves the same from scripts and notebooks on every platform, Windows
included.

``n_jobs`` follows the scikit-learn convention: ``1`` (or ``None``) runs
serially in this process, ``-1`` uses every core, ``k`` uses k processes.
Results come back in task order, so parallel and serial runs are identical.

Details that keep a parallel run indistinguishable from a serial one:

* **Warning filters travel with the work.**  Worker processes start with
  Python's default filters, so a warning the caller has silenced would
  otherwise reappear once per worker.  Each task runs under the caller's
  filters.
* **Workers are reused.**  Starting a process on Windows means importing
  NumPy, SciPy and pandas afresh, a few seconds each; the pool is kept for
  later calls with the same number of workers, and replaced if it breaks.
* **No temporary files.**  joblib, the usual choice, memory-maps arrays to a
  temporary folder whose clean-up prints spurious tracebacks on Windows.
  Tasks here are small, so they are simply sent whole.
"""

from __future__ import annotations

import atexit
import os
import warnings
from concurrent.futures import ProcessPoolExecutor
from concurrent.futures.process import BrokenProcessPool

__all__ = ["run_parallel"]

#: Windows caps a process pool at 61 workers.
_MAX_WORKERS = 61
_pools = {}


def _under_filters(filters, function, *args, **kwargs):
    with warnings.catch_warnings():
        warnings.filters[:] = filters
        return function(*args, **kwargs)


def _pool(workers):
    pool = _pools.get(workers)
    if pool is None:
        pool = _pools[workers] = ProcessPoolExecutor(max_workers=workers)
    return pool


@atexit.register
def _shutdown():
    for pool in _pools.values():
        pool.shutdown(wait=False, cancel_futures=True)
    _pools.clear()


def run_parallel(function, tasks, n_jobs=1, **shared):
    """``[function(*task, **shared) for task in tasks]``, on ``n_jobs`` processes."""
    tasks = list(tasks)
    if n_jobs in (None, 1) or len(tasks) < 2:
        return [function(*task, **shared) for task in tasks]
    if not isinstance(n_jobs, int) or n_jobs == 0 or n_jobs < -1:
        raise ValueError("n_jobs must be None, -1 or a positive integer")

    cores = os.cpu_count() or 1
    workers = min(cores if n_jobs == -1 else n_jobs, _MAX_WORKERS)
    filters = list(warnings.filters)
    for attempt in (1, 2):
        pool = _pool(workers)
        try:
            futures = [pool.submit(_under_filters, filters, function, *task, **shared)
                       for task in tasks]
            return [f.result() for f in futures]
        except BrokenProcessPool:
            # A worker died (killed, out of memory): start a fresh pool once.
            _pools.pop(workers, None)
            if attempt == 2:
                raise
