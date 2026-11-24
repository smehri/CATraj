"""Running independent pieces of work on several cores.

Epochs of a cross-validation, time blocks of an annotation, and the vessels
(or trips) of a compression are independent, so they can be spread over
processes. The standard library's ``concurrent.futures.ProcessPoolExecutor``
does the work, and it behaves the same from scripts and notebooks on every
platform, Windows included.

Two entry points:

* :func:`run_parallel` -- ``[function(*task, **shared) for task in tasks]``,
  for a module-level ``function`` and small tasks, one submission per task.
* :func:`parallel_map` -- ``[func(item) for item in items]`` for any
  callable, a closure or a function defined in a notebook included: it is
  pickled with cloudpickle.  The items are grouped into at most
  ``4 * n_jobs`` chunks balanced by ``sizes`` (largest first, each to the
  lightest chunk), so whatever ``func`` refers to -- the barriers of CPLS,
  typically -- is sent once per chunk, not once per item, and no worker is
  left alone with the longest tracks at the end.  :func:`map_groups` applies
  it to the groups of a dataframe; the ``n_jobs`` argument of the
  compression functions and of :func:`catraj.preprocessing.split_trips` uses
  it.  The public name is :func:`catraj.parallel_map`.

``n_jobs`` follows the scikit-learn convention: ``1`` (or ``None``) runs
serially in this process, ``-1`` uses every core, ``k`` uses k processes.
Results come back in the order of the tasks, so parallel and serial runs are
identical.

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
  The work here is simply sent whole.
"""

from __future__ import annotations

import atexit
import heapq
import os
import warnings
from concurrent.futures import ProcessPoolExecutor
from concurrent.futures.process import BrokenProcessPool

__all__ = ["run_parallel", "parallel_map", "map_groups", "resolve_n_jobs"]

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


def resolve_n_jobs(n_jobs, n_items=None):
    """The number of processes ``n_jobs`` asks for.

    ``None`` or ``1``: one, no parallelism; ``-1``: every core; ``k``: k.  At
    most 61 (the Windows limit), and at most ``n_items`` when that is given;
    never less than 1.
    """
    if n_jobs is None:
        return 1
    if not isinstance(n_jobs, int) or n_jobs == 0 or n_jobs < -1:
        raise ValueError("n_jobs must be None, -1 or a positive integer")
    workers = min((os.cpu_count() or 1) if n_jobs == -1 else n_jobs, _MAX_WORKERS)
    if n_items is not None:
        workers = min(workers, max(1, n_items))
    return workers


def _submit_all(workers, calls):
    """Results of ``function(*args, **kwargs)`` for every call, in order."""
    for attempt in (1, 2):
        pool = _pool(workers)
        try:
            futures = [pool.submit(function, *args, **kwargs) for function, args, kwargs in calls]
            return [f.result() for f in futures]
        except BrokenProcessPool:
            # A worker died (killed, out of memory): start a fresh pool once.
            _pools.pop(workers, None)
            if attempt == 2:
                raise


def run_parallel(function, tasks, n_jobs=1, **shared):
    """``[function(*task, **shared) for task in tasks]``, on ``n_jobs`` processes."""
    tasks = list(tasks)
    if n_jobs in (None, 1) or len(tasks) < 2:
        return [function(*task, **shared) for task in tasks]
    workers = resolve_n_jobs(n_jobs)
    filters = list(warnings.filters)
    return _submit_all(workers, [(_under_filters, (filters, function, *task), shared)
                                 for task in tasks])


def _chunks(sizes, n_chunks):
    """Indices of the items, split into ``n_chunks`` lists of about equal total size."""
    heap = [(0, c) for c in range(n_chunks)]
    out = [[] for _ in range(n_chunks)]
    for i in sorted(range(len(sizes)), key=lambda i: -sizes[i]):
        total, c = heapq.heappop(heap)
        out[c].append(i)
        heapq.heappush(heap, (total + sizes[i], c))
    return [c for c in out if c]


def _run_chunk(filters, payload):
    import cloudpickle

    func, items = cloudpickle.loads(payload)
    with warnings.catch_warnings():
        warnings.filters[:] = filters
        return [func(item) for item in items]


def parallel_map(func, items, n_jobs=1, sizes=None):
    """``[func(item) for item in items]``, on ``n_jobs`` processes.

    Parameters
    ----------
    func : callable
        Called once per item.  It, and everything it refers to, must pickle
        with cloudpickle (closures and notebook functions do); its result
        must pickle too.
    items : iterable
        One piece of work each, independent of the others.
    n_jobs : int or None
        Processes to use, as in :func:`resolve_n_jobs`; ``1`` (the default)
        or ``None`` runs the plain loop in this process.
    sizes : sequence of float, optional
        The expected cost of each item (the number of positions of a vessel,
        say), for balancing the chunks; all equal when omitted.

    Returns
    -------
    list
        The results, in the order of ``items``.
    """
    items = list(items)
    if sizes is not None and len(sizes) != len(items):
        raise ValueError(f"{len(sizes)} sizes for {len(items)} items")
    if n_jobs in (None, 1) or len(items) < 2:
        return [func(item) for item in items]
    workers = resolve_n_jobs(n_jobs)

    import cloudpickle

    chunks = _chunks(list(sizes) if sizes is not None else [1] * len(items),
                     min(len(items), 4 * workers))
    filters = list(warnings.filters)
    calls = [(_run_chunk, (filters, cloudpickle.dumps((func, [items[i] for i in chunk]))), {})
             for chunk in chunks]
    out = [None] * len(items)
    for chunk, res in zip(chunks, _submit_all(workers, calls)):
        for i, r in zip(chunk, res):
            out[i] = r
    return out


def map_groups(func, frame, by, n_jobs=1):
    """``func`` of every group of ``frame.groupby(by)``, in group order.

    The groups are balanced by their number of rows.
    """
    groups = [g for _, g in frame.groupby(by, sort=False)]
    return parallel_map(func, groups, n_jobs, sizes=[len(g) for g in groups])
