"""catraj._parallel: n_jobs gives exactly the result of the plain loop."""

import pickle

import geopandas as gpd
import numpy as np
import pandas as pd
import pytest
import shapely

from catraj.compression import (Barriers, compress, cpls_compress, segment_crossings,
                                two_stage_cpls_compress)
from catraj._parallel import _chunks, map_groups, parallel_map, resolve_n_jobs


def _fleet(n_vessels=7, seed=0):
    """Random walks of different lengths, one per vessel, and barriers on them."""
    rng = np.random.default_rng(seed)
    frames, barriers = [], []
    for v in range(n_vessels):
        n = int(rng.integers(40, 300))
        xy = np.cumsum(rng.normal(scale=50.0, size=(n, 2)) + [30.0, 5.0], axis=0)
        xy += [0.0, 20_000.0 * v]
        frames.append(pd.DataFrame({"vid": v, "x": xy[:, 0], "y": xy[:, 1],
                                    "t": 1.45e9 + 30.0 * np.arange(n),
                                    "speed": rng.uniform(0, 10, n)}))
        for i in rng.integers(0, n, 6):
            barriers.append(shapely.Point(xy[i] + rng.normal(scale=20, size=2))
                            .buffer(rng.uniform(20, 120)))
    df = pd.concat(frames, ignore_index=True)
    gdf = gpd.GeoDataFrame(df, geometry=gpd.points_from_xy(df.x, df.y), crs=32630)
    return gdf, Barriers(gpd.GeoSeries(barriers, crs=32630))


def test_resolve_n_jobs():
    assert resolve_n_jobs(None) == 1
    assert resolve_n_jobs(1) == 1
    assert resolve_n_jobs(8, n_items=3) == 3
    assert resolve_n_jobs(-1) >= 1
    assert resolve_n_jobs(4, n_items=0) == 1
    with pytest.raises(ValueError):
        resolve_n_jobs(0)


def test_chunks_cover_every_item_once_and_balance():
    sizes = [100, 1, 1, 1, 50, 50, 3, 7]
    chunks = _chunks(sizes, 3)
    assert sorted(i for c in chunks for i in c) == list(range(len(sizes)))
    totals = sorted(sum(sizes[i] for i in c) for c in chunks)
    assert totals[-1] == 100          # the largest item alone, nothing added to it


def test_parallel_map_keeps_the_order():
    items = list(range(37))
    assert parallel_map(lambda v: v * v, items, n_jobs=3,
                        sizes=[(i * 7) % 11 for i in items]) == [v * v for v in items]
    assert parallel_map(lambda v: v, [], n_jobs=3) == []
    with pytest.raises(ValueError):
        parallel_map(abs, [1, 2], n_jobs=2, sizes=[1])


def test_barriers_survive_pickling():
    _, barriers = _fleet()
    again = pickle.loads(pickle.dumps(barriers))
    assert len(again) == len(barriers) and again.crs == barriers.crs
    x0, y0 = np.array([0.0, 500.0]), np.array([0.0, 40_000.0])
    x1, y1 = x0 + 3000, y0 + 1000
    assert (again.hits(x0, y0, x1, y1) == barriers.hits(x0, y0, x1, y1)).all()


def test_cpls_compress_in_parallel_is_the_same():
    gdf, barriers = _fleet()
    one = cpls_compress(gdf, 150.0, barriers, by="vid", time_column="t")
    many = cpls_compress(gdf, 150.0, barriers, by="vid", time_column="t", n_jobs=3)
    pd.testing.assert_frame_equal(one, many)
    off = dict(avoid_self_intersection=False, keep_self_intersection=False, inversion=False)
    pd.testing.assert_frame_equal(
        cpls_compress(gdf, 150.0, barriers, by="vid", filter_window=None, **off),
        cpls_compress(gdf, 150.0, barriers, by="vid", filter_window=None, n_jobs=2, **off))


def test_compress_in_parallel_is_the_same():
    gdf, _ = _fleet()
    args = dict(thresholds={"speed": 2.0}, time_column="t", by="vid")
    pd.testing.assert_frame_equal(compress(gdf, 150.0, **args),
                                  compress(gdf, 150.0, n_jobs=3, **args))


def test_two_stage_in_parallel_is_the_same():
    gdf, barriers = _fleet()
    args = dict(time_column="t", by="vid")
    pd.testing.assert_frame_equal(two_stage_cpls_compress(gdf, 150.0, 1.0, barriers, **args),
                                  two_stage_cpls_compress(gdf, 150.0, 1.0, barriers,
                                                          n_jobs=3, **args))


def test_map_groups_with_a_notebook_style_closure():
    gdf, barriers = _fleet()
    count = lambda g: int(segment_crossings(g.x.to_numpy(), g.y.to_numpy(), barriers).sum())  # noqa: E731
    assert map_groups(count, gdf, "vid", n_jobs=3) == map_groups(count, gdf, "vid")
