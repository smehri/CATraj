import geopandas as gpd
import numpy as np
import pandas as pd
import pytest

from catraj.preprocessing import split_trips, trip_ids


def _track():
    """Sail 20 min, stay 40 min, sail 20 min, a 2-hour gap, sail 10 min; one fix a minute."""
    t, x = [], []
    clock, pos = 0.0, 0.0
    for kind, minutes in (("sail", 20), ("stay", 40), ("sail", 20), ("gap", 120), ("sail", 10)):
        if kind == "gap":
            clock += minutes * 60
            continue
        for _ in range(minutes):
            t.append(clock)
            x.append(pos)
            clock += 60
            pos += 300.0 if kind == "sail" else 0.0
    x = np.array(x)
    return x, np.zeros_like(x), np.array(t)


def test_trip_ids_split_at_stays_and_gaps():
    x, y, t = _track()
    trip, stay = trip_ids(x, y, t)
    assert (np.diff(trip) >= 0).all() and trip[0] == 0
    assert trip.max() == 3                      # sail | stay | sail | (gap) sail
    assert stay.sum() >= 40 and not stay[:19].any() and not stay[-10:].any()
    for k in np.unique(trip):                    # a piece is all stay or all movement
        assert len(set(stay[trip == k])) == 1


def test_trip_ids_rules_can_be_switched_off():
    x, y, t = _track()
    trip, stay = trip_ids(x, y, t, stay_radius=None, stay_duration=None)
    assert trip.max() == 1 and not stay.any()   # only the gap
    trip, _ = trip_ids(x, y, t, max_gap=None, stay_radius=None, stay_duration=None)
    assert trip.max() == 0
    trip, stay = trip_ids([], [], [])
    assert trip.size == 0 and stay.size == 0


def test_status_marks_stays():
    x, y, t = _track()
    status = np.zeros(x.size)
    status[:5] = 5                               # moored at the start
    trip, stay = trip_ids(x, y, t, stay_radius=None, stay_duration=None, status=status)
    assert stay[:5].all() and trip[5] == 1


def _fleet():
    x, y, t = _track()
    frames = [pd.DataFrame({"mmsi": m, "unix_time": t, "x": x + 1e5 * k, "y": y})
              for k, m in enumerate((227000001, 228000002, 229000003))]
    df = pd.concat(frames, ignore_index=True)
    return gpd.GeoDataFrame(df, geometry=gpd.points_from_xy(df.x, df.y), crs=32630)


def test_split_trips_per_vessel_and_in_parallel():
    fleet = _fleet()
    one = split_trips(fleet, by="mmsi", time_column="unix_time")
    many = split_trips(fleet, by="mmsi", time_column="unix_time", n_jobs=3)
    pd.testing.assert_frame_equal(one, many)
    assert list(one.columns) == list(fleet.columns) + ["trip", "stay"]
    assert (one.groupby("mmsi").trip.max() == 3).all()
    assert one.groupby(["mmsi", "trip"]).ngroups == 12


def test_split_trips_needs_a_unique_index():
    fleet = _fleet()
    with pytest.raises(ValueError):
        split_trips(fleet.set_index(np.zeros(len(fleet))), by="mmsi", time_column="unix_time")
