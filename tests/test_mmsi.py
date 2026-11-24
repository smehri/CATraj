import pandas as pd
import pytest

from catraj.preprocessing import mmsi_class, mmsi_classes


@pytest.mark.parametrize("mmsi, expected", [
    (227705102, "ship"),                                  # France, MID 227
    (229000000 + 12345, "ship"),                          # Malta, missing from the archive's list
    (217000001, "ship, unallocated MID"),                 # MID 217 is not allocated
    (123456789, "placeholder"),
    (111111111, "placeholder"),
    (2275000, "coast station"),                           # 002275000
    (22750001, "group of ships"),                         # 022750001
    (992271234, "aid to navigation"),
    (982271234, "craft of a parent ship"),
    (111227123, "SAR aircraft"),
    (970123456, "distress device (SART/MOB/EPIRB)"),
    (812345678, "handheld"),
    (1000000000, "not nine digits"),
    (0, "not nine digits"),
    (912345678, "unassigned pattern"),
])
def test_mmsi_class(mmsi, expected):
    assert mmsi_class(mmsi) == expected


def test_mmsi_classes_aligned():
    s = pd.Series([227705102, 123456789, 227705102], index=[10, 20, 30])
    out = mmsi_classes(s)
    assert list(out.index) == [10, 20, 30]
    assert list(out) == ["ship", "placeholder", "ship"]
