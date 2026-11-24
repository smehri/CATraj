"""MMSI check by ITU-R M.585.

Every MMSI has nine digits. A ship station's is ``MIDxxxxxx``, its first three
digits an allocated Maritime Identification Digit (MID, the flag state). Other
patterns belong to coast stations (``00MID...``), groups of ships
(``0MID...``), SAR aircraft (``111MID...``), aids to navigation (``99MID...``),
craft of a parent ship (``98MID...``), distress devices (``970/972/974...``)
and handhelds (``8...``). The MIDs below are the ITU's own table
(itu.int/gladapp/Allocation/MIDs, 2022); the Zenodo archive's country list is
incomplete and misses Malta 229, Panama 373-374, Singapore 566 and others.

An MMSI stored as an integer loses its leading zeros, so ``2275000`` is the
coast station ``002275000``; only values above 999,999,999 or at most 0 are
"not nine digits".
"""

from __future__ import annotations

import pandas as pd

__all__ = ["ITU_MIDS", "mmsi_class", "mmsi_classes"]

ITU_MIDS = frozenset(m for part in (
    "201-216,218-220,224-259,261-279,301,303-312,314,316,319,321,323,325,327,329-332,"
    "334,336,338-339,341,343,345,347-348,350-359,361-362,364,366-379,401,403,405,408,"
    "410,412-414,416-417,419,422-423,425,428,431-432,434,436-438,440-441,443,445,447,"
    "450-451,453,455,457,459,461,463,466,468,470-473,475,477-478,501,503,506,508,"
    "510-512,514-516,518,520,523,525,529,531,533,536,538,540,542,544,546,548,550,553,"
    "555,557,559,561,563-567,570,572,574,576-578,601,603,605,607-613,615-622,624-627,"
    "629-638,642,644-645,647,649-650,654-657,659-672,674-679,701,710,720,725,730,735,"
    "740,745,750,755,760,765,770,775").split(",")
    for m in range(int(part.split("-")[0]), int(part.split("-")[-1]) + 1))

_DISTRESS = "distress device (SART/MOB/EPIRB)"
# prefix -> (class, position of the MID; None where the pattern has none), first match wins
_PATTERNS = [("111", "SAR aircraft", 3), ("970", _DISTRESS, None), ("972", _DISTRESS, None),
             ("974", _DISTRESS, None), ("99", "aid to navigation", 2),
             ("98", "craft of a parent ship", 2), ("00", "coast station", 2),
             ("0", "group of ships", 1), ("8", "handheld", None)]
_PATTERNS += [(d, "ship", 0) for d in "234567"]


def mmsi_class(mmsi) -> str:
    """The ITU-R M.585 class of one MMSI, checking its MID wherever it has one.

    Returns ``"ship"`` for a ship station with an allocated MID; otherwise one
    of ``"not nine digits"``, ``"placeholder"`` (one repeated digit,
    123456789, 012345678), the class of another pattern, that class followed
    by ``", unallocated MID"``, or ``"unassigned pattern"``.
    """
    mmsi = int(mmsi)
    if not 0 < mmsi <= 999_999_999:
        return "not nine digits"
    s = f"{mmsi:09d}"                      # stored without its leading zeros
    if len(set(s)) == 1 or s in ("123456789", "012345678"):
        return "placeholder"
    for prefix, kind, at in _PATTERNS:
        if s.startswith(prefix):
            if at is not None and int(s[at:at + 3]) not in ITU_MIDS:
                return f"{kind}, unallocated MID"
            return kind
    return "unassigned pattern"


def mmsi_classes(mmsi) -> pd.Series:
    """:func:`mmsi_class` of every value of ``mmsi``, aligned with it.

    Each distinct MMSI is classified once, so millions of positions are fine.
    """
    mmsi = pd.Series(mmsi) if not isinstance(mmsi, pd.Series) else mmsi
    unique = pd.unique(mmsi.to_numpy())
    table = pd.Series([mmsi_class(m) for m in unique], index=unique, dtype=object)
    return pd.Series(table.reindex(mmsi.to_numpy()).to_numpy(), index=mmsi.index,
                     name="mmsi_class")
