"""Reading field descriptions from a data dictionary.

Source datasets often ship a plain-text description of their columns, because
the column names alone are opaque: ``Ff`` is mean wind speed in m/s, ``Td`` is
dewpoint temperature, ``id_windDirection`` is a compass point rather than a
bearing in degrees.  Nothing here is required -- a dataset whose column names
already say what they mean needs no dictionary -- but where one exists it makes
the difference between choosing ``Ff`` and choosing "Ff -- mean wind speed
(m/s)".

The format is the one used by the sample AIS and weather archives, and by many
other public datasets:

    Attribute       Data type           Description
    ------------------------------------------------------------------
    *  Ff           integer             Mean wind speed ... (meters per second)
    *  Td           double precision    Dewpoint temperature ... (C)

Leading ``*``, whitespace-separated columns, and a units hint in trailing
parentheses.  Anything that does not parse is skipped rather than raised: a
dictionary is a convenience, and a malformed line should not stop the work.
"""

from __future__ import annotations

import re
from pathlib import Path

__all__ = ["FieldDescription", "parse_data_dictionary", "describe_columns"]

#: Columns are separated by a tab or by two or more spaces -- the files mix
#: both, and a short name like ``T`` is followed by a single space then tabs.
_SEP = r"(?:\s*	\s*|\s{2,})"

_ROW = re.compile(
    r"^\s*\*\s+"                       # the bullet each entry starts with
    r"(?P<name>[A-Za-z_][\w]*)" + _SEP   # column name, then a column break
    + r"(?P<dtype>[A-Za-z][A-Za-z ]*?):?" + _SEP   # declared type
    + r"(?P<description>\S.*?)\s*$"      # the description itself
)

#: Units are conventionally the last parenthesised phrase, e.g. "(km)" or
#: "(meters per seconds)".  Long clauses are not units, so they are ignored.
_UNITS = re.compile(r"\(([^()]{1,28})\)\s*\.?\s*$")

_UNIT_WORDS = (
    "meter", "metre", "second", "minute", "hour", "degree", "knot", "km",
    "mm", "%", "percent", "mercury", "celsius", "°", "m/s", "kn",
)


class FieldDescription:
    """What a data dictionary says about one column."""

    __slots__ = ("name", "dtype", "description", "units")

    def __init__(self, name, dtype, description, units=None):
        self.name = name
        self.dtype = dtype
        self.description = description
        self.units = units

    @property
    def label(self):
        """A short human label: the column name plus what it measures."""
        summary = self.description
        # The first clause carries the meaning; the rest is usually placement
        # detail ("at a height of 10-12 meters above the earth's surface").
        for separator in (" at a height", " at 2 meter", " during the",
                          " between the", " over the", ", i.e.", " reduced to"):
            cut = summary.find(separator)
            if cut > 0:
                summary = summary[:cut]
                break
        summary = summary.rstrip(" .,")
        if self.units and self.units.lower() not in summary.lower():
            summary = f"{summary} ({self.units})"
        return f"{self.name} - {summary}" if summary else self.name

    def __repr__(self):
        return f"FieldDescription({self.name!r}, units={self.units!r})"


def _units_of(description):
    match = _UNITS.search(description)
    if not match:
        return None
    candidate = match.group(1).strip()
    lowered = candidate.lower()
    if any(word in lowered for word in _UNIT_WORDS):
        return candidate
    return None


def parse_data_dictionary(path):
    """Parse a field-description file into ``{column: FieldDescription}``.

    Several dataset blocks may share one file -- the sample AIS archive
    documents its dynamic, static and SAR tables together -- and all of them
    are read.  Where
    a name appears twice the first description wins, since the leading block is
    the one the file is named for.

    A file that cannot be read, or that contains nothing parseable, yields an
    empty mapping rather than raising: the dictionary is optional.
    """
    # These files are often Latin-1 rather than UTF-8 -- a degree sign in
    # "Air temperature (C)" is the usual giveaway -- so try UTF-8 first and
    # fall back rather than mangling it into replacement characters.
    text = None
    for encoding in ("utf-8", "cp1252", "latin-1"):
        try:
            text = Path(path).read_text(encoding=encoding)
            break
        except UnicodeDecodeError:
            continue
        except OSError:
            return {}
    if text is None:
        return {}

    fields = {}
    for line in text.splitlines():
        match = _ROW.match(line)
        if not match:
            continue
        name = match.group("name")
        if name in fields:
            continue
        description = match.group("description").strip()
        fields[name] = FieldDescription(
            name=name,
            dtype=match.group("dtype").strip().rstrip(":"),
            description=description,
            units=_units_of(description),
        )
    return fields


def describe_columns(columns, dictionary, truncated=10):
    """Label each column, falling back to the bare name when unknown.

    Shapefiles truncate field names to ten characters, so a dictionary entry
    for ``id_windDirection`` has to match a column called ``id_windDir``.  That
    prefix match is tried only when the exact name is absent.

    Returns ``{column: label}`` with an entry for every column given, so a
    caller can use it as a straight lookup.
    """
    labels = {}
    for column in columns:
        field = dictionary.get(column)
        if field is None and truncated:
            for name, candidate in dictionary.items():
                if name[:truncated] == column[:truncated] and len(name) > len(column):
                    field = candidate
                    break
        labels[column] = field.label if field else column
    return labels
