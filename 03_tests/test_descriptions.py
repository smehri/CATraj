import pytest

from catraj.annotation import describe_columns, parse_data_dictionary

from helpers import DATASETS

DICTIONARY = """\
Attribute       Data type           Description
------------------------------------------------------------------
*  Ff\t\tinteger             Mean wind speed at a height of 10-12 metres (meters per seconds)
*  T \t\tdouble precision    Air temperature at 2 metre height (°C)
*  id_windDirection   integer   Mean wind direction, compass point code
*  Ff\t\tinteger             a later duplicate that must not win
this line is not an entry
"""


@pytest.fixture
def dictionary(tmp_path):
    path = tmp_path / "params.txt"
    path.write_text(DICTIONARY, encoding="cp1252")  # degree sign, not UTF-8
    return parse_data_dictionary(path)


def test_entries_are_parsed(dictionary):
    assert set(dictionary) == {"Ff", "T", "id_windDirection"}
    assert dictionary["Ff"].dtype == "integer"
    assert dictionary["Ff"].units == "meters per seconds"
    assert dictionary["T"].units == "°C"


def test_first_description_wins(dictionary):
    assert dictionary["Ff"].description.startswith("Mean wind speed")


def test_label_keeps_the_meaning_and_drops_placement(dictionary):
    assert dictionary["Ff"].label == "Ff - Mean wind speed (meters per seconds)"


def test_truncated_shapefile_names_match_by_prefix(dictionary):
    labels = describe_columns(["Ff", "id_windDir", "unknown"], dictionary)
    assert labels["id_windDir"].startswith("id_windDirection - ")
    assert labels["unknown"] == "unknown"


def test_unreadable_file_is_an_empty_mapping(tmp_path):
    assert parse_data_dictionary(tmp_path / "missing.txt") == {}


@pytest.mark.data
def test_sample_weather_dictionary():
    fields = parse_data_dictionary(DATASETS / "context" / "Weather" / "Weather_parameters.txt")
    assert {"Ff", "T", "Td"} <= set(fields)
