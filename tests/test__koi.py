import numpy as np
import pandas as pd
import pytest
from testfixtures import LogCapture
from exo_mercat.koi import Koi


@pytest.fixture
def instance():
    return Koi()


def test__init(instance):
    assert instance.data is None
    assert instance.name is "koi"


def test__uniform_catalog(instance):
    data = {
        "kepid": [10797460, 10811496],
        "kepoi_name": ["K00752.02", "K00753.01"],
        "kepler_name": ["Kepler-227 c", "nan"],
        "koi_disposition": ["CONFIRMED", "CANDIDATE"],
        "ra_str": ["19h27m44.22s", "19h48m01.16s"],
        "dec_str": ["+48d08m29.9s", "+48d08m02.9s"],
    }
    data = pd.DataFrame(data)
    instance.data = data

    with LogCapture() as log:
        instance.uniform_catalog()
        assert "Catalog uniformed" in log.actual()[0][-1]

    assert list(instance.data.columns) == [
        "name",
        "alias",
        "aliasplanet",
        "disposition",
        "discoverymethod",
        "letter",
        "ra",
        "dec"
    ]

    assert (
        "KOI-752," in instance.data.at[0, "alias"]
    )  # KOI-752,KIC 10797460,Kepler-227,
    assert "nan" not in instance.data.at[0, "alias"]
    assert data.at[0, "kepler_name"].rstrip(" bcdefghi") in instance.data.at[0, "alias"]
    assert (
        data.at[0, "kepler_name"].rstrip(" bcdefghi")
        in instance.data.at[0, "alias"]
    )
    assert instance.data.at[0, "letter"] == "c"

    assert (
        "KOI-752.02" in instance.data.at[0, "aliasplanet"]
    )  # 'KOI-752.02,KIC 10797460 c,Kepler-227 c,'
    assert "nan" not in instance.data.at[0, "aliasplanet"]
    assert data.at[0, "kepler_name"] in instance.data.at[0, "aliasplanet"]

    assert "KOI-753," in instance.data.at[1, "alias"]  # KOI-753,KIC 10811496,
    assert "nan" not in instance.data.at[1, "alias"]

    assert instance.data.at[1, "letter"] == ".01"

    assert (
        "KOI-753.01," in instance.data.at[1, "aliasplanet"]
    )  # KIC 10811496.01,KOI-753.01,
    assert "nan" not in instance.data.at[1, "aliasplanet"]

    assert data.at[0, "ra_str"] in instance.data.at[0, "ra"]
    assert data.at[1, "dec_str"] in instance.data.at[1, "dec"]
    assert data.at[1, "koi_disposition"] in instance.data.at[1, "disposition"]
    assert "Transit" in instance.data.at[0, "discoverymethod"]


def test_convert_coordinates(instance):
    data = {
        "kepid": [10797460, 10811496],
        "kepoi_name": ["K00752.02", "K00753.01"],
        "kepler_name": ["Kepler-227 c", "nan"],
        "koi_disposition": ["CONFIRMED", "CANDIDATE"],
        "ra": ["19h27m44.22s", np.nan],
        "dec": ["+48d08m29.9s", ""],
    }
    data = pd.DataFrame(data)
    instance.data = data
    with LogCapture() as log:
        instance.convert_coordinates()
        assert "Converted coordinates" in log.actual()[0][-1]

    assert np.isclose(instance.data.at[0, "ra"], 291.9333)
    assert np.isclose(instance.data.at[0, "dec"], 48.1417)
    assert np.isnan(instance.data.at[1, "ra"])
    assert np.isnan(instance.data.at[1, "dec"])
