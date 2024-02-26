import numpy as np
import pandas as pd
import pytest
from testfixtures import LogCapture

from exo_mercat.epic import Epic


@pytest.fixture
def instance():
    return Epic()


def test__init(instance):
    assert instance.data is None
    assert instance.name is "epic"


def test__uniform_catalog(instance):
    data = {
        "pl_name": ["EPIC 212099230.01", "extra"],
        "discoverymethod": ["Transit", "extra"],
        "pl_orbper": [7.11047482, 0],
        "pl_orbpererr2": [np.nan, np.nan],
        "pl_orbpererr1": [np.nan, np.nan],
        "pl_orbsmax": [np.nan, np.nan],
        "pl_orbsmaxerr2": [np.nan, np.nan],
        "pl_orbsmaxerr1": [np.nan, np.nan],
        "pl_orbeccen": [np.nan, np.nan],
        "pl_orbeccenerr2": [np.nan, np.nan],
        "pl_orbeccenerr1": [np.nan, np.nan],
        "pl_orbincl": [np.nan, np.nan],
        "pl_orbinclerr2": [np.nan, np.nan],
        "pl_orbinclerr1": [np.nan, np.nan],
        "pl_radj": [np.nan, np.nan],
        "pl_radjerr2": [np.nan, np.nan],
        "pl_radjerr1": [np.nan, np.nan],
        "disc_year": [2016, np.nan],
        "rv_flag": [0, 0],
        "tran_flag": [1, 0],
        "ttv_flag": [0, 0],
        "pl_massj": [np.nan, np.nan],
        "pl_massjerr2": [np.nan, np.nan],
        "pl_massjerr1": [np.nan, np.nan],
        "pl_msinij": [np.nan, np.nan],
        "pl_msinijerr2": [np.nan, np.nan],
        "pl_msinijerr1": [np.nan, np.nan],
        "hostname": ["EPIC 212099230", ""],
        "st_age": [np.nan, np.nan],
        "st_ageerr1": [np.nan, np.nan],
        "st_ageerr2": [np.nan, np.nan],
        "st_mass": [0.99, np.nan],
        "st_masserr1": [0.14, np.nan],
        "st_masserr2": [-0.11, np.nan],
        "pl_refname": [
            "<a refstr=BARROS_ET_AL__2016 href=https://ui.adsabs.harvard.edu/abs/2016A&A...594A.100B/abstract target=ref>Barros et al. 2016</a>",
            np.nan,
        ],
        "k2_name": ["", np.nan],
        "default_flag": [1, 0],
        "pl_letter": ["", np.nan],
        "hd_name": ["", np.nan],
        "hip_name": ["nan", np.nan],
        "tic_id": ["TIC 178266267", np.nan],
        "gaia_id": ["Gaia DR2 665640392382991360", np.nan],
    }

    data = pd.DataFrame(data)
    instance.data = data

    expected_columns = [
        "catalog",
        "catalog_name",
        "catalog_host",
        "default_flag",
        "name",
        "discovery_method",
        "p",
        "p_min",
        "p_max",
        "a",
        "a_min",
        "a_max",
        "e",
        "e_min",
        "e_max",
        "i",
        "i_min",
        "i_max",
        "r",
        "r_min",
        "r_max",
        "discovery_year",
        "RV",
        "Transit",
        "TTV",
        "mass",
        "mass_min",
        "mass_max",
        "msini",
        "msini_min",
        "msini_max",
        "host",
        "Age (Gyrs)",
        "Age_max",
        "Age_min",
        "Mstar",
        "Mstar_max",
        "Mstar_min",
        "reference",
        "Kepler_host",
        "letter",
        "hd_name",
        "hip_name",
        "tic_id",
        "gaia_id",
        "pl_letter",
        "k2_name",
        "alias",
    ]
    with LogCapture() as log:
        instance.uniform_catalog()
        assert "Catalog uniformed" in log.actual()[0][-1]

    assert sorted(list(instance.data.columns)) == sorted(expected_columns)
    assert len(instance.data) == 1
    assert data.at[0, "gaia_id"] + "," in instance.data.at[0, "alias"]
    assert "nan" not in instance.data.at[0, "alias"]

    assert (data.at[0, "gaia_id"] + ",") in instance.data.at[0, "alias"] and (
            data.at[0, "tic_id"] + ","
    ) in instance.data.at[0, "alias"]
    assert ".01" in instance.data.at[0, "letter"]


def test__convert_coordinates(instance):
    assert instance.convert_coordinates() is None


def test__remove_theoretical_masses(instance):
    assert instance.remove_theoretical_masses() is None


def test__handle_reference_format(instance):
    data = {
        "name": ["TOI-942 b", "nan"],
        "reference": [
            "<a refstr=CARLEO_ET_AL__2021 href=https://ui.adsabs.harvard.edu/abs/2021A&A...645A..71C/abstract target=ref>Carleo et al. 2021</a>",
            "<a refstr=CARLEO_ET_AL__2021 href=https://ui.adsabs.harvard.edu/abs/2021A&A...645A..71C/abstract target=ref>Carleo et al. 2021</a>",
        ],
        "p": [4.32419, 2.2],
        "p_url": [
            "<a refstr=CARLEO_ET_AL__2021 href=https://ui.adsabs.harvard.edu/abs/2021A&A...645A..72C/abstract target=ref>Carleonew et al. 2021</a>",
            np.nan,
        ],
        "e": [0, np.nan],
        "a": [np.nan, np.nan],
        "a_url": [
            "<a refstr=CARLEO_ET_AL__2021 href=https://ui.adsabs.harvard.edu/abs/2021A&A...645A..72C/abstract target=ref>Carleonew et al. 2021</a>",
            "<a refstr=CARLEO_ET_AL__2021 href=https://ui.adsabs.harvard.edu/abs/2021A&A...645A..72C/abstract target=ref>Carleonew et al. 2021</a>",
        ],
        "i": [89.97, np.nan],
        "r": [0.429, np.nan],
        "mass": [2.6, np.nan],
        "msini": [np.nan, np.nan],
    }
    instance.data = pd.DataFrame(data)
    with LogCapture() as log:
        instance.handle_reference_format()
        assert "Reference columns uniformed" in log.actual()[0][-1]
    assert instance.data.at[0, "e_url"] == "2021A&A...645A..71C"
    assert instance.data.at[0, "p_url"] == "2021A&A...645A..72C"
    assert instance.data.at[0, "i_url"] == "2021A&A...645A..71C"
    assert instance.data.at[0, "r_url"] == "2021A&A...645A..71C"
    assert instance.data.at[0, "mass_url"] == "2021A&A...645A..71C"
    assert instance.data.at[0, "a_url"] == ""
    assert instance.data.at[0, "msini_url"] == ""
    assert instance.data.at[1, "e_url"] == ""
    assert instance.data.at[1, "p_url"] == ""  # originally nan
    assert instance.data.at[1, "i_url"] == ""
    assert (
            instance.data.at[1, "a_url"] == ""
    )  # despite being a_url originally non null, but the value is null
    assert instance.data.at[1, "mass_url"] == ""
    assert instance.data.at[1, "msini_url"] == ""


def test__assign_status(instance):
    # Create a sample DataFrame with some additional columns
    data = {"name": ["EPIC 212099230.01"], "disposition": ["CANDIDATE"]}
    df = pd.DataFrame(data)
    instance.data = df
    with LogCapture() as log:
        instance.assign_status()
        assert "Status column assigned" in log.actual()[0][-1]
        assert list(set(instance.data.status.values)) == ["CANDIDATE"]
