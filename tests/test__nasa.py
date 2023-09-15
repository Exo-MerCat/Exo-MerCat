import numpy as np
import pandas as pd
import pytest
from testfixtures import LogCapture

from exo_mercat.nasa import Nasa


@pytest.fixture
def instance():
    return Nasa()


def test__init(instance):
    assert instance.data is None
    assert instance.name is "nasa"


def test__uniform_catalog(instance):
    # Create a sample DataFrame with some additional columns
    data = {
        "pl_name": ["OGLE-2016-BLG-1227L b"],
        "discoverymethod": ["Microlensing"],
        "pl_orbper": [np.nan],
        "pl_orbpererr2": [np.nan],
        "pl_orbpererr1": [np.nan],
        "pl_orbsmax": [3.4],
        "pl_orbsmaxerr2": [-1.0],
        "pl_orbsmaxerr1": [2.1],
        "pl_orbeccen": [np.nan],
        "pl_orbeccenerr2": [np.nan],
        "pl_orbeccenerr1": [np.nan],
        "pl_orbincl": [np.nan],
        "pl_orbinclerr2": [np.nan],
        "pl_orbinclerr1": [np.nan],
        "pl_radj": [1.24],
        "pl_radjerr2": [np.nan],
        "pl_radjerr1": [np.nan],
        "disc_year": [2020],
        "disc_refname": [
            "<a refstr=HAN_ET_AL__2020 href=https://ui.adsabs.harvard.edu/abs/2020AJ....159...91H/abstract target=ref>Han et al. 2020</a>"
        ],
        "rv_flag": [0],
        "tran_flag": [0],
        "ttv_flag": [0],
        "pl_bmassj": [0.79],
        "pl_bmassjerr2": [-0.39],
        "pl_bmassjerr1": [1.3],
        "pl_bmassprov": ["Mass"],
        "hostname": ["OGLE-2016-BLG-1227L"],
        "st_age": [np.nan],
        "st_ageerr1": [np.nan],
        "st_ageerr2": [np.nan],
        "st_mass": [0.1],
        "st_masserr1": [0.17],
        "st_masserr2": [-0.05],
        "pl_radj_reflink": [
            "<a refstr=CALCULATED_VALUE href=/docs/composite_calc.html target=_blank>Calculated Value</a>"
        ],
        "pl_orbeccen_reflink": [np.nan],
        "pl_orbsmax_reflink": [
            "<a refstr=HAN_ET_AL__2020 href=https://ui.adsabs.harvard.edu/abs/2020AJ....159...91H/abstract target=ref>Han et al. 2020</a>"
        ],
        "pl_orbper_reflink": [np.nan],
        "pl_orbincl_reflink": [np.nan],
        "pl_bmassj_reflink": [
            "<a refstr=HAN_ET_AL__2020 href=https://ui.adsabs.harvard.edu/abs/2020AJ....159...91H/abstract target=ref>Han et al. 2020</a>"
        ],
        "hd_name": ["hd_name"],
        "hip_name": ["hip_name"],
        "tic_id": ["tic_id,second_id"],
        "gaia_id": [np.nan],
        "method_column": ["Transit Timing Variations"],
        "method_column1": ["Eclipse Timing Variations"],
        "method_column2": ["Pulsation Timing Variations"],
        "method_column3": ["Orbital Brightness Modulation"],
        "method_column4": ["Disk Kinematics"],
    }

    df = pd.DataFrame(data)
    instance.data = df
    with LogCapture() as log:
        instance.uniform_catalog()
        assert "Catalog uniformed" in log.actual()[0][-1]

    expected_columns = [
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
        "reference",
        "RV",
        "Transit",
        "TTV",
        "bestmass",
        "bestmass_min",
        "bestmass_max",
        "bestmass_provenance",
        "host",
        "Age (Gyrs)",
        "Age_max",
        "Age_min",
        "Mstar",
        "Mstar_max",
        "Mstar_min",
        "r_url",
        "e_url",
        "a_url",
        "p_url",
        "i_url",
        "bestmass_url",
    ]

    assert all(element in instance.data.columns for element in expected_columns)
    assert instance.data.at[0, "mass"] == data["pl_bmassj"]
    assert np.isnan(instance.data.at[0, "msini"])

    assert instance.data.at[0, "alias"] == "hd_name,hip_name,tic_id,second_id,"
    assert instance.data.at[0, "method_column"] == "TTV"
    assert instance.data.at[0, "method_column1"] == "TTV"
    assert instance.data.at[0, "method_column2"] == "Pulsar Timing"
    assert instance.data.at[0, "method_column3"] == "Other"
    assert instance.data.at[0, "method_column4"] == "Other"
    assert instance.data.at[0, "discovery_method"] == "Microlensing"


def test__sort_best_mass_to_mass_or_msini(instance):
    data = {
        "name": ["Mass_planet", "Msini_planet", "TheoreticalMass"],
        "bestmass": [1.0, 22.0, 10],
        "bestmass_max": [0.2, 0.4, 1.1],
        "bestmass_min": [0.3, 0.5, 1.1],
        "bestmass_url": ["http://somepaper.com", "http://someotherpaper.com", np.nan],
        "bestmass_provenance": ["Mass", "Msini", "M-R relationship"],
        "mass": [np.nan, np.nan, np.nan],
        "msini": [np.nan, np.nan, np.nan],
        "mass_min": [np.nan, np.nan, np.nan],
        "mass_max": [np.nan, np.nan, np.nan],
        "msini_min": [np.nan, np.nan, np.nan],
        "msini_max": [np.nan, np.nan, np.nan],
    }

    instance.data = pd.DataFrame(data)

    instance.sort_bestmass_to_mass_or_msini()
    assert instance.data.at[0, "mass"] == instance.data.at[0, "bestmass"]
    assert instance.data.at[0, "mass_min"] == instance.data.at[0, "bestmass_min"]
    assert instance.data.at[0, "mass_max"] == instance.data.at[0, "bestmass_max"]
    assert np.isnan(instance.data.at[0, "msini"])
    assert np.isnan(instance.data.at[0, "msini_min"])
    assert np.isnan(instance.data.at[0, "msini_max"])
    assert instance.data.at[1, "msini"] == instance.data.at[1, "bestmass"]
    assert instance.data.at[1, "msini_min"] == instance.data.at[1, "bestmass_min"]
    assert instance.data.at[1, "msini_max"] == instance.data.at[1, "bestmass_max"]
    assert np.isnan(instance.data.at[1, "mass"])
    assert np.isnan(instance.data.at[1, "mass_min"])
    assert np.isnan(instance.data.at[1, "mass_max"])

    assert np.isnan(instance.data.at[2, "mass"])
    assert np.isnan(instance.data.at[2, "mass_min"])
    assert np.isnan(instance.data.at[2, "mass_max"])
    assert np.isnan(instance.data.at[2, "msini"])
    assert np.isnan(instance.data.at[2, "msini_min"])
    assert np.isnan(instance.data.at[2, "msini_max"])
    data = {
        "name": ["Error_planet"],
        "bestmass": [1.0],
        "bestmass_max": [0.2],
        "bestmass_min": [0.3],
        "bestmass_url": [
            "http://somepaper.com",
        ],
        "bestmass_provenance": ["ExtraLabel"],
        "mass": [np.nan],
        "msini": [np.nan],
        "mass_min": [np.nan],
        "mass_max": [np.nan],
        "msini_min": [np.nan],
        "msini_max": [np.nan],
    }

    instance.data = pd.DataFrame(data)

    with pytest.raises(RuntimeError):
        instance.sort_bestmass_to_mass_or_msini()


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
    data = {
        "name": ["11 Oph b"],
    }
    df = pd.DataFrame(data)
    instance.data = df
    with LogCapture() as log:
        instance.assign_status()
        assert "Status column assigned" in log.actual()[0][-1]
        assert list(set(instance.data.status.values)) == ["CONFIRMED"]


def test_convert_coordinates(instance):
    assert instance.convert_coordinates() is None


def test__remove_theoretical_masses(instance):
    # Create a sample DataFrame with some additional columns
    data = {
        "name": ["11 Oph b"],
        "planet_status": ["Confirmed"],
        "mass": [21.0],
        "mass_min": [3.0],
        "mass_max": [3.0],
        "msini": [3.3],
        "msini_min": [0.0],
        "msini_max": [np.inf],
        "r": [22],
        "r_min": [2],
        "r_max": [2],
        "mass_url": ["Calculated"],
        "msini_url": ["Calculated"],
        "r_url": ["Calculated"],
    }
    df = pd.DataFrame(data)
    instance.data = df
    with LogCapture() as log:
        instance.remove_theoretical_masses()
        assert "Theoretical masses/radii removed" in log.actual()[0][-1]

    assert np.isnan(instance.data.at[0, "mass"])
    assert np.isnan(instance.data.at[0, "mass_min"])
    assert np.isnan(instance.data.at[0, "mass_max"])
    assert np.isnan(instance.data.at[0, "r"])
    assert np.isnan(instance.data.at[0, "r_min"])
    assert np.isnan(instance.data.at[0, "r_max"])
