import os
from datetime import date
from pathlib import Path
from unittest.mock import MagicMock, patch

import numpy as np
import pandas as pd
import pytest
from testfixtures import LogCapture

from exomercat.eu import Eu


@pytest.fixture
def instance():
    return Eu()


def test__init(instance):
    assert instance.data is None
    assert instance.name is "eu"


# def test__download_catalog(tmp_path, instance) -> None:
#     original_dir = os.getcwd()
#
#     os.chdir(tmp_path)  # Create a temporary in-memory configuration object
#
#     url = "http://exoplanet.eu/catalog/votable/?query_f=planet_status%3D%22retracted%22"
#     filename = "catalog"
#     expected_file_path = filename + date.today().strftime("%Y-%m-%d")+".csv"
#
#     # CASE 1: it downloads fine
#
#     with LogCapture() as log:
#         result = instance.download_catalog(url=url, filename=filename, local_date=date.today().strftime("%Y-%m-%d"))
#         assert "Catalog downloaded" in log.actual()[-1][-1]
#
#         assert result == Path(expected_file_path)
#
#     with LogCapture() as log:
#         result = instance.download_catalog(url=url, filename=filename,local_date=date.today().strftime("%Y-%m-%d"))
#         assert "Reading existing file" in log.actual()[0][-1]
#         assert "Catalog downloaded" in log.actual()[1][-1]
#
#         assert result == Path(expected_file_path)
#
#     os.remove(expected_file_path)
#
#         #
#         # os.remove(expected_file_path)
#     #     # open("catalog01-20-2024.csv", "w").close()
#         # # CASE 2.A : errors in downloading, takes local copy
#         # with LogCapture() as log:
#         #     result = instance.download_catalog(
#         #         url=url, filename=filename, timeout=0.00001
#         #     )
#         #     assert (
#         #         "Error fetching the catalog, taking a local copy: catalog01-20-2024.csv"
#         #         in log.actual()[1][-1]
#         #     )
#         #     assert "Catalog downloaded" in log.actual()[-1][-1]
#         #
#         #     # it gets another local file
#         #     assert result != Path(expected_file_path)
#         #     assert "catalog" in str(result)  # it contains the filename
#         #     assert "csv" in str(result)  # it is a csv file
#         # os.remove("catalog01-20-2024.csv")
#         #
#         # # # CASE 2.B : errors in downloading, raises error because no local copy
#         # with LogCapture() as log:
#         #     with pytest.raises(ValueError):
#         #         result = instance.download_catalog(
#         #             url=url, filename=filename, timeout=0.00001
#         #         )
#     os.chdir(original_dir)


def test__standardize_catalog(instance):
    # Create a sample DataFrame with some additional columns
    data = {
        "name": ["11 Oph b"],
        "planet_status": ["Confirmed"],
        "mass": [21.0],
        "mass_error_min": [3.0],
        "mass_error_max": [3.0],
        "mass_sini": [np.nan],
        "mass_sini_error_min": [np.nan],
        "mass_sini_error_max": [np.nan],
        "radius": [np.nan],
        "radius_error_min": [np.nan],
        "radius_error_max": [np.nan],
        "orbital_period": [730000.0],
        "orbital_period_error_min": [365000.0],
        "orbital_period_error_max": [365000.0],
        "semi_major_axis": [243.0],
        "semi_major_axis_error_min": [55.0],
        "semi_major_axis_error_max": [55.0],
        "eccentricity": [np.nan],
        "eccentricity_error_min": [np.nan],
        "eccentricity_error_max": [np.nan],
        "inclination": [np.nan],
        "inclination_error_min": [np.nan],
        "inclination_error_max": [np.nan],
        "angular_distance": [1.675862],
        "discovered": [2007.0],
        "updated": ["2018-06-18"],
        "omega": [np.nan],
        "omega_error_min": [np.nan],
        "omega_error_max": [np.nan],
        "tperi": [np.nan],
        "tperi_error_min": [np.nan],
        "tperi_error_max": [np.nan],
        "tconj": [np.nan],
        "tconj_error_min": [np.nan],
        "tconj_error_max": [np.nan],
        "tzero_tr": [np.nan],
        "tzero_tr_error_min": [np.nan],
        "tzero_tr_error_max": [np.nan],
        "tzero_tr_sec": [np.nan],
        "tzero_tr_sec_error_min": [np.nan],
        "tzero_tr_sec_error_max": [np.nan],
        "lambda_angle": [np.nan],
        "lambda_angle_error_min": [np.nan],
        "lambda_angle_error_max": [np.nan],
        "impact_parameter": [np.nan],
        "impact_parameter_error_min": [np.nan],
        "impact_parameter_error_max": [np.nan],
        "tzero_vr": [np.nan],
        "tzero_vr_error_min": [np.nan],
        "tzero_vr_error_max": [np.nan],
        "k": [np.nan],
        "k_error_min": [np.nan],
        "k_error_max": [np.nan],
        "temp_calculated": [np.nan],
        "temp_calculated_error_min": [np.nan],
        "temp_calculated_error_max": [np.nan],
        "temp_measured": [np.nan],
        "hot_point_lon": [np.nan],
        "geometric_albedo": [np.nan],
        "geometric_albedo_error_min": [np.nan],
        "geometric_albedo_error_max": [np.nan],
        "log_g": [np.nan],
        "publication": ["Published in a refereed paper"],
        "detection_type": ["Primary Transit#TTV"],
        "mass_measurement_type": ["Theoretical"],
        "radius_measurement_type": ["Theoretical"],
        "alternate_names": ["Oph 1622-2405 b"],
        "molecules": ["None"],
        "star_name": ["11 Oph"],
        "ra": [245.6041667],
        "dec": [-24.0872222],
        "mag_v": [np.nan],
        "mag_i": [np.nan],
        "mag_j": [np.nan],
        "mag_h": [np.nan],
        "mag_k": [14.03],
        "star_distance": [145.0],
        "star_distance_error_min": [20.0],
        "star_distance_error_max": [20.0],
        "star_metallicity": [np.nan],
        "star_metallicity_error_min": [np.nan],
        "star_metallicity_error_max": [np.nan],
        "star_mass": [0.0162],
        "star_mass_error_min": [0.005],
        "star_mass_error_max": [0.005],
        "star_radius": [np.nan],
        "star_radius_error_min": [np.nan],
        "star_radius_error_max": [np.nan],
        "star_sp_type": ["M9"],
        "star_age": [0.011],
        "star_age_error_min": [0.002],
        "star_age_error_max": [0.002],
        "star_teff": [2375.0],
        "star_teff_error_min": [175.0],
        "star_teff_error_max": [175.0],
        "star_detected_disc": ["None"],
        "star_magnetic_field": ["None"],
        "star_alternate_names": ["Oph 1622-2405, Oph 11A"],
    }

    expected_columns = [
        "discovery_method",
        "p",
        "p_max",
        "p_min",
        "a",
        "a_max",
        "a_min",
        "e",
        "e_max",
        "e_min",
        "i",
        "i_max",
        "i_min",
        "name",
        "Update",
        "discovery_year",
        "mass",
        "mass_max",
        "mass_min",
        "msini",
        "msini_max",
        "msini_min",
        "r",
        "r_max",
        "r_min",
        "MASSPROV",
        "RADPROV",
        "host",
        "reference",
        "alias",
    ]
    df = pd.DataFrame(data)
    instance.data = df
    with LogCapture() as log:
        instance.standardize_catalog()
        assert "Catalog standardized" in log.actual()[0][-1]

    assert all(element in instance.data.columns for element in expected_columns)

    assert all(
        element in instance.data.at[0, "alias"].split(",")
        for element in ["Oph 1622-2405 b", "Oph 1622-2405", "Oph 11A"]
    )
    assert instance.data.at[0, "discovery_method"] == "TTV"


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
        "MASSPROV": ["Theoretical"],
        "RADPROV": ["Theoretical"],
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


def test__assign_status(instance):
    # Create a sample DataFrame with some additional columns
    data = {
        "name": [
            "11 Oph b",
            "Candidate planet",
            "Unconfirmed planet",
            "Controversial planet",
            "Retracted planet",
        ],
        "planet_status": [
            "Confirmed",
            "Candidate",
            "Unconfirmed",
            "Controversial",
            "Retracted",
        ],
    }
    df = pd.DataFrame(data)
    instance.data = df
    with LogCapture() as log:
        instance.assign_status()
        assert "Status column assigned" in log.actual()[0][-1]

    assert list(instance.data.status) == [
        "CONFIRMED",
        "CANDIDATE",
        "CANDIDATE",
        "CANDIDATE",
        "FALSE POSITIVE",
    ]


def test__handle_reference_format(instance):
    df = pd.DataFrame(
        {
            "name": ["11 Oph b"],
            "e": [0.231],
            "mass": [np.nan],
            "msini": [16.1284],
            "i": [np.nan],
            "a": [1.29],
            "p": [326.03],
            "r": [np.nan],
        }
    )
    instance.data = df
    with LogCapture() as log:
        instance.handle_reference_format()
        assert "Reference columns standardized" in log.actual()[0][-1]
    assert list(instance.data.columns) == [
        "name",
        "e",
        "mass",
        "msini",
        "i",
        "a",
        "p",
        "r",
        "e_url",
        "mass_url",
        "msini_url",
        "i_url",
        "a_url",
        "p_url",
        "r_url",
    ]

    expected_result = [
        "11 Oph b",
        0.231,
        np.nan,
        16.1284,
        np.nan,
        1.29,
        326.03,
        np.nan,
        "eu",
        "",
        "eu",
        "",
        "eu",
        "eu",
        "",
    ]
    for d1, d2 in zip(list(instance.data.iloc[0]), expected_result):
        if pd.isna(d1) and pd.isna(d2):
            continue
        assert d1 == d2


def test_convert_coordinates(instance):
    assert instance.convert_coordinates() is None
