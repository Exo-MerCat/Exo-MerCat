import numpy as np
import pandas as pd
import pytest
from testfixtures import LogCapture
from exomercat.toi import Toi


@pytest.fixture
def instance():
    return Toi()


def test__init(instance):
    assert instance.data is None
    assert instance.name is "toi"


def test__standardize_catalog(instance):
    data = {
        "tid": [146589986, 149601557],
        "toi": [1032.01, 1033.01],
        "toipfx": [1032, 1033],
        "tfopwg_disp": ["PC", "FP"],
        "ra": [159.221506, 87.282586],
        "dec": [-42.541639, -60.498444],
        "pl_orbper": [5.664473, 29.0162503],
        "pl_orbpererr1": [0.0007689, 0.0001146],
        "pl_orbpererr2": [-0.0007689, -0.0001146],
        "pl_orbperlim": [0, 0],
        "pl_rade": [14.3057012, 9.84223],
        "pl_radeerr1": [0.8767856, np.nan],
        "pl_radeerr2": [-0.8767856, np.nan],
        "pl_radelim": [0, 0],
        "toi_created": ["2019-07-24 18:44:12", "2019-06-11 14:45:00"],
    }

    data = pd.DataFrame(data)
    instance.data = data

    with LogCapture() as log:
        instance.standardize_catalog()
        assert "Catalog standardized" in log.actual()[-1][-1]

    expected_result = {
        "tid": [146589986, 149601557],
        "toi": [1032.01, 1033.01],
        "toipfx": [1032, 1033],
        "tfopwg_disp": ["PC", "FP"],
        "ra": [159.221506, 87.282586],
        "dec": [-42.541639, -60.498444],
        "p": [5.664473, 29.0162503],
        "p_max": [0.0007689, 0.0001146],
        "p_min": [-0.0007689, -0.0001146],
        "pl_orbperlim": [0, 0],
        "pl_rade": [14.3057012, 9.84223],
        "pl_radeerr1": [0.8767856, np.nan],
        "pl_radeerr2": [-0.8767856, np.nan],
        "pl_radelim": [0, 0],
        "toi_created": ["2019-07-24 18:44:12", "2019-06-11 14:45:00"],
        "catalog": ["toi", "toi"],
        "TOI": ["TOI-1032.01", "TOI-1033.01"],
        "TIC_host": ["TIC 146589986", "TIC 149601557"],
        "TOI_host": ["TOI-1032", "TOI-1033"],
        "TIC_host2": ["TIC-146589986", "TIC-149601557"],
        "TOI_host2": ["TOI 1032", "TOI 1033"],
        "host": ["TIC 146589986", "TIC 149601557"],
        "letter": [".01", ".01"],
        "alias": [
            "2MASS J10365315-4232299,Gaia DR2 5392041345156957696,TIC-146589986,TOI 1032,TOI-1032,TYC 7718-01074-1,UCAC4 238-048613,WISE J103653.13-423229.8",
            "2MASS J05490779-6029549,Gaia DR2 4758940969330786432,TIC-149601557,TOI 1033,TOI-1033,TYC 8892-01686-1,UCAC4 148-006443,WISE J054907.80-602954.5",
        ],
        "alias_vizier": [
            "UCAC4 238-048613,2MASS J10365315-4232299,WISE J103653.13-423229.8,Gaia DR2 5392041345156957696,,,TYC 7718-01074-1",
            "UCAC4 148-006443,2MASS J05490779-6029549,WISE J054907.80-602954.5,Gaia DR2 4758940969330786432,,,TYC 8892-01686-1",
        ],
        "name": ["TOI-1032.01", "TOI-1033.01"],
        "catalog_name": ["TOI-1032.01", "TOI-1033.01"],
        "catalog_host": ["TIC 146589986", "TIC 149601557"],
        "mass": [np.nan, np.nan],
        "mass_min": [np.nan, np.nan],
        "mass_max": [np.nan, np.nan],
        "msini": [np.nan, np.nan],
        "msini_min": [np.nan, np.nan],
        "msini_max": [np.nan, np.nan],
        "a": [np.nan, np.nan],
        "a_min": [np.nan, np.nan],
        "a_max": [np.nan, np.nan],
        "e": [np.nan, np.nan],
        "e_min": [np.nan, np.nan],
        "e_max": [np.nan, np.nan],
        "i": [np.nan, np.nan],
        "i_min": [np.nan, np.nan],
        "i_max": [np.nan, np.nan],
        "Age (Gyrs)": [np.nan, np.nan],
        "Age_max": [np.nan, np.nan],
        "Age_min": [np.nan, np.nan],
        "Mstar": [np.nan, np.nan],
        "Mstar_max": [np.nan, np.nan],
        "Mstar_min": [np.nan, np.nan],
        "r": [1.2762714090201503, 0.8780664788214001],
        "r_min": [-0.078221708339808, np.nan],
        "r_max": [0.078221708339808, np.nan],
        "discovery_year": ["2019", "2019"],
        "discovery_method": ["Transit", "Transit"],
        "reference": ["toi", "toi"],
    }
    expected_df = pd.DataFrame(expected_result)
    assert (instance.data.columns == expected_df.columns).all()
    for col in [
        "tid",
        "toi",
        "toipfx",
        "tfopwg_disp",
        "ra",
        "dec",
        "toi_created",
        "catalog",
        "TOI",
        "TIC_host",
        "TOI_host",
        "TIC_host2",
        "TOI_host2",
        "host",
        "letter",
        "alias",
        "name",
        "catalog_name",
        "discovery_year",
        "discovery_method",
        "reference",
    ]:
        assert (instance.data[col] == expected_df[col]).all()
    for col in [
        "p",
        "p_max",
        "p_min",
        "pl_orbperlim",
        "pl_rade",
        "pl_radeerr1",
        "pl_radeerr2",
        "pl_radelim",
        "mass",
        "mass_min",
        "mass_max",
        "msini",
        "msini_min",
        "msini_max",
        "a",
        "a_min",
        "a_max",
        "e",
        "e_min",
        "e_max",
        "i",
        "i_min",
        "i_max",
        "Age (Gyrs)",
        "Age_max",
        "Age_min",
        "Mstar",
        "Mstar_max",
        "Mstar_min",
        "r",
        "r_min",
        "r_max",
    ]:
        assert np.isclose(instance.data.at[0, "r"], expected_df.at[0, "r"])


def test__convert_coordinates(instance):
    assert instance.convert_coordinates() is None


def test__remove_theoretical_masses(instance):
    assert instance.remove_theoretical_masses() is None


def test__handle_reference_format(instance):
    df = pd.DataFrame(
        {
            "name": ["TOI 1032.01"],
            "e": [np.nan],
            "mass": [np.nan],
            "msini": [np.nan],
            "i": [np.nan],
            "a": [np.nan],
            "p": [5.664473],
            "r": [1.2763],
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
        "TOI 1032.01",
        np.nan,
        np.nan,
        np.nan,
        np.nan,
        np.nan,
        5.664473,
        1.2763,
        "",
        "",
        "",
        "",
        "",
        "toi",
        "toi",
    ]
    for d1, d2 in zip(list(instance.data.iloc[0]), expected_result):
        if pd.isna(d1) and pd.isna(d2):
            continue
        assert d1 == d2


def test__assign_status(instance):
    # Create a sample DataFrame with some additional columns
    data = {
        "name": [
            "Candidate planet",
            "Confirmed planet",
            "Ambiguous planet",
            "False alarm planet",
            "False positive planet",
            "Known Planet",
            "unknown",
        ],
        "tfopwg_disp": ["PC", "CP", "APC", "FA", "FP", "KP", ""],
    }
    df = pd.DataFrame(data)
    instance.data = df
    with LogCapture() as log:
        instance.assign_status()
        assert "Status column assigned" in log.actual()[0][-1]

    assert list(instance.data.status) == [
        "CANDIDATE",
        "CONFIRMED",
        "CONTROVERSIAL",
        "FALSE POSITIVE",
        "FALSE POSITIVE",
        "CONFIRMED",
        "UNKNOWN",
    ]
