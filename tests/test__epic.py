import unittest
import io
import numpy as np
import pandas as pd
import pytest
import subprocess
import os, glob
from pathlib import Path, PosixPath
import logging
import requests.exceptions
from testfixtures import LogCapture
from exo_mercat.catalogs import uniform_string, Catalog
from unittest.mock import patch, MagicMock, Mock
from datetime import date

from exo_mercat.epic import Epic


@pytest.fixture
def instance():
    return Epic()


def test__init(instance):
    assert instance.data is None
    assert instance.name is "epic"


def test__uniform_catalog(instance):
    data = {
        "pl_name": ["HAT-P-56 b", "EPIC 201324549.01"],
        "pl_letter": ["b", ""],
        "default_flag": [1, 1],
        "k2_name": ["K2-20 b", np.nan],
        "epic_hostname": ["EPIC 202126852", "EPIC 201324549"],
        "epic_candname": ["EPIC 202126852 b", ""],
        "hostname": ["HAT-P-56", "EPIC 201324549"],
        "hd_name": ["HD 262389", np.nan],
        "hip_name": ["HIP 32209", np.nan],
        "tic_id": ["TIC 84339983", "TIC 38161230"],
        "gaia_id": ["Gaia DR2 3385777286999240576", "Gaia DR2 3796557953574638336"],
        "discoverymethod": ["Transit", "Transit"],
        "ra": [100.8480137, 171.2359519],
        "dec": [27.2521823, -2.0851597],
        "disposition": ["CONFIRMED", "FALSE POSITIVE"],
    }
    data = pd.DataFrame(data)
    instance.data = data

    with LogCapture() as log:
        instance.uniform_catalog()
        assert "Catalog uniformed" in log.actual()[0][-1]

    assert list(instance.data.columns) == [
        "pl_name",
        "pl_letter",
        "k2_name",
        "epic_hostname",
        "hostname",
        "hd_name",
        "hip_name",
        "tic_id",
        "gaia_id",
        "disposition",
        "ra",
        "dec",
        "discoverymethod",
        "default_flag",
        "Kepler_host",
        "name",
        "letter",
        "HD_letter",
        "HIP_letter",
        "TIC_letter",
        "GAIA_letter",
        "alias",
        "aliasplanet",
    ]
    assert (
        "K2-20," in instance.data.at[0, "alias"]
    )  # 'HIP 32209,K2-20,TIC 84339983,HD 262389,Gaia DR2 3385777286999240576,'
    assert "nan" not in instance.data.at[0, "alias"]
    assert data.at[0, "tic_id"] + "," in instance.data.at[0, "alias"]
    assert data.at[0, "hip_name"] + "," in instance.data.at[0, "alias"]
    assert data.at[0, "hd_name"] + "," in instance.data.at[0, "alias"]
    assert data.at[0, "gaia_id"] + "," in instance.data.at[0, "alias"]
    assert "b" in instance.data.at[0, "letter"]
    assert (
        data.at[0, "k2_name"] + "," in instance.data.at[0, "aliasplanet"]
    )  #'HD 262389 b,TIC 84339983 b,Gaia DR2 3385777286999240576 b,HIP 32209 b,K2-20 b,'
    assert (
        data.at[0, "tic_id"] + " " + data.at[0, "pl_letter"] + ","
        in instance.data.at[0, "aliasplanet"]
    )
    assert (
        data.at[0, "hip_name"] + " " + data.at[0, "pl_letter"] + ","
        in instance.data.at[0, "aliasplanet"]
    )
    assert (
        data.at[0, "hd_name"] + " " + data.at[0, "pl_letter"] + ","
        in instance.data.at[0, "aliasplanet"]
    )
    assert (
        data.at[0, "gaia_id"] + " " + data.at[0, "pl_letter"] + ","
        in instance.data.at[0, "aliasplanet"]
    )

    assert (
        "EPIC 201324549," in instance.data.at[1, "alias"]
    )  # '',Gaia DR2 3796557953574638336,EPIC 201324549,TIC 38161230,'
    assert data.at[1, "tic_id"] + "," in instance.data.at[1, "alias"]
    assert "nan" not in instance.data.at[1, "alias"]
    assert data.at[1, "gaia_id"] + "," in instance.data.at[1, "alias"]

    assert ".01" in instance.data.at[1, "letter"]
    assert (
        data.at[1, "gaia_id"] + ".01," in instance.data.at[1, "aliasplanet"]
    )  # Gaia DR2 3796557953574638336.01,EPIC 201324549.01,TIC 38161230.01,
    assert data.at[1, "tic_id"] + ".01," in instance.data.at[1, "aliasplanet"]


def test_convert_coordinates(instance):
    assert instance.convert_coordinates() is None
