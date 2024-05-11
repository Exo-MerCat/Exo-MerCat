import os
from datetime import date
from pathlib import Path, PosixPath
from unittest.mock import patch, MagicMock, Mock

import numpy as np
import pandas as pd
import pytest
from testfixtures import LogCapture

from exo_mercat.catalogs import Catalog


@pytest.fixture
def instance():
    return Catalog()


def test__init(instance):
    assert instance.data is None
    assert instance.name is "catalog"


def test__download_catalog(tmp_path, instance) -> None:
    original_dir = os.getcwd()

    os.chdir(tmp_path)  # Create a temporary in-memory configuration object
    url = "https://example.com/catalog.csv"
    filename = "catalog"
    # Read specific date
    expected_file_path = filename + '01-02-2024.csv'
    with LogCapture() as log:
        with pytest.raises(ValueError):
            result = instance.download_catalog(url=url, filename=filename,local_date='01-02-2024')
            assert "Could not find catalog with this specific date. Please check your date value." in log.actual()[0][-1]

    with patch("os.path.exists", MagicMock(return_value=True)):
        open(expected_file_path, "w").close()
        with LogCapture() as log:
            result = instance.download_catalog(url=url, filename=filename,local_date='01-02-2024')
            assert "Reading specific version: 01-02-2024" in log.actual()[0][-1]
            assert "Reading existing file" in log.actual()[1][-1]
            assert "Catalog downloaded" in log.actual()[2][-1]
            assert result == PosixPath(expected_file_path)
    os.remove(expected_file_path)

    expected_file_path = filename + date.today().strftime("%m-%d-%Y.csv")

    # Mock os.path.exists to simulate that the file already exists or not
    with patch("os.path.exists", MagicMock(return_value=True)):
        open(expected_file_path, "w").close()
        with LogCapture() as log:
            result = instance.download_catalog(url=url, filename=filename)
            assert "Reading existing file" in log.actual()[0][-1]
            assert "Catalog downloaded" in log.actual()[1][-1]

            assert result == PosixPath(expected_file_path)

    os.remove(expected_file_path)
    with patch("os.path.exists", MagicMock(return_value=False)):
        # CASE 1: it downloads fine
        with patch("requests.get", MagicMock()) as mock_run:
            mock_response = Mock()
            mock_response.status_code = 200
            mock_response.content = b"Mock content"
            mock_run.return_value = mock_response
            with LogCapture() as log:
                result = instance.download_catalog(url=url, filename=filename)
                assert "Catalog downloaded" in log.actual()[0][-1]

            assert result == Path(expected_file_path)
            mock_run.assert_called_once_with(url, timeout=None)

        os.remove(expected_file_path)
        open("cataloglocal_copy.csv", "w").close()
        # CASE 2.A : errors in downloading, takes local copy
        with LogCapture() as log:
            result = instance.download_catalog(
                url=url, filename=filename, timeout=0.00001
            )
            assert (
                    "Error fetching the catalog, taking a local copy: cataloglocal_copy.csv"
                    in log.actual()[0][-1]
            )
            assert "Catalog downloaded" in log.actual()[1][-1]

            # it gets another local file
            assert result != Path(expected_file_path)
            assert "catalog" in str(result)  # it contains the filename
            assert "csv" in str(result)  # it is a csv file
        os.remove("cataloglocal_copy.csv")

        # CASE 2.B : errors in downloading, raises error because no local copy
        with LogCapture() as log:
            with pytest.raises(ValueError):
                result = instance.download_catalog(
                    url=url, filename=filename, timeout=0.00001
                )
    os.chdir(original_dir)

def test__read_csv_catalog(instance):
    # Create a temporary in-memory configuration object
    instance.read_csv_catalog("tests/emc_test.csv")
    assert isinstance(instance.data, pd.DataFrame)


# def test__convert_datatypes(instance):
#     # Test when DataFrame conversion is successful
#     # Create a sample DataFrame
#     data = {
#         "col1": [1.0, 2.0, 3.0],
#         "col2": ["A", "B", "C"],
#         "col3": [True, False, True],
#         "col4": [1.4, 2, 3],
#     }
#     instance.data = pd.DataFrame(data)
#
#     # Call the convert_datatypes function
#     instance.convert_datatypes()
#     assert instance.data["col1"].dtype.type == np.int64
#     assert instance.data["col2"].dtype.type == str
#     assert instance.data["col3"].dtype.type == np.bool_
#     assert instance.data["col4"].dtype.type == np.float64
#


def test_keep_columns(instance):
    # Test the keep_columns function

    # Create a sample DataFrame with some additional columns
    data = {
        "name": ["HD 114762 b"],
        "catalog_name": ["HD 114762 b"],
        "catalog_host": ["HD 114762"],
        "discovery_method": ["Radial Velocity"],
        "ra": [198.0791667],
        "dec": [17.51694444],
        "p": [83.9151],
        "p_max": [np.nan],
        "p_min": [np.nan],
        "a": [0.353],
        "a_max": [np.nan],
        "a_min": [np.nan],
        "e": [0.3354],
        "e_max": [np.nan],
        "e_min": [np.nan],
        "i": [np.nan],
        "i_max": [np.nan],
        "i_min": [np.nan],
        "mass": [10.98],
        "mass_max": [np.nan],
        "mass_min": [np.nan],
        "msini": [np.nan],
        "msini_max": [np.nan],
        "msini_min": [np.nan],
        "r": [np.nan],
        "r_max": [np.nan],
        "r_min": [np.nan],
        "discovery_year": [1992],
        "alias": [np.nan],
        "a_url": ["oec"],
        "mass_url": ["oec"],
        "p_url": ["oec"],
        "msini_url": ["oec"],
        "r_url": ["oec"],
        "i_url": ["oec"],
        "e_url": ["oec"],
        "host": ["HD 114762"],
        "binary": ["S-type"],
        "letter": ["b"],
        "status": ["CONFIRMED"],
        "catalog": ["oec"],
        "original_catalog_status": ["oec: CONFIRMED"],
        "checked_catalog_status": ["oec: CONFIRMED"],
        "extra": ["extra"],
    }

    df = pd.DataFrame(data)
    instance.data = df
    with LogCapture() as log:
        instance.keep_columns()
        assert "Selected columns to keep" in log.actual()[0][-1]

    # Check if the DataFrame contains only the columns specified in the keep
    # list
    expected_columns = [
        "name",
        "catalog_name",
        "catalog_host",
        "discovery_method",
        "ra",
        "dec",
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
        "mass",
        "mass_max",
        "mass_min",
        "msini",
        "msini_max",
        "msini_min",
        "r",
        "r_max",
        "r_min",
        "discovery_year",
        "alias",
        "a_url",
        "mass_url",
        "p_url",
        "msini_url",
        "r_url",
        "i_url",
        "e_url",
        "host",
        "binary",
        "letter",
        "status",
        "catalog",
        "original_catalog_status",
        "checked_catalog_status",
    ]
    assert list(instance.data.columns) == expected_columns

    data = {
        "name": ["HD 114762 b"],
        "catalog_name": ["HD 114762 b"],
        "extra": ["extra"],
    }
    df = pd.DataFrame(data)
    instance.data = df
    with pytest.raises(KeyError):
        instance.keep_columns()


def test__identify_brown_dwarfs(instance):
    data = {
        "name": [
            "2MASS 0122-2439 B",
            "2MASS J0030-1450",
            "MOA 2015-BLG-337 a",
            "KOI-123.01",
            "DENIS J063001.4-184014 (bc)"
        ],
        "binary": ["", "", "", "",""],
        "letter": ["", "", "", "",""],
    }

    df = pd.DataFrame(data)
    instance.data = df
    # Call the identify_brown_dwarfs function
    with LogCapture() as log:
        instance.identify_brown_dwarfs()
        assert "Identified possible Brown Dwarfs" in log.actual()[0][-1]

    # Check if the 'letter' column is updated with 'BD' for relevant planet
    # names
    expected_result = {
        "name": [
            "2MASS 0122-2439 B",
            "2MASS J0030-1450",
            "MOA 2015-BLG-337 a",
            "KOI-123.01",
            "DENIS J063001.4-184014 (bc)"
        ],
        "binary": ["B", "", "a", "","bc"],
        "letter": ["BD", "BD", "BD", "","BD"],
    }
    expected_df = pd.DataFrame(expected_result)
    pd.testing.assert_frame_equal(df, expected_df)


def test__replace_known_mistakes(tmp_path, instance):
    original_dir = os.getcwd()

    os.chdir(tmp_path)  # Create a temporary in-memory configuration object
    os.mkdir("Logs/")
    # Create a temporary directory to store the fake_config.ini
    with open("replacements.ini", "w") as config_file:
        config_file.write("[NAMEtochangeNAME]\n")
        config_file.write("alf Tau b = Aldebaran b\n")
        config_file.write("not present = NOT PRESENT\n")
        config_file.write("[NAMEtochangeHOST]\n")
        config_file.write("alf Tau b = gam1 Leo\n")
        config_file.write("2MASS 0359+2009 b = 2MASS J03590986+2009361\n")
        config_file.write("[HOSTtochangeHOST]\n")
        config_file.write("gam1 Leo = gam01 Leo\n")
        config_file.write("not present = NOT PRESENT\n")
        config_file.write("[ra]\n")
        config_file.write("K2-2016-BLG-0005L = 269.879166677\n")
        config_file.write("not present = NOT PRESENT\n")
        config_file.write("[dec]\n")
        config_file.write("K2-2016-BLG-0005L = 269.879166677\n")
        config_file.write("[DROP]\n")
        config_file.write("name = Trojan\n")
        config_file.write("[NAMEtochangeBINARY]\n")
        config_file.write("XO-2N c = A\n")
        config_file.write("XO-2N b = A\n")
        config_file.write("not present = A\n")
    data = {
        "name": [
            "alf Tau b",
            "K2-2016-BLG-0005L",
            "XO-2N c",
            "Proxima Centauri b",
            "test",
            "XO-2N b",
            "2MASS 0359+2009 b",
            "Trojan",
        ],
        "host": [
            "gam1 Leo",
            "K2-2016-BLG-0005L",
            "XO-2N",
            "Proxima Centauri",
            "test",
            "XO-2N",
            "nametochangeHost",
            "",
        ],
        "ra": [0, 9, 0, 1, 0, 0, 0, 0],
        "dec": [0, 269.879166677, 0, 1, 0, 0, 0, 0],
        "binary": ["", "", "", "", "", "A", "", ""],
    }
    instance.data = pd.DataFrame(data)

    with LogCapture() as log:
        instance.replace_known_mistakes()
        assert "Known mistakes" in log.actual()[0][-1]

    assert instance.data["name"][0] == "Aldebaran b"
    assert instance.data["host"][0] == "gam01 Leo"
    assert instance.data["ra"][1] == 269.879166677
    assert instance.data["dec"][1] == 269.879166677
    assert instance.data["binary"][2] == "A"  # not executed in this function
    assert instance.data["binary"][4] == ""
    assert instance.data["binary"][5] == "A"
    assert "Trojan" not in instance.data.name.values
    assert instance.data["host"][6] == "2MASS J03590986+2009361"

    os.chdir(original_dir)



def test__make_errors_absolute(instance):
    # Test if the function makes errors absolute correctly

    # Create a sample DataFrame with both positive and negative error values
    data = {
        "p_max": [10, -20, 30],
        "a_max": [15, -25, 35],
        "e_max": [0.1, -0.2, 0.3],
        "i_max": [5.0, -10.0, 15.0],
        "r_max": [100.0, -200.0, 300.0],
        "msini_max": [50.0, -100.0, 150.0],
        "mass_max": [1000.0, -2000.0, 3000.0],
        "p_min": [1, -2, 3],
        "a_min": [1.5, -2.5, 3.5],
        "e_min": [0.01, -0.02, 0.03],
        "i_min": [0.5, -1.0, 1.5],
        "r_min": [10.0, -20.0, 30.0],
        "msini_min": [5.0, -10.0, 15.0],
        "mass_min": [100.0, -200.0, 300.0],
    }
    df = pd.DataFrame(data)

    # Set the instance's data attribute to the sample DataFrame
    instance.data = df.copy()

    # Call the make_errors_absolute function
    with LogCapture() as log:
        instance.make_errors_absolute()
        assert "Made all errors absolute values" in log.actual()[0][-1]

    # Check if all error columns are converted to absolute values correctly
    expected_result = {
        "p_max": [10, 20, 30],
        "a_max": [15, 25, 35],
        "e_max": [0.1, 0.2, 0.3],
        "i_max": [5.0, 10.0, 15.0],
        "r_max": [100.0, 200.0, 300.0],
        "msini_max": [50.0, 100.0, 150.0],
        "mass_max": [1000.0, 2000.0, 3000.0],
        "p_min": [1, 2, 3],
        "a_min": [1.5, 2.5, 3.5],
        "e_min": [0.01, 0.02, 0.03],
        "i_min": [0.5, 1.0, 1.5],
        "r_min": [10.0, 20.0, 30.0],
        "msini_min": [5.0, 10.0, 15.0],
        "mass_min": [100.0, 200.0, 300.0],
    }
    expected_df = pd.DataFrame(expected_result)
    pd.testing.assert_frame_equal(instance.data, expected_df)


def test__uniform_name_host_letter(instance):
    # Create a sample DataFrame with various names, hosts, and aliases
    data = {
        "name": ["planet1 b", "planet2 b", "planet3.01", "planet4.02"],
        "host": ["host1", "", "host3", ""],
        "alias": [
            "alias1 b,alias2",
            "alias3, alias4",
            "alias5.03, alias6.04",
            "alias7",
        ],
        "letter": ["b", "b", "", ""],
    }
    df = pd.DataFrame(data)

    # Set the instance's data attribute to the sample DataFrame
    instance.data = df

    with LogCapture() as log:
        # Call the uniform_name_host_letter function
        instance.uniform_name_host_letter()
        assert "name, host, letter columns uniformed" in log.actual()[0][-1]

    # Check if name, host, and letter columns are uniformly modified as
    # expected
    expected_result = {
        "name": ["planet1 b", "planet2 b", "planet3.01", "planet4.02"],
        "host": ["host1", "planet2", "host3", "planet4"],
        "alias": ["alias1,alias2", "alias3,alias4", "alias5,alias6", "alias7"],
        "letter": ["b", "b", ".01", ".02"],
    }
    expected_df = pd.DataFrame(expected_result)
    pd.testing.assert_frame_equal(instance.data, expected_df)


def test__check_mission_tables(instance):
    koi_test = {
        "kepid": [6922244, 4055765],
        "kepoi_name": ["K00010.01", "K00100.01"],
        "kepler_name": ["Kepler-8 b", ""],
        "koi_disposition": ["CONFIRMED", "CANDIDATE"],
        "ra": [281.288125, 291.17791666666665],
        "dec": [42.45108333333334, 39.19949999999999],
        "KOI": ["KOI-10.01", "KOI-100.01"],
        "KOI_host": ["KOI-10", "KOI-100"],
        "Kepler_host": ["Kepler-8", "nan"],
        "KIC_host": ["KIC 6922244", "KIC 4055765"],
        "letter": ["b", ".01"],
        "KIC": ["KIC 6922244 b", "KIC 4055765 .01"],
        "alias": ["Kepler-8,KIC 6922244,KOI-10,", "KIC 4055765,KOI-100,"],
        "aliasplanet": [
            "KIC 6922244 b,KOI-10.01,Kepler-8 b,",
            "KIC 4055765.01,KOI-100.01,",
        ],
        "name": ["KOI-10.01", "KOI-100.01"],
        "disposition": ["CONFIRMED", "CANDIDATE"],
        "discoverymethod": ["Transit", "Transit"],
    }

    # Create temporary CSV file with the data
    test_data_file = "test_koi.csv"
    pd.DataFrame(koi_test).to_csv(test_data_file, index=False)

    data = {
        "name": ["Kepler-8 b", "KIC 4055765.01", "weirdname"],
        "status": ["CONFIRMED", "FALSE", "Unknown"],
        "discovery_method": ["Transit", "nan", "nan"],
        "host": ["Kepler-8", "KOI-100", "Kepler-8"],
        "letter": ["b", ".01", "b"],
        "alias": ["onlypartial", "KIC 4055765,KOI-100,", ""],
    }

    instance.data = pd.DataFrame(data)

    with LogCapture() as log:
        # Call the uniform_name_host_letter function
        instance.check_mission_tables(test_data_file)
        assert "test_koi.csv checked" in log.actual()[0][-1]

    # Perform assertions to check if the data DataFrame is correctly updated
    assert instance.data.at[0, "status"] == "CONFIRMED"
    assert instance.data.at[1, "status"] == "CANDIDATE"
    assert instance.data.at[2, "status"] == "CONFIRMED"
    assert instance.data.at[0, "name"] == "Kepler-8 b"
    assert instance.data.at[1, "name"] == "KIC 4055765.01"
    assert instance.data.at[2, "name"] == "weirdname"
    assert instance.data.at[0, "discovery_method"] == "Transit"
    assert instance.data.at[1, "discovery_method"] == "Transit"
    assert instance.data.at[2, "discovery_method"] == "Transit"
    assert instance.data.at[0, "letter"] == "b"
    assert instance.data.at[1, "letter"] == ".01"
    assert instance.data.at[2, "letter"] == "b"
    assert instance.data.at[0, "host"] == "Kepler-8"
    assert instance.data.at[1, "host"] == "KOI-100"
    assert instance.data.at[2, "host"] == "Kepler-8"
    assert all(
        element in instance.data.at[0, "alias"].split(",")
        for element in [
            "onlypartial",
            "KIC 6922244",
            "KOI-10",
            "Kepler-8",
        ]
    )
    assert all(
        element in instance.data.at[1, "alias"].split(",")
        for element in ["KIC 4055765", "KOI-100"]
    )
    assert all(
        element in instance.data.at[2, "alias"].split(",")
        for element in [
            "KIC 6922244",
            "KOI-10",
            "Kepler-8",
        ]
    )

    # Clean up: Remove the temporary CSV file
    if os.path.exists(test_data_file):
        os.remove(test_data_file)


def test__fill_binary_column(instance):
    data = pd.DataFrame(
        {
            "name": [
                "KOI-1 b",
                "KOI-2 c",
                "KOI-3 d",
                "KOI-4 c",
                "Kepler-5 d",
                "NotBinary",
                "NotBinarybutFlag",
                "Notbinary2",
                "Notbinary1",
                "Notbinary3",
                "Kepler-21 A b",
                "Kepler-22 (AB) b",
                "Kepler-22 AB b",
            ],
            "host": [
                "Kepler-1 B",
                "Kepler-2 A b",
                "Kepler-3 A",
                "Kepler-4 (AB)",
                "Kepler-5 AB",
                "NotBinary",
                "cb_flag",
                "binaryflag1",
                "binaryflag2",
                "binaryflag3",
                "Kepler-21",
                "Kepler-22",
                "Kepler-22",
            ],
            "cb_flag": [0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0],
            "binaryflag": [0, 0, 0, 0, 0, 0, 0, 1, 2, 3, 0, 0, 0],
        }
    )
    instance.data = data
    # Call the function with the sample input data
    with LogCapture() as log:
        # Call the uniform_name_host_letter function
        instance.fill_binary_column()
        assert "Fixed planets orbiting binary stars" in log.actual()[0][-1]

    # Perform assertions to check if the binary column is correctly updated
    assert instance.data.at[0, "host"] == "Kepler-1"
    assert instance.data.at[1, "host"] == "Kepler-2"
    assert instance.data.at[2, "host"] == "Kepler-3"
    assert instance.data.at[3, "host"] == "Kepler-4"
    assert instance.data.at[4, "host"] == "Kepler-5"
    assert instance.data.at[5, "host"] == "NotBinary"
    assert instance.data.at[0, "binary"] == "B"
    assert instance.data.at[1, "binary"] == "A"
    assert instance.data.at[2, "binary"] == "A"
    assert instance.data.at[3, "binary"] == "AB"
    assert instance.data.at[4, "binary"] == "AB"
    assert instance.data.at[5, "binary"] == ""
    assert instance.data.at[6, "binary"] == "AB"
    assert instance.data.at[7, "binary"] == "AB"
    assert instance.data.at[8, "binary"] == "S-type"
    assert instance.data.at[9, "binary"] == "Rogue"
    assert instance.data.at[10, "binary"] == "A"
    assert instance.data.at[11, "binary"] == "AB"
    assert instance.data.at[12, "binary"] == "AB"


def test__create_catalogstatus_string(instance):
    # FIRST CALL, for ORIGINAL CATALOG STATUS
    data = pd.DataFrame(
        {
            "catalog": ["eu", "oec", "nasa"],
            "status": ["CANDIDATE", "CONFIRMED", "FALSE POSITIVE"],
        }
    )
    instance.data = data
    # Call the function with the sample input data
    with LogCapture() as log:
        # Call the uniform_name_host_letter function
        instance.create_catalogstatus_string("original_catalog_status")
        assert "original_catalog_status column created" in log.actual()[0][-1]

    assert instance.data.at[0, "original_catalog_status"] == "eu: CANDIDATE"
    assert instance.data.at[1, "original_catalog_status"] == "oec: CONFIRMED"
    assert instance.data.at[2, "original_catalog_status"] == "nasa: FALSE POSITIVE"

    # SECOND CALL, FOR CHECKED STATUS
    data = pd.DataFrame(
        {
            "catalog": ["eu", "oec", "nasa"],
            "status": ["FALSE POSITIVE", "CONFIRMED", "FALSE POSITIVE"],
        }
    )
    instance.data = data
    # Call the function with the sample input data
    with LogCapture() as log:
        # Call the uniform_name_host_letter function
        instance.create_catalogstatus_string("checked_catalog_status")
        assert "checked_catalog_status column created" in log.actual()[0][-1]

    assert instance.data.at[0, "checked_catalog_status"] == "eu: FALSE POSITIVE"
    assert instance.data.at[1, "checked_catalog_status"] == "oec: CONFIRMED"
    assert instance.data.at[2, "checked_catalog_status"] == "nasa: FALSE POSITIVE"


def test__make_uniform_alias_list(instance):
    data = pd.DataFrame(
        {
            "host": ["Kepler-1", "Kepler-1", "Kepler-1", "Kepler-1"],
            "alias": ["K001,Kepler-1", "anotheralias", np.NaN, ""],
        }
    )
    instance.data = data
    # Call the function with the sample input data
    with LogCapture() as log:
        # Call the uniform_name_host_letter function
        instance.make_uniform_alias_list()
        assert "Lists of aliases" in log.actual()[0][-1]

    assert all(
        element in ["anotheralias", "Kepler-1", "KOI-1"]
        for element in instance.data.at[0, "alias"].split(",")
    )


def test__fill_nan_on_coordinates(instance):
    # Sample input data (DataFrame) with missing values

    data = pd.DataFrame(
        {
            "name": ["EPIC 251809628.01", "HAT-P-56 b"],
            "ra": [np.nan, 100.848014],
            "rastr": ["", "06h43m23.52s"],
            "dec": ["", "27.252182"],
            "decstr": ["NaN", "+27d15m07.86s"],
        }
    )

    data = pd.DataFrame(data)
    instance.data = data
    # Call the function with the sample input data
    instance.fill_nan_on_coordinates()

    # Perform assertions to check if the coordinates are correctly converted
    assert np.isnan(instance.data.at[0, "ra"])
    assert np.isnan(instance.data.at[0, "dec"])
    assert instance.data.at[1, "ra"] == 100.848014
    assert instance.data.at[1, "dec"] == 27.252182


def test_print_catalog(instance):
    # Sample input data (DataFrame)
    data = pd.DataFrame(
        {
            "name": ["KOI-1", "KOI-2", "KOI-3"],
            "host": ["Kepler-1", "Kepler-2", "Kepler-3"],
            "alias": ["KOI-1", "Kepler-2 b", "KOI-3"],
        }
    )
    instance.data = data
    # Temporary in-memory file for testing
    temp_file = "test_print.csv"

    # Call the function with the sample input data and temporary file
    instance.print_catalog(temp_file)

    file_content = open(temp_file).read()

    # Expected output (DataFrame sorted by "name")
    expected_output = """name,host,alias\nKOI-1,Kepler-1,KOI-1\nKOI-2,Kepler-2,Kepler-2 b\nKOI-3,Kepler-3,KOI-3\n"""

    # Perform assertion to check if the content of the file matches the
    # expected output
    assert file_content == expected_output
    # Clean up: Remove the temporary CSV file
    if os.path.exists(temp_file):
        os.remove(temp_file)


# Define a list of function names that raise NotImplementedError
functions_raising_error = [
    "convert_coordinates",
    "remove_theoretical_masses",
    "handle_reference_format",
    "uniform_catalog",
    "assign_status",
]


@pytest.mark.parametrize("function_name", functions_raising_error)
def test__notimplemented(instance, function_name):
    func = getattr(instance, function_name)
    # Call the convert_coordinates function and expect NotImplementedError
    with pytest.raises(NotImplementedError):
        func()
