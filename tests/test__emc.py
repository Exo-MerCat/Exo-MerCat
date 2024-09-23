import os
from datetime import date

import numpy as np
import pandas as pd
import pytest
from pandas._testing import assert_frame_equal
from testfixtures import LogCapture
import socket
from exomercat.emc import Emc


@pytest.fixture
def instance():
    return Emc()


def test__init(instance):
    assert instance.data.empty is True
    assert instance.name is "exo_mercat"


def test__convert_coordinates(instance):
    assert instance.convert_coordinates() is None


def test__alias_as_host(tmp_path, instance):
    original_dir = os.getcwd()

    os.chdir(tmp_path)  # Create a temporary in-memory configuration object
    os.mkdir("Logs/")

    data = {
        "name": [
            "HD 110113 b",
            "HD 110113 c",
            "HD 110113 b",
            "HD 110113 c",
            "HD 110113 b",
            "HD 110113 c",
            "TOI-755.01",
            "TOI-755.02",
        ],
        "host": [
            "HD 110113",
            "HD 110113",
            "HD 110113",
            "HD 110113",
            "HD 110113",
            "HD 110113",
            "TIC 73228647",
            "TIC 73228647",
        ],
        "alias": [
            "TOI-755",
            "TOI-755",
            "HIP 61820,Gaia DR2 6133384959942131968,HD 110113,TIC 73228647",
            "HIP 61820,Gaia DR2 6133384959942131968,HD 110113,TIC 73228647",
            "HIP 61820,Gaia DR2 6133384959942131968,TOI-755,HD 110113,TYC 7783-2026-1",
            "HIP 61820,Gaia DR2 6133384959942131968,TOI-755,HD 110113,TYC 7783-2026-1",
            "HIP 61820,TOI-755,UCAC4 229-063280,WISE J124008.78-441843.3,TIC-73228647,TYC 7783-02026-1,"
            "2MASS J12400877-4418432,Gaia DR2 6133384959942131968",
            "HIP 61820,TOI-755,UCAC4 229-063280,WISE J124008.78-441843.3,TIC-73228647,TYC 7783-02026-1,"
            "2MASS J12400877-4418432,Gaia DR2 6133384959942131968",
        ],
    }

    instance.data = pd.DataFrame(data)

    original = list(instance.data.alias.drop_duplicates())
    # Call the identify_brown_dwarfs function
    with LogCapture() as log:
        instance.alias_as_host()
        log = pd.DataFrame(list(log), columns=["user", "info", "message"])
        assert (
            "Aliases labeled as hosts in some other entry checked. It happens 1 times."
            in log["message"].tolist()
        )

    final = list(instance.data.alias.drop_duplicates())

    assert len(final) == 1
    # the order of list_id is not important
    expected_alias = (
        "HIP 61820,TOI-755,UCAC4 229-063280,WISE J124008.78-441843.3,TIC-73228647,TYC 7783-02026-1,"
        "2MASS J12400877-4418432,Gaia DR2 6133384959942131968,TIC 73228647,TYC 7783-2026-1"
    )

    for i in instance.data.index:
        assert sorted(instance.data.at[i, "alias"].split(",")) == sorted(
            expected_alias.split(",")
        )

    assert os.path.exists("Logs/alias_as_host.txt")

    with open("Logs/alias_as_host.txt") as f:
        lines = f.readlines()
        assert lines == [
            "ALIAS: TIC 73228647 AS HOST:HD 110113\n",
        ]

    os.chdir(original_dir)


def test__check_binary_mismatch(tmp_path, instance):
    original_dir = os.getcwd()

    os.chdir(tmp_path)  # Create a temporary in-memory configuration object
    os.mkdir("Logs/")

    # # host
    # mismatches
    data = {
        "name": [
            "GJ 229 A c",  # no error
            "GJ 229 A c",  # no error
            "GJ 229 A c",  # no error
            "91 Aqr A b",  # FLAG 1
            "91 Aqr b",  # FLAG 1
            "HD 202206 (AB) c",  # FLAG 2
            "HD 202206 c",  # FLAG 2
            "ROXs 42B b",  # potential missed binary
            "HD 156846 b",  # only null and S-type
            "HD 156846 b",  # only null and S-type
            "HD 41004 A b",  # complex system (FLAG 2) with fix on coordinates
            "HD 41004 B b",  # complex system (FLAG 2) with fix on coordinates
            "HD 41004 b",  # complex system (FLAG 2) with fix on coordinates
        ],
        "host": [
            "GJ 229",
            "GJ 229",
            "GJ 229",
            "91 Aqr",
            "91 Aqr",
            "HD 202206",
            "HD 202206",
            "ROXs 42B",
            "HD 156846",
            "HD 156846",
            "HD 41004",
            "HD 41004",
            "HD 41004",
        ],
        "binary": [
            "A",
            "S-type",
            "S-type",
            "A",
            "",
            "A",
            "AB",
            "AB",
            "",
            "S-type",
            "A",
            "B",
            "S-type",
        ],
        "catalog": [
            "eu",
            "nasa",
            "other",
            "Catalog 1",
            "Catalog 3",
            "eu",
            "oec",
            "nasa",
            "oec",
            "eu",
            "eu",
            "eu",
            "nasa",
        ],
        "ra": [
            92.644231,
            92.644231,
            92.644232,
            10.0,
            20.0,
            10.0,
            10.0,
            10.0,
            260.141667,
            260.142337,
            1.222,
            1.23,
            1.222,
        ],
        "dec": [
            -21.864642,
            -21.864642,
            -21.864643,
            40.0,
            50.0,
            40.0,
            40.0,
            40.0,
            -19.333611,
            -19.334365,
            9.00,
            8.99,
            9.0001,
        ],
        "letter": ["c", "c", "c", "b", "b", "c", "c", "b", "b", "b", "b", "b", "b"],
    }

    expected_data = {
        "name": [
            "GJ 229 A c",
            "GJ 229 A c",
            "GJ 229 A c",
            "91 Aqr A b",
            "91 Aqr b",
            "HD 202206 (AB) c",
            "HD 202206 c",
            "ROXs 42B b",
            "HD 156846 b",
            "HD 156846 b",
            "HD 41004 A b",  # complex system (FLAG 2) with fix on coordinates
            "HD 41004 B b",  # complex system (FLAG 2) with fix on coordinates
            "HD 41004 b",  # complex system (FLAG 2) with fix on coordinates
        ],
        "host": [
            "GJ 229",
            "GJ 229",
            "GJ 229",
            "91 Aqr",
            "91 Aqr",
            "HD 202206",
            "HD 202206",
            "ROXs 42B",
            "HD 156846",
            "HD 156846",
            "HD 41004",
            "HD 41004",
            "HD 41004",
        ],
        "binary": [
            "A",
            "A",
            "A",
            "A",
            "A",
            "A",
            "AB",
            "AB",
            "S-type",
            "S-type",
            "A",
            "B",
            "A",
        ],
        "catalog": [
            "eu",
            "nasa",
            "other",
            "Catalog 1",
            "Catalog 3",
            "eu",
            "oec",
            "nasa",
            "oec",
            "eu",
            "eu",
            "eu",
            "nasa",
        ],
        "ra": [
            92.644231,
            92.644231,
            92.644232,
            10.0,
            20.0,
            10.0,
            10.0,
            10.0,
            260.141667,
            260.142337,
            1.222,
            1.23,
            1.222,
        ],
        "dec": [
            -21.864642,
            -21.864642,
            -21.864643,
            40.0,
            50.0,
            40.0,
            40.0,
            40.0,
            -19.333611,
            -19.334365,
            9.00,
            8.99,
            9.0001,
        ],
        "letter": ["c", "c", "c", "b", "b", "c", "c", "b", "b", "b", "b", "b", "b"],
        "binary_mismatch_flag": [0, 0, 0, 1, 1, 2, 2, 0, 1, 1, 2, 2, 2],
    }
    instance.data = pd.DataFrame(data)

    # Execute the function
    instance.check_binary_mismatch(keyword="host", tolerance=1.0 / 3600.0)
    assert_frame_equal(instance.data, pd.DataFrame(expected_data))

    ### Specific tests for main_id

    # mismatches
    data = {
        "name": [
            "91 Aqr A b",  # FLAG 1
            "91 Aqr b",  # FLAG 1
            "HD 202206 (AB) c",  # FLAG 2
            "HD 202206 c",  # FLAG 2
            "HD 156846 b",  # only null and S-type
            "HD 156846 b",  # only null and S-type
        ],
        "main_id": [
            "91 Aqr",
            "91 Aqr",
            "HD 202206",
            "HD 202206",
            "HD 156846",
            "HD 156846",
        ],
        "binary": ["A", "", "A", "AB", "", "S-type"],
        "catalog": ["Catalog 1", "Catalog 3", "eu", "oec", "oec", "eu"],
        "ra": [10.0, 20.0, 10.0, 10.0, 260.141667, 260.142337],
        "dec": [40.0, 50.0, 40.0, 40.0, -19.333611, -19.334365],
        "letter": ["b", "b", "c", "c", "b", "b"],
        "binary_mismatch_flag": [0, 0, 0, 1, 2, 2],
    }

    expected_data = {
        "name": [
            "91 Aqr A b",
            "91 Aqr b",
            "HD 202206 (AB) c",
            "HD 202206 c",
            "HD 156846 b",
            "HD 156846 b",
        ],
        "main_id": [
            "91 Aqr",
            "91 Aqr",
            "HD 202206",
            "HD 202206",
            "HD 156846",
            "HD 156846",
        ],
        "binary": ["A", "A", "A", "AB", "S-type", "S-type"],
        "catalog": ["Catalog 1", "Catalog 3", "eu", "oec", "oec", "eu"],
        "ra": [10.0, 20.0, 10.0, 10.0, 260.141667, 260.142337],
        "dec": [40.0, 50.0, 40.0, 40.0, -19.333611, -19.334365],
        "letter": ["b", "b", "c", "c", "b", "b"],
        "binary_mismatch_flag": [1, 1, 2, 2, 1, 1],
    }
    instance.data = pd.DataFrame(data)

    # Execute the function
    instance.check_binary_mismatch(keyword="main_id", tolerance=1.0 / 3600.0)
    assert_frame_equal(instance.data, pd.DataFrame(expected_data))

    ### Check the logs

    assert os.path.exists("Logs/check_binary_mismatch.txt")
    with open("Logs/check_binary_mismatch.txt") as f:
        lines = f.readlines()

    assert lines == [
        "*************************************************\n",
        "**** CHECKING BINARIES USING: host ****\n",
        "*************************************************\n",
        "\n",
        "Fixed S-type or null binary for 91 Aqr b. New binary value: A. WARNING: coordinate agreement exceeds tolerance. Maximum difference: 12.22406962954591 (binary_mismatch_flag = 1). Please check this system:\n",
        "         name    host binary letter    catalog    ra   dec\n",
        "3  91 Aqr A b  91 Aqr      A      b  Catalog 1  10.0  40.0\n",
        "4    91 Aqr b  91 Aqr   null      b  Catalog 3  20.0  50.0\n",
        "\n",
        "\n",
        "Fixed S-type or null binary for GJ 229 c. New binary value: A. \n",
        "\n",
        "\n",
        "Only S-type and null in the system for HD 156846 b. New binary value: "
        "S-type. WARNING: coordinate agreement exceeds tolerance. Maximum difference: "
        "0.0009839776494423166 (binary_mismatch_flag = 1) . Please check "
        "this system:\n",
        "          name       host  binary letter catalog          ra        dec\n",
        "8  HD 156846 b  HD 156846    null      b     oec  260.141667 -19.333611\n",
        "9  HD 156846 b  HD 156846  S-type      b      eu  260.142337 -19.334365\n",
        "\n",
        "\n",
        "WARNING: Either a complex system or a mismatch in the value of binary (binary_mismatch_flag = 2). Please check this system:  \n",
        "               name       host binary letter catalog\n",
        "5  HD 202206 (AB) c  HD 202206      A      c      eu\n",
        "6       HD 202206 c  HD 202206     AB      c     oec\n",
        "\n",
        "\n",
        "WARNING: Either a complex system or a mismatch in the value of binary (binary_mismatch_flag = 2). Please check this system:  \n",
        "            name      host  binary letter catalog\n",
        "10  HD 41004 A b  HD 41004       A      b      eu\n",
        "11  HD 41004 B b  HD 41004       B      b      eu\n",
        "12    HD 41004 b  HD 41004  S-type      b    nasa\n",
        "\n",
        "Fixed binary for complex system HD 41004b based on angular separation. New binary value: A.\n",
        "\n",
        "****host POTENTIAL BINARIES NOT TREATED HERE. They should be treated manually in replacements.ini ****\n",
        "MISSED POTENTIAL BINARY Key:ROXs 42B name: ROXs 42B b binary: AB catalog:nasa.\n",
        "*************************************************\n",
        "**** CHECKING BINARIES USING: main_id ****\n",
        "*************************************************\n",
        "\n",
        "WARNING: Flags are changing here compared to previous check. Previous value of binary_mismatch_flag was: [0]. Fixed S-type or null binary for 91 Aqr b. New binary value: A. WARNING: coordinate agreement exceeds tolerance. Maximum difference: 12.22406962954591 (binary_mismatch_flag = 1). Please check this system:\n",
        "         name main_id binary letter    catalog    ra   dec\n",
        "0  91 Aqr A b  91 Aqr      A      b  Catalog 1  10.0  40.0\n",
        "1    91 Aqr b  91 Aqr   null      b  Catalog 3  20.0  50.0\n",
        "\n",
        "\n",
        "WARNING: Flags are changing here compared to previous check. Previous value of binary_mismatch_flag was:[2]. Only S-type and "
        "null in the system for HD 156846 b. New binary value: S-type. WARNING: "
        "coordinate agreement exceeds tolerance. Maximum difference: "
        "0.0009839776494423166 (binary_mismatch_flag = 1) . Please check "
        "this system:\n",
        "          name    main_id  binary letter catalog          ra        dec\n",
        "4  HD 156846 b  HD 156846    null      b     oec  260.141667 -19.333611\n",
        "5  HD 156846 b  HD 156846  S-type      b      eu  260.142337 -19.334365\n",
        "\n",
        "\n",
        "WARNING: Flags are changing here compared to previous check. Previous value of binary_mismatch_flag was: [0, 1]. WARNING: Either "
        "a complex system or a mismatch in the value of binary (binary_mismatch_flag"
        " = 2). Please check this system:  \n",
        "               name    main_id binary letter catalog\n",
        "2  HD 202206 (AB) c  HD 202206      A      c      eu\n",
        "3       HD 202206 c  HD 202206     AB      c     oec\n",
        "\n",
        "****main_id POTENTIAL BINARIES NOT TREATED HERE. They should be treated manually in replacements.ini ****\n",
        # "MISSED POTENTIAL BINARY Key:ROXs 42B name: ROXs 42B b binary: AB catalog:nasa.\n",
    ]

    os.chdir(original_dir)


def test__prepare_columns_for_mainid_search(instance):
    data = pd.DataFrame(
        {
            "host": [
                "16 Cyg",
                "WASP-8",
                "21 Her",
                "HD 19994",
                "EPIC 203868608",
                "2MASS 1938+46",
                "2MASS 0103-55",
                "test",
                "test2",
            ],
            "binary": ["B", "A", "AB", "A", "AB", "AB", "AB", "S-type", "Rogue"],
            "alias": [
                "16 Cygni,Gaia DR2 2135550755683407232,HD 186427,HIP 96901,TIC 27533327,WDS J19418+5032",
                "2MASS J23593607-3501530,Gaia DR2 2312679845530628096,TIC 183532609,TIC-183532609,TOI-191,"
                "TYC 7522-00505-1,UCAC4 275-215468,WDS J23596-3502,WISE J235936.16-350153.1",
                "HD 147869,o Her",
                "94 Cet, 94 Cet A, Gaia DR2 3265335443260522112, HIP 14954, TIC 49845357",
                "Gaia DR2 6049656638390048896,TIC 98231712",
                "Kepler-451",
                "Delorme 1 (AB)",
                "testalias",
                "testalias2",
            ],
        }
    )

    instance.data = data
    instance.prepare_columns_for_mainid_search()

    expected = pd.DataFrame(
        {
            "host": [
                "16 Cyg",
                "WASP-8",
                "21 Her",
                "HD 19994",
                "EPIC 203868608",
                "2MASS 1938+46",
                "2MASS 0103-55",
                "test",
                "test2",
            ],
            "binary": ["B", "A", "AB", "A", "AB", "AB", "AB", "S-type", "Rogue"],
            "alias": [
                "16 Cygni,Gaia DR2 2135550755683407232,HD 186427,HIP 96901,TIC 27533327,WDS J19418+5032",
                "2MASS J23593607-3501530,Gaia DR2 2312679845530628096,TIC 183532609,TIC-183532609,TOI-191,"
                "TYC 7522-00505-1,UCAC4 275-215468,WDS J23596-3502,WISE J235936.16-350153.1",
                "HD 147869,o Her",
                "94 Cet, 94 Cet A, Gaia DR2 3265335443260522112, HIP 14954, TIC 49845357",
                "Gaia DR2 6049656638390048896,TIC 98231712",
                "Kepler-451",
                "Delorme 1 (AB)",
                "testalias",
                "testalias2",
            ],
            "hostbinary": [
                "16 Cyg B",
                "WASP-8 A",
                "21 Her AB",
                "HD 19994 A",
                "EPIC 203868608 AB",
                "2MASS 1938+46 AB",
                "2MASS 0103-55 AB",
                "test",
                "test2",
            ],
            "aliasbinary": [
                "16 Cygni B,Gaia DR2 2135550755683407232 B,HD 186427 B,HIP 96901 B,TIC 27533327 B,WDS J19418+5032 B",
                "2MASS J23593607-3501530 A,Gaia DR2 2312679845530628096 A,TIC 183532609 A,TIC-183532609 A,TOI-191 A,"
                "TYC 7522-00505-1 A,UCAC4 275-215468 A,WDS J23596-3502 A,WISE J235936.16-350153.1 A",
                "HD 147869 AB,o Her AB",
                "94 Cet A,94 Cet A A,Gaia DR2 3265335443260522112 A,HIP 14954 A,TIC 49845357 A",
                "Gaia DR2 6049656638390048896 AB,TIC 98231712 AB",
                "Kepler-451 AB",
                "Delorme 1 (AB) AB",
                "testalias",
                "testalias2",
            ],
            "hostbinary2": [
                "16 CygB",
                "WASP-8A",
                "21 HerAB",
                "HD 19994A",
                "EPIC 203868608AB",
                "2MASS 1938+46AB",
                "2MASS 0103-55AB",
                "test",
                "test2",
            ],
            "aliasbinary2": [
                "16 CygniB,Gaia DR2 2135550755683407232B,HD 186427B,HIP 96901B,TIC 27533327B,WDS J19418+5032B",
                "2MASS J23593607-3501530A,Gaia DR2 2312679845530628096A,TIC 183532609A,TIC-183532609A,TOI-191A,"
                "TYC 7522-00505-1A,UCAC4 275-215468A,WDS J23596-3502A,WISE J235936.16-350153.1A",
                "HD 147869AB,o HerAB",
                "94 CetA,94 Cet AA,Gaia DR2 3265335443260522112A,HIP 14954A,TIC 49845357A",
                "Gaia DR2 6049656638390048896AB,TIC 98231712AB",
                "Kepler-451AB",
                "Delorme 1 (AB)AB",
                "testalias",
                "testalias2",
            ],
            "main_id": ["", "", "", "", "", "", "", "", ""],
            "list_id": ["", "", "", "", "", "", "", "", ""],
            "angular_separation": ["", "", "", "", "", "", "", "", ""],
            "main_id_provenance": ["", "", "", "", "", "", "", "", ""],
            "main_id_ra": [
                np.nan,
                np.nan,
                np.nan,
                np.nan,
                np.nan,
                np.nan,
                np.nan,
                np.nan,
                np.nan,
            ],
            "main_id_dec": [
                np.nan,
                np.nan,
                np.nan,
                np.nan,
                np.nan,
                np.nan,
                np.nan,
                np.nan,
                np.nan,
            ],
            "angsep": [-1.0, -1.0, -1.0, -1.0, -1.0, -1.0, -1.0, -1.0, -1.0],
        }
    )
    assert sorted(instance.data.columns) == sorted(instance.data.columns)

    assert_frame_equal(
        instance.data[
            [
                "host",
                "binary",
                "hostbinary",
                "hostbinary2",
                "main_id",
                "main_id_ra",
                "main_id_dec",
                "angsep",
                "main_id_provenance",
                "angular_separation",
            ]
        ],
        expected[
            [
                "host",
                "binary",
                "hostbinary",
                "hostbinary2",
                "main_id",
                "main_id_ra",
                "main_id_dec",
                "angsep",
                "main_id_provenance",
                "angular_separation",
            ]
        ],
    )
    # the order of alias is not important
    for i in instance.data.index:
        assert sorted(instance.data.at[i, "alias"].split(",")) == sorted(
            expected.at[i, "alias"].split(",")
        )
        assert sorted(instance.data.at[i, "aliasbinary"].split(",")) == sorted(
            expected.at[i, "aliasbinary"].split(",")
        )
        assert sorted(instance.data.at[i, "aliasbinary2"].split(",")) == sorted(
            expected.at[i, "aliasbinary2"].split(",")
        )


def test__fill_mainid_provenance_column(instance):
    instance.data = pd.DataFrame({"main_id": ["*   4 Mon"], "main_id_provenance": [""]})
    instance.fill_mainid_provenance_column("SIMBAD")

    assert instance.data.main_id_provenance.values == ["SIMBAD"]


def test__simbad_list_host_search(tmp_path, instance):
    data = pd.DataFrame(
        {
            "host": ["HD 114762", "PSR B1620-26", "51 Peg", "16 Cyg", "nonexisting"],
            "main_id": ["", "", "", "", ""],
            "list_id": ["", "", "", "", ""],
            "main_id_ra": ["", "", "", "", ""],
            "main_id_dec": ["", "", "", "", ""],
            "angsep": [-1.0, -1.0, -1.0, -1.0, -1.0],
        }
    )

    instance.data = data

    # Call the function
    with LogCapture() as log:
        instance.simbad_list_host_search("host")
        log = pd.DataFrame(list(log), columns=["user", "info", "message"])

    expected = [
        "List of unique star names 5 of which successful SIMBAD queries 4",
    ]
    assert (pd.Series(expected).isin(log["message"].to_list())).all()

    # Expected output DataFrame
    expected_data = pd.DataFrame(
        {
            "host": [
                "HD 114762",
                "PSR B1620-26",
                "51 Peg",
                "16 Cyg",
                "nonexisting",
            ],
            "main_id": ["HD 114762", "PSR B1620-26", "*  51 Peg", "*  16 Cyg", ""],
            "list_id": [
                "HIP 64426,AP J13121982+1731016,Gaia DR3 3937211745905473024,TIC 138172859,SBC9 2406,LSPM J1312+1731,"
                "TYC 1454-315-1,ASCC  868467,2MASS J13121982+1731016,USNO-B1.0 1075-00258599,AG+17 1351,AGKR 11813,"
                "BD+18  2700,Ci 20  766,Ci 18 1695,FK5 5165,G  63-9,GC 17881,GCRV  7854,GEN# +1.00114762,HD 114762,"
                "HIC  64426,LFT  980,LHS  2693,LTT 13819,N30 3028,NLTT 33242,PM 13099+1747,PPM 129656,SAO 100458,"
                "SKY# 24417,SPOCS  555,TD1 16696,UBV   11938,UBV M  19151,YZ  17  4855,YZC 18  4855,uvby98 100114762,"
                "WISEA J131219.31+173101.6,Gaia DR1 3937211741607576576,WEB 11388,WDS J13123+1731A,HD 114762A,"
                "** PAT   47A,Gaia DR2 3937211745904553600",
                "PSR B1620-26,PSR J1623-2631,[BPH2004] CX 12,EQ J1623-2631",
                "GJ 882,Gaia DR3 2835207319109249920,TIC 139298196,PLX 5568,LSPM J2257+2046,TYC 1717-2193-1,"
                "ASCC  826013,2MASS J22572795+2046077,USNO-B1.0 1107-00589893,*  51 Peg,AG+20 2595,BD+19  5036,"
                "CSV 102222,GC 32003,GCRV 14411,GEN# +1.00217014,HD 217014,HIC 113357,HIP 113357,HR  8729,JP11  3558,"
                "LTT 16750,N30 5052,NLTT 55385,NSV 14374,PPM 114985,ROT  3341,SAO  90896,SKY# 43603,SPOCS  990,"
                "TD1 29480,UBV M  26734,UBV   19678,YPAC 218,YZ   0  1227,YZ  20  9382,uvby98 100217014,"
                "Gaia DR2 2835207319109249920,PLX 5568.00,IRAS 22550+2030,AKARI-IRC-V1 J2257280+204608,"
                "NAME Helvetios,WEB 20165,** RBR   21A,WDS J22575+2046A,CNS5 5664",
                "IDS 19392+5017 AB,CCDM J19418+5031AB,WDS J19418+5032AB,*  16 Cyg,ADS 12815 AB,** STFA   46,"
                "IRAS 19404+5024,IRAS F19404+5024",
                "",
            ],
            "main_id_ra": [
                198.082254,
                245.909257,
                344.366585,
                295.454542,
                "",
            ],
            "main_id_dec": [
                17.517119537310283,
                -26.531602499999998,
                20.768832511140005,
                50.52544444444444,
                "",
            ],
            "angsep": [0.0, 0.0, 0.0, 0.0, -1],
        }
    )
    # Assert the DataFrame content is as expected
    assert_frame_equal(
        instance.data[["host", "main_id", "main_id_ra", "main_id_dec", "angsep"]],
        expected_data[["host", "main_id", "main_id_ra", "main_id_dec", "angsep"]],
    )
    # the order of list_id is not important
    for i in instance.data.index:
        assert sorted(instance.data.at[i, "list_id"].split(",")) == sorted(
            expected_data.at[i, "list_id"].split(",")
        )


def test__simbad_list_alias_search(tmp_path, instance):
    data = pd.DataFrame(
        {
            "host": ["51 Peg", "2MASS J1207 A"],
            "alias": [
                "GJ 882,BD+19 5036,HIP 113357,HR 8729,HD 217014,51 Peg,Gaia DR2 2835207319109249920,Helvetios,"
                "SAO 90896,TYC 1717-2193-1",
                "2MASS J1207,2MASS J1207-39,2MASS J12073346-3932539,2MASS J1207B,2MASS1207-3932 B,"
                "2MASSW J1207334-393254,Gaia DR2 3459372646830687104,TIC 102076870,TWA 27b",
            ],
            "main_id": ["", ""],
            "list_id": ["", ""],
            "main_id_ra": ["", ""],
            "main_id_dec": ["", ""],
            "angsep": [-1.0, -1.0],
        }
    )

    instance.data = data  # Replace YourClassName with the actual class name
    # Call the function
    with LogCapture() as log:
        instance.simbad_list_alias_search("alias")
        log = pd.DataFrame(list(log), columns=["user", "info", "message"])
        assert (
            "WARNING, MULTIPLE ALIASES NOT IN AGREEMENT alias ['2MASS J12073346-3932539', 'TWA 27b'] main_id ["
            "'TWA 27', 'TWA 27B']" in log["message"].tolist()
        )

    # Expected output DataFrame
    expected_data = pd.DataFrame(
        {
            "host": ["51 Peg", "2MASS J1207 A"],
            "alias": [
                "GJ 882,BD+19 5036,HIP 113357,HR 8729,HD 217014,51 Peg,Gaia DR2 2835207319109249920,Helvetios,"
                "SAO 90896,TYC 1717-2193-1",
                "2MASS J1207,2MASS J1207-39,2MASS J12073346-3932539,2MASS J1207B,2MASS1207-3932 B,"
                "2MASSW J1207334-393254,Gaia DR2 3459372646830687104,TIC 102076870,TWA 27b",
            ],
            "main_id": ["*  51 Peg", "TWA 27"],
            "list_id": [
                "GJ 882,Gaia DR3 2835207319109249920,TIC 139298196,PLX 5568,LSPM J2257+2046,TYC 1717-2193-1,"
                "ASCC  826013,2MASS J22572795+2046077,USNO-B1.0 1107-00589893,*  51 Peg,AG+20 2595,BD+19  5036,"
                "CSV 102222,GC 32003,GCRV 14411,GEN# +1.00217014,HD 217014,HIC 113357,HIP 113357,HR  8729,JP11  3558,"
                "LTT 16750,N30 5052,NLTT 55385,NSV 14374,PPM 114985,ROT  3341,SAO  90896,SKY# 43603,SPOCS  990,"
                "TD1 29480,UBV M  26734,UBV   19678,YPAC 218,YZ   0  1227,YZ  20  9382,uvby98 100217014,"
                "Gaia DR2 2835207319109249920,PLX 5568.00,IRAS 22550+2030,AKARI-IRC-V1 J2257280+204608,"
                "NAME Helvetios,WEB 20165,** RBR   21A,WDS J22575+2046A,CNS5 5664",
                "** CVN   12A,2MASS J12073346-3932539,2MASSW J1207334-393254,DENIS J120733.4-393254,Gaia DR2 "
                "3459372646830687104,Gaia DR3 3459372646830687104,HIDDEN NAME 2M1207,HIDDEN NAME 2M1207A,"
                "TIC 102076870,TWA 27,TWA 27A,USNO-B1.0 0504-00258166,WDS J12076-3933A,WISE J120733.42-393254.2,"
                "WISEA J120733.41-393254.2",
            ],
            "main_id_ra": [344.366585, 181.88944813535],
            "main_id_dec": [20.768833, -39.54833795144],
            "angsep": [0.0, 0.0],
        }
    )

    # Assert the DataFrame content is as expected
    assert_frame_equal(
        instance.data[["host", "main_id", "main_id_ra", "main_id_dec", "angsep"]],
        expected_data[["host", "main_id", "main_id_ra", "main_id_dec", "angsep"]],
    )
    # the order of list_id is not important
    for i in instance.data.index:
        assert sorted(instance.data.at[i, "list_id"].split(",")) == sorted(
            expected_data.at[i, "list_id"].split(",")
        )


def test__get_host_info_from_simbad(instance):
    data = {
        "name": [
            "16 Cyg B b",
            "WASP-8 b",
            "21 Her (AB) b",
            "HD 19994 A b",
            "EPIC 203868608 (AB) b",
            "Kepler-451 b",
            "2MASS 0103-55 (AB) b",
        ],
        "host": [
            "16 Cyg",
            "WASP-8",
            "21 Her",
            "HD 19994",
            "EPIC 203868608",
            "2MASS 1938+46",
            "2MASS 0103-55",
        ],
        "binary": ["B", "A", "AB", "A", "AB", "AB", "AB"],
        "alias": [
            "16 Cygni,Gaia DR2 2135550755683407232,HD 186427,HIP 96901,TIC 27533327,WDS J19418+5032",
            "2MASS J23593607-3501530,Gaia DR2 2312679845530628096,TIC 183532609,TIC-183532609,TOI-191,"
            "TYC 7522-00505-1,UCAC4 275-215468,WDS J23596-3502,WISE J235936.16-350153.1",
            "HD 147869,o Her",
            "94 Cet, 94 Cet A, Gaia DR2 3265335443260522112, HIP 14954, TIC 49845357",
            "Gaia DR2 6049656638390048896,TIC 98231712",
            "Kepler-451",
            "Delorme 1 (AB)",
        ],
        "hostbinary": [
            "16 Cyg B",
            "WASP-8 A",
            "21 Her AB",
            "HD 19994 A",
            "EPIC 203868608 AB",
            "2MASS 1938+46 AB",
            "2MASS 0103-55 AB",
        ],
        "aliasbinary": [
            "16 Cygni B,Gaia DR2 2135550755683407232 B,HD 186427 B,HIP 96901 B,TIC 27533327 B,WDS J19418+5032 B",
            "2MASS J23593607-3501530 A,Gaia DR2 2312679845530628096 A,TIC 183532609 A,TIC-183532609 A,TOI-191 A,"
            "TYC 7522-00505-1 A,UCAC4 275-215468 A,WDS J23596-3502 A,WISE J235936.16-350153.1 A",
            "HD 147869 AB,o Her AB",
            "94 Cet A,94 Cet A A,Gaia DR2 3265335443260522112 A,HIP 14954 A,TIC 49845357 A",
            "Gaia DR2 6049656638390048896 AB,TIC 98231712 AB",
            "Kepler-451 AB",
            "Delorme 1 (AB) AB",
        ],
        "hostbinary2": [
            "16 CygB",
            "WASP-8A",
            "21 HerAB",
            "HD 19994A",
            "EPIC 203868608AB",
            "2MASS 1938+46AB",
            "2MASS 0103-55AB",
        ],
        "aliasbinary2": [
            "16 CygniB,Gaia DR2 2135550755683407232B,HD 186427B,HIP 96901B,TIC 27533327B,WDS J19418+5032B",
            "2MASS J23593607-3501530A,Gaia DR2 2312679845530628096A,TIC 183532609A,TIC-183532609A,TOI-191A,"
            "TYC 7522-00505-1A,UCAC4 275-215468A,WDS J23596-3502A,WISE J235936.16-350153.1A",
            "HD 147869AB,o HerAB",
            "94 CetA,94 Cet AA,Gaia DR2 3265335443260522112A,HIP 14954A,TIC 49845357A",
            "Gaia DR2 6049656638390048896AB,TIC 98231712AB",
            "Kepler-451AB",
            "Delorme 1 (AB)AB",
        ],
        "main_id": ["", "", "", "", "", "", ""],
        "list_id": ["", "", "", "", "", "", ""],
        "main_id_ra": [np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan],
        "main_id_dec": [np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan],
        "angsep": [-1.0, -1.0, -1.0, -1.0, -1.0, -1.0, -1.0],
        "angular_separation": ["", "", "", "", "", "", ""],
        "main_id_provenance": ["", "", "", "", "", "", ""],
    }
    # 16 Cyg B b found with HOST+ +BINARY
    # WASP-8 b found with ALIAS + BINARY
    # 21 Her (AB) b found with HOST+BINARY
    # HD 19994 A b found with ALIAS+BINARY
    # EPIC 203868608 (AB) b found with PURE HOST
    # Kepler-451 b found with PURE ALIAS
    # 2MASS 0103-55 (AB) b not found

    instance.data = pd.DataFrame(data)
    with LogCapture() as log:
        instance.get_host_info_from_simbad()
        log = pd.DataFrame(list(log), columns=["user", "info", "message"])

    expected = [
        "HOST+ +BINARY Simbad Check",
        "List of unique star names 7 of which successful SIMBAD queries 1",
        "Rows still missing main_id after host search 6",
        "ALIAS+ +BINARY Simbad Check",
        "Rows still missing main_id after alias search 5",
        "HOST+BINARY Simbad Check",
        "List of unique star names 5 of which successful SIMBAD queries 1",
        "Rows still missing main_id after host search 4",
        "ALIAS+BINARY Simbad Check",
        "Rows still missing main_id after alias search 3",
        "PURE HOST Simbad Check",
        "List of unique star names 3 of which successful SIMBAD queries 1",
        "Rows still missing main_id after host search 2",
        "PURE ALIAS Simbad Check",
        "Rows still missing main_id after alias search 1",
    ]
    assert (pd.Series(expected).isin(log["message"].to_list())).all()

    expected_output = {
        "list_id": [
            "CNS5 4878,LTT 15751,*  16 Cyg B,** STF 4046B,ADS 12815 B,AG+50 1408,AKARI-IRC-V1 J1941518+503102,ASCC  271120,"
            "BD+50  2848,CCDM J19418+5031B,GC 27285,GCRV 12084,GEN# +1.00186427,GJ 765.1 B,HD 186427,HIC  96901,"
            "HIP 96901,HR  7504,IDS 19392+5017 B,KIC 12069449,LSPM J1941+5031E,2MASS J19415198+5031032,NLTT 48138,"
            "PPM  37673,ROT  2840,SAO  31899,SKY# 36807,SPOCS  855,TIC 27533327,TYC 3565-1525-1,UBV   16780,"
            "UBV M  24082,USNO-B1.0 1405-00322540,USNO 890,WDS J19418+5032B,WEB 17005,WISEA J194151.82+503102.2,"
            "YZ  50  6150,Gaia DR3 2135550755683407232,Gaia DR2 2135550755683407232,Gaia DR1 2135550854464294784",
            "CD-35 16019A,** B 2511A,CCDM J23596-3502A,CD-35 16019,CPC 18 12094,CPD-35  9465,GSC 07522-00505,"
            "IDS 23544-3535 A,2MASS J23593607-3501530,PPM 304426,SAO 214901,SPOCS 3245,TIC 183532609,TOI-191,"
            "TYC 7522-505-1,UCAC2  16954660,UCAC3 110-468375,UCAC4 275-215468,WASP-8,WDS J23596-3502A,"
            "Gaia DR3 2312679845530628096,Gaia DR2 2312679845530628096,Gaia DR1 2312679841235149440",
            "* o Her,*  21 Her,AG+07 2044,BD+07  3164,FK5 1429,GC 22058,GCRV  9437,GEN# +1.00147869,GSC 00381-01598,"
            "HD 147869,HIC  80351,HIP 80351,HR  6111,2MASS J16241083+0656534,N30 3675,PMC 90-93  1008,PPM 162584,"
            "Renson 41690,SAO 121568,SBC7   573,SBC9 901,SKY# 29573,TD1 19139,TIC 369080491,TYC  381-1598-1,"
            "UBV   21316,UBV M  21407,uvby98 100147869,WEB 13596,YZ   7  7306,Gaia DR2 4439556480967874432,"
            "Gaia DR3 4439556480967874432,Gaia DR1 4439556476666646528",
            "LTT  1515,*  94 Cet,** HJ  663A,ADS  2406 A,AG-01  300,AKARI-IRC-V1 J0312465-011146,BD-01   457,"
            "CCDM J03128-0112A,CSI-01   457  1,CSI-01   457  3,CSI-01   457  2,FK5  116,GC  3838,GCRV  1775,"
            "GEN# +1.00019994,GJ 128,HD  19994,HIC  14954,HIP 14954,HR   962,IDS 03077-0134 A,IRAS 03102-0122,"
            "2MASS J03124644-0111458,N30  656,NLTT 10224,PLX  663,PMC 90-93    84,PPM 175267,ROT   431,SAO 130355,"
            "SKY#  4813,SPOCS  155,TD1  1984,TIC 49845357,TYC 4708-1423-1,UBV    3104,UCAC3 178-9414,"
            "UCAC4 445-004277,uvby98 100019994,WDS J03128-0112A,WEB  2887,WISEA J031246.58-011146.4,YZ  91   684,"
            "YZ   0  3372,[RHG95]   572,Gaia DR3 3265335443260522112,Gaia DR2 3265335443260522112,CNS5 807",
            "2MASS J16171898-2437186,AP J16171898-2437186,EPIC 203868608,TIC 98231712,UGCS J161718.97-243718.7,WISEA J161718.97-243718.9,"
            "Gaia DR2 6049656638390048896,Gaia DR3 6049656638390048896",
            "TYC 3556-3568-1,ASAS J193833+4604.0,ATO J294.6359+46.0664,GSC2.3 N2JF000803,GSC2 N0303123803,GSC 03556-03568,Kepler-451,KIC 9472174,LAMOST J193832.60+460359.1,LAMOST J193832.62+460359.1,LAMOST J193832.61+460359.1,2MASS J19383260+4603591,NSVS   5629361,TIC 271164763,UCAC3 273-158867,USNO-B1.0 1360-00318562,EQ J1938+4603,Gaia DR3 2080063931448749824,Gaia DR2 2080063931448749824",
            "",
        ],
        "main_id": [
            "*  16 Cyg B",
            "CD-35 16019",
            "* o Her",
            "*  94 Cet",
            "2MASS J16171898-2437186",
            "Kepler-451",
            "",
        ],
        "main_id_ra": [
            295.4665525,
            359.90029625,
            246.04511958333327,
            48.19348541666667,
            244.32909499999997,
            294.6358845833333,
            np.nan,
        ],
        "main_id_dec": [
            50.51752472222222,
            -35.0313675,
            6.948210277777778,
            -1.1960986111111112,
            -24.621871944444443,
            46.066426388888885,
            np.nan,
        ],
        "angsep": [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, -1],
        "angular_separation": ["", "", "", "", "", "", ""],
        "main_id_provenance": [
            "SIMBAD",
            "SIMBAD",
            "SIMBAD",
            "SIMBAD",
            "SIMBAD",
            "SIMBAD",
            "",
        ],
    }
    expected_output = pd.DataFrame(expected_output)

    # Assert the DataFrame content is as expected
    assert_frame_equal(
        instance.data[
            ["main_id", "main_id_ra", "main_id_dec", "angsep", "main_id_provenance"]
        ],
        expected_output[
            ["main_id", "main_id_ra", "main_id_dec", "angsep", "main_id_provenance"]
        ],
    )
    # the order of list_id is not important
    for i in instance.data.index:
        assert sorted(instance.data.at[i, "list_id"].split(",")) == sorted(
            expected_output.at[i, "list_id"].split(",")
        )


def test__get_coordinates_from_simbad(tmp_path, instance):
    original_dir = os.getcwd()

    os.chdir(tmp_path)  # Create a temporary in-memory configuration object
    os.mkdir("Logs/")
    data = {
        "hostbinary": ["51 Peg", "2MASS 0103-55 AB"],
        "main_id": ["", ""],
        "list_id": ["", ""],
        "ra": [344.3667, 15.9],
        "dec": [20.7689, -55.2656],
        "catalog": ["nasa", "eu"],
        "main_id_provenance": ["", ""],
        "main_id_ra": [np.nan, np.nan],
        "main_id_dec": [np.nan, np.nan],
        "angsep": [-1.0, -1.0],
    }
    expected_df = pd.DataFrame(
        {
            "hostbinary": ["51 Peg", "2MASS 0103-55 AB"],
            "main_id": ["*  51 Peg", ""],
            "list_id": [
                "LTT 16750,*  51 Peg,** RBR   21A,AG+20 2595,AKARI-IRC-V1 J2257280+204608,ASCC  826013,BD+19  5036,CNS5 5664,"
                "CSV 102222,GC 32003,GCRV 14411,GEN# +1.00217014,GJ 882,HD 217014,HIC 113357,HIP 113357,HR  8729,"
                "IRAS 22550+2030,JP11  3558,LSPM J2257+2046,2MASS J22572795+2046077,N30 5052,NAME Helvetios,"
                "NLTT 55385,NSV 14374,PLX 5568,PLX 5568.00,PPM 114985,ROT  3341,SAO  90896,SKY# 43603,SPOCS  990,"
                "TD1 29480,TIC 139298196,TYC 1717-2193-1,UBV M  26734,UBV   19678,USNO-B1.0 1107-00589893,"
                "uvby98 100217014,WDS J22575+2046A,WEB 20165,YPAC 218,YZ   0  1227,YZ  20  9382,"
                "Gaia DR3 2835207319109249920,Gaia DR2 2835207319109249920",
                "",
            ],
            "ra": [344.3667, 15.9],
            "dec": [20.7689, -55.2656],
            "catalog": ["nasa", "eu"],
            "main_id_ra": [344.36658535524, np.nan],
            "main_id_dec": [20.768832511140005, np.nan],
            "main_id_provenance": ["SIMBADCOORD", ""],
            "angsep": [0.45601200000000003, -1.0],
        }
    )
    instance.data = pd.DataFrame(data)
    with LogCapture() as log:
        instance.get_coordinates_from_simbad(1.0 / 3600.0)
        assert (
            "After coordinate check on SIMBAD at tolerance 1.0 arcsec, the residuals are: 1. Maximum angular "
            "separation: 0.45601200000000003" in log.actual()[-1][-1]
        )
    # import re
    # if not str(re.search("[\s\d][b-i]$", instance.data.at[0,'main_id'], re.M)) == "None":
    #     with LogCapture() as log:
    #         instance.polish_main_id()
    assert_frame_equal(
        instance.data[
            [
                "hostbinary",
                "main_id",
                "main_id_ra",
                "main_id_dec",
                "angsep",
                "main_id_provenance",
            ]
        ],
        expected_df[
            [
                "hostbinary",
                "main_id",
                "main_id_ra",
                "main_id_dec",
                "angsep",
                "main_id_provenance",
            ]
        ],
    )
    # the order of list_id is not important
    for i in instance.data.index:
        assert sorted(instance.data.at[i, "list_id"].split(",")) == sorted(
            expected_df.at[i, "list_id"].split(",")
        )

    os.chdir(original_dir)


def test__get_host_info_from_tic(instance):
    expected = pd.DataFrame(
        {
            "name": ["TOI-6775.01", "EPIC 251292508.01"],
            "catalog_name": ["TOI-6775.01", "EPIC 251292508.01"],
            "catalog_host": ["TIC 100263315", "EPIC 251292508"],
            "alias": [
                "2MASS J20260027-4855132,Gaia DR2 6668227036766532864,TIC-100263315,TOI-6775,UCAC4 206-182296,"
                "WISE J202600.26-485513.4",
                "Gaia DR2 630836794912890624,TIC 55361028",
            ],
            "host": ["TIC 100263315", "EPIC 251292508"],
            "main_id": ["TIC 100263315", "TIC 55361028"],
            "list_id": [
                "UCAC4 206-182296,2MASS J20260027-4855132,WISE J202600.26-485513.4,Gaia DR2 6668227036766532864",
                "UCAC4 530-049849,2MASS J09220352+1548296,WISE J092203.55+154828.6,Gaia DR2 630836794912890624",
            ],
            "main_id_ra": [306.501164, 140.51471830658],
            "main_id_dec": [-48.92036, 15.80819282436],
            "main_id_provenance": ["TIC", "TIC"],
            "angsep": [0.0, 0.0],
        }
    )

    data = {
        "name": ["TOI-6775.01", "EPIC 251292508.01"],
        "catalog_name": ["TOI-6775.01", "EPIC 251292508.01"],
        "catalog_host": ["TIC 100263315", "EPIC 251292508"],
        "alias": [
            "2MASS J20260027-4855132,Gaia DR2 6668227036766532864,TIC-100263315,TOI-6775,UCAC4 206-182296,"
            "WISE J202600.26-485513.4",
            "Gaia DR2 630836794912890624,TIC 55361028",
        ],
        "host": ["TIC 100263315", "EPIC 251292508"],
        "main_id": ["", ""],
        "list_id": ["", ""],
        "main_id_ra": ["", ""],
        "main_id_dec": ["", ""],
        "main_id_provenance": ["", ""],
        "angsep": [-1.0, -1.0],
    }

    instance.data = pd.DataFrame(data)
    with LogCapture() as log:
        instance.get_host_info_from_tic()
        log = pd.DataFrame(list(log), columns=["user", "info", "message"])

    expected_log = [
        "TIC host check",
        "List of unique star names with a TIC host 1 of which successful TIC queries 1",
        "Rows still missing main_id after TIC host search 1",
        "TIC alias check",
        "List of unique star names with a TIC alias 1 of which successful TIC queries 1",
        "Rows still missing main_id after TIC alias search 0",
    ]

    assert (pd.Series(expected_log).isin(log["message"].to_list())).all()
    #
    assert set(instance.data.columns) == set(expected.columns)
    assert_frame_equal(instance.data, expected)


def test__get_coordinates_info_from_tic(instance):
    data = pd.DataFrame(
        {
            "hostbinary": ["2MASS J1759-2739"],
            "ra": [269.95833335],
            "dec": [-27.660277799],
            "main_id_ra": [""],
            "main_id_dec": [""],
            "main_id": [""],
            "list_id": [""],
            "main_id_provenance": [""],
            "angsep": [-1],
        }
    )
    expected = pd.DataFrame(
        {
            "hostbinary": ["2MASS J1759-2739"],
            "ra": [269.958333],
            "dec": [-27.660278],
            "main_id_ra": [269.95822747273],
            "main_id_dec": [-27.66038023884],
            "main_id": ["TIC 1439998925"],
            "list_id": ["Gaia DR2 4062782845836539520"],
            "main_id_provenance": ["TICCOORD"],
            "angsep": [0.49996799999999997],
        }
    )
    instance.data = data
    tolerance = 1 / 3600
    with LogCapture() as log:
        instance.get_coordinates_from_tic(1.0 / 3600.0)
        log = pd.DataFrame(list(log), columns=["user", "info", "message"])

    expected_log = [
        "After coordinate check on TIC at tolerance 1.0 arcsec, the residuals are: 0. Maximum angular separation: 0.49996799999999997"
    ]

    assert (pd.Series(expected_log).isin(log["message"].to_list())).all()
    #
    assert set(instance.data.columns) == set(expected.columns)
    assert_frame_equal(instance.data, expected)

    pass


def test__check_coordinates(tmp_path, instance):
    original_dir = os.getcwd()

    os.chdir(tmp_path)  # Create a temporary in-memory configuration object
    os.mkdir("Logs/")
    data_successful = {
        "name": ["51 Peg b", "51 Peg b"],
        "host": ["51 Peg", "51 Peg"],
        "binary": ["", ""],
        "catalog": ["eu", "oec"],
        "status": ["CONFIRMED", "CONFIRMED"],
        "letter": ["b", "b"],
        "main_id": ["", ""],
        "ra": [344.3667, 344.36],
        "dec": [20.7689, 20.77],
    }

    instance.data = pd.DataFrame(data_successful)
    with LogCapture() as log:
        instance.check_coordinates(tolerance=0.03)
        assert "Found 0 mismatched RA" in log.actual()[0][-1]
        assert "Found 0 mismatched DEC" in log.actual()[1][-1]

        assert list(instance.data["coordinate_mismatch"]) == ["", ""]
    data_unsuccessful = {
        "name": ["51 Peg b", "51 Peg b"],
        "host": ["51 Peg", "51 Peg"],
        "binary": ["", ""],
        "catalog": ["eu", "oec"],
        "status": ["CONFIRMED", "CONFIRMED"],
        "letter": ["b", "b"],
        "main_id": ["", ""],
        "ra": [344.3667, 111.556],
        "dec": [20.7689, -22.77],
    }

    instance.data = pd.DataFrame(data_unsuccessful)
    with LogCapture() as log:
        instance.check_coordinates(tolerance=1 / 3600)
        assert "Found 1 mismatched RA" in log.actual()[0][-1]
        assert "Found 1 mismatched DEC" in log.actual()[1][-1]
        assert list(instance.data["coordinate_mismatch"]) == ["RADEC", "RADEC"]

    assert os.path.exists("Logs/check_coordinates.txt")

    with open("Logs/check_coordinates.txt") as f:
        lines = f.readlines()
        assert lines[0:4] == [
            "*** MISMATCH ON RA at tolerance 0.000278 *** \n",
            "       name    host binary letter catalog        ra\n",
            "0  51 Peg b  51 Peg             b      eu  344.3667\n",
            "1  51 Peg b  51 Peg             b     oec  111.5560\n",
        ]

        assert lines[4:8] == [
            "*** MISMATCH ON DEC at tolerance 0.000278 *** \n",
            "       name    host binary letter catalog      dec\n",
            "0  51 Peg b  51 Peg             b      eu  20.7689\n",
            "1  51 Peg b  51 Peg             b     oec -22.7700\n",
        ]

    os.chdir(original_dir)


def test__replace_old_new_identifier(instance):
    # tested in polish_main_id
    pass


def test__polish_main_id(tmp_path, instance):
    original_dir = os.getcwd()

    os.chdir(tmp_path)  # Create a temporary in-memory configuration object
    os.mkdir("Logs/")
    # Test data
    data = {
        "main_id": [
            "NAME V672 Lyr b",
            "BD+18  2050B",
            "BD+18  2050 B",  # already correct binary
            "*  51 Peg (AB)",  # test for (AB)
            "* 51 PegAB",  # test for AB and disagreement on binary
            "*  30 Ari B",
        ],
        "main_id_ra": [
            286.041048,
            132.508398,
            132.508398,
            344.366585,
            344.366585,
            39.240606,
        ],
        "main_id_dec": [
            36.632624,
            17.874189,
            17.874189,
            20.768833,
            20.768833,
            24.648056,
        ],
        "list_id": [
            "TrES-1,TOI-1236b,TOI-1236.01,HIDDEN NAME TrES-1b,NAME V672 Lyr b",
            "",
            "otherids",
            "",
            "ids",
            "unchanged id",
        ],
        "name": [
            "TrES-1 b",
            "EPIC 211839430.01",
            "EPIC 211839430.01",
            "51 Peg b",
            "51 Peg b",
            "30 Ari B b",
        ],
        "host": [
            "TrES-1",
            "EPIC 211839430",
            "EPIC 211839430",
            "51 Peg",
            "51 Peg",
            "30 Ari B",
        ],
        "catalog": ["eu", "epic", "toi", "nasa", "oec", "eu"],
        "binary": ["", "", "B", "S-type", "B", "B"],
    }
    instance.data = pd.DataFrame(data)

    # Save the test data to a temporary file for testing

    with LogCapture() as log:
        instance.polish_main_id()
        log = pd.DataFrame(list(log), columns=["user", "info", "message"])
        assert (
            "Removed planet/binary letter from main_id. It happens 6 times."
            in log["message"].tolist()
        )
    expected_data = {
        "main_id": [
            "V* V672 Lyr",
            "BD+18  2050",
            "BD+18  2050",
            "*  51 Peg",  # test for (AB)
            "*  51 Peg",  # test for AB
            "*  30 Ari B",  # weird one
        ],
        "main_id_ra": [
            286.04104817344,
            132.50595833333333,
            132.50595833333333,
            344.36658535524,
            344.36658535524,
            39.240606,
        ],
        "main_id_dec": [
            36.63262407786001,
            17.874583333333334,
            17.874583333333334,
            20.768832511140005,
            20.768832511140005,
            24.648056,
        ],
        "list_id": [
            "TrES-1,TOI-1236b,TOI-1236.01,HIDDEN NAME TrES-1b,NAME V672 Lyr b,TIC 120757718,2MASS J19040985+3637574,GSC 02652-01324,TYC 2652-1324-1,NAME TrES-1 Parent Star,** ADM    4A,** FAE    1A,WDS J19042+3638A,Gaia DR3 2098964849867337856,Gaia DR1 2098964845566110720,V* V672 Lyr,TOI-1236,Gaia DR2 2098964849867337856,KIC 875283",
            "YZ  18  3535,BD+18  2050,CCDM J08500+1752AB,WDS J08500+1752AB,IDS 08444+1815 AB,** KU   33,ADS  7030 AB,CSI+18  2050  1",
            "YZ  18  3535,BD+18  2050,CCDM J08500+1752AB,WDS J08500+1752AB,IDS 08444+1815 AB,** KU   33,ADS  7030 AB,CSI+18  2050  1,otherids",
            "NLTT 55385,HR  8729,Gaia DR3 2835207319109249920,BD+19  5036,CNS5 5664,PLX 5568.00,IRAS 22550+2030,NSV 14374,ROT  3341,LTT 16750,N30 5052,HD 217014,** RBR   21A,WDS J22575+2046A,PPM 114985,*  51 Peg,PLX 5568,YPAC 218,HIP 113357,UBV M  26734,UBV   19678,YZ   0  1227,YZ  20  9382,Gaia DR2 2835207319109249920,TIC 139298196,uvby98 100217014,TD1 29480,SPOCS  990,SKY# 43603,AKARI-IRC-V1 J2257280+204608,GEN# +1.00217014,JP11  3558,GJ 882,SAO  90896,AG+20 2595,LSPM J2257+2046,TYC 1717-2193-1,ASCC  826013,2MASS J22572795+2046077,USNO-B1.0 1107-00589893,NAME Helvetios,WEB 20165,HIC 113357,GC 32003,GCRV 14411,CSV 102222",
            "NLTT 55385,HR  8729,Gaia DR3 2835207319109249920,BD+19  5036,PLX 5568.00,IRAS 22550+2030,NSV 14374,ROT  3341,LTT 16750,N30 5052,HD 217014,** RBR   21A,WDS J22575+2046A,PPM 114985,*  51 Peg,PLX 5568,YPAC 218,HIP 113357,UBV M  26734,UBV   19678,YZ   0  1227,YZ  20  9382,Gaia DR2 2835207319109249920,TIC 139298196,uvby98 100217014,TD1 29480,SPOCS  990,SKY# 43603,AKARI-IRC-V1 J2257280+204608,GEN# +1.00217014,JP11  3558,GJ 882,SAO  90896,AG+20 2595,LSPM J2257+2046,TYC 1717-2193-1,ASCC  826013,2MASS J22572795+2046077,USNO-B1.0 1107-00589893,NAME Helvetios,WEB 20165,HIC 113357,GC 32003,GCRV 14411,CSV 102222,CNS5 5664,ids",
            "unchanged id",
        ],
        "name": [
            "TrES-1 b",
            "EPIC 211839430.01",
            "EPIC 211839430.01",
            "51 Peg b",
            "51 Peg b",
            "30 Ari B b",
        ],
        "host": [
            "TrES-1",
            "EPIC 211839430",
            "EPIC 211839430",
            "51 Peg",
            "51 Peg",
            "30 Ari B",
        ],
        "catalog": ["eu", "epic", "toi", "nasa", "oec", "eu"],
        "binary": ["", "B", "B", "AB", "B", "B"],
    }
    assert_frame_equal(
        instance.data[
            [
                "main_id",
                "main_id_ra",
                "main_id_dec",
                "name",
                "host",
                "catalog",
                "binary",
            ]
        ],
        pd.DataFrame(expected_data)[
            [
                "main_id",
                "main_id_ra",
                "main_id_dec",
                "name",
                "host",
                "catalog",
                "binary",
            ]
        ],
    )

    # check LIST_ID
    for i in instance.data.index:
        assert sorted(instance.data.at[i, "list_id"].split(",")) == sorted(
            pd.DataFrame(expected_data).at[i, "list_id"].split(",")
        )

    # Check if the log file was created and has the expected content
    assert os.path.exists("Logs/polish_main_id.txt")
    with open("Logs/polish_main_id.txt") as f:
        lines = f.readlines()
    assert  ['***** CHECK FOR PLANET LETTER IN MAIN_ID *****\n',
        'MAINID can be corrected NAME V672 Lyr b to V672 Lyr. \n',
        '\n',
        '***** CHECK FOR BINARY LETTER IN MAIN_ID *****\n',
        'MAINID can be corrected BD+18  2050B to BD+18  2050.  Binary value: B. Only '
        'S-type or null. Binary could be standardized.\n',
        'MAINID can be corrected BD+18  2050 B to BD+18  2050.  Binary value: B. '
        'Already correct.\n',
        'MAINID can be corrected *  51 Peg (AB) to *  51 Peg.  Binary value: AB. Only '
        'S-type or null. Binary could be standardized.\n',
        'MAINID can be corrected * 51 PegAB to * 51 Peg.  Binary value: AB. Binary '
        'value is not in agreement, please check:\n',
        '       name    host binary catalog\n',
        '4  51 Peg b  51 Peg      B     oec\n',
        'Weird MAINID found: *  30 Ari B but cannot be found when *  30 Ari. No '
        'replacement performed.\n'] == lines

    os.chdir(original_dir)


def test__fill_missing_mainid(instance):
    instance.data = pd.DataFrame(
        {
            "main_id": ["*  51 Peg", ""],
            "main_id_ra": [344.366585, ""],
            "main_id_dec": [20.768833, ""],
            "angsep": [0, -1.0],
            "angular_separation": ["", ""],
            "main_id_provenance": ["SIMBAD", ""],
            "catalog": ["nasa", "eu"],
            "host": ["51 Peg", "ZTFJ0407-00"],
            "ra": [344.366585, 61.954167],
            "dec": [20.768833, -0.121389],
        }
    )

    expected = pd.DataFrame(
        {
            "main_id": ["*  51 Peg", "ZTFJ0407-00"],
            "main_id_ra": [344.366585, 61.954167],
            "main_id_dec": [20.768833, -0.121389],
            "angsep": [0, -1.0],
            "angular_separation": ["nasa: 0.0", "eu: -1.0"],
            "main_id_provenance": ["SIMBAD", "eu"],
            "catalog": ["nasa", "eu"],
            "host": ["51 Peg", "ZTFJ0407-00"],
            "ra": [344.366585, 61.954167],
            "dec": [20.768833, -0.121389],
        }
    )
    instance.fill_missing_main_id()

    assert_frame_equal(instance.data, expected)


def test__check_same_host_different_id(tmp_path, instance):
    original_dir = os.getcwd()

    os.chdir(tmp_path)  # Create a temporary in-memory configuration object
    os.mkdir("Logs/")
    data = {
        "name": ["PSR B1620-26 AB b", "PSR B1620-26 AB b"],
        "catalog": ["eu", "oec"],
        "host": ["PSR B1620-26", "PSR B1620-26"],
        "binary": ["AB", "AB"],
        "hostbinary": ["PSR B1620-26 AB", "PSR B1620-26 AB"],
        "letter": ["b", "b"],
        "main_id": ["PSR B1620-26", "PSR B1620-26 AB"],
    }

    instance.data = pd.DataFrame(data)
    with LogCapture() as log:
        instance.check_same_host_different_id()
        assert "Checked if host is found" in log.actual()[0][-1]

    assert os.path.exists("Logs/post_main_id_query_checks.txt")

    with open("Logs/post_main_id_query_checks.txt") as f:
        lines = f.readlines()

    # FIRST SET OF DATA, SHOULD BE AN ERROR ON HOST+BINARY
    assert lines[0:3] == [
        "**************************************\n",
        "**** CHECK SAME HOST DIFFERENT ID ****\n",
        "**************************************\n",
    ]
    assert lines[3:7] == [
        "SAME HOST+BINARY DIFFERENT MAIN_ID\n",
        "        hostbinary          main_id binary catalog\n",
        "0  PSR B1620-26 AB     PSR B1620-26     AB      eu\n",
        "1  PSR B1620-26 AB  PSR B1620-26 AB     AB     oec\n",
    ]
    # SECOND SET, SAME HOST DIFFERENT MAIN_ID
    assert lines[7:11] == [
        "SAME HOST DIFFERENT MAIN_ID\n",
        "           host          main_id binary catalog\n",
        "0  PSR B1620-26     PSR B1620-26     AB      eu\n",
        "1  PSR B1620-26  PSR B1620-26 AB     AB     oec\n",
    ]
    assert len(lines) == 11
    os.chdir(original_dir)


def test__check_same_coords_different_id(tmp_path, instance):
    original_dir = os.getcwd()

    os.chdir(tmp_path)  # Create a temporary in-memory configuration object
    os.mkdir("Logs/")
    data = {
        "catalog": ["toi", "toi"],
        "host": ["TIC 10974783", "TIC 628103717"],
        "binary": ["", ""],
        "letter": [".01", ".01"],
        "main_id": ["TIC 10974783", "TIC 628103717"],
        "main_id_ra": [29.423042, 29.423375],
        "main_id_dec": [63.904389, 63.904589],
        "main_id_provenance": ["SIMBAD", "SIMBAD"],
    }

    instance.data = pd.DataFrame(data)
    with LogCapture() as log:
        instance.check_same_coords_different_id()
        assert "Checked if same coordinates found in main_ids" in log.actual()[0][-1]

    assert os.path.exists("Logs/post_main_id_query_checks.txt")

    with open("Logs/post_main_id_query_checks.txt") as f:
        lines = f.readlines()
    # FIRST SET OF DATA, SHOULD BE AN ERROR ON HOST+BINARY
    assert lines == [
        "****************************************\n",
        "**** CHECK SAME COORDS DIFFERENT ID ****\n",
        "****************************************\n",
        "FOUND SAME COORDINATES DIFFERENT MAINID\n",
        "            host        main_id binary letter catalog    angsep main_id_provenance\n",
        "0   TIC 10974783   TIC 10974783           .01     toi  0.000000             SIMBAD\n",
        "1  TIC 628103717  TIC 628103717           .01     toi  0.000248             SIMBAD\n",
        "FOUND SAME COORDINATES DIFFERENT MAINID\n",
        "            host        main_id binary letter catalog    angsep main_id_provenance\n",
        "0   TIC 10974783   TIC 10974783           .01     toi  0.000248             SIMBAD\n",
        "1  TIC 628103717  TIC 628103717           .01     toi  0.000000             SIMBAD\n",
    ]

    os.chdir(original_dir)


def test__group_by_list_id_check_main_id(tmp_path, instance):
    original_dir = os.getcwd()

    os.chdir(tmp_path)  # Create a temporary in-memory configuration object
    os.mkdir("Logs/")
    data = {
        "catalog": ["eu", "oec"],
        "status": ["CONFIRMED", "CONFIRMED"],
        "letter": ["b", "b"],
        "main_id": ["*  51 Peg", "TYC 1717-2193-1"],
        "list_id": [
            "LTT 16750,*  51 Peg,** RBR   21A,AG+20 2595,AKARI-IRC-V1 J2257280+204608,ASCC  826013,BD+19  5036,"
            "CSV 102222,GC 32003,GCRV 14411,GEN# +1.00217014,GJ   882,HD 217014,HIC 113357,HIP 113357,HR  8729,"
            "IRAS 22550+2030,JP11  3558,LSPM J2257+2046,2MASS J22572795+2046077,N30 5052,NAME Helvetios,NLTT 55385,"
            "NSV 14374,PLX 5568,PLX 5568.00,PPM 114985,ROT  3341,SAO  90896,SKY# 43603,SPOCS  990,TD1 29480,"
            "TIC 139298196,TYC 1717-2193-1,UBV M  26734,UBV   19678,USNO-B1.0 1107-00589893,uvby98 100217014,"
            "WDS J22575+2046A,WEB 20165,YPAC 218,YZ   0  1227,YZ  20  9382,Gaia DR3 2835207319109249920,"
            "Gaia DR2 2835207319109249920,CNS5 5664",
            "LTT 16750,*  51 Peg,** RBR   21A,AG+20 2595,AKARI-IRC-V1 J2257280+204608,ASCC  826013,BD+19  5036,"
            "CSV 102222,GC 32003,GCRV 14411,GEN# +1.00217014,GJ   882,HD 217014,HIC 113357,HIP 113357,HR  8729,"
            "IRAS 22550+2030,JP11  3558,LSPM J2257+2046,2MASS J22572795+2046077,N30 5052,NAME Helvetios,NLTT 55385,"
            "NSV 14374,PLX 5568,PLX 5568.00,PPM 114985,ROT  3341,SAO  90896,SKY# 43603,SPOCS  990,TD1 29480,"
            "TIC 139298196,TYC 1717-2193-1,UBV M  26734,UBV   19678,USNO-B1.0 1107-00589893,uvby98 100217014,"
            "WDS J22575+2046A,WEB 20165,YPAC 218,YZ   0  1227,YZ  20  9382,Gaia DR3 2835207319109249920,"
            "Gaia DR2 2835207319109249920,CNS5 5664",
        ],
    }

    instance.data = pd.DataFrame(data)
    with LogCapture() as log:
        instance.group_by_list_id_check_main_id()
        log = pd.DataFrame(list(log), columns=["user", "info", "message"])
        assert (
            "Planets that had a different main_id name but same SIMBAD alias: 1"
            in log["message"].tolist()
        )

    assert list(instance.data.main_id) == ["*  51 Peg", "*  51 Peg"]

    assert os.path.exists("Logs/post_main_id_query_checks.txt")

    with open("Logs/post_main_id_query_checks.txt") as f:
        lines = f.readlines()
        assert lines == [
            "****************************************\n",
            "**** GROUP BY LIST_ID CHECK MAIN_ID ****\n",
            "****************************************\n",
            "*** SAME LIST_ID, DIFFERENT MAIN_ID *** \n",
            "  catalog     status letter          main_id\n",
            "0      eu  CONFIRMED      b        *  51 Peg\n",
            "1     oec  CONFIRMED      b  TYC 1717-2193-1\n",
        ]

    os.chdir(original_dir)


def test__group_by_main_id_set_main_id_aliases(instance):
    data = {
        "host": ["51 Peg", "TYC 1717-2193-1 b"],
        "alias": [
            "GJ 882,BD+19 5036,HIP 113357,HR 8729,HD 217014,51 Peg,Gaia DR2 2835207319109249920,Helvetios,SAO 90896,"
            "TYC 1717-2193-1,nan",
            "51 Peg",
        ],
        "main_id": ["*  51 Peg", "*  51 Peg"],
        "list_id": [
            "LTT 16750,*  51 Peg,** RBR   21A,AG+20 2595,AKARI-IRC-V1 J2257280+204608,ASCC  826013,BD+19  5036,"
            "CSV 102222,GC 32003,GCRV 14411,GEN# +1.00217014,GJ   882,HD 217014,HIC 113357,HIP 113357,HR  8729,"
            "IRAS 22550+2030,JP11  3558,LSPM J2257+2046,2MASS J22572795+2046077,N30 5052,NAME Helvetios,NLTT 55385,"
            "NSV 14374,PLX 5568,PLX 5568.00,PPM 114985,ROT  3341,SAO  90896,SKY# 43603,SPOCS  990,TD1 29480,"
            "TIC 139298196,TYC 1717-2193-1,UBV M  26734,UBV   19678,USNO-B1.0 1107-00589893,uvby98 100217014,"
            "WDS J22575+2046A,WEB 20165,YPAC 218,YZ   0  1227,YZ  20  9382,Gaia DR3 2835207319109249920,"
            "Gaia DR2 2835207319109249920",
            "LTT 16750,*  51 Peg,** RBR   21A,AG+20 2595,AKARI-IRC-V1 J2257280+204608,ASCC  826013,BD+19  5036,"
            "CSV 102222,GC 32003,GCRV 14411,GEN# +1.00217014,GJ   882,HD 217014,HIC 113357,HIP 113357,HR  8729,"
            "IRAS 22550+2030,JP11  3558,LSPM J2257+2046,2MASS J22572795+2046077,N30 5052,NAME Helvetios,NLTT 55385,"
            "NSV 14374,PLX 5568,PLX 5568.00,PPM 114985,ROT  3341,SAO  90896,SKY# 43603,SPOCS  990,TD1 29480,"
            "TIC 139298196,TYC 1717-2193-1,UBV M  26734,UBV   19678,USNO-B1.0 1107-00589893,uvby98 100217014,"
            "WDS J22575+2046A,WEB 20165,YPAC 218,YZ   0  1227,YZ  20  9382,Gaia DR3 2835207319109249920,"
            "Gaia DR2 2835207319109249920",
        ],
    }

    instance.data = pd.DataFrame(data)
    with LogCapture() as log:
        instance.group_by_main_id_set_main_id_aliases()

    assert sorted(set(instance.data.at[0, "main_id_aliases"].split(","))) == sorted(
        [
            "*  51 Peg",
            "** RBR   21A",
            "2MASS J22572795+2046077",
            "51 Peg",
            "AG+20 2595",
            "AKARI-IRC-V1 J2257280+204608",
            "ASCC  826013",
            "BD+19  5036",
            "BD+19 5036",
            "CSV 102222",
            "GC 32003",
            "GCRV 14411",
            "GEN# +1.00217014",
            "GJ   882",
            "GJ 882",
            "Gaia DR2 2835207319109249920",
            "Gaia DR3 2835207319109249920",
            "HD 217014",
            "HIC 113357",
            "HIP 113357",
            "HR  8729",
            "HR 8729",
            "Helvetios",
            "IRAS 22550+2030",
            "JP11  3558",
            "LSPM J2257+2046",
            "LTT 16750",
            "N30 5052",
            "NAME Helvetios",
            "NLTT 55385",
            "NSV 14374",
            "PLX 5568",
            "PLX 5568.00",
            "PPM 114985",
            "ROT  3341",
            "SAO  90896",
            "SAO 90896",
            "SKY# 43603",
            "SPOCS  990",
            "TD1 29480",
            "TIC 139298196",
            "TYC 1717-2193-1",
            "UBV   19678",
            "UBV M  26734",
            "USNO-B1.0 1107-00589893",
            "WDS J22575+2046A",
            "WEB 20165",
            "YPAC 218",
            "YZ   0  1227",
            "YZ  20  9382",
            "uvby98 100217014",
        ]
    )


def test__cleanup_catalog(instance):
    data = {
        "name": ["51 Peg b", "anotherplanet"],
        "i_min": [1, np.inf],
        "i_max": [1, 0],
        "mass_min": [1, np.inf],
        "mass_max": [1, 0],
        "msini_min": [1, np.inf],
        "msini_max": [1, 0],
        "a_min": [1, np.inf],
        "a_max": [1, 0],
        "p_min": [1, np.inf],
        "p_max": [1, 0],
        "e_min": [1, np.inf],
        "e_max": [1, 0],
    }

    instance.data = pd.DataFrame(data)
    with LogCapture() as log:
        instance.cleanup_catalog()
        log = pd.DataFrame(list(log), columns=["user", "info", "message"])

        assert "Catalog cleared from zeroes and infinities." in log["message"].tolist()

    assert set(
        list(
            instance.data[instance.data.name == "anotherplanet"][
                [
                    "i_min",
                    "i_max",
                    "mass_min",
                    "mass_max",
                    "msini_min",
                    "msini_max",
                    "a_min",
                    "a_max",
                    "p_min",
                    "p_max",
                    "e_min",
                    "e_max",
                ]
            ].isna()
        )
    )


def test__group_by_period_check_letter(tmp_path, instance):
    original_dir = os.getcwd()

    os.chdir(tmp_path)  # Create a temporary in-memory configuration object
    os.mkdir("Logs/")
    # #### PERIOD
    # CASE 1: everything is in agreement
    data = {
        "p": [934.3, 934.3, 934.3],
        "a": [2.11, 2.11, 2.11],
        "main_id": ["*   6 Lyn", "*   6 Lyn", "*   6 Lyn"],
        "host": ["6 Lyn", "6 Lyn", "6 Lyn"],
        "binary": ["", "", ""],
        "letter": ["b", "b", "b"],
        "catalog": ["eu", "nasa", "oec"],
        "catalog_name": ["6 Lyn b", "6 Lyn b", "6 Lyn b"],
    }

    data = pd.DataFrame(data)
    expected_result = {
        "p": [934.3, 934.3, 934.3],
        "a": [2.11, 2.11, 2.11],
        "main_id": ["*   6 Lyn", "*   6 Lyn", "*   6 Lyn"],
        "host": ["6 Lyn", "6 Lyn", "6 Lyn"],
        "binary": ["", "", ""],
        "letter": ["b", "b", "b"],
        "catalog": ["eu", "nasa", "oec"],
        "catalog_name": ["6 Lyn b", "6 Lyn b", "6 Lyn b"],
    }

    expected_result = pd.DataFrame(expected_result)
    instance.data = data

    instance.group_by_period_check_letter()
    assert os.path.exists("Logs/group_by_period_check_letter.txt")

    with open("Logs/group_by_period_check_letter.txt") as f:
        lines = f.readlines()
        assert lines == [
            "TOTAL NUMBER OF GROUPS: 1\n",
        ]
    os.remove("Logs/group_by_period_check_letter.txt")
    assert sorted(instance.data.columns) == sorted(expected_result.columns)

    for col in instance.data.columns:
        if pd.isna(instance.data[col]).all() and pd.isna(expected_result[col]).all():
            continue
        assert (instance.data[col] == expected_result[col]).all()

    # CASE 2: one letter disagrees but the period is the same, so it can be
    # fixed
    data = {
        "p": [934.3, 934.3, 934.3],
        "a": [2.11, 2.11, 2.11],
        "main_id": ["*   6 Lyn", "*   6 Lyn", "*   6 Lyn"],
        "host": ["6 Lyn", "6 Lyn", "6 Lyn"],
        "binary": ["", "", ""],
        "letter": ["b", "b", ".01"],
        "catalog": ["eu", "nasa", "oec"],
        "catalog_name": ["6 Lyn b", "6 Lyn b", "6 Lyn b"],
    }

    data = pd.DataFrame(data)
    expected_result = {
        "p": [934.3, 934.3, 934.3],
        "a": [2.11, 2.11, 2.11],
        "main_id": ["*   6 Lyn", "*   6 Lyn", "*   6 Lyn"],
        "host": ["6 Lyn", "6 Lyn", "6 Lyn"],
        "binary": ["", "", ""],
        "letter": ["b", "b", "b"],
        "catalog": ["eu", "nasa", "oec"],
        "catalog_name": ["6 Lyn b", "6 Lyn b", "6 Lyn b"],
    }

    expected_result = pd.DataFrame(expected_result)
    instance.data = data

    instance.group_by_period_check_letter()

    assert os.path.exists("Logs/group_by_period_check_letter.txt")

    with open("Logs/group_by_period_check_letter.txt") as f:
        lines = f.readlines()

        assert lines == [
            "TOTAL NUMBER OF GROUPS: 1\n",
            "FIXABLE INCONSISTENT LETTER FOR SAME PERIOD \n",
            "     main_id binary letter catalog catalog_name      p\n",
            "0  *   6 Lyn             b      eu      6 Lyn b  934.3\n",
            "1  *   6 Lyn             b    nasa      6 Lyn b  934.3\n",
            "2  *   6 Lyn           .01     oec      6 Lyn b  934.3\n",
            "\n",
        ]
    os.remove("Logs/group_by_period_check_letter.txt")

    assert sorted(instance.data.columns) == sorted(expected_result.columns)

    for col in instance.data.columns:
        if pd.isna(instance.data[col]).all() and pd.isna(expected_result[col]).all():
            continue
        assert (instance.data[col].values == expected_result[col].values).all()

    # CASE 3: one letter is BD so force all to be BD
    data = {
        "p": [934.3, 934.3, 934.3],
        "a": [2.11, 2.11, 2.11],
        "main_id": ["*   6 Lyn", "*   6 Lyn", "*   6 Lyn"],
        "host": ["6 Lyn", "6 Lyn", "6 Lyn"],
        "binary": ["", "", ""],
        "letter": ["b", "b", "BD"],
        "catalog": ["eu", "nasa", "oec"],
        "catalog_name": ["6 Lyn b", "6 Lyn b", "6 Lyn b"],
    }

    data = pd.DataFrame(data)
    expected_result = {
        "p": [934.3, 934.3, 934.3],
        "a": [2.11, 2.11, 2.11],
        "main_id": ["*   6 Lyn", "*   6 Lyn", "*   6 Lyn"],
        "host": ["6 Lyn", "6 Lyn", "6 Lyn"],
        "binary": ["", "", ""],
        "letter": ["BD", "BD", "BD"],
        "catalog": ["eu", "nasa", "oec"],
        "catalog_name": ["6 Lyn b", "6 Lyn b", "6 Lyn b"],
    }

    expected_result = pd.DataFrame(expected_result)
    instance.data = data

    instance.group_by_period_check_letter()

    assert sorted(instance.data.columns) == sorted(expected_result.columns)

    for col in instance.data.columns:
        if pd.isna(instance.data[col]).all() and pd.isna(expected_result[col]).all():
            continue
        assert (instance.data[col].values == expected_result[col].values).all()

    assert os.path.exists("Logs/group_by_period_check_letter.txt")

    with open("Logs/group_by_period_check_letter.txt") as f:
        lines = f.readlines()
        assert lines == [
            "TOTAL NUMBER OF GROUPS: 1\n",
            "FORCED BD INCONSISTENT LETTER FOR SAME PERIOD \n",
            "     main_id binary letter catalog catalog_name      p\n",
            "0  *   6 Lyn             b      eu      6 Lyn b  934.3\n",
            "1  *   6 Lyn             b    nasa      6 Lyn b  934.3\n",
            "2  *   6 Lyn            BD     oec      6 Lyn b  934.3\n",
            "\n",
        ]

    os.remove("Logs/group_by_period_check_letter.txt")

    # SEMIMAJOR AXIS
    # CASE 1: everything is in agreement
    data = {
        "p": [np.nan, np.nan, np.nan],
        "a": [2.11, 2.11, 2.11],
        "main_id": ["*   6 Lyn", "*   6 Lyn", "*   6 Lyn"],
        "host": ["6 Lyn", "6 Lyn", "6 Lyn"],
        "binary": ["", "", ""],
        "letter": ["b", "b", "b"],
        "catalog": ["eu", "nasa", "oec"],
        "catalog_name": ["6 Lyn b", "6 Lyn b", "6 Lyn b"],
    }

    data = pd.DataFrame(data)
    expected_result = {
        "p": [np.nan, np.nan, np.nan],
        "a": [2.11, 2.11, 2.11],
        "main_id": ["*   6 Lyn", "*   6 Lyn", "*   6 Lyn"],
        "host": ["6 Lyn", "6 Lyn", "6 Lyn"],
        "binary": ["", "", ""],
        "letter": ["b", "b", "b"],
        "catalog": ["eu", "nasa", "oec"],
        "catalog_name": ["6 Lyn b", "6 Lyn b", "6 Lyn b"],
    }

    expected_result = pd.DataFrame(expected_result)
    instance.data = data

    instance.group_by_period_check_letter()
    assert os.path.exists("Logs/group_by_period_check_letter.txt")

    with open("Logs/group_by_period_check_letter.txt") as f:
        lines = f.readlines()
        assert lines == [
            "TOTAL NUMBER OF GROUPS: 1\n",
        ]
    os.remove("Logs/group_by_period_check_letter.txt")
    assert sorted(instance.data.columns) == sorted(expected_result.columns)

    for col in instance.data.columns:
        if pd.isna(instance.data[col]).all() and pd.isna(expected_result[col]).all():
            continue
        assert (instance.data[col] == expected_result[col]).all()

    # CASE 2: one letter disagrees but the period is the same, so it can be
    # fixed
    data = {
        "p": [np.nan, np.nan, np.nan],
        "a": [2.11, 2.11, 2.11],
        "main_id": ["*   6 Lyn", "*   6 Lyn", "*   6 Lyn"],
        "host": ["6 Lyn", "6 Lyn", "6 Lyn"],
        "binary": ["", "", ""],
        "letter": ["b", "b", ".01"],
        "catalog": ["eu", "nasa", "oec"],
        "catalog_name": ["6 Lyn b", "6 Lyn b", "6 Lyn b"],
    }

    data = pd.DataFrame(data)
    expected_result = {
        "p": [np.nan, np.nan, np.nan],
        "a": [2.11, 2.11, 2.11],
        "main_id": ["*   6 Lyn", "*   6 Lyn", "*   6 Lyn"],
        "host": ["6 Lyn", "6 Lyn", "6 Lyn"],
        "binary": ["", "", ""],
        "letter": ["b", "b", "b"],
        "catalog": ["eu", "nasa", "oec"],
        "catalog_name": ["6 Lyn b", "6 Lyn b", "6 Lyn b"],
    }

    expected_result = pd.DataFrame(expected_result)
    instance.data = data

    instance.group_by_period_check_letter()

    assert os.path.exists("Logs/group_by_period_check_letter.txt")

    with open("Logs/group_by_period_check_letter.txt") as f:
        lines = f.readlines()

        assert lines == [
            "TOTAL NUMBER OF GROUPS: 1\n",
            "FIXABLE INCONSISTENT LETTER FOR SAME SMA \n",
            "     main_id binary letter catalog catalog_name     a\n",
            "0  *   6 Lyn             b      eu      6 Lyn b  2.11\n",
            "1  *   6 Lyn             b    nasa      6 Lyn b  2.11\n",
            "2  *   6 Lyn           .01     oec      6 Lyn b  2.11\n",
            "\n",
        ]
    os.remove("Logs/group_by_period_check_letter.txt")

    assert sorted(instance.data.columns) == sorted(expected_result.columns)

    for col in instance.data.columns:
        if pd.isna(instance.data[col]).all() and pd.isna(expected_result[col]).all():
            continue
        assert (instance.data[col].values == expected_result[col].values).all()

    # CASE 3: one letter is BD so force all to be BD
    data = {
        "p": [np.nan, np.nan, np.nan],
        "a": [2.11, 2.11, 2.11],
        "main_id": ["*   6 Lyn", "*   6 Lyn", "*   6 Lyn"],
        "host": ["6 Lyn", "6 Lyn", "6 Lyn"],
        "binary": ["", "", ""],
        "letter": ["b", "b", "BD"],
        "catalog": ["eu", "nasa", "oec"],
        "catalog_name": ["6 Lyn b", "6 Lyn b", "6 Lyn b"],
    }

    data = pd.DataFrame(data)
    expected_result = {
        "p": [np.nan, np.nan, np.nan],
        "a": [2.11, 2.11, 2.11],
        "main_id": ["*   6 Lyn", "*   6 Lyn", "*   6 Lyn"],
        "host": ["6 Lyn", "6 Lyn", "6 Lyn"],
        "binary": ["", "", ""],
        "letter": ["BD", "BD", "BD"],
        "catalog": ["eu", "nasa", "oec"],
        "catalog_name": ["6 Lyn b", "6 Lyn b", "6 Lyn b"],
    }

    expected_result = pd.DataFrame(expected_result)
    instance.data = data

    instance.group_by_period_check_letter()

    assert sorted(instance.data.columns) == sorted(expected_result.columns)

    for col in instance.data.columns:
        if pd.isna(instance.data[col]).all() and pd.isna(expected_result[col]).all():
            continue
        assert (instance.data[col].values == expected_result[col].values).all()

    assert os.path.exists("Logs/group_by_period_check_letter.txt")

    with open("Logs/group_by_period_check_letter.txt") as f:
        lines = f.readlines()
        assert lines == [
            "TOTAL NUMBER OF GROUPS: 1\n",
            "FORCED BD INCONSISTENT LETTER FOR SAME SMA \n",
            "     main_id binary letter catalog catalog_name     a\n",
            "0  *   6 Lyn             b      eu      6 Lyn b  2.11\n",
            "1  *   6 Lyn             b    nasa      6 Lyn b  2.11\n",
            "2  *   6 Lyn            BD     oec      6 Lyn b  2.11\n",
            "\n",
        ]

    os.remove("Logs/group_by_period_check_letter.txt")

    os.chdir(original_dir)


def test__merge_into_single_entry(tmp_path):
    original_dir = os.getcwd()

    os.chdir(tmp_path)  # Create a temporary in-memory configuration object
    os.mkdir("Logs/")

    data = {
        "name": ["TOI-774.01", "WASP-55 b", "WASP-55 b", "WASP-55 b", "WASP-55 b"],
        "catalog_name": [
            "TOI-774.01",
            "WASP-55 b",
            "WASP-55 b",
            "WASP-55 b",
            "WASP-55 b",
        ],
        "host": ["TIC 294301883", "WASP-55", "WASP-55", "WASP-55", "WASP-55"],
        "discovery_method": ["Transit", "Transit", "Transit", "Transit", "Transit"],
        "ra": [203.758195, 203.758114, 203.7581951, 203.75814041666663, 203.7581951],
        "dec": [-17.503496, -17.503464, -17.5034961, -17.503479527777777, -17.5034961],
        "p": [4.4656343, 4.46563, 4.46563, 4.465633, 4.46563],
        "p_max": [2.4e-06, 4e-06, 4e-06, np.nan, 4e-06],
        "p_min": [2.4e-06, 4e-06, 4e-06, np.nan, 4e-06],
        "a": [np.nan, 0.0533, 0.0558, 0.0533, np.nan],
        "a_max": [np.nan, 0.0007, 0.0006, np.nan, np.nan],
        "a_min": [np.nan, 0.0007, 0.0006, np.nan, np.nan],
        "e": [np.nan, 0.0, 0.0, 0.0, 0.0],
        "e_max": [np.nan, 0.0, np.nan, np.nan, np.nan],
        "e_min": [np.nan, 0.0, np.nan, np.nan, np.nan],
        "i": [np.nan, 89.6, 89.6, 89.2, 89.6],
        "i_max": [np.nan, 0.2, 0.2, np.nan, 0.2],
        "i_min": [np.nan, 0.2, 0.2, np.nan, 0.2],
        "mass": [np.nan, 0.61, 0.61, 0.57, 0.61],
        "mass_max": [np.nan, 0.13, 0.13, np.nan, 0.13],
        "mass_min": [np.nan, 0.13, 0.13, np.nan, 0.13],
        "msini": [np.nan, 0.57, np.nan, np.nan, np.nan],
        "msini_max": [np.nan, 0.04, np.nan, np.nan, np.nan],
        "msini_min": [np.nan, 0.04, np.nan, np.nan, np.nan],
        "r": [1.3067001051730265, 1.33, 1.33, 1.3, 1.33],
        "r_max": [0.0570577303460527, 0.13, 0.13, np.nan, 0.13],
        "r_min": [0.0570577303460527, 0.13, 0.13, np.nan, 0.13],
        "discovery_year": [2019.0, 2012.0, 2012.0, 2011.0, 2012.0],
        "alias": [
            "1SWASP J133501.94-173012.7,2MASS J13350194-1730124,EPIC 212300977,Gaia DR2 3603529272750802560,"
            "TIC 294301883,TIC-294301883,TOI-774,TYC 6125-00113-1,TYC 6125-113-1,UCAC4 363-065216,WASP-55 A,"
            "WISE J133501.96-173012.7",
            "1SWASP J133501.94-173012.7,2MASS J13350194-1730124,EPIC 212300977,Gaia DR2 3603529272750802560,"
            "TIC 294301883,TIC-294301883,TOI-774,TYC 6125-00113-1,TYC 6125-113-1,UCAC4 363-065216,WASP-55 A,"
            "WISE J133501.96-173012.7",
            "1SWASP J133501.94-173012.7,2MASS J13350194-1730124,EPIC 212300977,Gaia DR2 3603529272750802560,"
            "TIC 294301883,TIC-294301883,TOI-774,TYC 6125-00113-1,TYC 6125-113-1,UCAC4 363-065216,WASP-55 A,"
            "WISE J133501.96-173012.7",
            "1SWASP J133501.94-173012.7,2MASS J13350194-1730124,EPIC 212300977,Gaia DR2 3603529272750802560,"
            "TIC 294301883,TIC-294301883,TOI-774,TYC 6125-00113-1,TYC 6125-113-1,UCAC4 363-065216,WASP-55 A,"
            "WISE J133501.96-173012.7",
            "1SWASP J133501.94-173012.7,2MASS J13350194-1730124,EPIC 212300977,Gaia DR2 3603529272750802560,"
            "TIC 294301883,TIC-294301883,TOI-774,TYC 6125-00113-1,TYC 6125-113-1,UCAC4 363-065216,WASP-55 A,"
            "WISE J133501.96-173012.7",
        ],
        "a_url": [np.nan, "eu", "2016MNRAS.457.4205S", "oec", np.nan],
        "mass_url": [np.nan, "eu", "2017AJ....153..136S", "oec", "2017AJ....153..136S"],
        "p_url": ["toi", "eu", "2017AJ....153..136S", "oec", "2017AJ....153..136S"],
        "msini_url": [np.nan, "eu", np.nan, np.nan, np.nan],
        "r_url": ["toi", "eu", "2017AJ....153..136S", "oec", "2017AJ....153..136S"],
        "i_url": [np.nan, "eu", "2017AJ....153..136S", "oec", "2017AJ....153..136S"],
        "e_url": [np.nan, "eu", "2017AJ....153..136S", "oec", "2017AJ....153..136S"],
        "binary": ["", "", "", "", ""],
        "letter": [".01", "b", "b", "b", "b"],
        "status": ["CONFIRMED", "CONFIRMED", "CONFIRMED", "CONFIRMED", "CONFIRMED"],
        "catalog": ["toi", "eu", "nasa", "oec", "epic"],
        "original_catalog_status": [
            "toi: CONFIRMED",
            "eu: CONFIRMED",
            "nasa: CONFIRMED",
            "oec: CONFIRMED",
            "epic: CONFIRMED",
        ],
        "checked_catalog_status": [
            "toi: CONFIRMED",
            "eu: CONFIRMED",
            "nasa: CONFIRMED",
            "oec: CONFIRMED",
            "epic: CONFIRMED",
        ],
        "binary_mismatch_flag": [0, 0, 0, 0, 0],
        "main_id": ["WASP-55", "WASP-55", "WASP-55", "WASP-55", "WASP-55"],
        "main_id_aliases": [
            "GALAH 150429002601133,Gaia DR3 3603529272750802560,TOI-774,RAVE J133502.0-173012,RAVE J133502.0-173013,"
            "TIC 294301883,Gaia DR2 3603529272750802560,WASP-55,TYC 6125-113-1,2MASS J13350194-1730124,"
            "1SWASP J133501.94-173012.7,EPIC 212300977,Gaia DR1 3603529272750802560",
            "GALAH 150429002601133,Gaia DR3 3603529272750802560,TOI-774,RAVE J133502.0-173012,RAVE J133502.0-173013,"
            "TIC 294301883,Gaia DR2 3603529272750802560,WASP-55,TYC 6125-113-1,2MASS J13350194-1730124,"
            "1SWASP J133501.94-173012.7,EPIC 212300977,Gaia DR1 3603529272750802560",
            "GALAH 150429002601133,Gaia DR3 3603529272750802560,TOI-774,RAVE J133502.0-173012,RAVE J133502.0-173013,"
            "TIC 294301883,Gaia DR2 3603529272750802560,WASP-55,TYC 6125-113-1,2MASS J13350194-1730124,"
            "1SWASP J133501.94-173012.7,EPIC 212300977,Gaia DR1 3603529272750802560",
            "GALAH 150429002601133,Gaia DR3 3603529272750802560,TOI-774,RAVE J133502.0-173012,RAVE J133502.0-173013,"
            "TIC 294301883,Gaia DR2 3603529272750802560,WASP-55,TYC 6125-113-1,2MASS J13350194-1730124,"
            "1SWASP J133501.94-173012.7,EPIC 212300977,Gaia DR1 3603529272750802560",
            "GALAH 150429002601133,Gaia DR3 3603529272750802560,TOI-774,RAVE J133502.0-173012,RAVE J133502.0-173013,"
            "TIC 294301883,Gaia DR2 3603529272750802560,WASP-55,TYC 6125-113-1,2MASS J13350194-1730124,"
            "1SWASP J133501.94-173012.7,EPIC 212300977,Gaia DR1 3603529272750802560,extraalias",
        ],
        "main_id_ra": [
            203.75814077726,
            203.75814077726,
            203.75814077726,
            203.75814077726,
            203.75814077726,
        ],
        "main_id_dec": [
            -17.50347989552,
            -17.50347989552,
            -17.50347989552,
            -17.50347989552,
            -17.50347989552,
        ],
        "angular_separation": [
            "toi: 0.0",
            "eu: 0.0",
            "nasa: 0.0",
            "oec: 0.0",
            "epic: 0.0",
        ],
        "angsep": [0.0, 0.0, 0.0, 0.0, 0.0],
        "main_id_provenance": ["SIMBAD", "SIMBAD", "SIMBAD", "SIMBAD", "SIMBAD"],
        "coordinate_mismatch": ["", "", "", "", ""],
    }

    data = pd.DataFrame(data)

    expected_result = {
        "nasa_name": ["WASP-55 b"],
        "toi_name": ["TOI-774.01"],
        "epic_name": ["WASP-55 b"],
        "eu_name": ["WASP-55 b"],
        "oec_name": ["WASP-55 b"],
        "host": ["WASP-55"],
        "letter": ["b"],
        "main_id": ["WASP-55"],
        "binary": [""],
        "main_id_ra": [203.75814077726],
        "main_id_dec": [-17.50347989552],
        "mass": [0.61],
        "mass_max": [0.13],
        "mass_min": [0.13],
        "mass_url": ["2017AJ....153..136S"],
        "MASSREL": [0.21311475409836067],
        "msini": [0.57],
        "msini_max": [0.04],
        "msini_min": [0.04],
        "msini_url": ["eu"],
        "MSINIREL": [0.07017543859649124],
        "p": [4.4656343],
        "p_max": [2.4e-06],
        "p_min": [2.4e-06],
        "p_url": ["toi"],
        "PERREL": [5.374376491151549e-07],
        "r": [1.3067001051730265],
        "r_max": [0.0570577303460527],
        "r_min": [0.0570577303460527],
        "r_url": ["toi"],
        "RADREL": [0.04366551293611277],
        "a": [0.0558],
        "a_max": [0.0006],
        "a_min": [0.0006],
        "a_url": ["2016MNRAS.457.4205S"],
        "AREL": [0.01075268817204301],
        "e": [0.0],
        "e_max": [0.0],
        "e_min": [0.0],
        "e_url": ["eu"],
        "EREL": [0.0],
        "i": [89.6],
        "i_max": [0.2],
        "i_min": [0.2],
        "i_url": ["2017AJ....153..136S"],
        "IREL": [0.0022321428571428575],
        "discovery_method": ["Transit"],
        "status": ["CONFIRMED"],
        "original_status_string": [
            "epic: CONFIRMED,eu: CONFIRMED,nasa: CONFIRMED,oec: CONFIRMED,toi: CONFIRMED"
        ],
        "checked_status_string": [
            "epic: CONFIRMED,eu: CONFIRMED,nasa: CONFIRMED,oec: CONFIRMED,toi: CONFIRMED"
        ],
        "confirmed": [5],
        "discovery_year": [2011.0],
        "main_id_aliases": [
            "GALAH 150429002601133,Gaia DR3 3603529272750802560,TOI-774,RAVE J133502.0-173012,RAVE J133502.0-173013,"
            "TIC 294301883,Gaia DR2 3603529272750802560,WASP-55,TYC 6125-113-1,2MASS J13350194-1730124,"
            "1SWASP J133501.94-173012.7,EPIC 212300977,Gaia DR1 3603529272750802560,extraalias"
        ],
        "catalog": ["epic,eu,nasa,oec,toi"],
        "angular_separation": ["epic: 0.0,eu: 0.0,nasa: 0.0,oec: 0.0,toi: 0.0"],
        "angular_separation_flag": [0],
        "main_id_provenance": ["SIMBAD"],
        "coordinate_mismatch": [""],
        "coordinate_mismatch_flag": [0],
        "duplicate_catalog_flag": [0],
        "duplicate_names": [""],
        "binary_mismatch_flag": ["0"],
    }
    expected_result = pd.DataFrame(expected_result)

    # TEST 1: Default data
    test1_data = data.copy(deep=True)
    test1_expected = expected_result.copy(deep=True)

    # TEST 2: Controversial status
    test2_data = data.copy(deep=True)
    test2_data["status"] = [
        "CONFIRMED",
        "CONFIRMED",
        "CANDIDATE",
        "CONFIRMED",
        "FALSE POSITIVE",
    ]
    test2_data["original_catalog_status"] = [
        "toi: CONFIRMED",
        "eu: CONFIRMED",
        "nasa: CANDIDATE",
        "oec: CONFIRMED",
        "epic: FALSE POSITIVE",
    ]
    test2_data["checked_catalog_status"] = [
        "toi: CONFIRMED",
        "eu: CONFIRMED",
        "nasa: CANDIDATE",
        "oec: CONFIRMED",
        "epic: FALSE POSITIVE",
    ]

    test2_expected = expected_result.copy(deep=True)
    test2_expected["status"] = ["CONTROVERSIAL"]
    test2_expected["original_status_string"] = [
        "epic: FALSE POSITIVE,eu: CONFIRMED,nasa: CANDIDATE,oec: CONFIRMED,toi: CONFIRMED"
    ]
    test2_expected["checked_status_string"] = [
        "epic: FALSE POSITIVE,eu: CONFIRMED,nasa: CANDIDATE,oec: CONFIRMED,toi: CONFIRMED"
    ]
    test2_expected["confirmed"] = [3]

    # TEST 3: discovery year scenarios (same discovery year; disagreeing
    # discovery year (checked in test1), no discovery year)
    test3_data = data.copy(deep=True)
    # reset previous values (sometimes copy changes the data for whatever
    # reason)
    test3_data["status"] = [
        "CONFIRMED",
        "CONFIRMED",
        "CONFIRMED",
        "CONFIRMED",
        "CONFIRMED",
    ]
    test3_data["original_catalog_status"] = [
        "toi: CONFIRMED",
        "eu: CONFIRMED",
        "nasa: CONFIRMED",
        "oec: CONFIRMED",
        "epic: CONFIRMED",
    ]
    test3_data["checked_catalog_status"] = [
        "toi: CONFIRMED",
        "eu: CONFIRMED",
        "nasa: CONFIRMED",
        "oec: CONFIRMED",
        "epic: CONFIRMED",
    ]

    test3_expected = expected_result.copy(deep=True)
    test3_expected["status"] = ["CONFIRMED"]
    test3_expected["original_status_string"] = [
        "epic: CONFIRMED,eu: CONFIRMED,nasa: CONFIRMED,oec: CONFIRMED,toi: CONFIRMED"
    ]
    test3_expected["checked_status_string"] = [
        "epic: CONFIRMED,eu: CONFIRMED,nasa: CONFIRMED,oec: CONFIRMED,toi: CONFIRMED"
    ]
    test3_expected["confirmed"] = [5]

    # 3.1 same discovery year
    test31_data = test3_data.copy(deep=True)
    test31_expected = test3_expected.copy(deep=True)
    test31_data["discovery_year"] = [2012.0, 2012.0, 2012.0, 2012.0, 2012.0]
    test31_expected["discovery_year"] = [2012]

    # 3.2 no discovery year
    test32_data = test3_data.copy(deep=True)
    test32_expected = test3_expected.copy(deep=True)
    test32_data["discovery_year"] = [np.nan, np.nan, np.nan, np.nan, np.nan]
    test32_expected["discovery_year"] = [""]

    # TEST 4: discovery method scenarios - different discovery methods,
    # including toi (TOI needs to be escluded)
    test4_data = data.copy(deep=True)
    # reset previous values (sometimes copy changes the data for whatever
    # reason)
    test4_data["status"] = [
        "CONFIRMED",
        "CONFIRMED",
        "CONFIRMED",
        "CONFIRMED",
        "CONFIRMED",
    ]
    test4_data["original_catalog_status"] = [
        "toi: CONFIRMED",
        "eu: CONFIRMED",
        "nasa: CONFIRMED",
        "oec: CONFIRMED",
        "epic: CONFIRMED",
    ]
    test4_data["checked_catalog_status"] = [
        "toi: CONFIRMED",
        "eu: CONFIRMED",
        "nasa: CONFIRMED",
        "oec: CONFIRMED",
        "epic: CONFIRMED",
    ]
    test4_data["discovery_year"] = [2019.0, 2012.0, 2012.0, 2011.0, 2012.0]

    test4_expected = expected_result.copy(deep=True)
    test4_expected["status"] = ["CONFIRMED"]
    test4_expected["original_status_string"] = [
        "epic: CONFIRMED,eu: CONFIRMED,nasa: CONFIRMED,oec: CONFIRMED,toi: CONFIRMED"
    ]
    test4_expected["checked_status_string"] = [
        "epic: CONFIRMED,eu: CONFIRMED,nasa: CONFIRMED,oec: CONFIRMED,toi: CONFIRMED"
    ]
    test4_expected["confirmed"] = [5]
    test4_expected["discovery_year"] = [2011]
    test4_data["discovery_method"] = ["Transit", "RV", "RV", "RV", "Imaging"]
    test4_expected["discovery_method"] = ["Imaging,RV"]

    # TEST 5: flags
    test5_data = data.copy(deep=True)
    # reset previous values (sometimes copy changes the data for whatever
    # reason)
    test5_data["status"] = [
        "CONFIRMED",
        "CONFIRMED",
        "CONFIRMED",
        "CONFIRMED",
        "CONFIRMED",
    ]
    test5_data["original_catalog_status"] = [
        "toi: CONFIRMED",
        "eu: CONFIRMED",
        "nasa: CONFIRMED",
        "oec: CONFIRMED",
        "epic: CONFIRMED",
    ]
    test5_data["checked_catalog_status"] = [
        "toi: CONFIRMED",
        "eu: CONFIRMED",
        "nasa: CONFIRMED",
        "oec: CONFIRMED",
        "epic: CONFIRMED",
    ]
    test5_data["discovery_year"] = [2019.0, 2012.0, 2012.0, 2011.0, 2012.0]
    test5_data["discovery_method"] = [
        "Transit",
        "Transit",
        "Transit",
        "Transit",
        "Transit",
    ]

    test5_expected = expected_result.copy(deep=True)
    test5_expected["status"] = ["CONFIRMED"]
    test5_expected["original_status_string"] = [
        "epic: CONFIRMED,eu: CONFIRMED,nasa: CONFIRMED,oec: CONFIRMED,toi: CONFIRMED"
    ]
    test5_expected["checked_status_string"] = [
        "epic: CONFIRMED,eu: CONFIRMED,nasa: CONFIRMED,oec: CONFIRMED,toi: CONFIRMED"
    ]
    test5_expected["confirmed"] = [5]
    test5_expected["discovery_year"] = [2011]
    test5_expected["discovery_method"] = ["Transit"]

    # 5.1 binary mismatch
    test51_data = test5_data.copy(deep=True)
    test51_expected = test5_expected.copy(deep=True)
    test51_data["binary_mismatch_flag"] = [0, 1, 2, 0, 1]
    test51_expected["binary_mismatch_flag"] = ["0,1,2"]

    # 5.2 coordinate mismatch
    test52_data = test5_data.copy(deep=True)
    test52_expected = test5_expected.copy(deep=True)
    test52_data["binary_mismatch_flag"] = [0, 0, 0, 0, 0]
    test52_expected["binary_mismatch_flag"] = ["0"]
    test52_data["coordinate_mismatch"] = ["RA", "DEC", "RADEC", "", ""]
    test52_expected["coordinate_mismatch"] = ["RA,DEC,RADEC"]
    test52_expected["coordinate_mismatch_flag"] = [2]

    # 5.3 coordinate mismatch part 2
    test53_data = test5_data.copy(deep=True)
    test53_expected = test5_expected.copy(deep=True)
    test53_data["coordinate_mismatch"] = ["RA", "", "", "", ""]
    test53_expected["coordinate_mismatch"] = ["RA"]
    test53_expected["coordinate_mismatch_flag"] = [1]

    # 5.4 angular separation mismatch
    test54_data = test5_data.copy(deep=True)
    test54_expected = test5_expected.copy(deep=True)
    test54_data["coordinate_mismatch"] = ["", "", "", "", ""]
    test54_expected["coordinate_mismatch"] = [""]
    test54_expected["coordinate_mismatch_flag"] = [0]
    test54_data["angular_separation"] = [
        "toi: 0.0",
        "eu: 0.350",
        "nasa: 0.0",
        "oec: 0.0",
        "epic: 0.0",
    ]
    test54_data["angsep"] = [0.0, 0.350, 0.0, 0.0, 0.0]
    test54_expected["angular_separation"] = [
        "epic: 0.0,eu: 0.350,nasa: 0.0,oec: 0.0,toi: 0.0"
    ]
    test54_expected["angular_separation_flag"] = [1]

    macro_tuple = [
        (test1_data, test1_expected),
        (test2_data, test2_expected),
        (test31_data, test31_expected),
        (test32_data, test32_expected),
        (test4_data, test4_expected),
        (test51_data, test51_expected),
        (test52_data, test52_expected),
        (test53_data, test53_expected),
        (test54_data, test54_expected),
    ]

    for data, expected in macro_tuple:
        result = Emc.merge_into_single_entry(data, "WASP-55", "", "b")

        assert sorted(result.columns) == sorted(expected.columns)

        for col in result.columns:
            if pd.isna(result[col]).all() and pd.isna(expected[col]).all():
                continue
            if col == "main_id_aliases":  # order here is not relevant
                assert sorted(result["main_id_aliases"][0].split(",")) == sorted(
                    expected["main_id_aliases"][0].split(",")
                )
            else:
                assert (result[col] == expected[col]).all()

    # TESTS WITH LOGGING
    # main id provenance
    # catalog duplicate flag

    # TEST 6: MAIN_ID PROVENANCE

    test6_data = data.copy(deep=True)
    # reset previous values (sometimes copy changes the data for whatever
    # reason)
    test6_data["status"] = [
        "CONFIRMED",
        "CONFIRMED",
        "CONFIRMED",
        "CONFIRMED",
        "CONFIRMED",
    ]
    test6_data["original_catalog_status"] = [
        "toi: CONFIRMED",
        "eu: CONFIRMED",
        "nasa: CONFIRMED",
        "oec: CONFIRMED",
        "epic: CONFIRMED",
    ]
    test6_data["checked_catalog_status"] = [
        "toi: CONFIRMED",
        "eu: CONFIRMED",
        "nasa: CONFIRMED",
        "oec: CONFIRMED",
        "epic: CONFIRMED",
    ]
    test6_data["discovery_year"] = [2019.0, 2012.0, 2012.0, 2011.0, 2012.0]
    test6_data["discovery_method"] = [
        "Transit",
        "Transit",
        "Transit",
        "Transit",
        "Transit",
    ]
    test6_data["angular_separation"] = [
        "toi: 0.0",
        "eu: 0.0",
        "nasa: 0.0",
        "oec: 0.0",
        "epic: 0.0",
    ]
    test6_data["angsep"] = [0.0, 0.0, 0.0, 0.0, 0.0]

    test6_expected = expected_result.copy(deep=True)
    test6_expected["status"] = ["CONFIRMED"]
    test6_expected["original_status_string"] = [
        "epic: CONFIRMED,eu: CONFIRMED,nasa: CONFIRMED,oec: CONFIRMED,toi: CONFIRMED"
    ]
    test6_expected["checked_status_string"] = [
        "epic: CONFIRMED,eu: CONFIRMED,nasa: CONFIRMED,oec: CONFIRMED,toi: CONFIRMED"
    ]
    test6_expected["confirmed"] = [5]
    test6_expected["discovery_year"] = [2011]
    test6_expected["discovery_method"] = ["Transit"]
    test6_expected["angular_separation"] = [
        "epic: 0.0,eu: 0.0,nasa: 0.0,oec: 0.0,toi: 0.0"
    ]
    test6_expected["angular_separation_flag"] = [0]

    test6_data["main_id_provenance"] = ["SIMBADCOORD", "TIC", "SIMBAD", "TICOORD", "eu"]
    test6_data["main_id_ra"] = [100, 90, 80, 50, 40]
    test6_data["main_id_dec"] = [100, 90, 80, 50, 40]
    test6_expected["main_id_provenance"] = ["SIMBAD"]
    test6_expected["main_id_ra"] = [80]
    test6_expected["main_id_dec"] = [80]

    result = Emc.merge_into_single_entry(test6_data, "WASP-55", "", "b")

    assert sorted(result.columns) == sorted(test6_expected.columns)

    for col in result.columns:
        if pd.isna(result[col]).all() and pd.isna(test6_expected[col]).all():
            continue
        if col == "main_id_aliases":  # order here is not relevant
            assert sorted(result["main_id_aliases"][0].split(",")) == sorted(
                test6_expected["main_id_aliases"][0].split(",")
            )
        else:
            assert (result[col] == test6_expected[col]).all()

    assert os.path.exists("Logs/merge_into_single_entry.txt")

    with open("Logs/merge_into_single_entry.txt") as f:
        lines = f.readlines()
        assert lines == [
            "\n",
            "WARNING, main_id_provenance not unique for  WASP-55  b\n",
            "  main_id_provenance  main_id_ra  main_id_dec angular_separation         "
            "p       a\n",
            "0        SIMBADCOORD         100          100           toi: 0.0  "
            "4.465634     NaN\n",
            "1                TIC          90           90            eu: 0.0  4.465630  "
            "0.0533\n",
            "2             SIMBAD          80           80          nasa: 0.0  4.465630  "
            "0.0558\n",
            "3            TICOORD          50           50           oec: 0.0  4.465633  "
            "0.0533\n",
            "4                 eu          40           40          epic: 0.0  "
            "4.465630     NaN\n",
        ]

        os.remove("Logs/merge_into_single_entry.txt")
        # TEST 7: MAIN_ID PROVENANCE

        test7_data = data.copy(deep=True)
        # reset previous values (sometimes copy changes the data for whatever
        # reason)
        test7_data["status"] = [
            "CONFIRMED",
            "CONFIRMED",
            "CONFIRMED",
            "CONFIRMED",
            "CONFIRMED",
        ]
        test7_data["original_catalog_status"] = [
            "toi: CONFIRMED",
            "eu: CONFIRMED",
            "nasa: CONFIRMED",
            "oec: CONFIRMED",
            "epic: CONFIRMED",
        ]
        test7_data["checked_catalog_status"] = [
            "toi: CONFIRMED",
            "eu: CONFIRMED",
            "nasa: CONFIRMED",
            "oec: CONFIRMED",
            "epic: CONFIRMED",
        ]
        test7_data["discovery_year"] = [2019.0, 2012.0, 2012.0, 2011.0, 2012.0]
        test7_data["discovery_method"] = [
            "Transit",
            "Transit",
            "Transit",
            "Transit",
            "Transit",
        ]
        test7_data["angular_separation"] = [
            "toi: 0.0",
            "eu: 0.0",
            "nasa: 0.0",
            "oec: 0.0",
            "epic: 0.0",
        ]
        test7_data["angsep"] = [0.0, 0.0, 0.0, 0.0, 0.0]
        test7_data["main_id_provenance"] = [
            "SIMBAD",
            "SIMBAD",
            "SIMBAD",
            "SIMBAD",
            "SIMBAD",
        ]
        test7_data["main_id_ra"] = [
            203.75814077726,
            203.75814077726,
            203.75814077726,
            203.75814077726,
            203.75814077726,
        ]
        test7_data["main_id_dec"] = [
            -17.50347989552,
            -17.50347989552,
            -17.50347989552,
            -17.50347989552,
            -17.50347989552,
        ]

        test7_expected = expected_result.copy(deep=True)
        test7_expected["status"] = ["CONFIRMED"]
        test7_expected["original_status_string"] = [
            "epic: CONFIRMED,eu: CONFIRMED,nasa: CONFIRMED,oec: CONFIRMED,toi: CONFIRMED"
        ]
        test7_expected["checked_status_string"] = [
            "epic: CONFIRMED,eu: CONFIRMED,nasa: CONFIRMED,oec: CONFIRMED,toi: CONFIRMED"
        ]
        test7_expected["confirmed"] = [5]
        test7_expected["discovery_year"] = [2011]
        test7_expected["discovery_method"] = ["Transit"]
        test7_expected["angular_separation"] = [
            "epic: 0.0,eu: 0.0,nasa: 0.0,oec: 0.0,toi: 0.0"
        ]
        test7_expected["angular_separation_flag"] = [0]
        test7_expected["main_id_provenance"] = ["SIMBAD"]
        test7_expected["main_id_ra"] = [203.75814077726]
        test7_expected["main_id_dec"] = [-17.50347989552]

        # add duplicate catalog
        test7_data["catalog"] = ["eu", "eu", "nasa", "oec", "epic"]
        test7_expected["duplicate_catalog_flag"] = [1]
        test7_expected["toi_name"] = ""
        test7_expected["eu_name"] = "TOI-774.01"  # the first one
        test7_expected["duplicate_names"] = [
            "eu: TOI-774.01,eu: WASP-55 b,nasa: WASP-55 b,oec: WASP-55 b,epic: WASP-55 b"
        ]
        test7_expected["catalog"] = "epic,eu,nasa,oec"
        result = Emc.merge_into_single_entry(test7_data, "WASP-55", "", "b")

        assert sorted(result.columns) == sorted(test7_expected.columns)

        for col in result.columns:
            if pd.isna(result[col]).all() and pd.isna(test7_expected[col]).all():
                continue
            if col == "main_id_aliases":  # order here is not relevant
                assert sorted(result["main_id_aliases"][0].split(",")) == sorted(
                    test6_expected["main_id_aliases"][0].split(",")
                )
            else:
                assert (result[col] == test7_expected[col]).all()

        assert os.path.exists("Logs/merge_into_single_entry.txt")

        with open("Logs/merge_into_single_entry.txt") as f:
            lines = f.readlines()
            assert lines == [
                "\n",
                "*** DUPLICATE ENTRY WASP-55  b ***\n",
                "  catalog catalog_name     status angular_separation         p       a\n",
                "0      eu   TOI-774.01  CONFIRMED           toi: 0.0  4.465634     NaN\n",
                "1      eu    WASP-55 b  CONFIRMED            eu: 0.0  4.465630  0.0533\n",
                "2    nasa    WASP-55 b  CONFIRMED          nasa: 0.0  4.465630  0.0558\n",
                "3     oec    WASP-55 b  CONFIRMED           oec: 0.0  4.465633  0.0533\n",
                "4    epic    WASP-55 b  CONFIRMED          epic: 0.0  4.465630     NaN\n",
            ]

    os.chdir(original_dir)


def test__group_by_letter_check_period(tmp_path, instance):
    original_dir = os.getcwd()

    os.chdir(tmp_path)  # Create a temporary in-memory configuration object
    os.mkdir("Logs/")

    # GENERALIZED DATASET AND EXPECTED DATASET
    data = {
        "name": ["6 Lyn b", "6 Lyn b"],
        "catalog_name": ["6 Lyn b", "6 Lyn b"],
        "discovery_method": ["Radial Velocity", "Radial Velocity"],
        "ra": [97.6958333, 97.6960311],
        "dec": [58.1627778, 58.1611753],
        "p": [934.3, 934.3],
        "p_max": [8.6, 8.6],
        "p_min": [8.6, 8.6],
        "a": [2.0, 2.0],
        "a_max": [0.1, 0.1],
        "a_min": [0.1, 0.1],
        "e": [0.073, 0.073],
        "e_max": [0.036, 0.036],
        "e_min": [0.036, 0.036],
        "i": [2.0, np.nan],
        "i_max": [79.0, np.nan],
        "i_min": [1.0, np.nan],
        "mass": [np.nan, np.nan],
        "mass_max": [np.nan, np.nan],
        "mass_min": [np.nan, np.nan],
        "msini": [2.01, 2.01],
        "msini_max": [0.077, 0.077],
        "msini_min": [0.077, 0.077],
        "r": [np.nan, np.nan],
        "r_max": [np.nan, np.nan],
        "r_min": [np.nan, np.nan],
        "discovery_year": [2008, 2008],
        "alias": [
            "2MASS J06304711+5809453",
            "2MASS J06304711+5809453,6 Lyncis",
        ],
        "a_url": ["eu", "2019AJ....157..149L"],
        "mass_url": ["", ""],
        "p_url": ["eu", "2019AJ....157..149L"],
        "msini_url": ["eu", "2019AJ....157..149L"],
        "r_url": ["", ""],
        "i_url": ["eu", ""],
        "e_url": ["eu", "2019AJ....157..149L"],
        "main_id": ["*   6 Lyn", "*   6 Lyn"],
        "host": ["6 Lyn", "6 Lyn"],
        "binary": ["", ""],
        "letter": ["b", "b"],
        "status": ["CONFIRMED", "CONFIRMED"],
        "catalog": ["eu", "nasa"],
        "original_catalog_status": ["eu: CONFIRMED", "nasa: CANDIDATE"],
        "checked_catalog_status": ["eu: CONFIRMED", "nasa: CONFIRMED"],
        "binary_mismatch_flag": [0, 0],
        "hostbinary": ["6 Lyn", "6 Lyn"],
        "RA": ["06 30 47.1075", "06 30 47.1075"],
        "DEC": ["+58 09 45.479", "+58 09 45.479"],
        "list_id": [
            "LTT 11856,*   6 Lyn",
            "LTT 11856,*   6 Lyn",
        ],
        "main_id_ra": [97.69628124999998, 97.69628124999998],
        "main_id_dec": [58.16263305555555, 58.16263305555555],
        "coordinate_mismatch": ["", ""],
        "angsep": [0.0, 0.0],
        "angular_separation": ["eu: 0.0", "nasa: 0.0"],
        "main_id_provenance": ["SIMBAD", "SIMBAD"],
        "main_id_aliases": [
            "2MASS J06304711+5809453,HR 2331",
            "2MASS J06304711+5809453,HR 2331",
        ],
    }

    data = pd.DataFrame(data)
    expected_result = {
        "main_id": ["*   6 Lyn"],
        "binary": [""],
        "letter": ["b"],
        "host": ["6 Lyn"],
        "angular_separation": ["eu: 0.0,nasa: 0.0"],
        "main_id_ra": [97.69628124999998],
        "main_id_dec": [58.16263305555555],
        "eu_name": ["6 Lyn b"],
        "nasa_name": ["6 Lyn b"],
        "oec_name": [""],
        "toi_name": [""],
        "epic_name": [""],
        "i_url": ["eu"],
        "i": [2.0],
        "i_min": [1.0],
        "i_max": [79.0],
        "IREL": [39.5],
        "checked_status_string": ["eu: CONFIRMED,nasa: CONFIRMED"],
        "original_status_string": ["eu: CONFIRMED,nasa: CANDIDATE"],
        "confirmed": [2],
        "status": ["CONFIRMED"],
        "discovery_year": [2008],
        "discovery_method": ["Radial Velocity"],
        "catalog": ["eu,nasa"],
        "main_id_aliases": ["2MASS J06304711+5809453,HR 2331"],
        "binary_mismatch_flag": ["0"],
        "coordinate_mismatch": [""],
        "coordinate_mismatch_flag": [0],
        "angular_separation_flag": [0],
        "main_id_provenance": ["SIMBAD"],
        "duplicate_catalog_flag": [0],
        "mass_url": [""],
        "mass": [np.nan],
        "mass_min": [np.nan],
        "mass_max": [np.nan],
        "MASSREL": [np.nan],
        "msini_url": ["2019AJ....157..149L"],
        "msini": [2.01],
        "msini_min": [0.077],
        "msini_max": [0.077],
        "MSINIREL": [0.03830845771144279],
        "r_url": [""],
        "r": [np.nan],
        "r_min": [np.nan],
        "r_max": [np.nan],
        "RADREL": [np.nan],
        "a_url": ["2019AJ....157..149L"],
        "a": [2.0],
        "a_min": [0.10],
        "a_max": [0.10],
        "AREL": [0.05],
        "p_url": ["2019AJ....157..149L"],
        "p": [934.3],
        "p_min": [8.6],
        "p_max": [8.6],
        "PERREL": [0.009204752220914053],
        "e_url": ["2019AJ....157..149L"],
        "e": [0.073],
        "e_min": [0.036],
        "e_max": [0.036],
        "EREL": [0.4931506849315068],
        "duplicate_names": [""],
        "merging_mismatch_flag": [0],
    }
    expected_result = pd.DataFrame(expected_result)

    # SUCCESSFUL MERGING CASES

    # CASE 1: P exists and it is in agreement. Regular Merge
    case1_data = data.copy(deep=True)
    case1_expected = expected_result.copy(deep=True)

    # CASE 2: No period, but sma. Merging can happen
    case2_data = data.copy(deep=True)
    case2_data["p"] = [np.nan, np.nan]
    case2_data["p_max"] = [np.nan, np.nan]
    case2_data["p_min"] = [np.nan, np.nan]
    case2_data["p_url"] = ["", ""]
    case2_expected = expected_result.copy(deep=True)
    case2_expected["p"] = np.nan
    case2_expected["p_min"] = np.nan
    case2_expected["p_max"] = np.nan
    case2_expected["p_url"] = ""
    case2_expected["PERREL"] = np.nan

    # run same tests for all successful cases
    macro_tuple = [(case1_data, case1_expected), (case2_data, case2_expected)]
    for data, expected in macro_tuple:
        instance.data = data
        instance.group_by_letter_check_period(verbose=True)
        assert os.path.exists("Logs/group_by_letter_check_period.txt")

        with open("Logs/group_by_letter_check_period.txt") as f:
            lines = f.readlines()
            assert lines == []
        os.remove("Logs/group_by_letter_check_period.txt")

        assert sorted(instance.data.columns) == sorted(expected.columns)

        instance.data = instance.data.convert_dtypes()
        expected = expected.convert_dtypes()
        for col in instance.data.columns:
            for row in instance.data.index:
                if pd.isna(instance.data.at[row, col]) and pd.isna(
                    expected.at[row, col]
                ):
                    continue
                try:
                    assert instance.data.at[row, col] == expected.at[row, col]
                except AssertionError:
                    assert np.isclose(instance.data.at[row, col], expected.at[row, col])

    # FALLBACK MERGING: merges into one, but no info on p or sma.
    # CASE 3: No period, but sma. Merging can happen
    case3_data = data.copy(deep=True)
    case3_data["p"] = [np.nan, np.nan]
    case3_data["p_max"] = [np.nan, np.nan]
    case3_data["p_min"] = [np.nan, np.nan]
    case3_data["p_url"] = ["", ""]
    case3_data["a"] = [np.nan, np.nan]
    case3_data["a_max"] = [np.nan, np.nan]
    case3_data["a_min"] = [np.nan, np.nan]
    case3_data["a_url"] = ["", ""]

    case3_expected = expected_result.copy(deep=True)
    case3_expected["p"] = np.nan
    case3_expected["p_min"] = np.nan
    case3_expected["p_max"] = np.nan
    case3_expected["p_url"] = ""
    case3_expected["PERREL"] = np.nan
    case3_expected["a"] = np.nan
    case3_expected["a_min"] = np.nan
    case3_expected["a_max"] = np.nan
    case3_expected["a_url"] = ""
    case3_expected["AREL"] = np.nan
    case3_expected["merging_mismatch_flag"] = 2

    # run same tests for fallback case (log text changes)
    macro_tuple = [(case3_data, case3_expected)]
    for data, expected in macro_tuple:
        instance.data = data
        instance.group_by_letter_check_period(verbose=True)
        assert os.path.exists("Logs/group_by_letter_check_period.txt")

        with open("Logs/group_by_letter_check_period.txt") as f:
            lines = f.readlines()
            assert lines == [
                "FALLBACK, MERGE (merging_mismatch_flag=2) \n",
                "     main_id binary letter catalog catalog_name\n",
                "0  *   6 Lyn             b      eu      6 Lyn b\n",
                "1  *   6 Lyn             b    nasa      6 Lyn b\n",
                "\n",
            ]
        os.remove("Logs/group_by_letter_check_period.txt")

        assert sorted(instance.data.columns) == sorted(expected.columns)

        instance.data = instance.data.convert_dtypes()
        expected = expected.convert_dtypes()
        for col in instance.data.columns:
            for row in instance.data.index:
                if pd.isna(instance.data.at[row, col]) and pd.isna(
                    expected.at[row, col]
                ):
                    continue
                try:
                    assert instance.data.at[row, col] == expected.at[row, col]
                except AssertionError:
                    assert np.isclose(instance.data.at[row, col], expected.at[row, col])

    # DISAGREEMENTS
    # CASE 4: Periods disagreeing, keep both
    case4_data = data.copy(deep=True)
    case4_data["p"] = [8.0, 15.0]
    case4_data["p_max"] = [0.5, 0.6]
    case4_data["p_min"] = [0.5, 0.6]
    case4_data["p_url"] = ["eu", "2019AJ....157..149L"]
    case4_data["a"] = [2.0, 2.0]
    case4_data["a_max"] = [0.1, 0.1]
    case4_data["a_min"] = [0.1, 0.1]
    case4_data["a_url"] = ["eu", "2019AJ....157..149L"]

    # create new ad hoc expected dataframe. Get columns from original data,
    # then add some
    case4_expected = pd.DataFrame(columns=expected_result.columns)
    for col in case4_data.columns:
        if col in case4_expected.columns:
            case4_expected[col] = case4_data[col]
    case4_expected["binary_mismatch_flag"] = ["0", "0"]

    case4_expected["main_id_ra"] = case4_data["main_id_ra"]
    case4_expected["main_id_dec"] = case4_data["main_id_dec"]
    case4_expected["eu_name"] = ["6 Lyn b", ""]
    case4_expected["nasa_name"] = ["", "6 Lyn b"]
    case4_expected["oec_name"] = ["", ""]
    case4_expected["toi_name"] = ["", ""]
    case4_expected["epic_name"] = ["", ""]
    case4_expected["IREL"] = [39.5, np.nan]
    case4_expected["checked_status_string"] = ["eu: CONFIRMED", "nasa: CONFIRMED"]
    case4_expected["original_status_string"] = ["eu: CONFIRMED", "nasa: CANDIDATE"]
    case4_expected["confirmed"] = [1, 1]
    case4_expected["coordinate_mismatch_flag"] = [0, 0]
    case4_expected["angular_separation_flag"] = [0, 0]
    case4_expected["duplicate_catalog_flag"] = [0, 0]
    case4_expected["MASSREL"] = [np.nan, np.nan]
    case4_expected["MSINIREL"] = [0.03830845771144279, 0.03830845771144279]
    case4_expected["RADREL"] = [np.nan, np.nan]
    case4_expected["AREL"] = [0.05, 0.05]
    case4_expected["PERREL"] = [0.0625, 0.04]
    case4_expected["EREL"] = [0.493151, 0.493151]
    case4_expected["duplicate_names"] = ["", ""]
    case4_expected["merging_mismatch_flag"] = [1, 1]

    # run same tests for case 4 (log text changes)
    macro_tuple = [(case4_data, case4_expected)]
    for data, expected in macro_tuple:
        instance.data = data
        instance.group_by_letter_check_period(verbose=True)
        assert os.path.exists("Logs/group_by_letter_check_period.txt")

        with open("Logs/group_by_letter_check_period.txt") as f:
            lines = f.readlines()
            assert lines == [
                "DISAGREEMENT (merging_mismatch_flag=1)\n",
                "     main_id binary letter catalog catalog_name     p\n",
                "0  *   6 Lyn             b      eu      6 Lyn b   8.0\n",
                "1  *   6 Lyn             b    nasa      6 Lyn b  15.0\n",
                "\n",
            ]
        os.remove("Logs/group_by_letter_check_period.txt")

        assert sorted(instance.data.columns) == sorted(expected.columns)

        instance.data = instance.data.convert_dtypes()
        expected = expected.convert_dtypes()
        for col in instance.data.columns:
            for row in instance.data.index:
                if pd.isna(instance.data.at[row, col]) and pd.isna(
                    expected.at[row, col]
                ):
                    continue
                try:
                    assert instance.data.at[row, col] == expected.at[row, col]
                except AssertionError:
                    assert np.isclose(instance.data.at[row, col], expected.at[row, col])

    # CASE 5: SMA disagreeing, keep both
    case5_data = data.copy(deep=True)
    case5_data["p"] = [np.nan, np.nan]
    case5_data["p_max"] = [np.nan, np.nan]
    case5_data["p_min"] = [np.nan, np.nan]
    case5_data["p_url"] = ["", ""]
    case5_data["a"] = [2.0, 4.0]
    case5_data["a_max"] = [0.1, 0.1]
    case5_data["a_min"] = [0.1, 0.1]
    case5_data["a_url"] = ["eu", "2019AJ....157..149L"]

    # create new ad hoc expected dataframe. Get columns from case 4 and change
    # some
    case5_expected = case4_expected.copy(deep=True)
    for col in case5_data.columns:
        if col in case5_expected.columns:
            case5_expected[col] = case5_data[col]
    case5_expected["binary_mismatch_flag"] = ["0", "0"]
    case5_expected["AREL"] = [0.05, 0.025]
    case5_expected["PERREL"] = [np.nan, np.nan]
    case5_expected["merging_mismatch_flag"] = [1, 1]

    # run same tests for case 5 (log text changes)
    macro_tuple = [(case5_data, case5_expected)]
    for data, expected in macro_tuple:
        instance.data = data
        instance.group_by_letter_check_period(verbose=True)
        assert os.path.exists("Logs/group_by_letter_check_period.txt")

        with open("Logs/group_by_letter_check_period.txt") as f:
            lines = f.readlines()
            assert lines == [
                "DISAGREEMENT (merging_mismatch_flag=1)\n",
                "     main_id binary letter catalog catalog_name    a\n",
                "0  *   6 Lyn             b      eu      6 Lyn b  2.0\n",
                "1  *   6 Lyn             b    nasa      6 Lyn b  4.0\n",
                "\n",
            ]
        os.remove("Logs/group_by_letter_check_period.txt")

        assert sorted(instance.data.columns) == sorted(expected.columns)

        instance.data = instance.data.convert_dtypes()
        expected = expected.convert_dtypes()
        for col in instance.data.columns:
            for row in instance.data.index:
                if pd.isna(instance.data.at[row, col]) and pd.isna(
                    expected.at[row, col]
                ):
                    continue
                try:
                    assert instance.data.at[row, col] == expected.at[row, col]
                except AssertionError:
                    assert np.isclose(instance.data.at[row, col], expected.at[row, col])

    os.chdir(original_dir)


def test__select_best_mass(instance):
    data = {
        "name": [
            "Msini case",
            "Mass case",
            "MsiniNULL case",
            "MassNULL case",
            "nan case",
        ],
        "mass": [10, 10, 10, np.nan, np.nan],
        "mass_min": [0.1, 1, 1, np.nan, np.nan],
        "mass_max": [0.1, 1, 1, np.nan, np.nan],
        "mass_url": ["url1", "url2", "url3", "", ""],
        "MASSREL": [0.01, 0.1, 0.1, np.nan, np.nan],
        "msini": [5, 10, np.nan, 5, np.nan],
        "msini_min": [0.1, 10, np.nan, 0.1, np.nan],
        "msini_max": [0.1, 10, np.nan, 0.1, np.nan],
        "MSINIREL": [0.005, 1, np.nan, 0.005, np.nan],
        "msini_url": ["url1", "url2", "", "url4", ""],
    }
    instance.data = pd.DataFrame(data)
    with LogCapture() as log:
        instance.select_best_mass()
        log = pd.DataFrame(list(log), columns=["user", "info", "message"])

        assert "Bestmass calculated." in log["message"].to_list()

    assert instance.data.at[0, "bestmass"] == 5
    assert instance.data.at[0, "bestmass_min"] == 0.1
    assert instance.data.at[0, "bestmass_max"] == 0.1
    assert instance.data.at[0, "bestmass_provenance"] == "Msini"
    assert instance.data.at[1, "bestmass"] == 10
    assert instance.data.at[1, "bestmass_min"] == 1
    assert instance.data.at[1, "bestmass_max"] == 1
    assert instance.data.at[1, "bestmass_provenance"] == "Mass"
    assert instance.data.at[2, "bestmass"] == 10
    assert instance.data.at[2, "bestmass_min"] == 1
    assert instance.data.at[2, "bestmass_max"] == 1
    assert instance.data.at[2, "bestmass_provenance"] == "Mass"
    assert instance.data.at[3, "bestmass"] == 5
    assert instance.data.at[3, "bestmass_min"] == 0.1
    assert instance.data.at[3, "bestmass_max"] == 0.1
    assert instance.data.at[3, "bestmass_provenance"] == "Msini"
    assert pd.isna(instance.data.at[4, "bestmass"])
    assert pd.isna(instance.data.at[4, "bestmass_min"])
    assert pd.isna(instance.data.at[4, "bestmass_max"])
    assert instance.data.at[4, "bestmass_provenance"] == ""


def test__set_exo_mercat_name(instance):
    data = {
        "main_id": ["*  51 Peg", "*  16 Cyg B", "HD 106515A"],
        "binary": ["", "B", "A"],
        "letter": ["b", "b", "b"],
    }
    instance.data = pd.DataFrame(data)
    with LogCapture() as log:
        instance.set_exo_mercat_name()
        log = pd.DataFrame(list(log), columns=["user", "info", "message"])

        assert "Exo-MerCat name assigned." in log["message"].tolist()
        assert list(instance.data["exo_mercat_name"]) == [
            "*  16 Cyg B b",
            "*  51 Peg b",
            "HD 106515 A b",
        ]


def test__fill_row_update(instance, tmp_path):
    original_dir = os.getcwd()

    os.chdir(tmp_path)  # Create a temporary in-memory configuration object
    os.mkdir("Exo-MerCat/")

    compar_data = {
        "exo_mercat_name": ["KMT-2019-BLG-2991 b", "*   6 Lyn b"],
        "host": ["MOA 2019-BLG-421", "6 Lyn"],
        "letter": ["b", "b"],
        "main_id": ["KMT-2019-BLG-2991", "*   6 Lyn"],
        "binary": ["", ""],
        "main_id_ra": [271.54545, 97.69628],
        "main_id_dec": [-27.485469, 58.16263],
        "mass": [0.54, 2.01],
        "mass_max": [0.37, 0.077],
        "mass_min": [0.37, 0.077],
        "mass_url": ["eu", "eu"],
    }
    compar_data = pd.DataFrame(compar_data)
    # CASE: no other exomercats available

    instance.data = compar_data.copy()
    instance.fill_row_update(local_date="2020-01-01")
    assert "row_update" in instance.data.columns
    assert instance.data.loc[0, "row_update"] == "2020-01-01"
    assert instance.data.loc[1, "row_update"] == "2020-01-01"

    # CASE: Data has changed

    new_data = {
        "exo_mercat_name": ["KMT-2019-BLG-2991 b", "*   6 Lyn b"],
        "host": ["MOA 2019-BLG-421", "6 Lyn"],
        "letter": ["b", "b"],
        "main_id": ["KMT-2019-BLG-2991", "*   6 Lyn"],
        "binary": ["", ""],
        "main_id_ra": [271.54545, 97.6962],
        "main_id_dec": [-27.485469, 58.1623],
        "mass": [0.54, 2.10],
        "mass_max": [0.37, 0.09],
        "mass_min": [0.37, 0.09],
        "mass_url": ["eu", "nasa"],
    }
    new_data = pd.DataFrame(new_data)
    pd.DataFrame(compar_data).to_csv("Exo-MerCat/exo-mercat_full2020-01-01.csv")
    instance.data = new_data.copy()
    instance.fill_row_update(local_date="")
    assert "row_update" in instance.data.columns
    assert instance.data.loc[0, "row_update"] == "2020-01-01"
    assert instance.data.loc[1, "row_update"] == date.today().strftime("%Y-%m-%d")
    os.chdir(original_dir)


def test__keep_columns(instance):
    # Test the keep_columns function

    # Create a sample DataFrame with some additional columns
    data = {
        "main_id": ["*   6 Lyn"],
        "binary": [""],
        "letter": ["b"],
        "main_id_ra": [97.69628320833333],
        "main_id_dec": [58.16263358333333],
        "oec_name": ["6 Lyn b"],
        "name": ["6 Lyn b"],
        "host": ["6 Lyn"],
        "i_url": ["oec"],
        "i": [2.0],
        "i_min": [1.0],
        "i_max": [79.0],
        "IREL": [39.5],
        "checked_status_string": ["oec: CONFIRMED"],
        "original_status_string": ["oec: CONFIRMED"],
        "confirmed": [1],
        "status": ["CONFIRMED"],
        "discovery_year": [2008.0],
        "discovery_method": ["Radial Velocity"],
        "catalog": ["oec"],
        "main_id_aliases": [
            "GEN# +1.00045410,UCAC4 741-044033,*   6 Lyn,USNO-B1.0 1481-00215981,CSI+58   932  1,HIP  31039,"
            "SKY# 11231,PPM  30486,GCRV  4140,CCDM J06309+5810A,uvby98 100045410,Gaia DR2 1004358968092652544,"
            "SAO  25771,YZ  58  4530,UBV    6436,[B10]  1627,LSPM J0630+5809,LTT 11856,WDS J06308+5810A,NLTT 16571,"
            "HR 2331,PLX 1499.00,BD+58   932,BD+58 932,TIC 444865362,HIC  31039,AP J06304711+5809453,IDS 06220+5814 "
            "A,TYC 3777-2071-1,SPOCS 2671,Gaia DR3 1004358968092652544,2MASS J06304711+5809453,DO 30475,GC  8416,"
            "PLX 1499,WEB  6178,AG+58  545,N30 1407,HIP 31039,6 Lyncis,IRAS 06264+5811,HD 45410,SAO 25771,"
            "UBV M  12104,HR  2331,HD  45410,ASCC  182129"
        ],
        "coordinate_mismatch": [""],
        "coordinate_mismatch_flag": [0],
        "angular_separation": ["oec: 0.0"],
        "angular_separation_flag": [0],
        "missing_simbad_flag": [0],
        "duplicate_catalog_flag": [0],
        "duplicate_names": [""],
        "mass_url": [np.nan],
        "mass": [np.nan],
        "mass_min": [np.nan],
        "mass_max": [np.nan],
        "MASSREL": [np.nan],
        "msini_url": ["oec"],
        "msini": [2.21],
        "msini_min": [0.16],
        "msini_max": [0.11],
        "MSINIREL": [0.07239819004524888],
        "r_url": [np.nan],
        "r": [np.nan],
        "r_min": [np.nan],
        "r_max": [np.nan],
        "RADREL": [np.nan],
        "a_url": ["oec"],
        "a": [2.18],
        "a_min": [0.06],
        "a_max": [0.05],
        "AREL": [0.027522935779816512],
        "p_url": ["oec"],
        "p": [874.774],
        "p_min": [8.47],
        "p_max": [16.27],
        "PERREL": [0.018599089593426415],
        "e_url": ["oec"],
        "e": [0.059],
        "e_min": [0.059],
        "e_max": [0.066],
        "EREL": [1.1186440677966103],
        "eu_name": [""],
        "nasa_name": [""],
        "toi_name": [""],
        "epic_name": [""],
        "bestmass": [2.21],
        "bestmass_min": [0.16],
        "bestmass_max": [0.11],
        "bestmass_url": ["oec"],
        "bestmass_provenance": ["Msini"],
        "exo_mercat_name": ["*   6 Lyn  b"],
        "main_id_provenance": ["SIMBAD"],
        "binary_mismatch_flag": [0],
        "merging_mismatch_flag": [0],
        "row_update": ["03-28-2024"],
    }

    df = pd.DataFrame(data)
    instance.data = df
    with LogCapture() as log:
        instance.keep_columns()
        log = pd.DataFrame(list(log), columns=["user", "info", "message"])
        assert "Selected columns to keep." in log["message"].tolist()

    # Check if the DataFrame contains only the columns specified in the keep
    # list
    expected_columns = [
        "exo_mercat_name",
        "nasa_name",
        "toi_name",
        "epic_name",
        "eu_name",
        "oec_name",
        "host",
        "letter",
        "main_id",
        "binary",
        "main_id_ra",
        "main_id_dec",
        "mass",
        "mass_max",
        "mass_min",
        "mass_url",
        "msini",
        "msini_max",
        "msini_min",
        "msini_url",
        "bestmass",
        "bestmass_max",
        "bestmass_min",
        "bestmass_url",
        "bestmass_provenance",
        "p",
        "p_max",
        "p_min",
        "p_url",
        "r",
        "r_max",
        "r_min",
        "r_url",
        "a",
        "a_max",
        "a_min",
        "a_url",
        "e",
        "e_max",
        "e_min",
        "e_url",
        "i",
        "i_max",
        "i_min",
        "i_url",
        "discovery_method",
        "status",
        "checked_status_string",
        "original_status_string",
        "confirmed",
        "discovery_year",
        "main_id_aliases",
        "catalog",
        "angular_separation",
        "angular_separation_flag",
        "main_id_provenance",
        "binary_mismatch_flag",
        "coordinate_mismatch",
        "coordinate_mismatch_flag",
        "duplicate_catalog_flag",
        "duplicate_names",
        "merging_mismatch_flag",
        "row_update",
    ]
    assert list(instance.data.columns) == expected_columns

    data = {
        "catalog_name": ["OEC"],
        "extra": ["extra"],
    }
    df = pd.DataFrame(data)
    instance.data = df
    with pytest.raises(KeyError):
        instance.keep_columns()


def test_remove_known_brown_dwarfs(tmp_path, instance):
    # Test if the function removes known brown dwarfs correctly
    original_dir = os.getcwd()

    os.chdir(tmp_path)  # Create a temporary in-memory configuration object
    os.mkdir("Exo-MerCat/")
    data = {
        "name": ["Planet1", "Planet2", "Planet3", "Planet4"],
        "mass": [15.0, 25.0, None, None],
        "msini": [18.0, 25.0, None, 18.0],
        "letter": ["A", "BD", "C", "BD"],
    }

    df = pd.DataFrame(data)
    instance.data = df
    local_date = ""
    instance.remove_known_brown_dwarfs(local_date, print_flag=True)

    # Check if known brown dwarfs have been removed from the DataFrame
    expected_result = {
        "name": ["Planet1", "Planet3", "Planet4"],
        "mass": [15.0, None, None],
        "msini": [18.0, None, 18.0],
        "letter": ["A", "C", "BD"],
    }
    expected_df = pd.DataFrame(expected_result)
    pd.testing.assert_frame_equal(
        instance.data.reset_index()[["name", "mass", "msini", "letter"]], expected_df
    )

    with open("Exo-MerCat/exo_mercat_brown_dwarfs.csv", "r") as file:
        f = file.readlines()
    assert "Planet2,25.0,25.0,BD\n" in f

    instance.data = df
    local_date = "2023-01-01"
    instance.remove_known_brown_dwarfs(local_date, print_flag=True)
    with open("Exo-MerCat/exo_mercat_brown_dwarfs2023-01-01.csv", "r") as file:
        f = file.readlines()
    assert "Planet2,25.0,25.0,BD\n" in f

    os.chdir(original_dir)


def test_save_catalog(instance, tmp_path):
    original_dir = os.getcwd()

    os.chdir(tmp_path)  # Create a temporary in-memory configuration object
    os.mkdir("Exo-MerCat/")
    # Sample input data (DataFrame)
    data = pd.DataFrame(
        {
            "name": ["KOI-1", "KOI-2", "KOI-3"],
            "host": ["Kepler-1", "Kepler-2", "Kepler-3"],
            "alias": ["KOI-1", "Kepler-2 b", "KOI-3"],
        }
    )
    # Expected output (DataFrame sorted by "name")
    expected_output = """name,host,alias\nKOI-1,Kepler-1,KOI-1\nKOI-2,Kepler-2,Kepler-2 b\nKOI-3,Kepler-3,KOI-3\n"""

    instance.data = data
    local_date = "2023-02-04"
    # Temporary in-memory file for testing
    temp_file1 = "Exo-MerCat/exo-mercat_full2023-02-04.csv"
    temp_file2 = "Exo-MerCat/exo-mercat_full.csv"
    # Call the function with the sample input data and temporary file
    instance.save_catalog(local_date, "_full")
    import glob

    assert os.path.exists(temp_file1)
    assert os.path.exists(temp_file2)
    file_content = open(temp_file1).read()
    assert file_content == expected_output
    file_content = open(temp_file2).read()
    assert file_content == expected_output
    # Clean up: Remove the temporary CSV file
    os.remove(temp_file1)
    os.remove(temp_file2)

    local_date = ""  # today
    # Temporary in-memory file for testing
    temp_file1 = "Exo-MerCat/exo-mercat" + date.today().strftime("%Y-%m-%d") + ".csv"
    temp_file2 = "Exo-MerCat/exo-mercat.csv"
    # Call the function with the sample input data and temporary file
    instance.save_catalog(local_date, "")
    assert os.path.exists(temp_file1)
    assert os.path.exists(temp_file2)
    file_content = open(temp_file1).read()
    assert file_content == expected_output
    file_content = open(temp_file2).read()
    assert file_content == expected_output
    # Cleaxn up: Remove the temporary CSV file
    os.remove(temp_file1)
    os.remove(temp_file2)

    os.chdir(original_dir)
