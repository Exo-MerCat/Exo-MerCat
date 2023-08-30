from unittest.mock import patch, MagicMock
from astropy.table import Table
import numpy as np
import pytest
import os

from pandas._testing import assert_frame_equal
from testfixtures import LogCapture

from exo_mercat.emc import (
    Emc,
    round_to_decimal,
    round_array_to_significant_digits,
    merge_into_single_entry,
    round_parameter_bin,
)
import pandas as pd


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
            "ups And b",
            "ups And c",
            "ups And d",
            "ups And b",
            "ups And c",
            "ups And e",
            "ups And A b",
            "ups And A c",
            "ups And A d",
            "ups And A e",
        ],
        "host": [
            "ups And",
            "ups And",
            "ups And",
            "ups And",
            "ups And",
            "ups And",
            "ups And",
            "ups And",
            "ups And",
            "ups And",
        ],
        "alias": [
            "TIC 189576919,Gaia DR2 348020448377061376,HD 9826,HIP 7513",
            "TIC 189576919,Gaia DR2 348020448377061376,HD 9826,HIP 7513",
            "TIC 189576919,Gaia DR2 348020448377061376,HD 9826,HIP 7513",
            "HD 9826",
            "HD 9826",
            "HD 9826",
            "nan",
            "nan",
            "nan",
            np.nan,
        ],
    }

    instance.data = pd.DataFrame(data)

    original = list(instance.data.alias.drop_duplicates())
    # Call the identify_brown_dwarfs function
    with LogCapture() as log:
        instance.alias_as_host()
        assert "Aliases labeled as hosts" in log.actual()[0][-1]

    final = list(instance.data.alias.drop_duplicates())

    assert len(final) == 1
    assert list(instance.data.alias.drop_duplicates()) == [
        "Gaia DR2 348020448377061376,HD 9826,HIP 7513,TIC 189576919"
    ]

    data = {
        "name": [
            "Kepler-396 b",
            "Kepler-396 c",
            "Kepler-396 b",
            "Kepler-396 c",
            "KOI-2672.01",
            "KOI-2672.02",
        ],
        "host": [
            "Kepler-396",
            "Kepler-396",
            "Kepler-396",
            "Kepler-396",
            "KOI-2672",
            "KOI-2672",
        ],
        "alias": [
            "Gaia DR2 2134847343119885440,KOI-2672,KIC 11253827,Kepler-396,TIC 27769688",
            "Gaia DR2 2134847343119885440,KOI-2672,KIC 11253827,Kepler-396,TIC 27769688",
            "KOI-2672,WISE J194431.87+485838.4,KIC 11253827,Kepler-396,2MASS J19443187+4858386",
            "KOI-2672,WISE J194431.87+485838.4,KIC 11253827,Kepler-396,2MASS J19443187+4858386",
            "Gaia DR2 2134847343119885440,KOI-2672,TYC 3565-2-1,KIC 11253827,Kepler-396,2MASS J19443187+4858386",
            "Gaia DR2 2134847343119885440,KOI-2672,TYC 3565-2-1,KIC 11253827,Kepler-396,2MASS J19443187+4858386",
        ],
    }

    instance.data = pd.DataFrame(data)
    instance.alias_as_host()

    assert (list(instance.data.alias.drop_duplicates())) == [
        "2MASS J19443187+4858386,Gaia DR2 2134847343119885440,KIC 11253827,KOI-2672,Kepler-396,TIC 27769688,TYC 3565-2-1,WISE J194431.87+485838.4"
    ]

    assert (
        len(
            instance.data.loc[instance.data.host == "Kepler-396", "host"]
            .drop_duplicates()
            .to_list()
        )
        == 1
    )

    assert os.path.exists("Logs/alias_as_host.txt")

    with open("Logs/alias_as_host.txt") as f:
        lines = f.readlines()
        assert lines == [
            "ALIAS: Kepler-396 AS HOST:KOI-2672\n",
            "ALIAS: KOI-2672 AS HOST:Kepler-396\n",
        ]

    os.chdir(original_dir)


def test__simbad_list_host_search(tmp_path, instance):
    data = pd.DataFrame(
        {
            "host": ["HD 114762", "PSR B1620-26", "51 Peg", "16 Cyg", "nonexisting"],
            "main_id": ["", "", "", "", ""],
            "IDS": ["", "", "", "", ""],
            "RA": ["", "", "", "", ""],
            "DEC": ["", "", "", "", ""],
        }
    )

    instance.data = data  # Replace YourClassName with the actual class name

    # Mock the Simbad.query_objects method with sample result_table
    with patch("astroquery.simbad.Simbad.query_objects") as mock_query_objects:
        mock_response = MagicMock()
        mock_response = Table(
            [
                ["HD 114762", "PSR B1620-26", "51 Peg", "16 Cyg"],
                ["HD 114762", "PSR B1620-26", "*  51 Peg", "*  16 Cyg"],
                [
                    "HD 114762|** PAT   47A|AG+17 1351|AGKR 11813|AP J13121982+1731016|ASCC  868467|BD+18  2700|Ci 20  766|Ci 18 1695|FK5 5165|GC 17881|GCRV  7854|GEN# +1.00114762|G  63-9|HD 114762A|HIC  64426|HIP  64426|LFT  980|LHS  2693|LSPM J1312+1731|LTT 13819|2MASS J13121982+1731016|N30 3028|NLTT 33242|PM 13099+1747|PPM 129656|SAO 100458|SBC9 2406|SKY# 24417|SPOCS  555|TD1 16696|TIC 138172859|TYC 1454-315-1|UBV   11938|UBV M  19151|USNO-B1.0 1075-00258599|uvby98 100114762|WDS J13123+1731A|WEB 11388|WISEA J131219.31+173101.6|YZC 18  4855|YZ  17  4855|Gaia DR2 3937211745904553600|Gaia DR3 3937211745905473024|Gaia DR1 3937211741607576576",
                    "PSR B1620-26|PSR J1623-2631|[BPH2004] CX 12|EQ J1623-2631",
                    "LTT 16750|*  51 Peg|** RBR   21A|AG+20 2595|AKARI-IRC-V1 J2257280+204608|ASCC  826013|BD+19  5036|CSV 102222|GC 32003|GCRV 14411|GEN# +1.00217014|GJ   882|HD 217014|HIC 113357|HIP 113357|HR  8729|IRAS 22550+2030|JP11  3558|LSPM J2257+2046|2MASS J22572795+2046077|N30 5052|NAME Helvetios|NLTT 55385|NSV 14374|PLX 5568|PLX 5568.00|PPM 114985|ROT  3341|SAO  90896|SKY# 43603|SPOCS  990|TD1 29480|TIC 139298196|TYC 1717-2193-1|UBV M  26734|UBV   19678|USNO-B1.0 1107-00589893|uvby98 100217014|WDS J22575+2046A|WEB 20165|YPAC 218|YZ   0  1227|YZ  20  9382|Gaia DR3 2835207319109249920|Gaia DR2 2835207319109249920",
                    "CCDM J19418+5031AB|*  16 Cyg|** STFA   46|ADS 12815 AB|IDS 19392+5017 AB|IRAS 19404+5024|IRAS F19404+5024|WDS J19418+5032AB",
                ],
                [
                    "13 12 19.7410",
                    "16 23 38.2218",
                    "22 57 27.9804",
                    "19 41 49.09",
                ],
                [
                    "+17 31 01.630",
                    "-26 31 53.769",
                    "+20 46 07.797",
                    "+50 31 31.6",
                ],
            ],
            names=("TYPED_ID", "MAIN_ID", "IDS", "RA", "DEC"),
        )

        mock_query_objects.return_value = mock_response

        # Call the function
        with LogCapture() as log:
            instance.simbad_list_host_search("host")
            assert (
                "List of unique star names 5 of which successful SIMBAD queries 4"
                in log.actual()[0][-1]
            )

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
                "IDS": [
                    "HD 114762|** PAT   47A|AG+17 1351|AGKR 11813|AP J13121982+1731016|ASCC  868467|BD+18  2700|Ci 20  766|Ci 18 1695|FK5 5165|GC 17881|GCRV  7854|GEN# +1.00114762|G  63-9|HD 114762A|HIC  64426|HIP  64426|LFT  980|LHS  2693|LSPM J1312+1731|LTT 13819|2MASS J13121982+1731016|N30 3028|NLTT 33242|PM 13099+1747|PPM 129656|SAO 100458|SBC9 2406|SKY# 24417|SPOCS  555|TD1 16696|TIC 138172859|TYC 1454-315-1|UBV   11938|UBV M  19151|USNO-B1.0 1075-00258599|uvby98 100114762|WDS J13123+1731A|WEB 11388|WISEA J131219.31+173101.6|YZC 18  4855|YZ  17  4855|Gaia DR2 3937211745904553600|Gaia DR3 3937211745905473024|Gaia DR1 3937211741607576576",
                    "PSR B1620-26|PSR J1623-2631|[BPH2004] CX 12|EQ J1623-2631",
                    "LTT 16750|*  51 Peg|** RBR   21A|AG+20 2595|AKARI-IRC-V1 J2257280+204608|ASCC  826013|BD+19  5036|CSV 102222|GC 32003|GCRV 14411|GEN# +1.00217014|GJ   882|HD 217014|HIC 113357|HIP 113357|HR  8729|IRAS 22550+2030|JP11  3558|LSPM J2257+2046|2MASS J22572795+2046077|N30 5052|NAME Helvetios|NLTT 55385|NSV 14374|PLX 5568|PLX 5568.00|PPM 114985|ROT  3341|SAO  90896|SKY# 43603|SPOCS  990|TD1 29480|TIC 139298196|TYC 1717-2193-1|UBV M  26734|UBV   19678|USNO-B1.0 1107-00589893|uvby98 100217014|WDS J22575+2046A|WEB 20165|YPAC 218|YZ   0  1227|YZ  20  9382|Gaia DR3 2835207319109249920|Gaia DR2 2835207319109249920",
                    "CCDM J19418+5031AB|*  16 Cyg|** STFA   46|ADS 12815 AB|IDS 19392+5017 AB|IRAS 19404+5024|IRAS F19404+5024|WDS J19418+5032AB",
                    "",
                ],
                "RA": [
                    "13 12 19.7410",
                    "16 23 38.2218",
                    "22 57 27.9804",
                    "19 41 49.09",
                    "",
                ],
                "DEC": [
                    "+17 31 01.630",
                    "-26 31 53.769",
                    "+20 46 07.797",
                    "+50 31 31.6",
                    "",
                ],
            }
        )

        # Assert the DataFrame content is as expected
        assert_frame_equal(instance.data, expected_data)


def test__simbad_list_alias_search(tmp_path, instance):
    data = pd.DataFrame(
        {
            "host": ["51 Peg"],
            "alias": [
                "GJ 882,BD+19 5036,HIP 113357,HR 8729,HD 217014,51 Peg,Gaia DR2 2835207319109249920,Helvetios,SAO 90896,TYC 1717-2193-1"
            ],
            "main_id": [""],
            "IDS": [""],
            "RA": [""],
            "DEC": [""],
        }
    )

    instance.data = data  # Replace YourClassName with the actual class name

    # Mock the Simbad.query_objects method with sample result_table
    with patch("astroquery.simbad.Simbad.query_objects") as mock_query_objects:
        mock_response = MagicMock()
        mock_response = Table(
            [
                ["TYC 1717-2193-1"],
                ["*  51 Peg"],
                [
                    "LTT 16750|*  51 Peg|** RBR   21A|AG+20 2595|AKARI-IRC-V1 J2257280+204608|ASCC  826013|BD+19  5036|CSV 102222|GC 32003|GCRV 14411|GEN# +1.00217014|GJ   882|HD 217014|HIC 113357|HIP 113357|HR  8729|IRAS 22550+2030|JP11  3558|LSPM J2257+2046|2MASS J22572795+2046077|N30 5052|NAME Helvetios|NLTT 55385|NSV 14374|PLX 5568|PLX 5568.00|PPM 114985|ROT  3341|SAO  90896|SKY# 43603|SPOCS  990|TD1 29480|TIC 139298196|TYC 1717-2193-1|UBV M  26734|UBV   19678|USNO-B1.0 1107-00589893|uvby98 100217014|WDS J22575+2046A|WEB 20165|YPAC 218|YZ   0  1227|YZ  20  9382|Gaia DR3 2835207319109249920|Gaia DR2 2835207319109249920"
                ],
                [
                    "22 57 27.9804",
                ],
                [
                    "+20 46 07.797",
                ],
            ],
            names=("TYPED_ID", "MAIN_ID", "IDS", "RA", "DEC"),
        )

        mock_query_objects.return_value = mock_response

        instance.simbad_list_alias_search("alias")

        # Expected output DataFrame
        expected_data = pd.DataFrame(
            {
                "host": ["51 Peg"],
                "alias": [
                    "GJ 882,BD+19 5036,HIP 113357,HR 8729,HD 217014,51 Peg,Gaia DR2 2835207319109249920,Helvetios,SAO 90896,TYC 1717-2193-1"
                ],
                "main_id": ["*  51 Peg"],
                "IDS": [
                    "LTT 16750|*  51 Peg|** RBR   21A|AG+20 2595|AKARI-IRC-V1 J2257280+204608|ASCC  826013|BD+19  5036|CSV 102222|GC 32003|GCRV 14411|GEN# +1.00217014|GJ   882|HD 217014|HIC 113357|HIP 113357|HR  8729|IRAS 22550+2030|JP11  3558|LSPM J2257+2046|2MASS J22572795+2046077|N30 5052|NAME Helvetios|NLTT 55385|NSV 14374|PLX 5568|PLX 5568.00|PPM 114985|ROT  3341|SAO  90896|SKY# 43603|SPOCS  990|TD1 29480|TIC 139298196|TYC 1717-2193-1|UBV M  26734|UBV   19678|USNO-B1.0 1107-00589893|uvby98 100217014|WDS J22575+2046A|WEB 20165|YPAC 218|YZ   0  1227|YZ  20  9382|Gaia DR3 2835207319109249920|Gaia DR2 2835207319109249920"
                ],
                "RA": [
                    "22 57 27.9804",
                ],
                "DEC": [
                    "+20 46 07.797",
                ],
            }
        )
        # Assert the DataFrame content is as expected
        assert_frame_equal(instance.data, expected_data)


def test__get_host_info_from_simbad(instance):
    data = {
        "name": ["HD 114762 b", "PSR B1620-26 b", "51 Peg b", "16 Cyg B b", "47 UMa b"],
        "host": ["HD 114762", "PSR B1620-26", "51 Peg", "16 Cyg", "47 UMa"],
        "binary": ["S-type", "AB", "", "B", ""],
        "alias": [
            "",
            "B1620-26",
            "GJ 882,BD+19 5036,HIP 113357,HR 8729,HD 217014,51 Peg,Gaia DR2 2835207319109249920,Helvetios,SAO 90896,TYC 1717-2193-1",
            "16 Cyg,WDS J19418+5032,16 Cygni",
            "Chalawan,HD 95128,47 Ursae Majoris,HIP 53721,TYC 3009-2703-1,GJ 407,BD+41 2147,Gaia DR2 777254360337133312,SAO 43557,47 UMa,HR 4277,2MASS J10592802+4025485",
        ],
    }

    instance.data = pd.DataFrame(data)
    with LogCapture() as log:
        instance.get_host_info_from_simbad()
        assert "HOST+BINARY Simbad Check" in log.actual()[0][-1]
        assert "Rows still missing main_id after host search" in log.actual()[2][-1]
        assert "ALIAS+BINARY Simbad Check" in log.actual()[3][-1]
        assert "Rows still missing main_id after alias search" in log.actual()[4][-1]
        assert "PURE HOST Simbad Check" in log.actual()[5][-1]
        assert "Rows still missing main_id after host search" in log.actual()[7][-1]
        assert "PURE ALIAS Simbad Check" in log.actual()[8][-1]
        assert "Rows still missing main_id after alias search" in log.actual()[9][-1]

        assert list(instance.data["hostbinary"]) == [
            "HD 114762",
            "PSR B1620-26 AB",
            "51 Peg",
            "16 Cyg B",
            "47 UMa",
        ]

        assert list(instance.data["aliasbinary"]) == [
            "",
            "B1620-26 AB",
            "GJ 882,BD+19 5036,HIP 113357,HR 8729,HD 217014,51 Peg,Gaia DR2 2835207319109249920,Helvetios,SAO 90896,TYC 1717-2193-1",
            "16 Cyg B,WDS J19418+5032 B,16 Cygni B",
            "Chalawan,HD 95128,47 Ursae Majoris,HIP 53721,TYC 3009-2703-1,GJ 407,BD+41 2147,Gaia DR2 777254360337133312,SAO 43557,47 UMa,HR 4277,2MASS J10592802+4025485",
        ]

    assert list(instance.data[instance.data.name == "51 Peg b"]["list_id"]) == [
        "LTT 16750,*  51 Peg,** RBR   21A,AG+20 2595,AKARI-IRC-V1 J2257280+204608,ASCC  826013,BD+19  5036,CSV 102222,GC 32003,GCRV 14411,GEN# +1.00217014,GJ   882,HD 217014,HIC 113357,HIP 113357,HR  8729,IRAS 22550+2030,JP11  3558,LSPM J2257+2046,2MASS J22572795+2046077,N30 5052,NAME Helvetios,NLTT 55385,NSV 14374,PLX 5568,PLX 5568.00,PPM 114985,ROT  3341,SAO  90896,SKY# 43603,SPOCS  990,TD1 29480,TIC 139298196,TYC 1717-2193-1,UBV M  26734,UBV   19678,USNO-B1.0 1107-00589893,uvby98 100217014,WDS J22575+2046A,WEB 20165,YPAC 218,YZ   0  1227,YZ  20  9382,Gaia DR3 2835207319109249920,Gaia DR2 2835207319109249920"
    ]
    assert np.isclose(instance.data.at[2, "ra_simbad"], 344.3667)
    assert np.isclose(instance.data.at[2, "dec_simbad"], 20.7689)


def test__set_common_alias(instance):
    data = {
        "host": ["51 Peg", "TYC 1717-2193-1 b"],
        "alias": [
            "GJ 882,BD+19 5036,HIP 113357,HR 8729,HD 217014,51 Peg,Gaia DR2 2835207319109249920,Helvetios,SAO 90896,TYC 1717-2193-1",
            "51 Peg",
        ],
        "main_id": ["*  51 Peg", "*  51 Peg"],
        "list_id": [
            "LTT 16750,*  51 Peg,** RBR   21A,AG+20 2595,AKARI-IRC-V1 J2257280+204608,ASCC  826013,BD+19  5036,CSV 102222,GC 32003,GCRV 14411,GEN# +1.00217014,GJ   882,HD 217014,HIC 113357,HIP 113357,HR  8729,IRAS 22550+2030,JP11  3558,LSPM J2257+2046,2MASS J22572795+2046077,N30 5052,NAME Helvetios,NLTT 55385,NSV 14374,PLX 5568,PLX 5568.00,PPM 114985,ROT  3341,SAO  90896,SKY# 43603,SPOCS  990,TD1 29480,TIC 139298196,TYC 1717-2193-1,UBV M  26734,UBV   19678,USNO-B1.0 1107-00589893,uvby98 100217014,WDS J22575+2046A,WEB 20165,YPAC 218,YZ   0  1227,YZ  20  9382,Gaia DR3 2835207319109249920,Gaia DR2 2835207319109249920",
            "LTT 16750,*  51 Peg,** RBR   21A,AG+20 2595,AKARI-IRC-V1 J2257280+204608,ASCC  826013,BD+19  5036,CSV 102222,GC 32003,GCRV 14411,GEN# +1.00217014,GJ   882,HD 217014,HIC 113357,HIP 113357,HR  8729,IRAS 22550+2030,JP11  3558,LSPM J2257+2046,2MASS J22572795+2046077,N30 5052,NAME Helvetios,NLTT 55385,NSV 14374,PLX 5568,PLX 5568.00,PPM 114985,ROT  3341,SAO  90896,SKY# 43603,SPOCS  990,TD1 29480,TIC 139298196,TYC 1717-2193-1,UBV M  26734,UBV   19678,USNO-B1.0 1107-00589893,uvby98 100217014,WDS J22575+2046A,WEB 20165,YPAC 218,YZ   0  1227,YZ  20  9382,Gaia DR3 2835207319109249920,Gaia DR2 2835207319109249920",
        ],
    }

    instance.data = pd.DataFrame(data)
    with LogCapture() as log:
        instance.set_common_alias()

    assert sorted(set(instance.data.at[0, "final_alias"].split(","))) == sorted(
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
        instance.check_coordinates()
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
        instance.check_coordinates()
        assert "Found 1 mismatched RA" in log.actual()[0][-1]
        assert "Found 1 mismatched DEC" in log.actual()[1][-1]
        assert list(instance.data["coordinate_mismatch"]) == ["RADEC", "RADEC"]

    assert os.path.exists("Logs/check_coordinates.txt")

    with open("Logs/check_coordinates.txt") as f:
        lines = f.readlines()
        assert (
            lines[0]
            == "*** MISMATCH ON RA ***        name    host binary letter catalog        ra\n"
        )
        assert lines[1] == "0  51 Peg b  51 Peg             b      eu  344.3667\n"
        assert lines[2] == "1  51 Peg b  51 Peg             b     oec  111.5560\n"

        assert (
            lines[3]
            == "*** MISMATCH ON DEC ***        name    host binary letter catalog      dec\n"
        )
        assert lines[4] == "0  51 Peg b  51 Peg             b      eu  20.7689\n"
        assert lines[5] == "1  51 Peg b  51 Peg             b     oec -22.7700\n"

    os.chdir(original_dir)


def test__get_coordinates_from_simbad(instance):
    data = {
        "name": ["51 Peg b"],
        "host": ["51 Peg"],
        "main_id": [""],
        "list_id": [""],
        "ra": [344.3667],
        "dec": [20.7689],
    }
    expected_df = pd.DataFrame(
        {
            "name": ["51 Peg b"],
            "host": ["51 Peg"],
            "main_id": ["*  51 Peg"],
            "list_id": [
                "LTT 16750,*  51 Peg,** RBR   21A,AG+20 2595,AKARI-IRC-V1 J2257280+204608,ASCC  826013,BD+19  5036,CSV 102222,GC 32003,GCRV 14411,GEN# +1.00217014,GJ   882,HD 217014,HIC 113357,HIP 113357,HR  8729,IRAS 22550+2030,JP11  3558,LSPM J2257+2046,2MASS J22572795+2046077,N30 5052,NAME Helvetios,NLTT 55385,NSV 14374,PLX 5568,PLX 5568.00,PPM 114985,ROT  3341,SAO  90896,SKY# 43603,SPOCS  990,TD1 29480,TIC 139298196,TYC 1717-2193-1,UBV M  26734,UBV   19678,USNO-B1.0 1107-00589893,uvby98 100217014,WDS J22575+2046A,WEB 20165,YPAC 218,YZ   0  1227,YZ  20  9382,Gaia DR3 2835207319109249920,Gaia DR2 2835207319109249920"
            ],
            "ra": [344.3667],
            "dec": [20.7689],
            "angular_separation": [0.00012667086528514546],
            "ra_simbad": [344.36658535524],
            "dec_simbad": [20.768832511140005],
        }
    )
    instance.data = pd.DataFrame(data)
    with LogCapture() as log:
        instance.get_coordinates_from_simbad()
        assert (
            "After coordinate check at tolerance 0.0005 residuals: 0. Maximum angular separation: 0.00012667086528514546"
            in log.actual()[-1][-1]
        )
    assert_frame_equal(instance.data, expected_df)


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

    data = {
        "name": ["PSR B1620-26 AB b", "PSR B1620-26 AB b"],
        "catalog": ["eu", "oec"],
        "host": ["PSR B1620-26", "PSR B1620-26"],
        "binary": ["AB", "AB"],
        "hostbinary": ["PSR B1620-26", "PSR B1620-26 AB"],
        "letter": ["b", "b"],
        "main_id": ["PSR B1620-26", "PSR B1620-26 AB"],
    }

    instance.data = pd.DataFrame(data)
    with LogCapture() as log:
        instance.check_same_host_different_id()
        assert "Checked if host is found" in log.actual()[0][-1]

    assert os.path.exists("Logs/same_host_different_id.txt")

    with open("Logs/same_host_different_id.txt") as f:
        lines = f.readlines()
        assert (
            lines[0]
            == "PSR B1620-26 AB main_id: ['PSR B1620-26', 'PSR B1620-26 AB'] binary: ['AB', 'AB'] Catalog: ['eu', 'oec']\n"
        )
        assert (
            lines[1]
            == "PSR B1620-26 main_id: ['PSR B1620-26', 'PSR B1620-26 AB'] binary: ['AB', 'AB'] Catalog: ['eu', 'oec']\n"
        )
    os.chdir(original_dir)


def test__polish_main_id(tmp_path, instance):
    original_dir = os.getcwd()

    os.chdir(tmp_path)  # Create a temporary in-memory configuration object
    os.mkdir("Logs/")
    # Test data
    data = {
        "main_id": [
            "HD 131977b",
            "VB 10 b",
            "*  51 Peg",
        ]
    }
    instance.data = pd.DataFrame(data)

    # Save the test data to a temporary file for testing

    with LogCapture() as log:
        instance.polish_main_id()
        assert (
            "Removed planet letter from main_id. It happens 2 times."
            in log.actual()[0][-1]
        )

    # Assert the expected outputs
    assert instance.data["main_id"].tolist() == [
        "HD 131977",
        "VB 10",
        "*  51 Peg",
    ]

    # Check if the log file was created and has the expected content
    assert os.path.exists("Logs/main_id_correction.txt")
    with open("Logs/main_id_correction.txt") as f:
        lines = f.readlines()
        assert lines[0] == "MAINID corrected HD 131977b to HD 131977\n"
        assert lines[1] == "MAINID corrected VB 10 b to VB 10 \n"

    os.chdir(original_dir)


def test__check_binary_mismatch(tmp_path, instance):
    original_dir = os.getcwd()

    os.chdir(tmp_path)  # Create a temporary in-memory configuration object
    os.mkdir("Logs/")

    # host
    # mismatches
    data = {
        "name": [
            "GJ 229 A c",
            "GJ 229 A c",
            "GJ 229 A c",
            "91 Aqr A b",
            "91 Aqr b",
            "HD 202206 (AB) c",
            "HD 202206 c",
            "ROXs 42B b",
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
        ],
        "binary": ["A", "S-type", "S-type", "A", "", "A", "AB", "AB"],
        "catalog": [
            "eu",
            "nasa",
            "wrongcoord",
            "Catalog 1",
            "Catalog 3",
            "eu",
            "oec",
            "nasa",
        ],
        "ra": [92.644231, 92.643599, 111, 10.0, 20.0, 10.0, 10.0, 10.0],
        "dec": [-21.864642, -21.867723, 111, 40.0, 50.0, 40.0, 40.0, 40.0],
        "letter": ["c", "c", "c", "b", "b", "c", "c", "b"],
    }
    instance.data = pd.DataFrame(data)

    # Execute the function
    instance.check_binary_mismatch(keyword="host")

    # Assert the expected 'potential_binary_mismatch' values after correction
    assert instance.data.loc[0, "potential_binary_mismatch"] == 0
    assert instance.data.loc[1, "binary"] == "A"
    assert instance.data.loc[2, "potential_binary_mismatch"] == 1
    assert instance.data.loc[4, "binary"] == "A"
    assert instance.data.loc[4, "potential_binary_mismatch"] == 1
    assert instance.data.loc[5, "potential_binary_mismatch"] == 2
    assert instance.data.loc[6, "potential_binary_mismatch"] == 2

    assert os.path.exists("Logs/binary_mismatch.txt")
    with open("Logs/binary_mismatch.txt") as f:
        lines = f.readlines()
        assert lines[0] == "****host****\n"
        assert (
            lines[2]
            == "****host+letter THAT COULD BE UNIFORMED (only if S-type or null)****\n"
        )
        assert (
            lines[3]
            == "GJ 229 NAME:['GJ 229 A c', 'GJ 229 A c', 'GJ 229 A c'] HOST:['GJ 229', 'GJ 229', 'GJ 229'] LETTER:c BINARY:['A', 'S-type', 'S-type'] CATALOG:['eu', 'nasa', 'wrongcoord'] WARNING, Coordinate Mismatch (potential_binary_mismatch 1) RA: [92.644231, 92.643599, 111.0] DEC:[-21.864642, -21.867723, 111.0] \n"
        )
        assert "WARNING, Coordinate Mismatch (potential_binary_mismatch 1)" in lines[4]
        assert (
            "host+letter THAT ARE INCONSISTENTLY LABELED (Potential Mismatch 2)."
            in lines[6]
        )
        assert (
            lines[8]
            == "HD 202206 NAME:['HD 202206 (AB) c', 'HD 202206 c'] HOST: ['HD 202206', 'HD 202206'] LETTER:c BINARY:['A', 'AB'] CATALOG:['eu', 'oec']\n"
        )
        assert (
            lines[9]
            == "****host POTENTIAL BINARIES NOT TREATED HERE. They should be treated manually in replacements.ini ****\n"
        )
        assert (
            lines[10]
            == "MISSED POTENTIAL BINARY Key:ROXs 42B name: ROXs 42B b binary: AB.\n"
        )
    os.chdir(original_dir)


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
        assert "Catalog cleared from zeroes and infinities" in log.actual()[0][-1]

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


def test__fix_letter_by_period(tmp_path, instance):
    original_dir = os.getcwd()

    os.chdir(tmp_path)  # Create a temporary in-memory configuration object
    os.mkdir("Logs/")

    ## CASE 1: everything is in agreement
    data = {
        "p": [934.3, 934.3, 934.3],
        "a": [2.11, 2.11, 2.11],
        "main_id": ["*   6 Lyn", "*   6 Lyn", "*   6 Lyn"],
        "host": ["6 Lyn", "6 Lyn", "6 Lyn"],
        "binary": ["", "", ""],
        "letter": ["b", "b", "b"],
    }

    data = pd.DataFrame(data)
    expected_result = {
        "p": [934.3, 934.3, 934.3],
        "a": [2.11, 2.11, 2.11],
        "main_id": ["*   6 Lyn", "*   6 Lyn", "*   6 Lyn"],
        "host": ["6 Lyn", "6 Lyn", "6 Lyn"],
        "binary": ["", "", ""],
        "letter": ["b", "b", "b"],
        "working_period_group": [156, 156, 156],
        "working_sma_group": [156, 156, 156],
    }

    expected_result = pd.DataFrame(expected_result)
    instance.data = data

    instance.fix_letter_by_period()

    assert sorted(instance.data.columns) == sorted(expected_result.columns)

    for col in instance.data.columns:
        if pd.isna(instance.data[col]).all() and pd.isna(expected_result[col]).all():
            continue
        assert (instance.data[col] == expected_result[col]).all()

    ## CASE 2: one letter disagrees but the period is the same, so it can be fixed
    data = {
        "p": [934.3, 934.3, 934.3],
        "a": [2.11, 2.11, 2.11],
        "main_id": ["*   6 Lyn", "*   6 Lyn", "*   6 Lyn"],
        "host": ["6 Lyn", "6 Lyn", "6 Lyn"],
        "binary": ["", "", ""],
        "letter": ["b", "b", ".01"],
    }

    data = pd.DataFrame(data)
    expected_result = {
        "p": [934.3, 934.3, 934.3],
        "a": [2.11, 2.11, 2.11],
        "main_id": ["*   6 Lyn", "*   6 Lyn", "*   6 Lyn"],
        "host": ["6 Lyn", "6 Lyn", "6 Lyn"],
        "binary": ["", "", ""],
        "letter": ["b", "b", "b"],
        "working_period_group": [156, 156, 156],
        "working_sma_group": [156, 156, 156],
    }

    expected_result = pd.DataFrame(expected_result)
    instance.data = data

    instance.fix_letter_by_period()

    assert sorted(instance.data.columns) == sorted(expected_result.columns)

    for col in instance.data.columns:
        if pd.isna(instance.data[col]).all() and pd.isna(expected_result[col]).all():
            continue
        assert (instance.data[col].values == expected_result[col].values).all()

    assert os.path.exists("Logs/fixed_letters.txt")

    with open("Logs/fixed_letters.txt") as f:
        lines = f.readlines()
        assert lines == [
            "CONTROVERSIAL LETTER ENTRY *   6 Lyn  PERIOD [934.3, 934.3, 934.3] LETTER ['b', 'b', '.01']\n",
            "-> FIXABLE\n",
        ]
    os.remove("Logs/fixed_letters.txt")
    ## CASE 3: one letter is BD so force all to be BD
    data = {
        "p": [934.3, 934.3, 934.3],
        "a": [2.11, 2.11, 2.11],
        "main_id": ["*   6 Lyn", "*   6 Lyn", "*   6 Lyn"],
        "host": ["6 Lyn", "6 Lyn", "6 Lyn"],
        "binary": ["", "", ""],
        "letter": ["b", "b", "BD"],
    }

    data = pd.DataFrame(data)
    expected_result = {
        "p": [934.3, 934.3, 934.3],
        "a": [2.11, 2.11, 2.11],
        "main_id": ["*   6 Lyn", "*   6 Lyn", "*   6 Lyn"],
        "host": ["6 Lyn", "6 Lyn", "6 Lyn"],
        "binary": ["", "", ""],
        "letter": ["BD", "BD", "BD"],
        "working_period_group": [156, 156, 156],
        "working_sma_group": [156, 156, 156],
    }

    expected_result = pd.DataFrame(expected_result)
    instance.data = data

    instance.fix_letter_by_period()

    assert sorted(instance.data.columns) == sorted(expected_result.columns)

    for col in instance.data.columns:
        if pd.isna(instance.data[col]).all() and pd.isna(expected_result[col]).all():
            continue
        assert (instance.data[col].values == expected_result[col].values).all()

    assert os.path.exists("Logs/fixed_letters.txt")

    with open("Logs/fixed_letters.txt") as f:
        lines = f.readlines()
        assert lines == [
            "CONTROVERSIAL LETTER ENTRY *   6 Lyn  PERIOD [934.3, 934.3, 934.3] LETTER ['b', 'b', 'BD']\n",
            "-> FORCED BD\n",
        ]

    os.remove("Logs/fixed_letters.txt")

    os.chdir(original_dir)


def test__group_by_letter_check_period(tmp_path, instance):
    original_dir = os.getcwd()

    os.chdir(tmp_path)  # Create a temporary in-memory configuration object
    os.mkdir("Logs/")

    ### SUCCESSFUL MERGING CASES

    # CASE 1: P exists and it is in agreement. Regular Merge
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
        "i_url": ["nasa", ""],
        "e_url": ["eu", "2019AJ....157..149L"],
        "main_id": ["*   6 Lyn", "*   6 Lyn"],
        "host": ["6 Lyn", "6 Lyn"],
        "binary": ["", ""],
        "letter": ["b", "b"],
        "status": ["CONFIRMED", "CONFIRMED"],
        "catalog": ["eu", "nasa"],
        "Catalogstatus": ["eu: CONFIRMED", "nasa: CONFIRMED"],
        "potential_binary_mismatch": [0, 0],
        "hostbinary": ["6 Lyn", "6 Lyn"],
        "RA": ["06 30 47.1075", "06 30 47.1075"],
        "DEC": ["+58 09 45.479", "+58 09 45.479"],
        "list_id": [
            "LTT 11856,*   6 Lyn",
            "LTT 11856,*   6 Lyn",
        ],
        "ra_simbad": [97.69628124999998, 97.69628124999998],
        "dec_simbad": [58.16263305555555, 58.16263305555555],
        "coordinate_mismatch": ["", ""],
        "angular_separation": [0.0, 0.0],
        "final_alias": [
            "2MASS J06304711+5809453,HR 2331",
            "2MASS J06304711+5809453,HR 2331",
        ],
        "working_period_group": [156, 156],
        "working_sma_group": [156, 156],
    }

    data = pd.DataFrame(data)
    expected_result = {
        "main_id": ["*   6 Lyn"],
        "binary": [""],
        "letter": ["b"],
        "host": ["6 Lyn"],
        "angular_separation": ["0.0"],
        "ra_official": [97.69628124999998],
        "dec_official": [58.16263305555555],
        "eu_name": ["6 Lyn b"],
        "nasa_name": ["6 Lyn b"],
        "oec_name": [""],
        "i_url": ["nasa"],
        "i": [2.0],
        "i_min": [1.0],
        "i_max": [79.0],
        "IREL": [39.5],
        "status_string": ["eu: CONFIRMED,nasa: CONFIRMED"],
        "confirmed": [2],
        "status": ["CONFIRMED"],
        "discovery_year": [2008],
        "discovery_method": ["Radial Velocity"],
        "catalog": ["eu,nasa"],
        "final_alias": ["2MASS J06304711+5809453,HR 2331"],
        "potential_binary_mismatch": ["0"],
        "coordinate_mismatch": [""],
        "coordinate_mismatch_flag": [0],
        "angular_separation_flag": [0],
        "duplicate_flag": [0],
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
    }
    expected_result = pd.DataFrame(expected_result)
    instance.data = data
    instance.group_by_letter_check_period(verbose=True)

    assert sorted(instance.data.columns) == sorted(expected_result.columns)

    instance.data = instance.data.convert_dtypes()
    expected_result = expected_result.convert_dtypes()
    for col in instance.data.columns:
        for row in instance.data.index:
            if pd.isna(instance.data.at[row, col]) and pd.isna(
                expected_result.at[row, col]
            ):
                continue
            try:
                assert instance.data.at[row, col] == expected_result.at[row, col]
            except AssertionError:
                assert np.isclose(
                    instance.data.at[row, col], expected_result.at[row, col]
                )

    # CASE 2: No period, but sma. Merging can happen
    update_data = {
        "p": [np.nan, np.nan],
        "p_max": [np.nan, np.nan],
        "p_min": [np.nan, np.nan],
        "working_period_group": [-1, -1],
    }
    update_df = pd.DataFrame(update_data)
    update_df.index = data.index
    data.update(update_df)
    instance.data = data
    instance.group_by_letter_check_period(verbose=False)

    assert sorted(instance.data.columns) == sorted(expected_result.columns)

    instance.data = instance.data.convert_dtypes()
    expected_result = expected_result.convert_dtypes()
    for col in instance.data.columns:
        for row in instance.data.index:
            if pd.isna(instance.data.at[row, col]) and pd.isna(
                expected_result.at[row, col]
            ):
                continue
            try:
                assert instance.data.at[row, col] == expected_result.at[row, col]
            except AssertionError:
                assert np.isclose(
                    instance.data.at[row, col], expected_result.at[row, col]
                )

    ## FALLBACK MERGING: merges into one, but no info on p or sma.
    # CASE 2: No period, but sma. Merging can happen
    update_data = {
        "p": [np.nan, np.nan],
        "p_max": [np.nan, np.nan],
        "p_min": [np.nan, np.nan],
        "a": [np.nan, np.nan],
        "a_max": [np.nan, np.nan],
        "a_min": [np.nan, np.nan],
        "working_period_group": [-1, -1],
        "working_sma_group": [-1, -1],
    }
    update_df = pd.DataFrame(update_data)
    update_df.index = data.index
    data.update(update_df)
    instance.data = data
    instance.group_by_letter_check_period(verbose=False)

    expected_result_updated = {
        "p": [np.nan],
        "p_max": [np.nan],
        "p_min": [np.nan],
        "PREL": [np.nan],
        "a": [np.nan],
        "a_max": [np.nan],
        "a_min": [np.nan],
        "AREL": [np.nan],
    }
    update_df = pd.DataFrame(expected_result_updated)
    update_df.index = expected_result.index
    expected_result.update(expected_result_updated)

    assert sorted(instance.data.columns) == sorted(expected_result.columns)

    instance.data = instance.data.convert_dtypes()
    expected_result = expected_result.convert_dtypes()
    for col in instance.data.columns:
        for row in instance.data.index:
            if pd.isna(instance.data.at[row, col]) and pd.isna(
                expected_result.at[row, col]
            ):
                continue
            try:
                assert instance.data.at[row, col] == expected_result.at[row, col]
            except AssertionError:
                assert np.isclose(
                    instance.data.at[row, col], expected_result.at[row, col]
                )

    assert os.path.exists("Logs/contrasting_periods.txt")

    with open("Logs/contrasting_periods.txt") as f:
        lines = f.readlines()
        assert lines == [
            "FALLBACK, MERGE *   6 Lyn b\n",
        ]

    os.remove("Logs/contrasting_periods.txt")

    ## UNSUCCESSFUL MERGING
    # CASE 3: Disagreement on period, keep both
    update_data = {
        "p": [8.0, 12.0],
        "p_max": [0.5, 0.6],
        "p_min": [0.5, 0.6],
        "working_period_group": [156, 134],
    }
    update_df = pd.DataFrame(update_data)
    update_df.index = data.index
    data.update(update_df)

    instance.data = data
    instance.group_by_letter_check_period(verbose=False)
    expected_result = {
        "eu_name": ["6 Lyn b", ""],
        "nasa_name": ["", "6 Lyn b"],
        "oec_name": ["", ""],
        "confirmed": [1, 1],
        "discovery_method": ["Radial Velocity", "Radial Velocity"],
        "p": [8.0, 12.0],
        "p_max": [0.5, 0.6],
        "p_min": [0.5, 0.6],
        "PERREL": [0.0625, 0.05],
        "a": [2.0, 2.0],
        "a_max": [0.1, 0.1],
        "a_min": [0.1, 0.1],
        "AREL": [0.05, 0.05],
        "e": [0.073, 0.073],
        "e_max": [0.036, 0.036],
        "e_min": [0.036, 0.036],
        "EREL": [0.4931506849315068, 0.4931506849315068],
        "i": [2.0, np.nan],
        "i_max": [79.0, np.nan],
        "i_min": [1.0, np.nan],
        "IREL": [39.5, np.nan],
        "mass": [np.nan, np.nan],
        "mass_max": [np.nan, np.nan],
        "mass_min": [np.nan, np.nan],
        "MASSREL": [np.nan, np.nan],
        "msini": [2.01, 2.01],
        "msini_max": [0.077, 0.077],
        "msini_min": [0.077, 0.077],
        "MSINIREL": [0.03830845771144279, 0.03830845771144279],
        "r": [np.nan, np.nan],
        "r_max": [np.nan, np.nan],
        "r_min": [np.nan, np.nan],
        "RADREL": [np.nan, np.nan],
        "discovery_year": [2008, 2008],
        "a_url": ["eu", "2019AJ....157..149L"],
        "mass_url": ["", ""],
        "p_url": ["eu", "2019AJ....157..149L"],
        "msini_url": ["eu", "2019AJ....157..149L"],
        "r_url": ["", ""],
        "i_url": ["nasa", ""],
        "e_url": ["eu", "2019AJ....157..149L"],
        "main_id": ["*   6 Lyn", "*   6 Lyn"],
        "host": ["6 Lyn", "6 Lyn"],
        "binary": ["", ""],
        "letter": ["b", "b"],
        "status": ["CONFIRMED", "CONFIRMED"],
        "status_string": ["eu: CONFIRMED", "nasa: CONFIRMED"],
        "catalog": ["eu", "nasa"],
        "potential_binary_mismatch": ["0", "0"],
        "ra_official": [97.69628124999998, 97.69628124999998],
        "dec_official": [58.16263305555555, 58.16263305555555],
        "coordinate_mismatch": ["", ""],
        "coordinate_mismatch_flag": [0, 0],
        "angular_separation": ["0.0", "0.0"],
        "angular_separation_flag": [0, 0],
        "duplicate_flag": [0, 0],
        "final_alias": [
            "2MASS J06304711+5809453,HR 2331",
            "2MASS J06304711+5809453,HR 2331",
        ],
    }
    expected_result = pd.DataFrame(expected_result)

    assert sorted(instance.data.columns) == sorted(expected_result.columns)

    instance.data = instance.data.convert_dtypes()
    expected_result = expected_result.convert_dtypes()

    for col in instance.data.columns:
        for row in instance.data.index:
            if pd.isna(instance.data.at[row, col]) and pd.isna(
                expected_result.at[row, col]
            ):
                continue
            try:
                assert instance.data.at[row, col] == expected_result.at[row, col]
            except AssertionError:
                assert np.isclose(
                    instance.data.at[row, col], expected_result.at[row, col]
                )

    assert os.path.exists("Logs/contrasting_periods.txt")

    with open("Logs/contrasting_periods.txt") as f:
        lines = f.readlines()
        assert lines == [
            "DISAGREEMENT *   6 Lyn b PERIOD [8.0, 12.0]\n",
        ]

    os.remove("Logs/contrasting_periods.txt")

    # CASE 4: Disagreement on SMA, keep both
    update_data = {
        "p": [np.nan, np.nan],
        "p_max": [np.nan, np.nan],
        "p_min": [np.nan, np.nan],
        "working_period_group": [-1, -1],
        "a": [2.0, 40],
        "a_max": [0.1, 0.1],
        "a_min": [0.1, 0.1],
        "working_sma_group": [156, 134],
    }
    update_df = pd.DataFrame(update_data)
    update_df.index = data.index
    data.update(update_df)
    instance.data = data

    expected_result_updated = {
        "p": [np.nan, np.nan],
        "p_max": [np.nan, np.nan],
        "p_min": [np.nan, np.nan],
        "PREL": [np.nan, np.nan],
        "a": [2.0, 40],
        "a_max": [0.1, 0.1],
        "a_min": [0.1, 0.1],
        "AREL": [0.05, 0.0025],
    }
    update_df = pd.DataFrame(expected_result_updated)
    update_df.index = expected_result.index
    expected_result.update(expected_result_updated)

    instance.group_by_letter_check_period(verbose=False)

    assert sorted(instance.data.columns) == sorted(expected_result.columns)

    instance.data = instance.data.convert_dtypes()
    expected_result = expected_result.convert_dtypes()

    for col in instance.data.columns:
        for row in instance.data.index:
            if pd.isna(instance.data.at[row, col]) and pd.isna(
                expected_result.at[row, col]
            ):
                continue
            try:
                assert instance.data.at[row, col] == expected_result.at[row, col]
            except AssertionError:
                assert np.isclose(
                    instance.data.at[row, col], expected_result.at[row, col]
                )

    assert os.path.exists("Logs/contrasting_periods.txt")

    with open("Logs/contrasting_periods.txt") as f:
        lines = f.readlines()
        assert lines == [
            "DISAGREEMENT *   6 Lyn b SMA [2.0, 40.0]\n",
        ]

    os.chdir(original_dir)


def test__potential_duplicates_after_merging(tmp_path, instance):
    original_dir = os.getcwd()

    os.chdir(tmp_path)  # Create a temporary in-memory configuration object
    os.mkdir("Logs/")

    data = {
        "exo_mercat_name": ["*  14 Her b", "*  14 Her c", "*  14 Her c"],
        "nasa_name": ["14 Her b", "", "HD 145675 c"],
        "eu_name": ["14 Her b", "14 Her c", ""],
        "oec_name": ["14 Her b", "", ""],
        "host": ["14 Her", "14 Her", "14 Her"],
        "letter": ["b", "c", "c"],
        "main_id": ["*  14 Her", "*  14 Her", "*  14 Her"],
        "binary": ["", "", ""],
        "coordinate_mismatch_flag": [0, 0, 0],
        "duplicate_flag": [0, 0, 0],
    }
    expected_result = {
        "exo_mercat_name": ["*  14 Her b", "*  14 Her c", "*  14 Her c"],
        "nasa_name": ["14 Her b", "", "HD 145675 c"],
        "eu_name": ["14 Her b", "14 Her c", ""],
        "oec_name": ["14 Her b", "", ""],
        "host": ["14 Her", "14 Her", "14 Her"],
        "letter": ["b", "c", "c"],
        "main_id": ["*  14 Her", "*  14 Her", "*  14 Her"],
        "binary": ["", "", ""],
        "coordinate_mismatch_flag": [0, 0, 0],
        "duplicate_flag": [0, 0, 0],
        "emc_duplicate_flag": [0, 1, 1],
    }

    expected_result = pd.DataFrame(expected_result)
    instance.data = pd.DataFrame(data)
    with LogCapture() as log:
        instance.potential_duplicates_after_merging()
        assert "Checked duplicates after merging" in log.actual()[0][-1]

    assert sorted(instance.data.columns) == sorted(expected_result.columns)

    instance.data = instance.data.convert_dtypes()
    expected_result = expected_result.convert_dtypes()
    for col in instance.data.columns:
        for row in instance.data.index:
            if pd.isna(instance.data.at[row, col]) and pd.isna(
                expected_result.at[row, col]
            ):
                continue
            assert instance.data.at[row, col] == expected_result.at[row, col]

    assert os.path.exists("Logs/potential_duplicates_after_merging.txt")

    with open("Logs/potential_duplicates_after_merging.txt") as f:
        lines = f.readlines()
        assert lines == [
            "MAINID *  14 Her  c\n",
        ]

    os.chdir(original_dir)


def test__set_common_host(tmp_path, instance):
    original_dir = os.getcwd()

    os.chdir(tmp_path)  # Create a temporary in-memory configuration object
    os.mkdir("Logs/")
    data = {
        "host": ["51 Peg", "TYC 1717-2193-1"],
        "catalog": ["eu", "oec"],
        "status": ["CONFIRMED", "CONFIRMED"],
        "letter": ["b", "b"],
        "main_id": ["*  51 Peg", "*  51 Peg"],
        "list_id": [
            "LTT 16750,*  51 Peg,** RBR   21A,AG+20 2595,AKARI-IRC-V1 J2257280+204608,ASCC  826013,BD+19  5036,CSV 102222,GC 32003,GCRV 14411,GEN# +1.00217014,GJ   882,HD 217014,HIC 113357,HIP 113357,HR  8729,IRAS 22550+2030,JP11  3558,LSPM J2257+2046,2MASS J22572795+2046077,N30 5052,NAME Helvetios,NLTT 55385,NSV 14374,PLX 5568,PLX 5568.00,PPM 114985,ROT  3341,SAO  90896,SKY# 43603,SPOCS  990,TD1 29480,TIC 139298196,TYC 1717-2193-1,UBV M  26734,UBV   19678,USNO-B1.0 1107-00589893,uvby98 100217014,WDS J22575+2046A,WEB 20165,YPAC 218,YZ   0  1227,YZ  20  9382,Gaia DR3 2835207319109249920,Gaia DR2 2835207319109249920",
            "LTT 16750,*  51 Peg,** RBR   21A,AG+20 2595,AKARI-IRC-V1 J2257280+204608,ASCC  826013,BD+19  5036,CSV 102222,GC 32003,GCRV 14411,GEN# +1.00217014,GJ   882,HD 217014,HIC 113357,HIP 113357,HR  8729,IRAS 22550+2030,JP11  3558,LSPM J2257+2046,2MASS J22572795+2046077,N30 5052,NAME Helvetios,NLTT 55385,NSV 14374,PLX 5568,PLX 5568.00,PPM 114985,ROT  3341,SAO  90896,SKY# 43603,SPOCS  990,TD1 29480,TIC 139298196,TYC 1717-2193-1,UBV M  26734,UBV   19678,USNO-B1.0 1107-00589893,uvby98 100217014,WDS J22575+2046A,WEB 20165,YPAC 218,YZ   0  1227,YZ  20  9382,Gaia DR3 2835207319109249920,Gaia DR2 2835207319109249920",
        ],
    }

    instance.data = pd.DataFrame(data)
    with LogCapture() as log:
        instance.set_common_host()
        assert (
            "Planets that had a different host name but same SIMBAD alias: 1"
            in log.actual()[0][-1]
        )

    assert list(instance.data.host) == ["51 Peg", "51 Peg"]

    assert os.path.exists("Logs/check_alias.txt")

    with open("Logs/check_alias.txt") as f:
        lines = f.readlines()
        assert (
            lines[0]
            == "HOST ['51 Peg', 'TYC 1717-2193-1']['eu', 'oec']['CONFIRMED', 'CONFIRMED']['b', 'b'] ID ['*  51 Peg', '*  51 Peg']"
        )

    os.chdir(original_dir)


def test__select_best_mass(instance):
    data = {
        "name": ["Msini case", "Mass case"],
        "mass": [10, 10],
        "mass_min": [0.1, 1],
        "mass_max": [0.1, 1],
        "mass_url": ["url1", "url2"],
        "MASSREL": [0.01, 0.1],
        "msini": [5, np.nan],
        "msini_min": [0.1, np.nan],
        "msini_max": [0.1, np.nan],
        "MSINIREL": [0.005, np.nan],
        "msini_url": ["url1", "url2"],
    }
    instance.data = pd.DataFrame(data)
    with LogCapture() as log:
        instance.select_best_mass()
        assert "Bestmass calculated" in log.actual()[0][-1]

    assert instance.data.at[0, "bestmass"] == 5
    assert instance.data.at[0, "bestmass_min"] == 0.1
    assert instance.data.at[0, "bestmass_max"] == 0.1
    assert instance.data.at[0, "bestmass_provenance"] == "Msini"
    assert instance.data.at[1, "bestmass"] == 10
    assert instance.data.at[1, "bestmass_min"] == 1
    assert instance.data.at[1, "bestmass_max"] == 1
    assert instance.data.at[1, "bestmass_provenance"] == "Mass"


def test__set_exo_mercat_name(instance):
    data = {
        "main_id": ["*  51 Peg", "*  16 Cyg B", "HD 106515A"],
        "binary": ["", "B", "A"],
        "letter": ["b", "b", "b"],
    }
    instance.data = pd.DataFrame(data)
    with LogCapture() as log:
        instance.set_exo_mercat_name()
        assert "Exo-MerCat name assigned." in log.actual()[0][-1]
        assert list(instance.data["exo_mercat_name"]) == [
            "*  16 Cyg B b",
            "*  51 Peg b",
            "HD 106515 A b",
        ]


def test__keep_columns(instance):
    # Test the keep_columns function

    # Create a sample DataFrame with some additional columns
    data = {
        "main_id": ["*   6 Lyn"],
        "binary": [""],
        "letter": ["b"],
        "angular_separation": ["[0.0]"],
        "ra_official": [97.69628320833333],
        "dec_official": [58.16263358333333],
        "oec_name": ["6 Lyn b"],
        "name": ["6 Lyn b"],
        "host": ["6 Lyn"],
        "i_url": ["oec"],
        "i": [2.0],
        "i_min": [1.0],
        "i_max": [79.0],
        "IREL": [39.5],
        "status_string": ["oec: CONFIRMED"],
        "confirmed": [1],
        "status": ["CONFIRMED"],
        "discovery_year": [2008.0],
        "discovery_method": ["Radial Velocity"],
        "catalog": ["['oec']"],
        "final_alias": [
            "GEN# +1.00045410,UCAC4 741-044033,*   6 Lyn,USNO-B1.0 1481-00215981,CSI+58   932  1,HIP  31039,SKY# 11231,PPM  30486,GCRV  4140,CCDM J06309+5810A,uvby98 100045410,Gaia DR2 1004358968092652544,SAO  25771,YZ  58  4530,UBV    6436,[B10]  1627,LSPM J0630+5809,LTT 11856,WDS J06308+5810A,NLTT 16571,HR 2331,PLX 1499.00,BD+58   932,BD+58 932,TIC 444865362,HIC  31039,AP J06304711+5809453,IDS 06220+5814 A,TYC 3777-2071-1,SPOCS 2671,Gaia DR3 1004358968092652544,2MASS J06304711+5809453,DO 30475,GC  8416,PLX 1499,WEB  6178,AG+58  545,N30 1407,HIP 31039,6 Lyncis,IRAS 06264+5811,HD 45410,SAO 25771,UBV M  12104,HR  2331,HD  45410,ASCC  182129"
        ],
        "coordinate_mismatch": [""],
        "coordinate_mismatch_flag": [0],
        "angular_separation_flag": [0],
        "duplicate_flag": [0],
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
        "eu_name": [np.nan],
        "nasa_name": [np.nan],
        "bestmass": [2.21],
        "bestmass_min": [0.16],
        "bestmass_max": [0.11],
        "bestmass_url": ["oec"],
        "bestmass_provenance": ["Msini"],
        "exo_mercat_name": ["*   6 Lyn  b"],
        "emc_duplicate_flag": [0],
    }

    df = pd.DataFrame(data)
    instance.data = df
    with LogCapture() as log:
        instance.keep_columns()
        assert "Selected columns to keep" in log.actual()[0][-1]

    # Check if the DataFrame contains only the columns specified in the keep list
    expected_columns = [
        "exo_mercat_name",
        "nasa_name",
        "eu_name",
        "oec_name",
        "host",
        "letter",
        "main_id",
        "binary",
        "ra_official",
        "dec_official",
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
        "status_string",
        "confirmed",
        "discovery_year",
        "final_alias",
        "catalog",
        "angular_separation",
        "angular_separation_flag",
        "coordinate_mismatch",
        "coordinate_mismatch_flag",
        "duplicate_flag",
        "emc_duplicate_flag",
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


def test__round_to_decimal():
    # Test cases with various input values
    test_cases = [
        (12345, 12300.0),  # Test for an order of magnitude <= 100
        (678.9, 700.0),  # Test for an order of magnitude between 100 and 1000
        (0.012345, 0.01),  # Test for an order of magnitude > 1000
        (1000, 1000),  # Test with number having an order of magnitude equal to 1
    ]

    for number, expected_output in test_cases:
        result = round_to_decimal(number)
        assert result == expected_output


def test__round_array_to_significant_digits():
    # Test cases with various input arrays
    test_cases = [
        (
            [1.2345, 6.789, 0.012345, 0, 1000],
            [1.0, 7.0, 0.01, -1, 1000.0],
        ),  # Normal test case
        (
            [100, 200, 300, 0, 400],
            [100.0, 200.0, 300.0, -1, 400.0],
        ),  # Test with only integers
        (
            [0.00124, 0.00235, 0.00333],
            [0.001, 0.002, 0.003],
        ),  # Test with values having an order of magnitude < 1
        (
            [
                103087,
                5098779,
            ],
            [103000, 5100000],
        ),  # Test with values requiring rounding to 2 significant digits
        ([0], [-1]),  # Test with an array containing only 0
        ([], []),  # Test with an empty array
    ]

    for numbers, expected_output in test_cases:
        result = round_array_to_significant_digits(numbers)
        assert result == expected_output


def test__round_parameter_bin():
    data = pd.Series([326.03, 52160.0, 3765.0, 0.59862739, 0.43, np.nan, 818000.0])
    rounded_series = round_parameter_bin(data)
    assert sorted(rounded_series.values) == [-1, 2, 8, 137, 187, 240, 297]


#
def test__merge_into_single_entry(tmp_path):
    original_dir = os.getcwd()

    os.chdir(tmp_path)  # Create a temporary in-memory configuration object
    os.mkdir("Logs/")

    data = {
        "name": ["6 Lyn b", "6 Lyn b", "6 Lyn b"],
        "catalog_name": ["6 Lyn b", "6 Lyn b", "6 Lyn b"],
        "discovery_method": ["Radial Velocity", "Radial Velocity", "Radial Velocity"],
        "ra": [97.6958333, 97.6960311, 97.69628320833333],
        "dec": [58.1627778, 58.1611753, 58.16263358333333],
        "p": [934.3, 934.3, 874.774],
        "p_max": [8.6, 8.6, 16.27],
        "p_min": [8.6, 8.6, 8.47],
        "a": [2.11, 2.11, 2.18],
        "a_max": [0.11, 0.11, 0.05],
        "a_min": [0.11, 0.11, 0.06],
        "e": [0.073, 0.073, 0.059],
        "e_max": [0.036, 0.036, 0.066],
        "e_min": [0.036, 0.036, 0.059],
        "i": [np.nan, np.nan, 2.0],
        "i_max": [np.nan, np.nan, 79.0],
        "i_min": [np.nan, np.nan, 1.0],
        "mass": [np.nan, np.nan, np.nan],
        "mass_max": [np.nan, np.nan, np.nan],
        "mass_min": [np.nan, np.nan, np.nan],
        "msini": [2.01, 2.01, 2.21],
        "msini_max": [0.077, 0.077, 0.11],
        "msini_min": [0.077, 0.077, 0.16],
        "r": [np.nan, np.nan, np.nan],
        "r_max": [np.nan, np.nan, np.nan],
        "r_min": [np.nan, np.nan, np.nan],
        "discovery_year": [2008, 2008, 2008],
        "alias": [
            "2MASS J06304711+5809453,6 Lyncis,BD+58 932,Gaia DR2 1004358968092652544,HD 45410,HIP 31039,HR 2331,SAO 25771,TIC 444865362,TYC 3777-2071-1",
            "2MASS J06304711+5809453,6 Lyncis,BD+58 932,Gaia DR2 1004358968092652544,HD 45410,HIP 31039,HR 2331,SAO 25771,TIC 444865362,TYC 3777-2071-1",
            "2MASS J06304711+5809453,6 Lyncis,BD+58 932,Gaia DR2 1004358968092652544,HD 45410,HIP 31039,HR 2331,SAO 25771,TIC 444865362,TYC 3777-2071-1",
        ],
        "a_url": ["eu", "2019AJ....157..149L", "oec"],
        "mass_url": ["", "", ""],
        "p_url": ["eu", "2019AJ....157..149L", "oec"],
        "msini_url": ["eu", "2019AJ....157..149L", "oec"],
        "r_url": ["", "", ""],
        "i_url": ["", "", "oec"],
        "e_url": ["eu", "2019AJ....157..149L", "oec"],
        "main_id": ["*   6 Lyn", "*   6 Lyn", "*   6 Lyn"],
        "host": ["6 Lyn", "6 Lyn", "6 Lyn"],
        "binary": ["", "", ""],
        "letter": ["b", "b", "b"],
        "status": ["CONFIRMED", "CONFIRMED", "CONFIRMED"],
        "catalog": ["eu", "nasa", "oec"],
        "Catalogstatus": ["eu: CONFIRMED", "nasa: CONFIRMED", "oec: CONFIRMED"],
        "potential_binary_mismatch": [0, 0, 0],
        "hostbinary": ["6 Lyn", "6 Lyn", "6 Lyn"],
        "RA": ["06 30 47.1075", "06 30 47.1075", "06 30 47.1075"],
        "DEC": ["+58 09 45.479", "+58 09 45.479", "+58 09 45.479"],
        "list_id": [
            "LTT 11856,*   6 Lyn",
            "LTT 11856,*   6 Lyn",
            "LTT 11856,*   6 Lyn",
        ],
        "ra_simbad": [97.69628124999998, 97.69628124999998, 97.69628124999998],
        "dec_simbad": [58.16263305555555, 58.16263305555555, 58.16263305555555],
        "coordinate_mismatch": ["", "", ""],
        "angular_separation": [0.0, 0.0, 0.0],
        "final_alias": [
            "2MASS J06304711+5809453,HR 2331",
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
        "angular_separation": ["0.0"],
        "ra_official": [97.69628124999998],
        "dec_official": [58.16263305555555],
        "eu_name": ["6 Lyn b"],
        "nasa_name": ["6 Lyn b"],
        "oec_name": ["6 Lyn b"],
        "i_url": ["oec"],
        "i": [2.0],
        "i_min": [1.0],
        "i_max": [79.0],
        "IREL": [39.5],
        "status_string": ["eu: CONFIRMED,nasa: CONFIRMED,oec: CONFIRMED"],
        "confirmed": [3],
        "status": ["CONFIRMED"],
        "discovery_year": [2008],
        "discovery_method": ["Radial Velocity"],
        "catalog": ["eu,nasa,oec"],
        "final_alias": ["2MASS J06304711+5809453,HR 2331"],
        "potential_binary_mismatch": ["0"],
        "coordinate_mismatch": [""],
        "coordinate_mismatch_flag": [0],
        "angular_separation_flag": [0],
        "duplicate_flag": [0],
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
        "a_url": ["oec"],
        "a": [2.18],
        "a_min": [0.06],
        "a_max": [0.05],
        "AREL": [0.027522935779816512],
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
    }
    expected_result = pd.DataFrame(expected_result)
    result = merge_into_single_entry(data, "*   6 Lyn", "", "b")

    assert sorted(result.columns) == sorted(expected_result.columns)

    for col in result.columns:
        if pd.isna(result[col]).all() and pd.isna(expected_result[col]).all():
            continue
        assert (result[col] == expected_result[col]).all()

    # test controversial status, BD overwriting, unknown discovery year, unknown discovery method,angular separation flag, coordinate mismatch
    # Additional data to update (condensed form)
    update_data = {
        "discovery_method": ["Default", "", ""],
        "status": ["CONFIRMED", "CONFIRMED", "FALSE POSITIVE"],
        "Catalogstatus": ["eu: CONFIRMED", "nasa: CONFIRMED", "oec: FALSE POSITIVE"],
        "discovery_year": [2008, 2010, 2008],
        "coordinate_mismatch": ["RA", "DEC", ""],
        "angular_separation": [0.0, 0.012, 0.0],
    }

    # Create the update DataFrame
    update_df = pd.DataFrame(update_data)

    # Set the index to match data_df for proper alignment
    update_df.index = data.index

    # Update data_df with values from update_df
    data.update(update_df)

    updated_expected = {
        "main_id": ["*   6 Lyn"],
        "status_string": ["eu: CONFIRMED,nasa: CONFIRMED,oec: FALSE POSITIVE"],
        "confirmed": [2],
        "status": ["CONTROVERSIAL"],
        "discovery_method": [""],
        "discovery_year": [2008],
        "potential_binary_mismatch": ["0"],
        "coordinate_mismatch": ["RA,DEC"],
        "coordinate_mismatch_flag": [2],
        "angular_separation": ["0.0,0.012"],
        "angular_separation_flag": [1],
        "duplicate_flag": [0],
    }

    # Create the update DataFrame
    update_expected_df = pd.DataFrame(updated_expected)

    # Set the index to match data_df for proper alignment
    update_expected_df.index = expected_result.index

    # Update data_df with values from update_df
    expected_result.update(update_expected_df)

    result = merge_into_single_entry(data, "*   6 Lyn", "", "b")

    assert sorted(result.columns) == sorted(expected_result.columns)

    for col in result.columns:
        if pd.isna(result[col]).all() and pd.isna(expected_result[col]).all():
            continue
        assert (result[col] == expected_result[col]).all()

    # test coordinate mismatch part 2
    update_data = {
        "Catalogstatus": ["eu: CONFIRMED", "oec: CONFIRMED", "oec: FALSE POSITIVE"],
        "coordinate_mismatch": ["RA", "", ""],
        "catalog": ["eu", "oec", "oec"],
    }

    # Create the update DataFrame
    update_df = pd.DataFrame(update_data)

    # Set the index to match data_df for proper alignment
    update_df.index = data.index

    # Update data_df with values from update_df
    data.update(update_df)
    # nan discovery year
    data["discovery_year"] = [np.nan for disc in data["discovery_year"]]

    updated_expected = {
        "status_string": ["eu: CONFIRMED,oec: CONFIRMED,oec: FALSE POSITIVE"],
        "nasa_name": [""],
        "status": "CONTROVERSIAL",
        "coordinate_mismatch": ["RA"],
        "coordinate_mismatch_flag": [1],
        "duplicate_flag": [1],
        "catalog": ["eu,oec"],
        "discovery_year": [""],
    }

    # Create the update DataFrame
    update_expected_df = pd.DataFrame(updated_expected)

    # Set the index to match data_df for proper alignment
    update_expected_df.index = expected_result.index

    # Update data_df with values from update_df
    expected_result.update(update_expected_df)

    result = merge_into_single_entry(data, "*   6 Lyn", "", "b")

    assert sorted(result.columns) == sorted(expected_result.columns)

    for col in result.columns:
        if pd.isna(result[col]).all() and pd.isna(expected_result[col]).all():
            continue
        assert (result[col] == expected_result[col]).all()

    assert os.path.exists("Logs/duplicate_entries.txt")

    with open("Logs/duplicate_entries.txt") as f:
        lines = f.readlines()
        assert lines == [
            "DUPLICATE ENTRY *   6 Lyn ['b', 'b', 'b'] CATALOGS ['eu', 'oec', 'oec'] STATUS ['CONFIRMED', 'CONFIRMED', 'FALSE POSITIVE']\n",
        ]

    os.chdir(original_dir)
