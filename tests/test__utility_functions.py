import gzip
import os
import xml.etree.ElementTree as ElementTree

import numpy as np
import pandas as pd
from pandas._testing import assert_frame_equal

import pyvo
from astropy.table import Table
from exo_mercat.utility_functions import UtilityFunctions
import pytest


@pytest.fixture
def instance():
    return UtilityFunctions()


def test__init(instance):
    assert isinstance(instance, UtilityFunctions)


def test__service_files_initialization(instance, tmp_path):
    # Change current working directory to the temporary folder
    original_dir = os.getcwd()

    os.chdir(tmp_path)

    # Call the function
    instance.service_files_initialization()

    # Assert that the folders are created
    assert os.path.exists("Exo-MerCat")
    assert os.path.exists("InputSources")
    assert os.path.exists("UniformSources")
    assert os.path.exists("Logs")

    # Create some dummy files in the Logs folder
    with open("Logs/file1.txt", "w") as f:
        f.write("Dummy content")
    with open("Logs/file2.txt", "w") as f:
        f.write("Dummy content")

    # Call the function
    instance.service_files_initialization()

    # Assert that the Logs folder is empty
    logs_files = os.listdir("Logs/")
    assert len(logs_files) == 0

    os.chdir(original_dir)


def test__find_const(instance):
    expected_constants = {
        "alfa ": "alf ",
        "beta ": "bet ",
        "gamma ": "gam ",
        "delta ": "del ",
        "zeta ": "zet ",
        "teta ": "tet ",
        "iota ": "iot ",
        "kappa ": "kap ",
        "lambda ": "lam ",
        "omicron ": "omi ",
        "sigma ": "sig ",
        "upsilon ": "ups ",
        "omega ": "ome ",
        "alpha ": "alf ",
        "epsilon ": "eps ",
        "theta ": "tet ",
        "mu ": "miu ",
        "nu ": "niu ",
        "xi ": "ksi ",
        "chi ": "khi ",
        "Alfa ": "alf ",
        "Beta ": "bet ",
        "Gamma ": "gam ",
        "Delta ": "del ",
        "Eps ": "eps ",
        "Zeta ": "zet ",
        "Eta ": "eta ",
        "Teta ": "tet ",
        "Iota ": "iot ",
        "Kappa ": "kap ",
        "Lambda ": "lam ",
        "Miu ": "miu ",
        "Niu ": "niu ",
        "Ksi ": "ksi ",
        "Omicron ": "omi ",
        "Pi ": "pi ",
        "Rho ": "rho ",
        "Sigma ": "sig ",
        "Upsilon ": "ups ",
        "Phi ": "phi ",
        "Khi ": "khi ",
        "Psi ": "psi ",
        "Omega ": "ome ",
        "Alpha ": "alf ",
        "Bet ": "bet ",
        "Gam ": "gam ",
        "Del ": "del ",
        "Epsilon ": "eps ",
        "Zet ": "zet ",
        "Theta ": "tet ",
        "Iot ": "iot ",
        "Kap ": "kap ",
        "Lam ": "lam ",
        "Mu ": "miu ",
        "Nu ": "niu ",
        "Xi ": "ksi ",
        "Omi ": "omi ",
        "Sig ": "sig ",
        "Ups ": "ups ",
        "Chi ": "khi ",
        "Ome ": "ome ",
        "Andromedae": "And",
        "Antliae": "Ant",
        "Apodis": "Aps",
        "Aquarii": "Aqr",
        "Aquilae": "Aql",
        "Arae": "Ara",
        "Arietis": "Ari",
        "Aurigae": "Aur",
        "Bootis": "Boo",
        "Bo&ouml;": "Boo",
        "Caeli": "Cae",
        "Camelopardalis": "Cam",
        "Cancri": "Cnc",
        "Canum Venaticorum": "CVn",
        "Canis Majoris": "CMa",
        "Canis Minoris": "CMi",
        "Capricorni": "Cap",
        "Carinae": "Car",
        "Cassiopeiae": "Cas",
        "Centauri": "Cen",
        "Cephei": "Cep",
        "Cepi": "Cep",
        "Ceti": "Cet",
        "Chamaeleontis": "Cha",
        "Circini": "Cir",
        "Columbae": "Col",
        "Comae Berenices": "Com",
        "Coronae Australis": "CrA",
        "Coronae Borealis": "CrB",
        "Corvi": "Crv",
        "Crateris": "Crt",
        "Crucis": "Cru",
        "Cygni": "Cyg",
        "Delphini": "Del",
        "Doradus": "Dor",
        "Draconis": "Dra",
        "Equulei": "Equ",
        "Eridani": "Eri",
        "Fornacis": "For",
        "Geminorum": "Gem",
        "Gruis": "Gru",
        "Herculis": "Her",
        "Horologii": "Hor",
        "Hydrae": "Hya",
        "Hydri": "Hyi",
        "Indi": "Ind",
        "Lacertae": "Lac",
        "Leonis": "Leo",
        "Leonis Minoris": "LMi",
        "Leporis": "Lep",
        "Librae": "Lib",
        "Lupi": "Lup",
        "Lyncis": "Lyn",
        "Lyrae": "Lyr",
        "Mensae": "Men",
        "Microscopii": "Mic",
        "Monocerotis": "Mon",
        "Muscae": "Mus",
        "Normae": "Nor",
        "Octantis": "Oct",
        "Ophiuchi": "Oph",
        "Orionis": "Ori",
        "Pavonis": "Pav",
        "Pegasi": "Peg",
        "Persei": "Per",
        "Phoenicis": "Phe",
        "Pictoris": "Pic",
        "Piscium": "Psc",
        "Piscis Austrini": "PsA",
        "Puppis": "Pup",
        "Pyxidis": "Pyx",
        "Reticuli": "Ret",
        "Sagittae": "Sge",
        "Sagittarii": "Sgr",
        "Scorpii": "Sco",
        "Sculptoris": "Scl",
        "Scuti": "Sct",
        "Serpentis": "Ser",
        "Sextantis": "Sex",
        "Tauri": "Tau",
        "Telescopii": "Tel",
        "Trianguli": "Tri",
        "Trianguli Australis": "TrA",
        "Tucanae": "Tuc",
        "Ursae Majoris": "UMa",
        "Uma": "UMa",
        "Ursae Minoris": "UMi",
        "Umi": "UMi",
        "Velorum": "Vel",
        "Virginis": "Vir",
        "Volantis": "Vol",
        "Vulpeculae": "Vul",
        # "2M ": "2MASS ",
        "KOI ": "KOI-",
        "Kepler ": "Kepler-",
        "BD ": "BD",
        # "OGLE-": "OGLE ",
        # "MOA-": "MOA ",
        # "gam 1 ": "gam ",
        # "EPIC-": "EPIC ",
        # "Pr 0": "Pr ",
        # "TOI ": "TOI-",
        "kepler": "Kepler",
        # "Gliese": "GJ",
        "p ": "pi ",
    }

    actual_constants = instance.find_const()
    assert expected_constants == actual_constants


def test__read_config(instance, tmp_path):
    original_dir = os.getcwd()

    os.chdir(tmp_path)  # Create a temporary in-memory configuration object
    # Create a temporary directory to store the fake_config.ini
    with open("input_sources.ini", "w") as config_file:
        config_file.write("[nasa]\n")
        config_file.write(
            "url = https://exoplanetarchive.ipac.caltech.edu/TAP/sync?query=select+*+from+pscomppars&format=csv\n"
        )
        config_file.write("file = InputSources/nasa_init\n")

    # Call the function to read the config file
    config = instance.read_config()

    # Assert that the function returns a ConfigParser object
    assert isinstance(config, dict)

    # Assert that specific configuration parameters exist
    # Example: Assuming your input_sources.ini has a [General] section with a
    # 'timeout' parameter
    assert "nasa" in config.keys()
    assert "url" in config["nasa"].keys()
    assert "file" in config["nasa"].keys()
    assert (
            config["nasa"]["url"]
            == "https://exoplanetarchive.ipac.caltech.edu/TAP/sync?query=select+*+from+pscomppars&format=csv"
    )
    os.chdir(original_dir)


def test__read_config_replacements(instance, tmp_path):
    original_dir = os.getcwd()

    os.chdir(tmp_path)  # Create a temporary in-memory configuration object
    # Create a temporary directory to store the fake_config.ini
    with open("replacements.ini", "w") as config_file:
        config_file.write("[NAME]\n")
        config_file.write("alf Tau b = Aldebaran b\n")
        config_file.write("[HOST]\n")
        config_file.write("gam1 Leo = gam01 Leo\n")
        config_file.write("[ra]\n")
        config_file.write("K2-2016-BLG-0005L = 269.879166677\n")
        config_file.write("[DROP]\n")
        config_file.write("name = Trojan\n")
        config_file.write("[BINARY]\n")
        config_file.write("XO-2N c = A\n")

    replacements = instance.read_config_replacements("NAME")
    assert replacements["alf Tau b"] == "Aldebaran b"
    replacements = instance.read_config_replacements("HOST")
    assert replacements["gam1 Leo"] == "gam01 Leo"
    replacements = instance.read_config_replacements("ra")
    assert replacements["K2-2016-BLG-0005L"] == "269.879166677"
    replacements = instance.read_config_replacements("DROP")
    assert replacements["name"] == "Trojan"
    replacements = instance.read_config_replacements("BINARY")
    assert replacements["XO-2N c"] == "A"
    assert replacements["XO-2N c"] != "N"
    os.chdir(original_dir)


@pytest.mark.parametrize(
    ("string", "expected"),
    [
        ("2M 0219-39 b", "2MASS J0219-39 b"),
        ("2M1510A b", "2MASS J1510A b"),
        ("2MASS J0326-2102", "2MASS J0326-2102"),
        ("2M1510A a", "2MASS J1510A"),
        ("Gliese 49 b", "GJ 49 b"),
        ("Gl 378 b", "GJ 378 b"),
        ("K00473.01", "KOI-473.01"),
        ("TOI 1064.01", "TOI-1064.01"),
        ("VHS 1256-1257 b", "VHS J1256-1257 b"),
        ("KMT-2017-BLG-0165L", "KMT-2017-BLG-0165"),
        ("KMT-2019-BLG-1339/OGLE-2019/BLG-1019 b", "KMT-2019-BLG-1339"),
        ("OGLE--2018-BLG-0516 b", "OGLE 2018-BLG-0516 b"),
        ("OGLE-2005-071", "OGLE 2005-071"),
        ("MOA-2016-BLG-339L", "MOA 2016-BLG-339"),
        # random general one that should not change
        ("WASP-184 b", "WASP-184 b"),
    ],
)
def test__uniform_string(instance, string, expected):
    assert instance.uniform_string(string) == expected


def test__round_to_decimal(instance):
    # Test cases with various input values
    test_cases = [
        (12345, 12300.0),  # Test for an order of magnitude <= 100
        (678.9, 700.0),  # Test for an order of magnitude between 100 and 1000
        (0.012345, 0.01),  # Test for an order of magnitude > 1000
        (1000, 1000),  # Test with number having an order of magnitude equal to 1
    ]

    for number, expected_output in test_cases:
        result = instance.round_to_decimal(number)
        assert result == expected_output


# def test__round_array_to_significant_digits(instance):
#     # Test cases with various input arrays
#     test_cases = [
#         (
#             [1.2345, 6.789, 0.012345, 0, 1000],
#             [1.0, 7.0, 0.01, -1, 1000.0],
#         ),  # Normal test case
#         (
#             [100, 200, 300, 0, 400],
#             [100.0, 200.0, 300.0, -1, 400.0],
#         ),  # Test with only integers
#         (
#             [0.00124, 0.00235, 0.00333],
#             [0.001, 0.002, 0.003],
#         ),  # Test with values having an order of magnitude < 1
#         (
#             [
#                 103087,
#                 5098779,
#             ],
#             [103000, 5100000],
#         ),  # Test with values requiring rounding to 2 significant digits
#         ([0], [-1]),  # Test with an array containing only 0
#         ([], []),  # Test with an empty array
#     ]
#
#     for numbers, expected_output in test_cases:
#         result = instance.round_array_to_significant_digits(numbers=numbers)
#         assert result == expected_output

#
# def test__round_parameter_bin(instance):
#     data = pd.Series([326.03, 52160.0, 3765.0, 0.59862739, 0.43, np.nan, 818000.0])
#     rounded_series = instance.round_parameter_bin(data)
#     assert sorted(rounded_series.values) == [-1, 2, 8, 137, 187, 240, 297]
#
#
# #


def test__calculate_working_p_sma(instance):
    group = pd.DataFrame(
        {
            "name": ["same planet", "same planet", "same planet", "null planet"],
            "p": [9886, 1572, 1500, np.nan],
            "a": [2.5, 1.9, 2, np.nan],
        }
    )
    returned_group = UtilityFunctions.calculate_working_p_sma(group, tolerance=0.1)
    expected_group = pd.DataFrame(
        {
            "name": ["same planet", "same planet", "same planet", "null planet"],
            "p": [9886, 1572, 1500, np.nan],
            "a": [2.5, 1.9, 2.0, np.nan],
            "working_p": [9886.0, 1500.0, 1500.0, -1.0],
            "working_a": [2.5, 2.0, 2.0, -1.0],
        }
    )
    expected_group = expected_group.sort_values(by="p")
    assert_frame_equal(returned_group, expected_group)


def fill_xml_string():
    xml_string = ("<systems>\n"
                  "            <system>\n"
                  "            	<name>11 Com</name>\n"
                  "            	<rightascension>12 20 43.0255</rightascension>\n"
                  "            	<declination>+17 47 34.3392</declination>\n"
                  "            	<distance errorminus=\"1.7\" errorplus=\"1.7\">88.9</distance>\n"
                  "            	<star>\n"
                  "            		<name>11 Com</name>\n"
                  "            		<name>11 Comae Berenices</name>\n"
                  "            		<name>HD 107383</name>\n"
                  "            		<name>HIP 60202</name>\n"
                  "            		<name>TYC 1445-2560-1</name>\n"
                  "            		<name>SAO 100053</name>\n"
                  "            		<name>HR 4697</name>\n"
                  "            		<name>BD+18 2592</name>\n"
                  "            		<name>2MASS J12204305+1747341</name>\n"
                  "            		<name>Gaia DR2 3946945413106333696</name>\n"
                  "            		<mass errorminus=\"0.3\" errorplus=\"0.3\">2.7</mass>\n"
                  "            		<radius errorminus=\"2\" errorplus=\"2\">19</radius>\n"
                  "            		<magV>4.74</magV>\n"
                  "            		<magB errorminus=\"0.02\" errorplus=\"0.02\">5.74</magB>\n"
                  "            		<magJ errorminus=\"0.334\" errorplus=\"0.334\">2.943</magJ>\n"
                  "            		<magH errorminus=\"0.268\" errorplus=\"0.268\">2.484</magH>\n"
                  "            		<magK errorminus=\"0.346\" errorplus=\"0.346\">2.282</magK>\n"
                  "            		<temperature errorminus=\"100\" errorplus=\"100\">4742</temperature>\n"
                  "            		<metallicity errorminus=\"0.09\" errorplus=\"0.09\">-0.35</metallicity>\n"
                  "            		<spectraltype>G8 III</spectraltype>\n"
                  "            		<planet>\n"
                  "            			<name>11 Com b</name>\n"
                  "            			<name>Gaia DR2 3946945413106333696 b</name>\n"
                  "            			<name>HIP 60202 b</name>\n"
                  "            			<name>HD 107383 b</name>\n"
                  "            			<name>TYC 1445-2560-1 b</name>\n"
                  "            			<list>Confirmed planets</list>\n"
                  "            			<mass errorminus=\"1.5\" errorplus=\"1.5\" type=\"msini\">19.4</mass>\n"
                  "            			<period errorminus=\"0.32\" errorplus=\"0.32\">326.03</period>\n"
                  "            			<semimajoraxis errorminus=\"0.05\" errorplus=\"0.05\">1.29</semimajoraxis>\n"
                  "            			<eccentricity errorminus=\"0.005\" errorplus=\"0.005\">0.231</eccentricity>\n"
                  "            			<periastron errorminus=\"1.5\" errorplus=\"1.5\">94.8</periastron>\n"
                  "            			<periastrontime errorminus=\"1.6\" errorplus=\"1.6\">2452899.6</periastrontime>\n"
                  "<description>11 Com b is a brown dwarf-mass companion to the intermediate-mass star 11 Comae "
                  "Berenices.</description>\n"
                  "            			<discoverymethod>RV</discoverymethod>\n"
                  "            			<lastupdate>15/09/20</lastupdate>\n"
                  "            			<discoveryyear>2008</discoveryyear>\n"
                  "            		</planet>\n"
                  "            	</star>\n"
                  "            	<videolink>http://youtu.be/qyJXJJDrEDo</videolink>\n"
                  "            	<constellation>Coma Berenices</constellation>\n"
                  "            </system>\n"
                  "            <system>\n"
                  "    	<name>16 Cygni</name>\n"
                  "    	<rightascension>19 41 48.95343</rightascension>\n"
                  "    	<declination>+50 31 30.2153</declination>\n"
                  "    	<distance errorminus=\"0.016\" errorplus=\"0.016\">21.146</distance>\n"
                  "    	<binary>\n"
                  "    		<name>16 Cygni</name>\n"
                  "    		<name>16 Cyg</name>\n"
                  "    		<name>WDS J19418+5032</name>\n"
                  "    		<separation unit=\"arcsec\">39.56</separation>\n"
                  "    		<separation errorminus=\"0.6\" errorplus=\"0.6\" unit=\"AU\">836.5</separation>\n"
                  "    		<positionangle>133.30</positionangle>\n"
                  "    		<binary>\n"
                  "    			<name>16 Cygni AC</name>\n"
                  "    			<name>16 Cyg AC</name>\n"
                  "    			<name>WDS J19418+5032 A</name>\n"
                  "    			<separation unit=\"arcsec\">3.4</separation>\n"
                  "    			<separation unit=\"AU\">72</separation>\n"
                  "    			<positionangle>209</positionangle>\n"
                  "    			<star>\n"
                  "    				<name>16 Cygni A</name>\n"
                  "    				<name>16 Cyg A</name>\n"
                  "    				<name>HD 186408</name>\n"
                  "    				<name>HIP 96895</name>\n"
                  "    				<name>TYC 3565-1524-1</name>\n"
                  "    				<name>SAO 31898</name>\n"
                  "    				<name>HR 7503</name>\n"
                  "    				<name>Gliese 765.1 A</name>\n"
                  "    				<name>GJ 765.1 A</name>\n"
                  "    				<name>BD+50 2847</name>\n"
                  "    				<name>2MASS J19414896+5031305</name>\n"
                  "    				<name>KIC 12069424</name>\n"
                  "    				<name>WDS J19418+5032 Aa</name>\n"
                  "    				<magB>6.59</magB>\n"
                  "    				<magV>5.95</magV>\n"
                  "    				<magJ>5.09</magJ>\n"
                  "    				<magH>4.72</magH>\n"
                  "    				<magK>4.43</magK>\n"
                  "    				<spectraltype>G2V</spectraltype>\n"
                  "    				<mass errorminus=\"0.02\" errorplus=\"0.02\">1.11</mass>\n"
                  "    				<radius errorminus=\"0.008\" errorplus=\"0.008\">1.243</radius>\n"
                  "    				<temperature errorminus=\"50\" errorplus=\"50\">5825</temperature>\n"
                  "    				<metallicity errorminus=\"0.026\" errorplus=\"0.026\">0.096</metallicity>\n"
                  "    				<age errorminus=\"0.4\" errorplus=\"0.4\">6.8</age>\n"
                  "    			</star>\n"
                  "    			<star>\n"
                  "    				<name>16 Cygni C</name>\n"
                  "    				<name>16 Cyg C</name>\n"
                  "    				<name>WDS J19418+5032 Ab</name>\n"
                  "    				<mass>0.17</mass>\n"
                  "    				<magV>13</magV>\n"
                  "    				<spectraltype>M</spectraltype>\n"
                  "    			</star>\n"
                  "    		</binary>\n"
                  "    		<star>\n"
                  "    			<name>16 Cygni B</name>\n"
                  "    			<name>16 Cyg B</name>\n"
                  "    			<name>HD 186427</name>\n"
                  "    			<name>HIP 96901</name>\n"
                  "    			<name>TYC 3565-1525-1</name>\n"
                  "    			<name>SAO 31899</name>\n"
                  "    			<name>HR 7504</name>\n"
                  "    			<name>Gliese 765.1 B</name>\n"
                  "    			<name>GJ 765.1 B</name>\n"
                  "    			<name>BD+50 2848</name>\n"
                  "    			<name>2MASS J19415198+5031032</name>\n"
                  "    			<name>KIC 12069449</name>\n"
                  "    			<mass errorminus=\"0.02\" errorplus=\"0.02\">1.07</mass>\n"
                  "    			<radius errorminus=\"0.007\" errorplus=\"0.007\">1.127</radius>\n"
                  "    			<magB>6.86</magB>\n"
                  "    			<magV>6.20</magV>\n"
                  "    			<magJ>4.99</magJ>\n"
                  "    			<magH>4.70</magH>\n"
                  "    			<magK>4.65</magK>\n"
                  "    			<temperature errorminus=\"50\" errorplus=\"50\">5750</temperature>\n"
                  "    			<metallicity errorminus=\"0.021\" errorplus=\"0.021\">0.052</metallicity>\n"
                  "    			<spectraltype>G2V</spectraltype>\n"
                  "    			<age errorminus=\"0.4\" errorplus=\"0.4\">6.8</age>\n"
                  "    			<planet>\n"
                  "    				<name>16 Cygni B b</name>\n"
                  "    				<name>16 Cyg B b</name>\n"
                  "    				<name>HD 186427 B b</name>\n"
                  "    				<list>Confirmed planets</list>\n"
                  "    				<mass errorminus=\"0.05\" errorplus=\"0.05\" type=\"msini\">1.77</mass>\n"
                  "    				<period errorminus=\"0.6\" errorplus=\"0.6\">799.5</period>\n"
                  "    				<semimajoraxis errorminus=\"0.01\" errorplus=\"0.01\">1.72</semimajoraxis>\n"
                  "    				<eccentricity errorminus=\"0.011\" errorplus=\"0.011\">0.689</eccentricity>\n"
                  "    				<periastron errorminus=\"2.1\" errorplus=\"2.1\">83.4</periastron>\n"
                  "    				<periastrontime errorminus=\"1.6\" errorplus=\"1.6\">2450539.3</periastrontime>\n"
                  "    				<inclination errorminus=\"1\" errorplus=\"1\">45</inclination>\n"
                  "<description>16 Cygni is a hierarchical triple system. The star is in the Kepler field of view. In "
                  "an active search for extra-terrestrial intelligence a radio message has been sent to this system "
                  "on May 24 1999. It will reach the system in 2069.</description>\n"
                  "    				<discoverymethod>RV</discoverymethod>\n"
                  "    				<lastupdate>15/09/22</lastupdate>\n"
                  "    				<discoveryyear>1996</discoveryyear>\n"
                  "    				<list>Planets in binary systems, S-type</list>\n"
                  "    			</planet>\n"
                  "    		</star>\n"
                  "    	</binary>\n"
                  "    	<constellation>Cygnus</constellation>\n"
                  "    </system>\n"
                  "    <system>\n"
                  "    	<name>2M 1938+4603</name>\n"
                  "    	<name>Kepler-451</name>\n"
                  "    	<name>KIC 9472174</name>\n"
                  "    	<rightascension>19 38 32.6</rightascension>\n"
                  "    	<declination>+46 03 59</declination>\n"
                  "    	<distance errorminus=\"6.4\" errorplus=\"6.4\">400.8</distance>\n"
                  "    	<binary>\n"
                  "    		<name>2M 1938+4603</name>\n"
                  "    		<name>Kepler-451</name>\n"
                  "    		<name>KIC 9472174</name>\n"
                  "    		<name>2MASS J19383260+4603591</name>\n"
                  "    		<name>WISE J193832.61+460358.9</name>\n"
                  "    		<name>TYC 3556-3568-1</name>\n"
                  "    		<name>NSVS 5629361</name>\n"
                  "    		<period errorminus=\"0.000000005\" errorplus=\"0.000000005\">0.125765282</period>\n"
                  "    		<inclination errorminus=\"0.20\" errorplus=\"0.20\">69.45</inclination>\n"
                  "<transittime errorminus=\"0.00003\" errorplus=\"0.00003\" unit=\"BJD\">2455276.60843</transittime>\n"
                  "    		<semimajoraxis errorminus=\"0.00007\" errorplus=\"0.00007\">0.00414</semimajoraxis>\n"
                  "    		<star>\n"
                  "    			<name>2M 1938+4603 A</name>\n"
                  "    			<name>Kepler-451 A</name>\n"
                  "    			<name>KIC 9472174 A</name>\n"
                  "    			<name>2MASS J19383260+4603591 A</name>\n"
                  "    			<name>WISE J193832.61+460358.9 A</name>\n"
                  "    			<name>TYC 3556-3568-1 A</name>\n"
                  "    			<name>NSVS 5629361 A</name>\n"
                  "    			<magB errorminus=\"0.10\" errorplus=\"0.10\">12.17</magB>\n"
                  "    			<magV errorminus=\"0.24\" errorplus=\"0.24\">12.69</magV>\n"
                  "    			<magR errorminus=\"0.08\" errorplus=\"0.08\">12.46</magR>\n"
                  "    			<magI>12.399</magI>\n"
                  "    			<magJ errorminus=\"0.022\" errorplus=\"0.022\">12.757</magJ>\n"
                  "    			<magH errorminus=\"0.020\" errorplus=\"0.020\">12.889</magH>\n"
                  "    			<magK errorminus=\"0.029\" errorplus=\"0.029\">12.955</magK>\n"
                  "    			<spectraltype>sdBV</spectraltype>\n"
                  "    			<mass errorminus=\"0.03\" errorplus=\"0.03\">0.48</mass>\n"
                  "    			<temperature errorminus=\"106\" errorplus=\"106\">29564</temperature>\n"
                  "    			<radius errorminus=\"0.004\" errorplus=\"0.004\">0.223</radius>\n"
                  "    		</star>\n"
                  "    		<star>\n"
                  "    			<name>2M 1938+4603 B</name>\n"
                  "    			<name>Kepler-451 B</name>\n"
                  "    			<name>KIC 9472174 B</name>\n"
                  "    			<name>2MASS J19383260+4603591 B</name>\n"
                  "    			<name>WISE J193832.61+460358.9 B</name>\n"
                  "    			<name>TYC 3556-3568-1 B</name>\n"
                  "    			<name>NSVS 5629361 B</name>\n"
                  "    			<spectraltype>M</spectraltype>\n"
                  "    			<mass errorminus=\"0.01\" errorplus=\"0.01\">0.12</mass>\n"
                  "    			<radius errorminus=\"0.003\" errorplus=\"0.003\">0.158</radius>\n"
                  "    		</star>\n"
                  "    		<planet>\n"
                  "    			<name>2M 1938+4603 b</name>\n"
                  "    			<name>2M 1938+4603 (AB) b</name>\n"
                  "    			<name>Kepler-451 b</name>\n"
                  "    			<name>Kepler-451 (AB) b</name>\n"
                  "    			<name>KIC 9472174 b</name>\n"
                  "    			<name>KIC 9472174 (AB) b</name>\n"
                  "    			<period errorminus=\"2\" errorplus=\"2\">416</period>\n"
                  "    			<semimajoraxis errorminus=\"0.02\" errorplus=\"0.02\">0.92</semimajoraxis>\n"
                  "    			<mass errorminus=\"0.1\" errorplus=\"0.1\">1.9</mass>\n"
                  "    			<discoverymethod>timing</discoverymethod>\n"
                  "    			<discoveryyear>2015</discoveryyear>\n"
                  "    			<lastupdate>15/06/11</lastupdate>\n"
                  "<description>2M 1938+4603 is an eclipsing post-common envelope binary in the Kepler field."
                  "Variations in the timing of the eclipses of the binary star have been used to infer the presence "
                  "of a giant planet in a circumbinary orbit.</description>\n"
                  "    			<list>Confirmed planets</list>\n"
                  "    			<list>Planets in binary systems, P-type</list>\n"
                  "    		</planet>\n"
                  "    	</binary>\n"
                  "    	<constellation>Cygnus</constellation>\n"
                  "    </system>\n"
                  "    <system>\n"
                  "    		<planet>\n"
                  "    			<name>Example of Rogue planet</name>\n"
                  "    			<list>Confirmed planets</list>\n"
                  "    			<list>Planets in binary systems, P-type</list>\n"
                  "    		</planet>\n"
                  "    	<constellation>Cygnus</constellation>\n"
                  "    </system>\n"
                  "            </systems>")
    return xml_string


def test__convert_xmlfile_to_csvfile(instance):
    xml_string = fill_xml_string()
    # Parse the XML string and create an XML file
    root = ElementTree.fromstring(xml_string)
    tree = ElementTree.ElementTree(root)

    # Write to a compressed XML file (output.xml.gz)
    with gzip.open("output.xml.gz", "wb") as file:
        tree.write(file, encoding="utf-8", xml_declaration=True)
    instance.convert_xmlfile_to_csvfile(file_path="output.xml.gz")
    data = pd.read_csv("output.csv")
    expected_columns = [
        "name",
        "binaryflag",
        "mass",
        "masstype",
        "mass_min",
        "mass_max",
        "radius",
        "radius_min",
        "radius_max",
        "period",
        "period_min",
        "period_max",
        "semimajoraxis",
        "semimajoraxis_min",
        "semimajoraxis_max",
        "eccentricity",
        "eccentricity_min",
        "eccentricity_max",
        "periastron",
        "longitude",
        "ascendingnode",
        "inclination",
        "inclination_min",
        "inclination_max",
        "temperature",
        "age",
        "discoverymethod",
        "discoveryyear",
        "lastupdate",
        "system_rightascension",
        "system_declination",
        "system_distance",
        "hoststar_mass",
        "hoststar_radius",
        "hoststar_metallicity",
        "hoststar_temperature",
        "hoststar_age",
        "hoststar_magJ",
        "hoststar_magI",
        "hoststar_magU",
        "hoststar_magR",
        "hoststar_magB",
        "hoststar_magV",
        "hoststar_magH",
        "hoststar_magK",
        "hoststar_spectraltype",
        "alias",
        "list",
    ]
    assert all(element in data.columns for element in expected_columns)
    assert (
            data.at[0, "alias"]
            == "11 Com,11 Comae Berenices,HD 107383,HIP 60202,TYC 1445-2560-1,SAO 100053,HR 4697,BD+18 2592,"
               "2MASS J12204305+1747341,Gaia DR2 3946945413106333696"
    )
    assert data.at[0, "system_rightascension"] == "12 20 43.0255"
    assert data.at[0, "system_distance"] == 88.9
    assert data.at[0, "mass_min"] == 1.5
    assert data.at[0, "list"] == "Confirmed planets"
    assert "S-type" in data.at[1, "list"]
    assert data.at[1, "binaryflag"] == 2
    assert "P-type" in data.at[2, "list"]
    assert data.at[2, "binaryflag"] == 1
    assert data.at[3, "binaryflag"] == 3
    os.remove("output.csv")
    os.remove("output.xml.gz")
    # Intentionally pass an invalid XML file path to the function
    with pytest.raises(Exception) as e:
        instance.convert_xmlfile_to_csvfile(file_path="invalid_path_to_xml_file.xml.gz")

    # Check if the logging.error is called and the message is logged
    assert "No such file or directory" in str(e.value)


def test__convert_discovery_methods(instance):
    # Sample data
    data = pd.DataFrame(
        {
            "discovery_method": [
                "Primary Transit#TTV",
                "Transit Timing Variations",
                "Eclipse Timing Variations",
                "Primary Transit",
                "Pulsar",
                "Pulsation Timing Variations",
                "Timing",
                "disk kinematics",
                "Kinematic",
                "Disk Kinematics",
                "Orbital Brightness Modulation",
                "astrometry",
                "microlensing",
                "imaging",
                "transit",
                "timing",
                "RV",
                None,
            ]
        }
    )

    expected_result = pd.DataFrame(
        {
            "discovery_method": [
                "TTV",
                "TTV",
                "TTV",
                "Transit",
                "Pulsar Timing",
                "Pulsar Timing",
                "Pulsar Timing",
                "Other",
                "Other",
                "Other",
                "Other",
                "Astrometry",
                "Microlensing",
                "Imaging",
                "Transit",
                "Pulsar Timing",
                "Radial Velocity",
                "",
            ]
        }
    )

    # Apply the conversion function
    result = instance.convert_discovery_methods(data)

    # Check if the result matches the expected output
    assert result.equals(expected_result)


def test__perform_query(instance):
    #### SIMBAD #####
    # SEARCH ON NAME (host+ + binary, host+binary, pure host)
    expected = pd.DataFrame(
        {
            "hostbinary": ["16 Cyg B", "21 HerAB", "EPIC 203868608"],
            "main_id": ["*  16 Cyg B", "* o Her", "2MASS J16171898-2437186"],
            "ra_2": [295.46655282394, 246.04511980308, 244.32909535559997],
            "dec_2": [50.51752473081, 6.94821049715, -24.6218720689],
            "ids": [
                "GJ 765.1 B|HIP 96901|Gaia DR3 2135550755683407232|TIC 27533327|TYC 3565-1525-1|ASCC  271120|2MASS "
                "J19415198+5031032|USNO-B1.0 1405-00322540|*  16 Cyg B|ADS 12815 B|AG+50 1408|BD+50  2848|CCDM "
                "J19418+5031B|GC 27285|GCRV 12084|GEN# +1.00186427|HD 186427|HIC  96901|HR  7504|IDS 19392+5017 B|LTT "
                "15751|NLTT 48138|PPM  37673|ROT  2840|SAO  31899|SKY# 36807|SPOCS  855|UBV   16780|UBV M  24082|USNO "
                "890|YZ  50  6150|Gaia DR2 2135550755683407232|LSPM J1941+5031E|WDS J19418+5032B|AKARI-IRC-V1 "
                "J1941518+503102|** STF 4046B|WISEA J194151.82+503102.2|Gaia DR1 2135550854464294784|WEB 17005|KIC "
                "12069449",
                "HIP 80351|Gaia DR3 4439556480967874432|TIC 369080491|SBC9 901|*  21 Her|* o Her|AG+07 2044|BD+07  "
                "3164|FK5 1429|GC 22058|GCRV  9437|GEN# +1.00147869|GSC 00381-01598|HD 147869|HIC  80351|HR  6111|N30 "
                "3675|PMC 90-93  1008|PPM 162584|SAO 121568|SBC7   573|SKY# 29573|TD1 19139|TYC  381-1598-1|UBV   "
                "21316|UBV M  21407|YZ   7  7306|uvby98 100147869|Renson 41690|2MASS J16241083+0656534|Gaia DR1 "
                "4439556476666646528|WEB 13596|Gaia DR2 4439556480967874432",
                "Gaia DR3 6049656638390048896|AP J16171898-2437186|TIC 98231712|2MASS J16171898-2437186|UGCS " "J161718.97-243718.7|WISEA J161718.97-243718.9|EPIC 203868608|Gaia DR2 6049656638390048896",
            ],
            "angsep": [0.0, 0.0, 0.0],
        }
    )

    list_of_hosts = pd.DataFrame()
    list_of_hosts["hostbinary"] = ["16 Cyg B", "21 HerAB", "EPIC 203868608"]
    service = pyvo.dal.TAPService("http://simbad.u-strasbg.fr:80/simbad/sim-tap")

    t2 = Table.from_pandas(list_of_hosts)
    query = """SELECT t.*, basic.main_id, basic.ra as ra_2,basic.dec as dec_2, ids.ids as ids FROM TAP_UPLOAD.tab as 
    t LEFT OUTER JOIN ident ON ident.id = t.hostbinary LEFT OUTER JOIN basic ON ident.oidref = basic.oid LEFT OUTER 
    JOIN ids ON basic.oid = ids.oidref"""
    table = UtilityFunctions.perform_query(service, query, uploads_dict={"tab": t2})

    assert set(table.columns) == set(expected.columns)
    assert_frame_equal(table[['hostbinary','main_id','ra_2','dec_2','angsep']], expected[['hostbinary','main_id','ra_2','dec_2','angsep']])
    for i in table.index:
        assert sorted(table.at[i, "ids"].split("|")) == sorted(
            expected.at[i, "ids"].split("|")
        )

    # SEARCH ON ALIAS (alias+ + binary, alias+binary,pure alias)
    expected = pd.DataFrame(
        {
            "ind": [0, 1, 2],
            "alias": ["WDS J23596-3502 A", "94 CetA", "Kepler-451"],
            "main_id": ["CD-35 16019", "*  94 Cet", "Kepler-451"],
            "ra_2": [359.90029661347, 48.19348569878, 294.63588492334],
            "dec_2": [-35.03136767178, -1.1960988358299998, 46.066426453320005],
            "ids": [
                "Gaia DR3 2312679845530628096|SPOCS 3245|Gaia DR2 2312679845530628096|** B 2511A|CCDM "
                "J23596-3502A|WDS J23596-3502A|TYC 7522-505-1|Gaia DR1 2312679841235149440|CD-35 16019A|WASP-8|CD-35 "
                "16019|CPC 18 12094|CPD-35  9465|GSC 07522-00505|PPM 304426|SAO 214901|UCAC2  16954660|UCAC3 "
                "110-468375|2MASS J23593607-3501530|IDS 23544-3535 A|UCAC4 275-215468|TIC 183532609|TOI-191",
                "GJ 128|HIP 14954|Gaia DR3 3265335443260522112|TIC 49845357|PLX  663|*  94 Cet|AG-01  300|BD-01   "
                "457|CSI-01   457  1|FK5  116|GC  3838|GCRV  1775|HD  19994|HIC  14954|HR   962|LTT  1515|N30  "
                "656|NLTT 10224|PMC 90-93    84|PPM 175267|ROT   431|SAO 130355|SKY#  4813|SPOCS  155|TD1  1984|UBV   "
                " 3104|YZ  91   684|YZ   0  3372|CCDM J03128-0112A|ADS  2406 A|WDS J03128-0112A|** HJ  663A|GEN# "
                "+1.00019994|IDS 03077-0134 A|Gaia DR2 3265335443260522112|TYC 4708-1423-1|IRAS "
                "03102-0122|AKARI-IRC-V1 J0312465-011146|CSI-01   457  3|[RHG95]   572|UCAC3 178-9414|UCAC4 "
                "445-004277|2MASS J03124644-0111458|CSI-01   457  2|WISEA J031246.58-011146.4|uvby98 100019994|WEB  "
                "2887",
                "LAMOST J193832.60+460359.1|LAMOST J193832.62+460359.1|LAMOST J193832.61+460359.1|Gaia DR3 "
                "2080063931448749824|ATO J294.6359+46.0664|TIC 271164763|GSC 03556-03568|UCAC3 273-158867|2MASS "
                "J19383260+4603591|USNO-B1.0 1360-00318562|GSC2 N0303123803|GSC2.3 N2JF000803|TYC 3556-3568-1|ASAS "
                "J193833+4604.0|NSVS   5629361|Gaia DR2 2080063931448749824|Kepler-451|EQ J1938+4603|KIC 9472174",
            ],
            "angsep": [0.0, 0.0, 0.0],
        }
    )
    alias_df = pd.DataFrame(columns=["ind", "alias"])
    alias_df["ind"] = [0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 2]
    alias_df["alias"] = [
        "2MASS J23593607-3501530 A",
        "Gaia DR2 2312679845530628096 A",
        "TIC 183532609 A",
        "TIC-183532609 A",
        "TOI-191 A",
        "TYC 7522-00505-1 A",
        "UCAC4 275-215468 A",
        "WDS J23596-3502 A",
        "WISE J235936.16-350153.1 A",
        "94 CetA",
        "94 Cet AA",
        "Gaia DR2 3265335443260522112A",
        "HIP 14954A",
        "TIC 49845357A",
        "Kepler-451",
    ]

    t2 = Table.from_pandas(alias_df)
    query = (
        """SELECT t.*, basic.main_id, basic.ra as ra_2,basic.dec as dec_2, ids.ids FROM TAP_UPLOAD.tab as t LEFT 
        OUTER JOIN ident ON ident.id = t.alias LEFT OUTER JOIN basic ON ident.oidref = basic.oid LEFT OUTER JOIN ids 
        ON basic.oid = ids.oidref""",
    )
    table = UtilityFunctions.perform_query(service, query, uploads_dict={"tab": t2})
    assert set(table.columns) == set(expected.columns)
    table = table.reset_index(drop=True)

    assert_frame_equal(table[['alias','main_id','ra_2','dec_2','angsep']], expected[['alias','main_id','ra_2','dec_2','angsep']])
    for i in table.index:
        assert sorted(table.at[i, "ids"].split("|")) == sorted(
            expected.at[i, "ids"].split("|")
        )

    # SELECT EXTRA ALIAS

    query = """SELECT  basic.main_id, basic.ra as ra_2,basic.dec as dec_2, ids.ids
        FROM ident JOIN basic ON ident.oidref = basic.oid LEFT OUTER JOIN ids ON basic.oid = ids.oidref
        WHERE id = '*  16 Cyg B'"""

    table = UtilityFunctions.perform_query(service, query)
    expected = pd.DataFrame(
        {
            "main_id": ["*  16 Cyg B"],
            "ra_2": [295.46655282394],
            "dec_2": [50.51752473081],
            "ids": [
                "GJ 765.1 B|HIP 96901|Gaia DR3 2135550755683407232|TIC 27533327|TYC 3565-1525-1|ASCC  271120|2MASS "
                "J19415198+5031032|USNO-B1.0 1405-00322540|*  16 Cyg B|ADS 12815 B|AG+50 1408|BD+50  2848|CCDM "
                "J19418+5031B|GC 27285|GCRV 12084|GEN# +1.00186427|HD 186427|HIC  96901|HR  7504|IDS 19392+5017 B|LTT "
                "15751|NLTT 48138|PPM  37673|ROT  2840|SAO  31899|SKY# 36807|SPOCS  855|UBV   16780|UBV M  24082|USNO "
                "890|YZ  50  6150|Gaia DR2 2135550755683407232|LSPM J1941+5031E|WDS J19418+5032B|AKARI-IRC-V1 "
                "J1941518+503102|** STF 4046B|WISEA J194151.82+503102.2|Gaia DR1 2135550854464294784|WEB 17005|KIC "
                "12069449",
            ],
            "angsep": [0.0],
        }
    )

    assert set(table.columns) == set(expected.columns)
    assert_frame_equal(table[['main_id','ra_2','dec_2','angsep']], expected[['main_id','ra_2','dec_2','angsep']])
    for i in table.index:
        assert sorted(table.at[i, "ids"].split("|")) == sorted(
            expected.at[i, "ids"].split("|")
        )

    #     # SEARCH ON COORDINATES
    expected = pd.DataFrame(
        {
            "main_id": ["*  51 Peg"],
            "dec_2": [20.768832511140005],
            "ra_2": [344.36658535524],
            "type": ["PM*"],
            "hostbinary": ["51 Peg"],
            "ra": [344.3667],
            "dec": [20.7689],
            "angsep": [0.45601200000000003],
            "selected": [1],
        }
    )

    data = {
        "hostbinary": ["51 Peg"],
        "ra": [344.3667],
        "dec": [20.7689],
    }
    tolerance = 1 / 3600  # arcsec in degrees
    t2 = Table.from_pandas(pd.DataFrame(data))
    query = (
            """SELECT basic.main_id, basic.dec as dec_2,basic.ra as ra_2, basic.otype as type, t.hostbinary, t.ra, 
            t.dec FROM basic JOIN TAP_UPLOAD.tab AS t on 1=CONTAINS(POINT('ICRS',basic.ra, basic.dec),   
            CIRCLE('ICRS', t.ra, t.dec,"""
            + str(tolerance)
            + """)) """
    )
    table = UtilityFunctions.perform_query(service, query, uploads_dict={"tab": t2})
    #in this case must run also calculate_angsep
    table = UtilityFunctions.calculate_angsep(table)

    assert set(table.columns) == set(expected.columns)
    print(table, expected)
    assert_frame_equal(table, expected)

    ###### TIC #####
    #     # SEARCH ON TIC
    expected = pd.DataFrame(
        {
            "ra_2": ["306.50116406605"],
            "dec_2": ["-48.92035980163"],
            "GAIA": ["Gaia DR2 6668227036766532864"],
            "UCAC4": ["UCAC4 206-182296"],
            "2MASS": ["2MASS J20260027-4855132"],
            "WISEA": ["WISE J202600.26-485513.4"],
            "TIC": ["100263315"],
            "KIC": [""],
            "HIP": [""],
            "TYC": [""],
            "host": ["100263315"],
            "main_id": ["TIC 100263315"],
            "ids": [
                "UCAC4 206-182296,2MASS J20260027-4855132,WISE J202600.26-485513.4,Gaia DR2 6668227036766532864"
            ],
            "angsep": [0.0],
        }
    )
    service = pyvo.dal.TAPService(" http://TAPVizieR.u-strasbg.fr/TAPVizieR/tap/")
    list_of_hosts = pd.DataFrame()
    list_of_hosts["host"] = [100263315]
    t2 = Table.from_pandas(list_of_hosts)
    query = """SELECT tic.RAJ2000 as ra_2, tic.DEJ2000 as dec_2,tic.GAIA, tic.UCAC4, tic."2MASS", tic.WISEA, tic.TIC, 
    tic.KIC, tic.HIP, tic.TYC, t.*  FROM "IV/38/tic" as tic JOIN TAP_UPLOAD.tab as t ON tic.TIC = t.host"""

    table = UtilityFunctions.perform_query(service, query, uploads_dict={"tab": t2})
    assert set(table.columns) == set(expected.columns)
    assert_frame_equal(table, expected)

    #
    #     # SEARCH ON TIC COORDINATES (upload of table is a bit buggy so we do without)
    data = pd.DataFrame(
        {
            "hostbinary": ["EPIC 251345848"],
            "ra": [140.405406],
            "dec": [20.727007],
        }
    )
    expected = pd.DataFrame(
        {
            "ra_2": [140.40548276307],
            "dec_2": [20.72704944739],
            "GAIA": ["Gaia DR2 637589853596386560"],
            "UCAC4": ["UCAC4 554-044757"],
            "2MASS": ["2MASS J09213732+2043373"],
            "WISEA": ["WISE J092137.30+204337.2"],
            "TIC": ["86119727"],
            "KIC": [""],
            "HIP": [""],
            "TYC": [""],
            "main_id": ["TIC 86119727"],
            "ids": [
                "UCAC4 554-044757,2MASS J09213732+2043373,WISE J092137.30+204337.2,Gaia DR2 637589853596386560"
            ],
            "angsep": [0.30024],
            "hostbinary": ["EPIC 251345848"],
            "ra": [140.405406],
            "dec": [20.727007],
            "selected": [1],
        }
    )
    t2 = pd.DataFrame(data)

    table = pd.DataFrame()
    for ind in t2.index:
        query = (
                """SELECT tic.RAJ2000 as ra_2, tic.DEJ2000 as dec_2,tic.GAIA, tic.UCAC4, tic."2MASS", tic.WISEA, 
                tic.TIC, tic.KIC, tic.HIP, tic.TYC  FROM "IV/38/tic" as tic  WHERE 1=CONTAINS(POINT('ICRS',tic.RAJ2000, tic.DEJ2000),   CIRCLE('ICRS',""" + str(
            t2.at[ind, 'ra']) + """,""" + str(t2.at[ind, 'dec']) + ""","""
                + str(tolerance)
                + """))"""
        )
        single_table = UtilityFunctions.perform_query(service, query, uploads_dict={})
        single_table['hostbinary'] = t2.at[ind, 'hostbinary']
        single_table['ra'] = t2.at[ind, 'ra']
        single_table['dec'] = t2.at[ind, 'dec']
        single_table = UtilityFunctions.calculate_angsep(single_table)

        table = pd.concat([table, single_table])
    table = table.drop_duplicates()

    assert set(table.columns) == set(expected.columns)
    assert_frame_equal(table, expected)
