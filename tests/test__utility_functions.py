import gzip
import os
import xml.etree.ElementTree as ET

import numpy as np
import pandas as pd

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
        "2M ": "2MASS ",
        "KOI ": "KOI-",
        "Kepler ": "Kepler-",
        "BD ": "BD",
        "OGLE-": "OGLE ",
        "MOA-": "MOA ",
        "gam 1 ": "gam ",
        "EPIC-": "EPIC ",
        "Pr 0": "Pr ",
        "TOI ": "TOI-",
        "kepler": "Kepler",
        "Gliese": "GJ",
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
    # Example: Assuming your input_sources.ini has a [General] section with a 'timeout' parameter
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
        ("WASP-184 b", "WASP-184 b"),  # random general one that should not change
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


def fill_xml_string():
    xml_string = """<systems>
            <system>
            	<name>11 Com</name>
            	<rightascension>12 20 43.0255</rightascension>
            	<declination>+17 47 34.3392</declination>
            	<distance errorminus="1.7" errorplus="1.7">88.9</distance>
            	<star>
            		<name>11 Com</name>
            		<name>11 Comae Berenices</name>
            		<name>HD 107383</name>
            		<name>HIP 60202</name>
            		<name>TYC 1445-2560-1</name>
            		<name>SAO 100053</name>
            		<name>HR 4697</name>
            		<name>BD+18 2592</name>
            		<name>2MASS J12204305+1747341</name>
            		<name>Gaia DR2 3946945413106333696</name>
            		<mass errorminus="0.3" errorplus="0.3">2.7</mass>
            		<radius errorminus="2" errorplus="2">19</radius>
            		<magV>4.74</magV>
            		<magB errorminus="0.02" errorplus="0.02">5.74</magB>
            		<magJ errorminus="0.334" errorplus="0.334">2.943</magJ>
            		<magH errorminus="0.268" errorplus="0.268">2.484</magH>
            		<magK errorminus="0.346" errorplus="0.346">2.282</magK>
            		<temperature errorminus="100" errorplus="100">4742</temperature>
            		<metallicity errorminus="0.09" errorplus="0.09">-0.35</metallicity>
            		<spectraltype>G8 III</spectraltype>
            		<planet>
            			<name>11 Com b</name>
            			<name>Gaia DR2 3946945413106333696 b</name>
            			<name>HIP 60202 b</name>
            			<name>HD 107383 b</name>
            			<name>TYC 1445-2560-1 b</name>
            			<list>Confirmed planets</list>
            			<mass errorminus="1.5" errorplus="1.5" type="msini">19.4</mass>
            			<period errorminus="0.32" errorplus="0.32">326.03</period>
            			<semimajoraxis errorminus="0.05" errorplus="0.05">1.29</semimajoraxis>
            			<eccentricity errorminus="0.005" errorplus="0.005">0.231</eccentricity>
            			<periastron errorminus="1.5" errorplus="1.5">94.8</periastron>
            			<periastrontime errorminus="1.6" errorplus="1.6">2452899.6</periastrontime>
            			<description>11 Com b is a brown dwarf-mass companion to the intermediate-mass star 11 Comae Berenices.</description>
            			<discoverymethod>RV</discoverymethod>
            			<lastupdate>15/09/20</lastupdate>
            			<discoveryyear>2008</discoveryyear>
            		</planet>
            	</star>
            	<videolink>http://youtu.be/qyJXJJDrEDo</videolink>
            	<constellation>Coma Berenices</constellation>
            </system>
            <system>
    	<name>16 Cygni</name>
    	<rightascension>19 41 48.95343</rightascension>
    	<declination>+50 31 30.2153</declination>
    	<distance errorminus="0.016" errorplus="0.016">21.146</distance>
    	<binary>
    		<name>16 Cygni</name>
    		<name>16 Cyg</name>
    		<name>WDS J19418+5032</name>
    		<separation unit="arcsec">39.56</separation>
    		<separation errorminus="0.6" errorplus="0.6" unit="AU">836.5</separation>
    		<positionangle>133.30</positionangle>
    		<binary>
    			<name>16 Cygni AC</name>
    			<name>16 Cyg AC</name>
    			<name>WDS J19418+5032 A</name>
    			<separation unit="arcsec">3.4</separation>
    			<separation unit="AU">72</separation>
    			<positionangle>209</positionangle>
    			<star>
    				<name>16 Cygni A</name>
    				<name>16 Cyg A</name>
    				<name>HD 186408</name>
    				<name>HIP 96895</name>
    				<name>TYC 3565-1524-1</name>
    				<name>SAO 31898</name>
    				<name>HR 7503</name>
    				<name>Gliese 765.1 A</name>
    				<name>GJ 765.1 A</name>
    				<name>BD+50 2847</name>
    				<name>2MASS J19414896+5031305</name>
    				<name>KIC 12069424</name>
    				<name>WDS J19418+5032 Aa</name>
    				<magB>6.59</magB>
    				<magV>5.95</magV>
    				<magJ>5.09</magJ>
    				<magH>4.72</magH>
    				<magK>4.43</magK>
    				<spectraltype>G2V</spectraltype>
    				<mass errorminus="0.02" errorplus="0.02">1.11</mass>
    				<radius errorminus="0.008" errorplus="0.008">1.243</radius>
    				<temperature errorminus="50" errorplus="50">5825</temperature>
    				<metallicity errorminus="0.026" errorplus="0.026">0.096</metallicity>
    				<age errorminus="0.4" errorplus="0.4">6.8</age>
    			</star>
    			<star>
    				<name>16 Cygni C</name>
    				<name>16 Cyg C</name>
    				<name>WDS J19418+5032 Ab</name>
    				<mass>0.17</mass>
    				<magV>13</magV>
    				<spectraltype>M</spectraltype>
    			</star>
    		</binary>
    		<star>
    			<name>16 Cygni B</name>
    			<name>16 Cyg B</name>
    			<name>HD 186427</name>
    			<name>HIP 96901</name>
    			<name>TYC 3565-1525-1</name>
    			<name>SAO 31899</name>
    			<name>HR 7504</name>
    			<name>Gliese 765.1 B</name>
    			<name>GJ 765.1 B</name>
    			<name>BD+50 2848</name>
    			<name>2MASS J19415198+5031032</name>
    			<name>KIC 12069449</name>
    			<mass errorminus="0.02" errorplus="0.02">1.07</mass>
    			<radius errorminus="0.007" errorplus="0.007">1.127</radius>
    			<magB>6.86</magB>
    			<magV>6.20</magV>
    			<magJ>4.99</magJ>
    			<magH>4.70</magH>
    			<magK>4.65</magK>
    			<temperature errorminus="50" errorplus="50">5750</temperature>
    			<metallicity errorminus="0.021" errorplus="0.021">0.052</metallicity>
    			<spectraltype>G2V</spectraltype>
    			<age errorminus="0.4" errorplus="0.4">6.8</age>
    			<planet>
    				<name>16 Cygni B b</name>
    				<name>16 Cyg B b</name>
    				<name>HD 186427 B b</name>
    				<list>Confirmed planets</list>
    				<mass errorminus="0.05" errorplus="0.05" type="msini">1.77</mass>
    				<period errorminus="0.6" errorplus="0.6">799.5</period>
    				<semimajoraxis errorminus="0.01" errorplus="0.01">1.72</semimajoraxis>
    				<eccentricity errorminus="0.011" errorplus="0.011">0.689</eccentricity>
    				<periastron errorminus="2.1" errorplus="2.1">83.4</periastron>
    				<periastrontime errorminus="1.6" errorplus="1.6">2450539.3</periastrontime>
    				<inclination errorminus="1" errorplus="1">45</inclination>
    				<description>16 Cygni is a hierarchical triple system. The star is in the Kepler field of view. In an active search for extra-terrestrial intelligence a radio message has been sent to this system on May 24 1999. It will reach the system in 2069.</description>
    				<discoverymethod>RV</discoverymethod>
    				<lastupdate>15/09/22</lastupdate>
    				<discoveryyear>1996</discoveryyear>
    				<list>Planets in binary systems, S-type</list>
    			</planet>
    		</star>
    	</binary>
    	<constellation>Cygnus</constellation>
    </system>
    <system>
    	<name>2M 1938+4603</name>
    	<name>Kepler-451</name>
    	<name>KIC 9472174</name>
    	<rightascension>19 38 32.6</rightascension>
    	<declination>+46 03 59</declination>
    	<distance errorminus="6.4" errorplus="6.4">400.8</distance>
    	<binary>
    		<name>2M 1938+4603</name>
    		<name>Kepler-451</name>
    		<name>KIC 9472174</name>
    		<name>2MASS J19383260+4603591</name>
    		<name>WISE J193832.61+460358.9</name>
    		<name>TYC 3556-3568-1</name>
    		<name>NSVS 5629361</name>
    		<period errorminus="0.000000005" errorplus="0.000000005">0.125765282</period>
    		<inclination errorminus="0.20" errorplus="0.20">69.45</inclination>
    		<transittime errorminus="0.00003" errorplus="0.00003" unit="BJD">2455276.60843</transittime>
    		<semimajoraxis errorminus="0.00007" errorplus="0.00007">0.00414</semimajoraxis>
    		<star>
    			<name>2M 1938+4603 A</name>
    			<name>Kepler-451 A</name>
    			<name>KIC 9472174 A</name>
    			<name>2MASS J19383260+4603591 A</name>
    			<name>WISE J193832.61+460358.9 A</name>
    			<name>TYC 3556-3568-1 A</name>
    			<name>NSVS 5629361 A</name>
    			<magB errorminus="0.10" errorplus="0.10">12.17</magB>
    			<magV errorminus="0.24" errorplus="0.24">12.69</magV>
    			<magR errorminus="0.08" errorplus="0.08">12.46</magR>
    			<magI>12.399</magI>
    			<magJ errorminus="0.022" errorplus="0.022">12.757</magJ>
    			<magH errorminus="0.020" errorplus="0.020">12.889</magH>
    			<magK errorminus="0.029" errorplus="0.029">12.955</magK>
    			<spectraltype>sdBV</spectraltype>
    			<mass errorminus="0.03" errorplus="0.03">0.48</mass>
    			<temperature errorminus="106" errorplus="106">29564</temperature>
    			<radius errorminus="0.004" errorplus="0.004">0.223</radius>
    		</star>
    		<star>
    			<name>2M 1938+4603 B</name>
    			<name>Kepler-451 B</name>
    			<name>KIC 9472174 B</name>
    			<name>2MASS J19383260+4603591 B</name>
    			<name>WISE J193832.61+460358.9 B</name>
    			<name>TYC 3556-3568-1 B</name>
    			<name>NSVS 5629361 B</name>
    			<spectraltype>M</spectraltype>
    			<mass errorminus="0.01" errorplus="0.01">0.12</mass>
    			<radius errorminus="0.003" errorplus="0.003">0.158</radius>
    		</star>
    		<planet>
    			<name>2M 1938+4603 b</name>
    			<name>2M 1938+4603 (AB) b</name>
    			<name>Kepler-451 b</name>
    			<name>Kepler-451 (AB) b</name>
    			<name>KIC 9472174 b</name>
    			<name>KIC 9472174 (AB) b</name>
    			<period errorminus="2" errorplus="2">416</period>
    			<semimajoraxis errorminus="0.02" errorplus="0.02">0.92</semimajoraxis>
    			<mass errorminus="0.1" errorplus="0.1">1.9</mass>
    			<discoverymethod>timing</discoverymethod>
    			<discoveryyear>2015</discoveryyear>
    			<lastupdate>15/06/11</lastupdate>
    			<description>2M 1938+4603 is an eclipsing post-common envelope binary in the Kepler field. Variations in the timing of the eclipses of the binary star have been used to infer the presence of a giant planet in a circumbinary orbit.</description>
    			<list>Confirmed planets</list>
    			<list>Planets in binary systems, P-type</list>
    		</planet>
    	</binary>
    	<constellation>Cygnus</constellation>
    </system>
    <system>
    		<planet>
    			<name>Example of Rogue planet</name>
    			<list>Confirmed planets</list>
    			<list>Planets in binary systems, P-type</list>
    		</planet>
    	<constellation>Cygnus</constellation>
    </system>
            </systems>"""
    return xml_string


def test__convert_xmlfile_to_csvfile(instance):
    xml_string = fill_xml_string()
    # Parse the XML string and create an XML file
    root = ET.fromstring(xml_string)
    tree = ET.ElementTree(root)

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
        == "11 Com,11 Comae Berenices,HD 107383,HIP 60202,TYC 1445-2560-1,SAO 100053,HR 4697,BD+18 2592,2MASS J12204305+1747341,Gaia DR2 3946945413106333696"
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
        data = pd.DataFrame({
            'discovery_method': [
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
                None
            ]
        })

        expected_result = pd.DataFrame({
            'discovery_method': [
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
                ""
            ]
        })

        # Apply the conversion function
        result = instance.convert_discovery_methods(data)

        # Check if the result matches the expected output
        assert (result.equals(expected_result))


