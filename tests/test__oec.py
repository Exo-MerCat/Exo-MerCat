import gzip
import os
from datetime import date
from pathlib import Path, PosixPath
from unittest.mock import MagicMock, patch, Mock
import requests

import numpy as np
import pandas as pd
import pytest
from testfixtures import LogCapture

from exomercat.oec import Oec


def fill_xml_string():
    xml_string = """
    <systems>
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
                    <description>11 Com b is a brown dwarf-mass companion to the intermediate-mass star 
                    11 Comae Berenices.</description>
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
                    <description>16 Cygni is a hierarchical triple system. The star is in the Kepler 
                    field of view. In an active search for extra-terrestrial intelligence a radio 
                    message has been sent to this system on May 24 1999. It will reach the system 
                    in 2069.</description>
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
                <description>2M 1938+4603 is an eclipsing post-common envelope binary in the Kepler field. 
                Variations in the timing of the eclipses of the binary star have been used to infer the 
                presence of a giant planet in a circumbinary orbit.</description>
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


@pytest.fixture
def instance():
    return Oec()


def test__init(instance):
    assert instance.data == None
    assert instance.name == "oec"


def test__download_catalog(tmp_path, instance) -> None:
    original_dir = os.getcwd()

    os.chdir(tmp_path)

    url = "https://raw.githubusercontent.com/OpenExoplanetCatalogue/open_exoplanet_catalogue/refs/heads/master/systems/HD%20154857.xml"
    filename = "oec"
    # DATE IS TODAY AND FILE DOES NOT EXIST
    expected_file_path = filename+date.today().strftime("%Y-%m-%d")+".csv"
    expected_file_path_xml = filename + date.today().strftime("%Y-%m-%d")+".xml"
    # CASE A: download the file
    with LogCapture() as log:
        result = instance.download_catalog(url=url, filename=filename,local_date=date.today().strftime("%Y-%m-%d"))
    log = pd.DataFrame(list(log), columns=["user", "info", "message"])
    assert "Catalog downloaded." in log["message"].to_list()
    assert isinstance(pd.read_csv(expected_file_path), pd.DataFrame)

    # CASE B: file already exists, no need to download
    with LogCapture() as log:
        result = instance.download_catalog(url=url, filename=filename, local_date=date.today().strftime("%Y-%m-%d"))
    log = pd.DataFrame(list(log), columns=["user", "info", "message"])
    assert "Reading existing file downloaded in date: " + date.today().strftime("%Y-%m-%d") in log["message"].to_list()
    assert isinstance(pd.read_csv(expected_file_path), pd.DataFrame)


    os.rename(expected_file_path,'oec2024-10-02.csv')
    os.rename(expected_file_path_xml,'oec2024-10-02.xml')

    # CASE C: file corrupted
    with LogCapture() as log:
        with   patch("pandas.read_csv", side_effect=pd.errors.ParserError):  # Simulate corrupted file
                result = instance.download_catalog(
                    url=url, filename=filename, local_date=date.today().strftime("%Y-%m-%d"))
        log = pd.DataFrame(list(log), columns=["user", "info", "message"])
    assert(
                    "File "+expected_file_path+" downloaded, but corrupted. Removing file..."
                    in log['message'].to_list()
        )

        # CASE C.1: We can find old file
    assert ("Error fetching the catalog, taking a local copy: oec2024-10-02.csv") in log['message'].to_list()
    assert isinstance(pd.read_csv("oec2024-10-02.csv"), pd.DataFrame)

    os.remove('oec2024-10-02.csv')
    with LogCapture() as log:
        with   patch("requests.get", side_effect=requests.exceptions.ConnectionError):  # Simulate error in download
            with pytest.raises(ConnectionError) as exc_info:
                result = instance.download_catalog(
                    url=url, filename=filename, local_date=date.today().strftime("%Y-%m-%d"))
        log = pd.DataFrame(list(log), columns=["user", "info", "message"])

    # CASE C.2: cannot find alternative
    assert (
                    "The catalog could not be downloaded and there is no backup catalog available." == str(exc_info.value)
        )
    # CASE D: no available catalog at specific date
    with LogCapture() as log:
        with pytest.raises(ValueError) as exc_info:
            result = instance.download_catalog(
                    url=url, filename=filename, local_date='2024-01-01')
        log = pd.DataFrame(list(log), columns=["user", "info", "message"])

    assert (
            "Could not find catalog with this specific date. Please check your date value."  == str(exc_info.value)
    )
    # REPEAT TO CHECK .xml.gz case
    with LogCapture() as log:
        with pytest.raises(ValueError) as exc_info:
            result = instance.download_catalog(
                    url=url+'.gz', filename=filename, local_date='2024-01-01')
        log = pd.DataFrame(list(log), columns=["user", "info", "message"])

    assert (
            "Could not find catalog with this specific date. Please check your date value."  == str(exc_info.value)
    )

    os.chdir(original_dir)



def test__standardize_catalog(instance):
    # Create a sample DataFrame with some additional columns
    data = {
        "alias": [
            "11 UMi,11 Ursae Minoris,Pherkard,Pherkad Minor,HD 136726,HIP 74793,TYC 4414-2315-1,SAO 8207,HR 5714,BD+72 678,2MASS J15170588+7149258,Gaia DR2 1696798367260229376"
        ],
        "name": ["11 UMi b"],
        "binaryflag": [0],
        "mass": [11.2],
        "masstype": ["msini"],
        "mass_min": [0.245],
        "mass_max": [0.245],
        "radius": [np.nan],
        "radius_min": [np.nan],
        "radius_max": [np.nan],
        "period": [516.22],
        "period_min": [3.25],
        "period_max": [3.25],
        "semimajoraxis": [1.54],
        "semimajoraxis_min": [0.07],
        "semimajoraxis_max": [0.07],
        "eccentricity": [0.08],
        "eccentricity_min": [0.03],
        "eccentricity_max": [0.03],
        "periastron": [117.63],
        "longitude": ["None"],
        "ascendingnode": [np.nan],
        "inclination": [np.nan],
        "inclination_min": [np.nan],
        "inclination_max": [np.nan],
        "temperature": [np.nan],
        "age": [np.nan],
        "discoverymethod": ["RV"],
        "discoveryyear": [2009.0],
        "lastupdate": ["15/09/20"],
        "system_rightascension": ["15 17 05.88899"],
        "system_declination": ["+71 49 26.0466"],
        "system_distance": [122.1],
        "hoststar_mass": [np.nan],
        "hoststar_radius": [np.nan],
        "hoststar_metallicity": [np.nan],
        "hoststar_temperature": [np.nan],
        "hoststar_age": [np.nan],
        "hoststar_magJ": [np.nan],
        "hoststar_magI": [np.nan],
        "list": ["Confirmed planets"],
    }

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
        "mass",
        "mass_min",
        "mass_max",
        "msini",
        "msini_min",
        "msini_max",
        "ra",
        "dec",
    ]

    df = pd.DataFrame(data)
    instance.data = df
    with LogCapture() as log:
        instance.standardize_catalog()
        assert "Catalog standardized" in log.actual()[0][-1]

    assert instance.data.at[0, "catalog"] == "oec"
    assert instance.data.at[0, "catalog_name"] == "11 UMi b"
    assert np.isnan(instance.data.at[0, "longitude"])
    assert np.isnan(instance.data.at[0, "mass"])
    assert np.isnan(instance.data.at[0, "mass_min"])
    assert np.isnan(instance.data.at[0, "mass_max"])
    assert instance.data.at[0, "msini"] == 11.2
    assert instance.data.at[0, "msini_min"] == 0.245
    assert instance.data.at[0, "msini_max"] == 0.245

    assert all(element in instance.data.columns for element in expected_columns)
    assert all(
        element in instance.data.at[0, "alias"].split(",")
        for element in [
            "11 UMi",
            "11 Ursae Minoris",
            "Pherkard",
            "Pherkad Minor",
            "HD 136726",
            "HIP 74793",
            "TYC 4414-2315-1",
            "SAO 8207",
            "HR 5714",
            "BD+72 678",
            "2MASS J15170588+7149258",
            "Gaia DR2 1696798367260229376",
        ]
    )
    assert instance.data.at[0, "discovery_method"] == "Radial Velocity"

    data = {
        "name": ["KOI-123.01", "Planet b", "Number9"],
        "alias": ["", "", ""],
        "mass": [11.2, np.nan, 11],
        "masstype": ["msini", "mass", "msini"],
        "mass_min": [0.245, 0.1, 0.1],
        "mass_max": [0.245, 0.1, 0.1],
        "discovery_method": ["RV", np.nan, "nan"],
    }
    df = pd.DataFrame(data)
    instance.data = df
    instance.standardize_catalog()

    assert instance.data.at[0, "host"] == "KOI-123"
    assert instance.data.at[1, "host"] == "Planet"
    assert instance.data.at[2, "host"] == "Number9"
    assert instance.data.at[0, "discovery_method"] == "Radial Velocity"
    assert instance.data.at[1, "discovery_method"] == ""
    assert instance.data.at[2, "discovery_method"] == ""


def test__remove_theoretical_masses(instance):
    assert instance.remove_theoretical_masses() == None


def test__assign_status(instance):
    # Create a sample DataFrame with some additional columns
    data = {
        "name": [
            "11 Oph b",
            "Confirmed planet S-type",
            "Controversial planet",
            "Retracted planet",
            "KOI",
        ],
        "list": [
            "Confirmed planets",
            "Confirmed planets, Planets in binary systems, S-type",
            "Controversial, Planets in binary systems",
            "Retracted planet candidate",
            "Kepler Objects of Interest",
        ],
    }
    df = pd.DataFrame(data)
    instance.data = df
    with LogCapture() as log:
        instance.assign_status()
        assert "Status column assigned" in log.actual()[0][-1]

    assert list(instance.data.status) == [
        "CONFIRMED",
        "CONFIRMED",
        "CANDIDATE",
        "FALSE POSITIVE",
        "CANDIDATE",
    ]


def test__handle_reference_format(instance):
    df = pd.DataFrame(
        {
            "name": ["11 Oph b"],
            "e": [0.231],
            "mass": [np.nan],
            "msini": [19.400],
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
        19.400,
        np.nan,
        1.29,
        326.03,
        np.nan,
        "oec",
        "",
        "oec",
        "",
        "oec",
        "oec",
        "",
    ]
    for d1, d2 in zip(list(instance.data.iloc[0]), expected_result):
        if pd.isna(d1) and pd.isna(d2):
            continue
        assert d1 == d2


def test_convert_coordinates(instance):
    data = {
        "kepid": [10797460, 10811496],
        "kepoi_name": ["K00752.02", "K00753.01"],
        "kepler_name": ["Kepler-227 c", "nan"],
        "koi_disposition": ["CONFIRMED", "CANDIDATE"],
        "ra": ["19h27m44.22s", np.nan],
        "dec": ["+48d08m29.9s", ""],
    }
    data = pd.DataFrame(data)
    instance.data = data
    with LogCapture() as log:
        instance.convert_coordinates()
        assert "Converted coordinates" in log.actual()[0][-1]

    assert np.isclose(instance.data.at[0, "ra"], 291.9333)
    assert np.isclose(instance.data.at[0, "dec"], 48.1417)
    assert np.isnan(instance.data.at[1, "ra"])
    assert np.isnan(instance.data.at[1, "dec"])
