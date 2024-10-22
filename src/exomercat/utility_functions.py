import configparser
from datetime import datetime
import gzip
import os
import re
import xml.etree.ElementTree as ElementTree
from pathlib import Path
from typing import Union
import socket
import sys
import logging
import glob
import numpy as np
import pandas as pd
import pyvo
from astropy.table import Table

from astropy import units as u
from astropy.coordinates import SkyCoord


class UtilityFunctions:
    """
    A class that contains utility functions that can be used in other modules.
    """

    def __init__(self) -> None:
        """
        Initialize the UtilityFunction class.

        :param self: An instance of the UtilityFunction class
        :type self: UtilityFunctions
        :return: None
        :rtype: None
        """
        pass

    @staticmethod
    def service_files_initialization() -> None:
        """
        Initialize the directory structure for the exoplanet catalog processing.

        Creates the following directories if they don't exist: 'Exo-MerCat/'; 'InputSources/'; 'StandardizedSources/'; 'Logs/'.

        Clears all files in the 'Logs/' directory.

        :return: None
        :rtype: None
        """

        # Create Exo-MerCat folder if it does not exist
        if not os.path.exists("Exo-MerCat/"):
            os.makedirs("Exo-MerCat")

        # Create InputSources folder if it does not exist
        if not os.path.exists("InputSources/"):
            os.makedirs("InputSources")

        # Create StandardizedSources folder if it does not exist
        if not os.path.exists("StandardizedSources/"):
            os.makedirs("StandardizedSources")

        # Create Logs folder if it does not exist and remove all files
        if not os.path.exists("Logs/"):
            os.makedirs("Logs")
        os.system("rm Logs/*")

    @staticmethod
    def ping_simbad_vizier() -> str:
        """
        Test the connection to SIMBAD and VizieR services.

        Attempts to perform a simple query on both SIMBAD and VizieR services
        to check if they are accessible and responding.

        :return: A string containing the status of both connection attempts.
        :rtype: str
        """
        # Test SIMBAD
        error_str = ""
        data = {
            "hostbinary": ["51 Peg"],
            "host": ["51 Peg"],
        }
        list_of_hosts = pd.DataFrame(data)

        service = pyvo.dal.TAPService("http://simbad.cds.unistra.fr/simbad/sim-tap")

        t2 = Table.from_pandas(list_of_hosts)

        query = "SELECT t.*, basic.main_id, basic.ra as ra_2,basic.dec as dec_2, ids.ids as ids FROM TAP_UPLOAD.tab as t LEFT OUTER JOIN ident ON ident.id = t.hostbinary LEFT OUTER JOIN basic ON ident.oidref = basic.oid LEFT OUTER JOIN ids ON basic.oid = ids.oidref"
        # Set socket timeout
        timeout = 100
        socket.setdefaulttimeout(timeout)
        try:
            table = service.run_sync(query, uploads={"tab": t2}, timeout=timeout)
            table = table.to_table()
            if len(table.to_pandas())==1:
                error_str += "Ping to SIMBAD\t\t\tOK. \n"
            else:
                error_str += "Ping to SIMBAD\t\t\tFAILED. \n"
        except:
            error_str += "Ping to SIMBAD\t\t\tFAILED. \n"

        # Test VizieR
        service = pyvo.dal.TAPService("http://TAPVizieR.cds.unistra.fr/TAPVizieR/tap/")
        data = {
            "hostbinary": ["TIC 50365310"],
            "host": ["TIC 50365310"],
        }
        list_of_hosts = pd.DataFrame(data)

        list_of_hosts["host"] = (
            list_of_hosts["host"].str.replace("TIC ", "").str.replace("TIC-", "")
        ).astype(int)
        t2 = Table.from_pandas(list_of_hosts)

        query = 'SELECT tc.*, RAJ2000 as ra_2, DEJ2000 as dec_2, GAIA, UCAC4, "2MASS", WISEA, TIC, KIC, HIP, TYC  FROM "IV/39/tic82" AS db JOIN TAP_UPLOAD.tab AS tc ON db.TIC = tc.host'
        # Set socket timeout
        timeout = 100
        socket.setdefaulttimeout(timeout)

        try:
            table = service.run_sync(query, uploads={"tab": t2}, timeout=timeout)
            table = table.to_table()
            if len(table.to_pandas())==1:
                error_str += "Ping to VizieR\t\t\tOK."
            else:
                error_str += "Ping to VizieR\t\t\tFAILED."
        except:
            error_str += "Ping to VizieR\t\t\tFAILED."

        return error_str

    @staticmethod
    def get_common_nomenclature() -> dict:
        """
        Provide a mapping of astronomical constants and abbreviations.

        :return: A dictionary mapping full names to abbreviated forms for various
                 astronomical terms, constellations, and catalog prefixes.
        :rtype: dict
        """

        constants = {
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

        return constants

    @staticmethod
    def read_config() -> dict:
        """
        Read and parse the 'input_sources.ini' configuration file.

        :return: A dictionary of the configuration parameters.
        :rtype: dict
        """

        # Read the input_sources.ini file
        config = configparser.RawConfigParser(
            inline_comment_prefixes="#", delimiters=("=")
        )
        config.read("input_sources.ini")

        # Save into dictionary
        output_dict = {s: dict(config.items(s)) for s in config.sections()}
        return output_dict

    @staticmethod
    def read_config_replacements(section: str) -> dict:
        """
        Read and parse a specific section of the 'replacements.ini' configuration file.

        :param section: Specify which section of the replacements
        :type section: str
        :return: A dictionary containing the custom replacements
        :rtype: dict
        """

        # Read the replacements.ini file
        config = configparser.RawConfigParser(inline_comment_prefixes="#")
        config.optionxform = str
        config.read("replacements.ini")
        # Return the section as a dictionary
        config_replace = dict(config.items(section))
        return config_replace

    @staticmethod
    def standardize_string(name: str) -> str:
        """
        Standardize the format of exoplanet and star names.

        Applies various rules to correct common formatting inconsistencies in
        exoplanet and star naming conventions.

        :param name: Specify the string to standardize
        :type name: str
        :return: The standardized string
        :rtype: str
        """

        name = name.replace("'", "").replace('"', "")
        if "K0" in name[:2]:
            name = "KOI-" + name.lstrip("K").lstrip("0")
        if "TOI " in name[:4]:
            name = "TOI-" + name[4:].lstrip(" ")
        if not str(re.match("2M[\\d ]", name, re.M)) == "None":
            name = "2MASS J" + name[2:].lstrip()
            name = name.replace("JJ", "J").replace("J ", "J")
        if "Gliese" in name:
            name = name.replace("Gliese ", "GJ ")
        if not str(re.match("VHS \\d", name, re.M)) == "None":
            name = name.replace("VHS ", "VHS J")
        if "Gl " in name:
            name = name.replace("Gl ", "GJ ")
        if "KMT-" in name:
            name = name.rstrip("L").replace(":", "-")
        if "MOA-" in name:
            name = name.replace("MOA-", "MOA ").rstrip("L")
        if "OGLE--" in name:
            name = name.replace("OGLE--", "OGLE ").rstrip("L")
        if "OGLE" in name:
            name = name.replace("OGLE-", "OGLE ").rstrip("L")
        if "KMT-" in name:
            name = name.split("/")[0]
        # if "CoRoT-" in name:
        #     name = name.replace("-", " ")
        if "2MASS" in name:
            name = name.rstrip(" a")
        return name

    @staticmethod
    def calculate_working_p_sma(group: pd.DataFrame, tolerance: float) -> pd.DataFrame:
        """
        Calculate working period and semi-major axis values for a group of exoplanets.
        
        :param group: The input DataFrame containing columns 'p' and 'a'.
        :type group: pd.DataFrame
        :param tolerance: The tolerance factor used in calculations.
        :type tolerance: float
        :return: The DataFrame with 'working_p' and 'working_a' values calculated.
        :rtype: pd.DataFrame
        """

        # Sort the group by column 'p'
        group = group.sort_values(by="p")
        # Initialize 'working_p' and 'working_a' columns with NaN values
        group["working_p"] = np.nan
        group["working_a"] = np.nan

        for i in group.index:
            # Calculate 'working_p' based on tolerance
            if group.loc[i, "working_p"] != group.loc[i, "working_p"]:
                group.loc[
                    abs(group.p - group.at[i, "p"]) <= tolerance * group.at[i, "p"],
                    "working_p",
                ] = group.at[i, "p"]

            # Calculate 'working_a' based on tolerance
            if group.loc[i, "working_a"] != group.loc[i, "working_a"]:
                group.loc[
                    abs(group.a - group.at[i, "a"]) <= tolerance * group.at[i, "a"],
                    "working_a",
                ] = group.at[i, "a"]

        # Fill NaN values in 'working_a' and 'working_p' with -1
        group.loc[:, "working_a"] = group.loc[:, "working_a"].fillna(-1)
        group.loc[:, "working_p"] = group.loc[:, "working_p"].fillna(-1)
        
        return group

    @staticmethod
    def get_parameter(treeobject: ElementTree.Element, parameter: str) -> str:
        """
        Extract a parameter value from an XML ElementTree object.

        :param treeobject: An ElementTree object.
        :type treeobject: ElementTree.Element
        :param parameter: A string representing the name of an element in the XML file.
        :type parameter: str
        :returns: A string containing the parameter value.
        :rtype: str
        """
        if parameter == "alias":
            alias = treeobject.findall("*/name")
            ret = ",".join([a.text for a in alias])
        else:
            try:
                ret = treeobject.findtext("./" + parameter).strip()
            except BaseException:
                ret = ""
        return ret

    @staticmethod
    def get_attribute(
        treeobject: ElementTree.Element, parameter: str, attrib: str
    ) -> str:
        """
        Extract an attribute value from a specific element in an XML ElementTree object.

        :param treeobject: An ElementTree object, which is the root of a parsed XML file.
        :type treeobject: ElementTree.Element
        :param parameter: A string representing the name of an element in the XML file.
        :type parameter: str
        :param attrib: A string representing one of that element's attributes.
        :type attrib: str
        :returns: A string containing the value of the attribute.
        :rtype: str
        """
        if (
            treeobject.find("./" + parameter) is not None
            and attrib in treeobject.find("./" + parameter).attrib
        ):
            return treeobject.find("./" + parameter).attrib[attrib]
        else:
            return ""

    @staticmethod
    def get_parameter_all(treeobject: ElementTree.Element, parameter: str) -> str:
        """
        Extract all occurrences of a parameter from an XML ElementTree object.

        :param treeobject: An ElementTree object, which is the root of a parsed XML file.
        :type treeobject: ElementTree.Element
        :param parameter: A string representing the name of an element in the XML file.
        :type parameter: str
        :returns: A string containing all values in `treeobject` for the supplied `parameter`.
        :rtype: str
        """

        ret = ",".join([x.text for x in treeobject.iter(parameter)])

        return ret

    @staticmethod
    def convert_xmlfile_to_csvfile(
        file_path: Union[Path, str], output_file: str
    ) -> None:
        """
        Convert an XML file containing exoplanet data to a CSV file.
      
        :param file_path: The file path of the XML file to be converted.
        :type file_path: Union[Path, str]
        :returns: None
        :rtype: None
        """
        
        fields = [
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
        if ".xml.gz" in file_path:
            input_file = gzip.open(Path(file_path), "r")
            table = ElementTree.parse(input_file)
        else:  
            # if it ends in .xml. Strings ending in other extensions are forbidden in parent function
            table = ElementTree.parse(file_path)

        # Create an empty DataFrame to store the extracted data
        tab = pd.DataFrame()

        # Iterate over each system in the XML file
        for system in table.findall(".//system"):
            # Find all planets and stars in the system
            planets = system.findall(".//planet")
            stars = system.findall(".//star")

            # Iterate over each planet in the system
            for planet in planets:
                parameters = pd.DataFrame(
                    [UtilityFunctions.get_parameter(system, "alias")], columns=["alias"]
                )

                # Extract the specified fields from the planet element
                for field in fields:
                    parameters[field] = None
                    parameters[field] = UtilityFunctions.get_parameter(planet, field)
                    parameters.alias = UtilityFunctions.get_parameter(system, "alias")

                    if field[0:7] == "system_":
                        parameters[field] = UtilityFunctions.get_parameter(
                            system, field[7:]
                        )
                    elif field[0:9] == "hoststar_":
                        parameters[field] = UtilityFunctions.get_parameter(
                            stars, field[9:]
                        )
                    elif field == "list":
                        parameters[field] = UtilityFunctions.get_parameter_all(
                            planet, field
                        )
                    elif field == "masstype":
                        parameters[field] = UtilityFunctions.get_attribute(
                            planet, field[0:-4], "type"
                        )
                    elif field[-4:] == "_min":
                        parameters[field] = UtilityFunctions.get_attribute(
                            planet, field[0:-4], "errorminus"
                        )
                    elif field[-4:] == "_max":
                        parameters[field] = UtilityFunctions.get_attribute(
                            planet, field[0:-4], "errorplus"
                        )

                parameters.binaryflag = 0
                if planet in system.findall(".//binary/planet"):
                    # P type planets
                    parameters.binaryflag = 1
                if planet in system.findall(".//binary/star/planet"):
                    # S type planets
                    parameters.binaryflag = 2
                if len(stars) == 0:
                    # rogue planets
                    parameters.binaryflag = 3

                tab = pd.concat([tab, parameters], sort=False)

        tab.to_csv(output_file)

    @staticmethod
    def convert_discovery_methods(data: pd.DataFrame) -> pd.DataFrame:
        """
        Convert the discovery methods in the DataFrame to standardized values.

        :param data: The DataFrame containing the discovery methods.
        :type data: pd.DataFrame
        :return: The DataFrame with the discovery methods converted.
        :rtype: pd.DataFrame
        """
        # Fill missing values with empty string and replace "nan" with empty string
        data["discovery_method"] = (
            data["discovery_method"].fillna("").replace("nan", "")
        )
        # Convert specific discovery methods to standardized values
        data.loc[
            data.discovery_method == "Primary Transit#TTV", "discovery_method"
        ] = "TTV"
        data.loc[
            data.discovery_method == "Transit Timing Variations", "discovery_method"
        ] = "TTV"
        data.loc[
            data.discovery_method == "Eclipse Timing Variations", "discovery_method"
        ] = "TTV"
        data.loc[
            data.discovery_method == "Primary Transit", "discovery_method"
        ] = "Transit"
        data.loc[
            data.discovery_method == "Pulsar", "discovery_method"
        ] = "Pulsar Timing"
        data.loc[
            data.discovery_method == "Pulsation Timing Variations", "discovery_method"
        ] = "Pulsar Timing"
        data.loc[
            data.discovery_method == "Timing", "discovery_method"
        ] = "Pulsar Timing"
        data.loc[
            data.discovery_method == "disk kinematics", "discovery_method"
        ] = "Other"
        data.loc[data.discovery_method == "Kinematic", "discovery_method"] = "Other"
        data.loc[
            data.discovery_method == "Disk Kinematics", "discovery_method"
        ] = "Other"
        data.loc[
            data.discovery_method == "Orbital Brightness Modulation", "discovery_method"
        ] = "Other"
        data.loc[
            data.discovery_method == "astrometry", "discovery_method"
        ] = "Astrometry"
        data.loc[
            data.discovery_method == "microlensing", "discovery_method"
        ] = "Microlensing"
        data.loc[data.discovery_method == "imaging", "discovery_method"] = "Imaging"
        data.loc[data.discovery_method == "transit", "discovery_method"] = "Transit"
        data.loc[
            data.discovery_method == "timing", "discovery_method"
        ] = "Pulsar Timing"
        data.loc[data.discovery_method == "RV", "discovery_method"] = "Radial Velocity"

        return data

    @staticmethod
    def perform_query(service, query, uploads_dict=None) -> pd.DataFrame:
        """
        Perform a query using the given service and query.

        :param service: The service object used to perform the query.
        :type service: object
        :param query: The query string.
        :type query: str
        :param uploads_dict: A dictionary of uploads. Defaults to None.
        :type uploads_dict: dict, optional

        :return: The result of the query as a DataFrame.
        :rtype: pd.DataFrame
        """
        # Set socket timeout
        timeout = 100000
        socket.setdefaulttimeout(timeout)

        # Perform the query
        if uploads_dict is None:
            uploads_dict = {}
        table = service.run_sync(query, uploads=uploads_dict, timeout=timeout)

        # Convert the table to a DataFrame
        if len(table) > 0:
            table = table.to_table().to_pandas()
            # table=table[table.otype.str.contains('\*')] # IF DECOMMENTED, ADD
            # OTYPE BACK IN THE QUERY

            # If TIC case, treat things differently
            if "TIC" in table.columns:
                table = table.astype(str)
                table["main_id"] = "TIC " + table["TIC"].replace("", "<NA>")
                table["UCAC4"] = "UCAC4 " + table["UCAC4"].replace("", "<NA>")
                table["2MASS"] = "2MASS J" + table["2MASS"].replace("", "<NA>")
                table["WISEA"] = "WISE " + table["WISEA"].replace("", "<NA>")
                table["GAIA"] = "Gaia DR2 " + table["GAIA"].replace("", "<NA>")
                table["KIC"] = "KIC " + table["KIC"].replace("", "<NA>")
                table["HIP"] = "HIP " + table["HIP"].replace("", "<NA>")
                table["TYC"] = "TYC " + table["TYC"].replace("", "<NA>")

                for col in table.columns:
                    table.loc[table[col].str.contains("<NA>"), col] = ""
                    table["ids"] = table[
                        ["UCAC4", "2MASS", "WISEA", "GAIA", "KIC", "HIP", "TYC"]
                    ].agg(",".join, axis=1)
                    table["ids"] = table["ids"].map(lambda x: x.lstrip(",").rstrip(","))

            # Add default angsep if found via name and not by coordinates
            table["angsep"] = 0.0  # default value

            table = table[table.main_id != ""]
            table = table.applymap(lambda x: x.strip() if isinstance(x, str) else x)

            return table.reset_index(drop=True)
        else:
            return pd.DataFrame()

    @staticmethod
    def calculate_angsep(table) -> pd.DataFrame:
        """
        Calculate angular separations between coordinates in a DataFrame.

        :param table: A pandas DataFrame containing columns 'ra', 'dec', 'ra_2', 'dec_2'.
        :type table: pd.DataFrame
        :return: A modified DataFrame with the angular separation calculated and selected rows.
        :rtype: pd.DataFrame
        """
        # Convert ra and dec columns to float
        table["ra"] = table["ra"].map(lambda x: float(x))
        table["dec"] = table["dec"].map(lambda x: float(x))
        table["ra_2"] = table["ra_2"].map(lambda x: float(x))
        table["dec_2"] = table["dec_2"].map(lambda x: float(x))

        # Calculate angular separation
        for row in table.iterrows():
            r = row[1]
            c1 = SkyCoord(
                float(r["ra"]),
                float(r["dec"]),
                frame="icrs",
                unit=(u.degree, u.degree),
            )
            c2 = SkyCoord(
                float(r.ra_2), float(r.dec_2), frame="icrs", unit=(u.degree, u.degree)
            )
            angsep = c2.separation(c1).degree
            table.at[row[0], "angsep"] = angsep

        # Initialize 'selected' column
        table["selected"] = 0

        # Convert 'angsep' values and scale
        table["angsep"] = table["angsep"].map(lambda x: np.round(float(x), 8) * 3600)

        # Group by 'hostbinary' and select closest entry
        for hostbin, group in table.groupby("hostbinary"):
            if len(group) > 1:
                # If more than one entry, remove planet entries
                for i in group.index:
                    if (
                        str(re.search("[\\s\\d][b-i]$", group.main_id[i], re.M))
                        != "None"
                    ):
                        group = group.drop(i)

                selected = group[group.angsep == min(group.angsep)].head(1)

            else:
                selected = group.copy()
            table.loc[selected.index, "selected"] = 1

        # Filter out unselected rows
        table = table[table.selected == 1]
        return table

    def load_standardized_catalog(filename: str, local_date: str) -> pd.DataFrame:
        """
        Load a standardized catalog file for a given date. If not found,
        it falls back to the most recent available version.

        :param filename: The base filename of the catalog.
        :type filename: str
        :param local_date: The date for which to load the catalog (format: YYYY-MM-DD).
        :type local_date: str
        :return: A DataFrame containing the loaded catalog data.
        :rtype: pd.DataFrame
        :raises ValueError: If no suitable catalog file can be found.
        """
        file_path_str = filename + local_date + ".csv"
        # File already exists
        if os.path.exists(file_path_str):
            logging.info("Reading existing standardized file: " + file_path_str)
        else:
            # Cannot find, try most recent local copy
            if len(glob.glob(filename + "*.csv")) > 0:
                li = list(glob.glob(filename + "*.csv"))
                li = [re.search(r"\d\d\d\d-\d\d-\d\d", l)[0] for l in li]
                li = [datetime.strptime(l, "%Y-%m-%d") for l in li]

                # Get the most recent compared to the current date. Get only the ones earlier than the date
                local_date_datetime = datetime.strptime(
                    re.search(r"\d\d\d\d-\d\d-\d\d", file_path_str)[0], "%Y-%m-%d"
                )
                li = [l for l in li if l < local_date_datetime]
                compar_date = max(li).strftime("%Y-%m-%d")
                file_path_str = filename + compar_date + ".csv"

                logging.warning(
                    "Error fetching the standardized catalog, taking a local copy: %s",
                    file_path_str,
                )
            else:
                # Cannot find any previous copy
                raise ValueError(
                    "Could not find catalog with this specific date. Please check your date value."
                )

        return pd.read_csv(file_path_str)

    @staticmethod
    def print_progress_bar(iteration, total, prefix="", suffix="", length=50, fill="█"):
        """
        Print a progress bar to the console.

        :param iteration: Current iteration (Int).
        :param total: Total iterations (Int).
        :param prefix: Prefix string (Str).
        :param suffix: Suffix string (Str).
        :param length: Character length of bar (Int).
        :param fill: Bar fill character (Str).
        """
        percent = ("{0:.1f}").format(100 * (iteration / float(total)))
        filled_length = int(length * iteration // total)
        bar = fill * filled_length + "-" * (length - filled_length)
        sys.stdout.write(f"\r{prefix} |{bar}| {percent}% {suffix}")
        sys.stdout.flush()  # Flush to ensure it prints out immediately
