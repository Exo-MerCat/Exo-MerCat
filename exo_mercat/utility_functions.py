import configparser
import gzip
import os
import re
import xml.etree.ElementTree as ElementTree
from pathlib import Path
from typing import Union
import socket

import numpy as np
import pandas as pd
from astropy import units as u
from astropy.coordinates import SkyCoord


class UtilityFunctions:
    """
    This is a class that contains utility functions that can be used in other modules.
    """

    def __init__(self) -> None:
        """
        The __init__ function is called when the class is instantiated.
        :param self: Represent the instance of the class
        :type self: UtilityFunctions
        :return: None
        :rtype: None
        """
        pass

    @staticmethod
    def service_files_initialization() -> None:
        """
        Creates the `Exo-MerCat`, `InputSources`, `UniformedSources`, and `Logs` folders if they do not exist,
        and deletes all files in the `Logs` folder.
        :return: None
        :rtype: None
        """
        # Create Exo-MerCat folder if it does not exist
        if not os.path.exists("Exo-MerCat/"):
            os.makedirs("Exo-MerCat")

        # Create InputSources folder if it does not exist
        if not os.path.exists("InputSources/"):
            os.makedirs("InputSources")

        # Create UniformSources folder if it does not exist
        if not os.path.exists("UniformSources/"):
            os.makedirs("UniformSources")

        # Create Logs folder if it does not exist and remove all files
        if not os.path.exists("Logs/"):
            os.makedirs("Logs")
        os.system("rm Logs/*")

    @staticmethod
    def find_const() -> dict:
        """
        Returns a dictionary containing a mapping of common astronomical constants to their abbreviated forms.

        :return: A dictionary with astronomical constants as keys and their abbreviated forms as values.
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
        Reads the `input_sources.ini` file and returns a dictionary of the configuration parameters.
        :return: A dictionary of the configuration parameters.
        :rtype: dict
        """

        # read the input_sources.ini file
        config = configparser.RawConfigParser(
            inline_comment_prefixes="#", delimiters=("=")
        )
        config.read("input_sources.ini")
        #
        output_dict = {s: dict(config.items(s)) for s in config.sections()}
        return output_dict

    @staticmethod
    def read_config_replacements(section: str) -> dict:
        """
        The read_config_replacements function reads the replacements.ini file and returns a dictionary of
        replacement values for use in the replace_text function.

        :param section: Specify which section of the replacements
        :return: A dictionary containing the custom replacements
        """
        # read the replacements.ini file
        config = configparser.RawConfigParser(inline_comment_prefixes="#")
        config.optionxform = str
        config.read("replacements.ini")
        # return the section as a dictionary
        config_replace = dict(config.items(section))
        return config_replace

    @staticmethod
    def uniform_string(name: str) -> str:
        """
        The uniform_string function takes a string as input and returns the same string with some common formatting
        errors corrected. The function is used to correct for inconsistencies in the naming of exoplanets, which can be
        caused by different sources using different naming conventions.

        :param name: Specify the string to uniform
        :return: The uniformed string
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
        Calculate working parameters 'working_p' and 'working_a' based on the input group DataFrame.

        Sorts the group by column 'p', calculates 'working_p' and 'working_a' values based on tolerance.

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
        Parses a parameter from an XML ElementTree object.

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
        Parses the ElementTree object for a parameter and gets the desired attribute.

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
        """ "
        Parses the ElementTree object for a list of parameters.

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
    def convert_xmlfile_to_csvfile(file_path: Union[Path, str]) -> None:
        """
        Converts an XML file to a CSV file, extracting specific fields from the XML data.

        :param file_path: The file path of the XML file to be converted.
        :type file_path: Union[Path, str]
        :returns: None
        :rtype: None
        """
        #
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

        # Open the input file and parse it as XML
        input_file = gzip.open(Path(file_path), "r")
        table = ElementTree.parse(input_file)
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

        new_file_path = Path(str(file_path[:-6] + "csv"))
        tab.to_csv(new_file_path)

    @staticmethod
    def convert_discovery_methods(data: pd.DataFrame) -> pd.DataFrame:
        """
        Convert the discovery methods in the DataFrame to standardized values.

        :param data: The DataFrame containing the discovery methods.
        :type data: pandas.DataFrame
        :return: The DataFrame with the discovery methods converted.
        :rtype: pandas.DataFrame
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
    def perform_query(service, query, uploads_dict=None):
        """
        Perform a query using the given service and query.

        :param service: The service object used to perform the query.
        :type service: object
        :param query: The query string.
        :type query: str
        :param uploads_dict: A dictionary of uploads. Defaults to None.
        :type uploads_dict: dict, optional

        :return: The result of the query as a DataFrame.
        :rtype: pandas.DataFrame
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

            # if TIC case, treat things differently
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

            # add default angsep if found via name and not by coordinates
            table["angsep"] = 0.0  # default value

            table = table[table.main_id != ""]

            return table.reset_index(drop=True)
        else:
            return pd.DataFrame()

    @staticmethod
    def calculate_angsep(table):
        """
        Calculates the angular separation between two points based on their coordinates.

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
