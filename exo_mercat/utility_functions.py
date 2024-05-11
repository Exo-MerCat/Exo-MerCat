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
    def __init__(self):
        pass

    @staticmethod
    def service_files_initialization() -> None:
        """
        The service_files_initialization function creates the Exo-MerCat, InputSources, UniformedSources and Logs
        folders if they do not exist, and deletes all files in the Logs folder.
        """
        # CREATE OUTPUT FOLDERS
        if not os.path.exists("Exo-MerCat/"):
            os.makedirs("Exo-MerCat")

        if not os.path.exists("InputSources/"):
            os.makedirs("InputSources")

        if not os.path.exists("UniformSources/"):
            os.makedirs("UniformSources")

        # CREATE LOG FOLDER
        if not os.path.exists("Logs/"):
            os.makedirs("Logs")
        os.system("rm Logs/*")

    @staticmethod
    def find_const() -> dict:
        """

        :return: A dictionary containing all possible abbreviations and all possible full names.
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
        The read_config function reads the input_sources.ini file and returns a dictionary of
        the configuration parameters.
        :return a dictionary containing the configuration parameters
        """
        config = configparser.RawConfigParser(
            inline_comment_prefixes="#", delimiters=("=")
        )
        config.read("input_sources.ini")
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
        config = configparser.RawConfigParser(inline_comment_prefixes="#")
        config.optionxform = str
        config.read("replacements.ini")
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
    def round_to_decimal(number: float) -> float:
        """
        Round a number to an appropriate number of decimal places based on its order of magnitude.

        This function takes a numeric input 'number' and rounds it to an appropriate number of
        decimal places based on its order of magnitude. The function calculates the order of
        magnitude of the input number and determines the rounding strategy based on that.

        :param number: The number to be rounded.
        :return: the rounded number
        """
        order_of_magnitude = 10 ** np.floor(np.log10(abs(number)))
        if order_of_magnitude <= 100:
            rounded_number = round(number / order_of_magnitude, 0)
        elif order_of_magnitude <= 1000:
            rounded_number = round(number / order_of_magnitude, 1)
        else:
            rounded_number = round(number / order_of_magnitude, 2)
        return rounded_number * order_of_magnitude

    # @staticmethod
    # def round_array_to_significant_digits(numbers: list) -> list:
    #     """
    #     Round an array of numbers to their significant digits with special handling for zero values.
    #
    #     This function takes an array of numeric inputs 'numbers' and rounds each number to its
    #     appropriate significant digits using the round_to_decimal function. Zero values are
    #     replaced with -1 to distinguish them from rounded numbers.
    #
    #     :param numbers: The array of numbers to be rounded.
    #     :return: The array of rounded numbers with special handling for zero values.
    #     """
    #     rounded_numbers = []
    #     for num in numbers:
    #         if num != 0:
    #             rounded_num = UtilityFunctions.round_to_decimal(num)
    #             rounded_numbers.append(rounded_num)
    #         else:
    #             rounded_numbers.append(-1)
    #     return rounded_numbers
    #

    @staticmethod
    def calculate_working_p_sma(group: pd.DataFrame, tolerance: float) -> pd.DataFrame:
        """difference of 10%"""
        group = group.sort_values(by="p")
        group["working_p"] = np.nan
        group["working_a"] = np.nan
        for i in group.index:
            if group.loc[i, "working_p"] != group.loc[i, "working_p"]:
                group.loc[
                    abs(group.p - group.at[i, "p"]) <= tolerance * group.at[i, "p"],
                    "working_p",
                ] = group.at[i, "p"]
            if group.loc[i, "working_a"] != group.loc[i, "working_a"]:
                group.loc[
                    abs(group.a - group.at[i, "a"]) <= tolerance * group.at[i, "a"],
                    "working_a",
                ] = group.at[i, "a"]
        group.loc[:, "working_a"] = group.loc[:, "working_a"].fillna(-1)
        group.loc[:, "working_p"] = group.loc[:, "working_p"].fillna(-1)
        return group

    # @staticmethod
    # def round_parameter_bin(parameter_series: pd.Series) -> pd.Series:
    #     """
    #     Round values in a pandas Series to bins based on their order of magnitude.
    #
    #     This function takes a pandas Series 'parameter_series' containing numeric values and rounds
    #     each value to a bin based on its order of magnitude. It calculates the order of magnitude
    #     for each value, defines variable bins based on the calculated order of magnitude, and
    #     assigns the corresponding bin label to each value.
    #
    #     :param parameter_series: A pandas Series containing numeric values to be rounded to bins.
    #     :return: A pandas Series with bin labels corresponding to each value's order of magnitude.
    #
    #     """
    #
    #     # Calculate the order of magnitude
    #     order_of_magnitude = np.log10(parameter_series.fillna(0))
    #
    #     # Define variable bins based on the order of magnitude
    #     # You can adjust these bins according to your specific needs
    #     bins = np.linspace(
    #         np.log10(parameter_series.min() * 0.9),
    #         np.log10(1.1 * parameter_series.max()),
    #         300,
    #     )
    #     # bin_labels = [f"[{10 ** bins[i]}, {10 ** bins[i + 1]})" for i in range(len(bins) - 1)]
    #     # Apply pcut with variable bins
    #     return pd.cut(order_of_magnitude, bins=bins, labels=False).fillna(-1)

    @staticmethod
    def get_parameter(treeobject: ElementTree.Element, parameter: str) -> str:
        """
        The getParameter parses a parameter from an XML ElementTree object.

        :param treeobject: An ElementTree object
        :param parameter: a string representing the name of an element in the XML file
        :returns: A string containing the parameter value
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
        The getAttribute function parses the ElementTree object for a parameter and gets the desired attribute.

        :param treeobject: an ElementTree object, which is the root of a parsed XML file
        :param parameter: a string representing the name of an element in the XML file
        :param attrib: a string representing one of that element's attributes
        :returns: A string containing the value of the attribute
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
        The getParameter_all function parses the ElementTree object for a list of parameters.

        :param treeobject: an ElementTree object, which is the root of a parsed XML file
        :param parameter: a string representing the name of an element in the XML file
        :returns: A list of all values in treeobject for the supplied parameter
        """
        # try:

        ret = ",".join([x.text for x in treeobject.iter(parameter)])
        # except BaseException: #pragma: no cover
        #     ret = "" # pragma: no cover
        return ret

    @staticmethod
    def convert_xmlfile_to_csvfile(file_path: Union[Path, str]) -> None:
        """
        The convert_xmlfile_to_csvfile function takes a file path to an XML file and converts it into a CSV file. The
        function uses the gzip library to open the XML files, which are compressed. The ElementTree library is used
        to parse the XML files into tables that can be read by Pandas. A list of fields is created that will be used
        as column headers in the final CSV table. The input_file variable opens up the specified xml_file using gzip
        and reads it in as an object that can be parsed by ElementTree's parse method, which creates a tree structure
        from each element in the xml_file (elements

        :param file_path: The path to the file
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
        input_file = gzip.open(Path(file_path), "r")
        table = ElementTree.parse(input_file)
        tab = pd.DataFrame()

        # read the catalog from XML to Pandas
        for system in table.findall(".//system"):
            planets = system.findall(".//planet")
            stars = system.findall(".//star")

            for planet in planets:
                parameters = pd.DataFrame(
                    [UtilityFunctions.get_parameter(system, "alias")], columns=["alias"]
                )

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
        data["discovery_method"] = (
            data["discovery_method"].fillna("").replace("nan", "")
        )
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

        timeout = 100000
        socket.setdefaulttimeout(timeout)

        if uploads_dict is None:
            uploads_dict = {}
        table = service.run_sync(query, uploads=uploads_dict, timeout=timeout)
        if len(table)>0:
            table = table.to_table().to_pandas()
            # table=table[table.otype.str.contains('\*')] # IF DECOMMENTED, ADD
            # OTYPE BACK IN THE QUERY
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
            table["angsep"] = 0.0 #default value

            table = table[table.main_id != ""]

            return table.reset_index(drop=True)
        else:
            return pd.DataFrame()

    @staticmethod
    def calculate_angsep(table):
        table['ra']=table["ra"].map(
            lambda x: float(x))
        table['dec']=table['dec'].map(
            lambda x: float(x))
        table['ra_2']=table['ra_2'].map(
            lambda x: float(x))
        table['dec_2']=table['dec_2'].map(
            lambda x: float(x))

        for row in table.iterrows():
            r = row[1]
            c1 = SkyCoord(
                float(r["ra"]),
                float(r["dec"]),
                frame="icrs",
                unit=(u.degree, u.degree),
            )
            c2 = SkyCoord(float(r.ra_2), float(r.dec_2), frame="icrs", unit=(u.degree, u.degree))
            angsep = c2.separation(c1).degree
            table.at[row[0], "angsep"] = angsep

        table["selected"] = 0

        table["angsep"] = table["angsep"].map(
            lambda x: np.round(float(x), 8) * 3600
        )

        for hostbin, group in table.groupby("hostbinary"):
            if len(group) > 1:
                # if you find more than one, remove the planet entries
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

        table = table[table.selected == 1]
        return table