import glob
import logging
import os
import re
from datetime import date
from pathlib import Path
from typing import Union

import numpy as np
import pandas as pd
import requests
import unidecode

from exo_mercat.utility_functions import UtilityFunctions as Utils


class Catalog:
    def __init__(self) -> None:
        """
        The __init__ function is called when the class is instantiated. It sets up the object with a data attribute
        that will be used to store the catalog data, and a name attribute that can be used to refer to this
        particular instance of Catalog.
        """
        self.data = None
        self.name = "catalog"

    def download_catalog(self, url: str, filename: str, local_date: str = '', timeout: float = None) -> Path:
        """
        The download_catalog function downloads the catalog from a given url and saves it to a file.
        If the file already exists, it will not be downloaded again.


        :param url: Specify the url of the catalog to be downloaded
        :param filename: Specify the name of the file to be downloaded
        :param timeout:  Specify the timeout
        :return: The string of the file path of the catalog

        """
        if local_date !='':
            file_path_str = filename + local_date+ ".csv"
            if len(glob.glob(file_path_str)) == 0:
                raise ValueError("Could not find catalog with this specific date. Please check your date value.")
            else:
                logging.info('Reading specific version: '+local_date)
        else:
            file_path_str = filename + date.today().strftime("%m-%d-%Y") + ".csv"
        if os.path.exists(file_path_str):
            logging.info("Reading existing file")
        else:
            try:
                result = requests.get(url, timeout=timeout)
                with open(file_path_str, "wb") as f:
                    f.write(result.content)

            except (
                    OSError,
                    IOError,
                    FileNotFoundError,
                    ConnectionError,
                    ValueError,
                    TypeError,
                    TimeoutError,
                    requests.exceptions.ConnectionError,
                    requests.exceptions.SSLError,
                    requests.exceptions.Timeout,
                    requests.exceptions.ConnectTimeout,
                    requests.exceptions.HTTPError,
            ):
                if len(glob.glob(filename + "*.csv")) > 0:
                    file_path_str = glob.glob(filename + "*.csv")[0]

                    logging.warning(
                        "Error fetching the catalog, taking a local copy: %s",
                        file_path_str,
                    )
                else:
                    raise ValueError("Could not find previous catalogs")
        logging.info("Catalog downloaded.")

        return Path(file_path_str)

    def read_csv_catalog(self, file_path_str: Union[Path, str]) -> None:
        """
        The read_csv_catalog function reads in a csv file and stores it as a pandas dataframe.

        :param file_path_str: Specify the file path of the csv file
        :return: A pandas dataframe
        """
        self.data = pd.read_csv(file_path_str, low_memory=False)

    # def convert_datatypes(self) -> None:
    #     """
    #     The convert_datatypes function converts the data types of all columns in the DataFrame to more memory
    #     efficient types. This is done by iterating through each column and checking if it can be converted
    #     to a more space-efficient type. If so, then that conversion is made.
    #     """
    #     self.data = self.data.convert_dtypes()
    #     logging.info("Converted datatypes.")

    def keep_columns(self) -> None:
        """
        The keep_columns function removes all columns from the dataframe except for those specified in the keep list.
        """
        keep = [
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
        try:
            self.data = self.data[keep]
        # check that all columns exist, otherwise raise an error
        except KeyError:
            raise KeyError("Not all columns exist")
        logging.info("Selected columns to keep.")

    def identify_brown_dwarfs(self) -> None:
        """
        The identify_brown_dwarfs function identifies possible brown dwarfs in the dataframe.
        It does this by checking if the last character of a planet name is a number or if it ends
        with an uppercase letter. If so, it fills the letter cell with 'BD' to filter it out later.
        The function excludes KOI-like objects by avoid the patterns ".0d" with d being a digit.
        """
        for i in self.data.index:
            # known weird candidates
            if "PSR B1257+12" not in self.data.at[i, "name"]:
                if not str(re.search("\\d$", self.data.at[i, "name"], re.M)) == "None":
                    if self.data.at[i, "name"][-3:-1] != ".0":
                        self.data.at[i, "letter"] = "BD"
                if (
                        not str(re.search("[aABCD]$", self.data.at[i, "name"], re.M))
                            == "None"
                ):
                    self.data.at[i, "letter"] = "BD"
                    self.data.at[i, "binary"] = self.data.at[i, "name"][
                                                -1:
                                                ]  # so that we avoid binary systems to get merged

                # 03/27/2024 add special case for problematic triple BD system DENIS J063001.4-184014 (bc)
                # and all those whose name ends with parenthesis
                if  len(re.findall(r'\(.*?\)$', self.data.at[i, "name"]))>0:
                    self.data.at[i, "letter"] = "BD"
                    self.data.at[i, "binary"] = re.findall(r'\(.*?\)$', self.data.at[i, "name"])[0].strip('(').strip(')')

        logging.info("Identified possible Brown Dwarfs (no letter for planet name).")

    def replace_known_mistakes(self) -> None:
        """
        The replace function replaces the values in the dataframe with those specified in replacements.ini
        """
        const = Utils.find_const()
        config_name_for_name = Utils.read_config_replacements("NAMEtochangeNAME")
        config_name_for_host = Utils.read_config_replacements("NAMEtochangeHOST")
        config_host_for_host = Utils.read_config_replacements("HOSTtochangeHOST")

        config_replace = Utils.read_config_replacements("DROP")
        for check, lis in config_replace.items():
            for drop in lis.split(","):
                self.data = self.data[
                    ~(self.data[check].str.contains(drop.strip(), na=False))
                ]

        # check unused replacements
        f = open("Logs/replace_known_mistakes.txt", "a")
        f.write("**** UNUSED REPLACEMENTS FOR " + self.name + " ****\n")

        for name in config_name_for_name.keys():
            if len(self.data[self.data.name == name]) == 0:
                f.write("NAME for NAME: " + name + "\n")
            else:
                self.data.loc[self.data.name == name, "name"] = config_name_for_name[
                    name
                ]

        for name in config_name_for_host.keys():
            if len(self.data[self.data.name == name]) == 0:
                f.write("NAME for HOST: " + name + "\n")
            else:
                self.data.loc[self.data.name == name, "host"] = config_name_for_host[
                    name
                ]

        for host in config_host_for_host.keys():
            if len(self.data[self.data.host == host]) == 0:
                f.write("HOST for HOST: " + host + "\n")
            else:
                self.data.loc[self.data.host == host, "host"] = config_host_for_host[
                    host
                ]

        for coord in ["ra", "dec"]:
            config_replace = Utils.read_config_replacements(coord)
            for name in config_replace.keys():
                if len(self.data.loc[self.data.host == name]) == 0:
                    f.write(coord + ": " + name + "\n")
                else:
                    self.data.loc[self.data.host == name, coord] = float(
                        config_replace[name]
                    )

        for j in self.data.index:
            for i in const.keys():
                if i in self.data.loc[j, "name"]:
                    self.data.loc[j, "name"] = self.data.loc[j, "name"].replace(
                        i, const[i]
                    )
            for i in const.keys():
                if i in self.data.loc[j, "host"]:
                    self.data.loc[j, "host"] = self.data.loc[j, "host"].replace(
                        i, const[i]
                    )

        config_binary = Utils.read_config_replacements("NAMEtochangeBINARY")

        # check unused replacements
        for binary in config_binary.keys():
            if len(self.data[self.data.name == binary]) == 0:
                f.write("BINARY: " + binary + "\n")
            else:
                self.data.loc[self.data.name == binary, "binary"] = config_binary[
                    binary
                ].replace("NaN", "")
        f.close()

        self.data["name"] = self.data["name"].apply(unidecode.unidecode)
        self.data["name"] = self.data.name.apply(lambda x: " ".join(x.split()))
        self.data = self.data.reset_index(drop=True)
        logging.info("Known mistakes replaced.")

    def remove_theoretical_masses(self):
        """
        The remove_theoretical_masses function removes the theoretical masses from the dataframe.
        It is not implemented here because it is catalog-dependent.
        """
        raise NotImplementedError

    def handle_reference_format(self):
        """
        The handle_reference_format function is used to create a url for each reference in the references list.
        It is not implemented here because it is catalog-dependent.
        """
        raise NotImplementedError

    def uniform_catalog(self):
        """
        The uniform_catalog function is used to standardize the dataframe columns and values.
        It is not implemented here because it is catalog-dependent.
        """
        raise NotImplementedError


    def make_errors_absolute(self) -> None:
        """
        The make_errors_absolute function takes in a DataFrame and returns a DataFrame where all the columns related
        to errors are made absolute.
        """
        numerics = ["int16", "int32", "int64", "float16", "float32", "float64"]
        for c in [
            c
            for c in [
                "p_max",
                "a_max",
                "e_max",
                "i_max",
                "r_max",
                "msini_max",
                "mass_max",
                "p_min",
                "a_min",
                "e_min",
                "i_min",
                "r_min",
                "msini_min",
                "mass_min",
            ]
            if self.data[c].dtype in numerics
        ]:
            self.data[c] = self.data[c].abs()
        logging.info("Made all errors absolute values.")

    def uniform_name_host_letter(self) -> None:
        """
        The uniform_name_host_letter function uniforms the names, hosts, and aliases.
        """

        self.data["name"] = self.data.name.apply(lambda x: Utils.uniform_string(x))
        ind = self.data[self.data.host == ""].index
        self.data["host"] = self.data.host.replace("", np.nan).fillna(self.data.name)

        for identifier in self.data.loc[ind, "host"]:
            if not str(re.search("(\\.0)\\d$", identifier, re.M)) == "None":
                self.data.loc[self.data.host == identifier, "host"] = identifier[
                                                                      :-3
                                                                      ].strip()
            if not str(re.search(" [b-z]$", identifier, re.M)) == "None":
                self.data.loc[self.data.host == identifier, "host"] = identifier[
                                                                      :-1
                                                                      ].strip()
        self.data["host"] = self.data.host.apply(lambda x: Utils.uniform_string(x))

        for i in self.data.index:
            polished_alias = ""
            for al in self.data.at[i, "alias"].split(","):
                if not str(re.search(" [b-z]$", al, re.M)) == "None":
                    al = al[:-1]
                if not str(re.search("(\\.0)\\d$", al, re.M)) == "None":
                    al = al[:-3]
                if al != "":
                    polished_alias = (
                            polished_alias
                            + ","
                            + Utils.uniform_string(al.lstrip().rstrip())
                    )
            self.data.at[i, "alias"] = polished_alias.lstrip(",")

        for identifier in self.data.name:
            if not str(re.search("(\\.0)\\d$", identifier, re.M)) == "None":
                self.data.loc[self.data.name == identifier, "letter"] = identifier[-3:]

                # self.data.loc[self.data.name == identifier, "letter"] = (
                #     identifier[-1:]
                #     .replace("1", "b")
                #     .replace("2", "c")
                #     .replace("3", "d")
                #     .replace("4", "e")
                #     .replace("5", "f")
                #     .replace("6", "g")
                #     .replace("7", "h")
                #     .replace("8", "i")
                # )
                # self.data.loc[self.data.name == identifier, "name"] = (
                #     identifier.replace(".01", " b")
                #     .replace(".02", " c")
                #     .replace(".03", " d")
                #     .replace(".04", " e")
                #     .replace(".05", " f")
                #     .replace(".06", " g")
                #     .replace(".07", " h")
                #     .replace(".08", " i")
                # )
            else:
                self.data.loc[self.data.name == identifier, "letter"] = identifier[-1:]
                # self.data.loc[self.data.name == identifier, "host"] = identifier[:-3].strip()

        logging.info("name, host, letter columns uniformed.")

    def assign_status(self):
        """
        The assign_status function assigns a status to each planet based on the status column.
        It is not implemented here because it is catalog-dependent.
        """
        raise NotImplementedError

    def check_mission_tables(self, table_path_str: str) -> None:
        """
        The check_mission_tables function checks the dataframe for any objects that have a name
        that matches an entry in the KOI, TESS or EPIC catalogs. If there is a match, it will update the
        status of that object to whatever status is listed in the KOI, TESS and EPIC catalogs and update its coordinates
        if they are missing from the dataframe.

        :param table_path_str: the string containing path to the table.
        """
        self.data.status = self.data.status.fillna("")
        tab = pd.read_csv(table_path_str)
        tab = tab.fillna("")

        for index in self.data.index:
            name = self.data.at[index, "name"]
            final_alias_total = self.data.at[index, "alias"].split(",")
            sub = tab[tab.aliasplanet.str.contains(name + ",", regex=False)]

            sub = sub.drop_duplicates().reset_index()
            if len(sub) > 0:
                if self.data.at[index, "status"] != sub.at[0, "disposition"]:
                    self.data.at[index, "status"] = sub.at[0, "disposition"]
                # if not list(sub.name)[0] == "":
                #     self.data.at[index, "name"] = sub.at[0, "name"]
                if self.data.at[index, "discovery_method"] in [
                    "nan",
                    "Unknown",
                    "Default",
                ]:
                    self.data.at[index, "discovery_method"] = sub.at[
                        0, "discoverymethod"
                    ]

                for internal_alias in sub.alias:
                    for internal_al in internal_alias.split(","):
                        if internal_al not in final_alias_total:
                            final_alias_total.append(internal_al)

            host = self.data.at[index, "host"]
            letter = self.data.at[index, "letter"]
            # check hosts
            sub = tab[tab.alias.str.contains(host + ",", regex=False)]
            sub = sub[sub.letter == letter]
            sub = sub.drop_duplicates().reset_index()
            if len(sub) > 0:
                if self.data.at[index, "status"] != sub.at[0, "disposition"]:
                    self.data.at[index, "status"] = sub.at[0, "disposition"]
                # if not list(sub.name)[0] == "":
                #     self.data.at[index, "name"] = sub.at[0, "name"]
                if self.data.at[index, "discovery_method"] == "nan":
                    self.data.at[index, "discovery_method"] = sub.at[
                        0, "discoverymethod"
                    ]

                for internal_alias in sub.alias:
                    for internal_al in internal_alias.split(","):
                        internal_al = (
                            Utils.uniform_string(internal_al)
                            .replace(" b", "")
                            .replace(" c", "")
                            .replace(" d", "")
                            .replace(" e", "")
                            .replace(" f", "")
                            .replace(" g", "")
                            .replace(" h", "")
                            .replace(".01", "")
                            .replace(".02", "")
                            .replace(".03", "")
                            .replace(".04", "")
                            .replace(".05", "")
                            .replace(".06", "")
                            .replace(".07", "")
                        )
                        if internal_al not in final_alias_total:
                            final_alias_total.append(internal_al)

            self.data.at[index, "alias"] = ",".join(
                [x for x in set(final_alias_total) if x != "nan"]
            )
        # if there are still empty status strings, use special keyword
        # preliminary i.e. it hasn't been updated yet

        self.data["status"] = self.data.status.replace("", "PRELIMINARY")

        logging.info(table_path_str + " checked.")

    def fill_binary_column(self) -> None:
        """
        The fill_binary_column function fills the binary column of the dataframe with
        the appropriate values. It does this by checking if there is a binary letter at the end of that host
        column. If so, it strips out that letter and puts it into its own column called binary.
        If not, nothing happens.
        """
        self.data = self.data.reset_index()
        self.data["binary"] = ""
        for i in self.data.index:
            #cleanup host if planet name is there
            if len(re.findall(r"([\s\d][b-z])$", self.data.at[i, "host"])) > 0:
                print('planet name in host', self.data.at[i,'host'])
                self.data.at[i, "host"] = self.data.at[i, "host"][:-1].strip()

            if self.data.at[i,'binary']=='':
                # CIRCUMBINARY NAME or HOST
                if (len(re.findall(r"[\s\d](AB)\s[b-z]$", self.data.at[i,'name'])) > 0) or (
                        len(re.findall(r"[\s\d](\(AB\))\s[b-z]$", self.data.at[i,'name'])) > 0) or (len(re.findall(r"[\s\d](AB)$", self.data.at[i,'host'])) > 0) or (
                        len(re.findall(r"[\s\d](\(AB\))$", self.data.at[i,'host'])) > 0) or (
                        len(re.findall(r"[\s\d](AB)$", self.data.at[i,'host'])) > 0):
                    self.data.at[i,'binary'] = "AB"
                    self.data.at[i, 'host'] = self.data.at[i, 'host'].replace('(AB)', '').replace('AB', '').strip()

                # SIMPLE BINARY NAME
                if (
                        len(re.findall(r"[\s\d][ABCNS][\s\d][b-z]$", self.data.at[i,'name']))
                        > 0
                ):
                    self.data.at[i,'binary'] = self.data.at[i,'name'][-3:-2]


                # SIMPLE BINARY HOST
                if len(re.findall(r"[\d\s][ABCSN]$", self.data.at[i,'host'])) > 0:
                    self.data.at[i, 'binary'] =self.data.at[i,'host'][-1:].strip()
                    self.data.at[i,'host'] = self.data.at[i,'host'][:-1].strip()

       #clean the host column
        self.data["host"] = self.data.host.apply(
            lambda x: " ".join(x.strip().strip(".").strip(" (").split())
        )

        if "cb_flag" in self.data.columns:
            # SPECIFIC TO NASA
            self.data.loc[self.data.cb_flag == 1, "binary"] = "AB"

        if "binaryflag" in self.data.columns:
            # SPECIFIC TO OEC
            # if unknown host star, be less specific with S-type, otherwise
            # keep the known letter
            self.data.loc[self.data.binaryflag == 2, "binary"] = self.data.loc[
                self.data.binaryflag == 2, "binary"
            ].replace("", "S-type")

            self.data.loc[self.data.binaryflag == 1, "binary"] = self.data.loc[
                self.data.binaryflag == 1, "binary"
            ].replace("", "AB")
            self.data.loc[self.data.binaryflag == 3, "binary"] = self.data.loc[
                self.data.binaryflag == 3, "binary"
            ].replace("", "Rogue")

        logging.info("Fixed planets orbiting binary stars.")

    def create_catalogstatus_string(self, string: str) -> None:
        """
        The create_catalogstatus_string function creates a new column in the dataframe
        which is a concatenation of the Catalog and status columns. Depending on when it is called,
        it can be either formed by the "original" status provided by the catalog, or the "checked"
        status which is the one EMC picks after checking with the KOI/K2 catalogs.
        """
        self.data[string] = self.data.catalog + ": " + self.data.status.fillna("")
        logging.info(string + " column created.")

    def make_uniform_alias_list(self) -> None:
        """
        The make_uniform_alias_list function takes in a dataframe and returns a list of aliases for each host.
        The function first groups the data by host, then creates a set of all the aliases associated with that
        host. The set is filtered to remove any None or NaN values, as well as removing the host name from this list.
        Finally, it iterates through each row in the groupby object and sets its alias value equal to this new list.
        """
        for host, group in self.data.groupby(by="host"):
            final_alias = ""

            for al in group.alias:
                if al not in [np.nan, "NaN", "nan"]:
                    final_alias = final_alias + "," + al
            self.data.loc[self.data.host == host, "alias"] = ",".join(
                [Utils.uniform_string(x) for x in set(final_alias.split(",")) if x]
            )
        logging.info("Lists of aliases uniformed.")

    def convert_coordinates(self) -> None:
        """
        The coordinates function takes the RA and Dec columns of a dataframe, and converts them to decimal degrees.
        It is not implemented here because it is catalog-dependent.
        """
        raise NotImplementedError

    def fill_nan_on_coordinates(self) -> None:
        """
        The coordinates function takes the RA and Dec columns of a dataframe,
        and converts them to decimal degrees. It replaces any
        missing values with NaN. Finally, it uses SkyCoord to convert from hour angles and
        degrees into decimal degrees.
        Currently only used for Open Exoplanet Catalogue, KOI catalogs.
        UPDATE: EPIC now has degrees already.
        """

        self.data["ra"] = pd.to_numeric(
            self.data.ra.replace("nan", np.nan).replace("", np.nan)
        )
        self.data["dec"] = pd.to_numeric(
            self.data.dec.replace("nan", np.nan).replace("", np.nan)
        )

        logging.info("Filled empty coordinates with nan.")

    def print_catalog(self, filename: Union[str, Path]) -> None:
        """
        The print_cat function prints the dataframe to a csv file.
        It takes one argument, filename, which is the name of the file you want to print it as.

        :param filename: The location of the file to be written


        """
        # self.data = self.data.sort_values(by="exo_mercat_name")
        self.data.to_csv(filename, index=None)
        logging.info("Printed catalog.")
