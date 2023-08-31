import glob
import logging
import re
from datetime import date
from pathlib import Path
from typing import Union
import os
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

    def download_catalog(self, url: str, filename: str, timeout: float = None) -> Path:
        """
        The download_catalog function downloads the catalog from a given url and saves it to a file.
        If the file already exists, it will not be downloaded again.


        :param url: Specify the url of the catalog to be downloaded
        :param filename: Specify the name of the file to be downloaded
        :param timeout:  Specify the timeout
        :return: The string of the file path of the catalog

        """
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
            ) as e:
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

    def convert_datatypes(self) -> None:
        """
        The convert_datatypes function converts the data types of all columns in the DataFrame to more memory
        efficient types. This is done by iterating through each column and checking if it can be converted
        to a more space-efficient type. If so, then that conversion is made.
        """
        self.data = self.data.convert_dtypes()
        logging.info("Converted datatypes.")

    def keep_columns(self) -> None:
        """
        The keep_columns function removes all columns from the dataframe except for those specified in the keep list.
        """
        keep = [
            "name",
            "catalog_name",
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
            if not "PSR B1257+12" in self.data.at[i, "name"]:  # known weird candidates
                if not str(re.search("\d$", self.data.at[i, "name"], re.M)) == "None":
                    if self.data.at[i, "name"][-3:-1] != ".0":
                        self.data.at[i, "letter"] = "BD"
                if (
                    not str(re.search("[aABCD]$", self.data.at[i, "name"], re.M))
                    == "None"
                ):
                    self.data.at[i, "letter"] = "BD"
        logging.info("Identified possible Brown Dwarfs (no letter for planet name).")

    def replace_known_mistakes(self) -> None:
        """
        The replace function replaces the values in the dataframe with those specified in replacements.ini
        """
        const = Utils.find_const()
        config_name = Utils.read_config_replacements("NAME")
        config_host = Utils.read_config_replacements("HOST")
        # config_hd = read_config_replacements("HD")

        # check unused replacements
        f = open("Logs/unused_replacements.txt", "a")
        f.write("****" + self.name + "****\n")
        for name in config_name.keys():
            if len(self.data[self.data.name == name]) == 0:
                f.write("NAME: " + name + "\n")

        for host in config_host.keys():
            if len(self.data[self.data.host == host]) == 0:
                f.write("HOST: " + host + "\n")
        # for hd in config_hd.keys():
        #     if len(self.data[self.data.name == hd])==0:
        #         f.write('HD: '+ hd+'\n')
        for coord in ["ra", "dec"]:
            config_replace = Utils.read_config_replacements(coord)
            for name in config_replace.keys():
                if len(self.data.loc[self.data.host == name]) == 0:
                    f.write(coord + ": " + name + "\n")
                else:
                    if (
                        abs(
                            self.data.loc[self.data.host == name, coord]
                            - float(config_replace[name])
                        ).all()
                        <= 0.01
                    ):
                        f.write(
                            coord
                            + ": "
                            + name
                            + " "
                            + str(
                                max(
                                    abs(
                                        self.data.loc[self.data.host == name][coord]
                                        - float(config_replace[name])
                                    )
                                )
                            )
                            + "\n"
                        )

        f.close()

        # f = open("Logs/performed_replacements.txt", "a")
        # f.write("****" + self.name + "****\n")
        for j in self.data.index:
            for i in const.keys():
                if i in self.data.loc[j, "name"]:
                    self.data.loc[j, "name"] = self.data.loc[j, "name"].replace(
                        i, const[i]
                    )
                    # f.write("NAME: " + i + " to " + const[i] + "\n")
            for i in const.keys():
                if i in self.data.loc[j, "host"]:
                    self.data.loc[j, "host"] = self.data.loc[j, "host"].replace(
                        i, const[i]
                    )
                    # f.write("HOST: " + i + " to " + const[i] + "\n")
            for i in config_name.keys():
                if i in self.data.loc[j, "name"]:
                    self.data.loc[j, "name"] = self.data.loc[j, "name"].replace(
                        i, config_name[i]
                    )
                    # f.write("NAME: " + i + " to " + config_name[i] + "\n")
            for i in config_host.keys():
                if i in self.data.loc[j, "host"]:
                    self.data.loc[j, "host"] = self.data.loc[j, "host"].replace(
                        i, config_host[i]
                    )
                    # f.write("HOST: " + i + " to " + config_host[i] + "\n")

        for repl_searchname in ["ra", "dec"]:
            config_replace = Utils.read_config_replacements(repl_searchname)
            for name, change in config_replace.items():
                try:
                    self.data.loc[self.data.host == name, repl_searchname] = float(
                        change
                    )
                    # f.write(repl_searchname + ": " + name + " to " + change + "\n")
                except BaseException:
                    pass

        config_replace = Utils.read_config_replacements("DROP")
        for check, lis in config_replace.items():
            for drop in lis.split(","):
                self.data = self.data[
                    ~(self.data[check].str.contains(drop.strip(), na=False))
                ]
                # f.write("DROP: " + check + ":" + drop + "\n")

        # f.close()
        self.data["name"] = self.data["name"].apply(unidecode.unidecode)
        self.data["name"] = self.data.name.apply(lambda x: " ".join(x.split()))
        self.data = self.data.reset_index(drop=True)
        logging.info("Known mistakes replaced.")

    def check_known_binary_mismatches(self) -> None:
        """
        The check_known_binary_mismatches function checks for known mismatches between the binary names in the
        dataframe and those in the config file. If there is a mismatch, it will replace it with what is specified
        in the config file. It also writes to two files: one that lists all of the replacements performed, and
        another that lists all the replacements not used.
        """
        config_binary = Utils.read_config_replacements("BINARY")
        f = open("Logs/unused_replacements.txt", "a")
        # check unused replacements
        for binary in config_binary.keys():
            if len(self.data[self.data.name == binary]) == 0:
                f.write("BINARY: " + binary + "\n")
            else:
                if (
                    config_binary[binary].replace("NaN", "")
                    in self.data[self.data.name == binary].binary.tolist()
                ):
                    f.write("BINARY ALREADY PRESENT: " + binary + "\n")
        f.close()

        # f = open("Logs/performed_replacements.txt", "a")
        for name in config_binary.keys():
            self.data.loc[self.data.name == name, "binary"] = config_binary[
                name
            ].replace("NaN", "")
            # f.write("BINARY: " + name + " to " + config_binary[name] + "\n")
        f.close()

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

    def remove_known_brown_dwarfs(self, print_flag: bool) -> None:
        """
        The remove_known_brown_dwarfs function removes all known brown dwarfs from the dataframe.
        It does this by checking if the mass of a planet is less than 20 Mjup, and if it isn't,
        then it will be removed from the dataframe. If print_flag is set to True, then a csv file will be created
        with all of these planets in them.

        :param print_flag: Specify whether the function should print out a list of brown dwarfs
        """
        if print_flag:
            self.data[
                (
                    self.data.mass.fillna(self.data.msini.fillna(0))
                    .replace("", 0)
                    .astype(float)
                    > 20.0
                )
                #   | (self.data.letter == "BD")
            ].to_csv("UniformSources/" + self.name + "_brown_dwarfs.csv")

        self.data = self.data[
            (
                self.data.mass.fillna(self.data.msini.fillna(0))
                .replace("", 0)
                .astype(float)
                <= 20.0
            )
            #    & (self.data.letter != "BD")
        ]
        self.data[(self.data.letter == "BD")].to_csv(
            "UniformSources/" + self.name + "_possible_brown_dwarfs.csv"
        )

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
            if not str(re.search("(\.0)\\d$", identifier, re.M)) == "None":
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
                if not str(re.search("(\.0)\\d$", al, re.M)) == "None":
                    al = al[:-3]
                if al != "":
                    polished_alias = (
                        polished_alias
                        + ","
                        + Utils.uniform_string(al.lstrip().rstrip())
                    )
            self.data.at[i, "alias"] = polished_alias.lstrip(",")

        for identifier in self.data.name:
            if not str(re.search("(\.0)\\d$", identifier, re.M)) == "None":
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

    def check_koiepic_tables(self, table_path_str: str) -> None:
        """
        The check_epic_table function checks the dataframe for any objects that have a name
        that matches an entry in the KOI or EPIC catalogs. If there is a match, it will update the
        status of that object to whatever status is listed in the KOI and EPIC catalogs and update its coordinates
        if they are missing from the dataframe.

        :param table_path_str: the string containing path to the table.
        """

        tab = pd.read_csv(table_path_str)
        tab = tab.fillna("")

        for index in self.data.index:
            name = self.data.at[index, "name"]
            final_alias_total = self.data.at[index, "alias"].split(",")
            sub = tab[tab.aliasplanet.str.contains(name + ",")]
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
            sub = tab[tab.alias.str.contains(host + ",")]
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

        logging.info("status from " + table_path_str + " checked.")

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
            # protect in case planet name in host column
            if (
                not str(re.search(r"([ABCNLS][\s\d][a-z])$", self.data.at[i, "host"]))
                == "None"
            ):
                self.data.at[i, "host"] = self.data.at[i, "host"][:-1].strip()

            # strip the binary letter and put it in binary column
            if (
                not str(re.search(r"[\s\d][ABCNS]\s", self.data.at[i, "name"]))
                == "None"
            ):
                self.data.at[i, "binary"] = self.data.at[i, "name"][-3:-2]

            # specific for circumbinary planets (AB)
            if (
                not str(re.search(r"[\s\d](AB)\s[a-z]$", self.data.at[i, "name"]))
                == "None"
            ):
                self.data.at[i, "binary"] = "AB"

            if (
                not str(re.search(r"[\s\d](\(AB\))\s[a-z]$", self.data.at[i, "name"]))
                == "None"
            ):
                self.data.at[i, "binary"] = "AB"

            # strip the binary letter and put it in binary column
            if not str(re.search(r"[\s\d][ABCNS]$", self.data.at[i, "host"])) == "None":
                self.data.at[i, "binary"] = self.data.at[i, "host"][-1:]
                self.data.at[i, "host"] = self.data.at[i, "host"][:-1].strip()

            # specific for circumbinary planets (AB)
            if not str(re.search(r"[\s\d](AB)$", self.data.at[i, "host"])) == "None":
                self.data.at[i, "binary"] = "AB"
                self.data.at[i, "host"] = self.data.at[i, "host"][:-3].strip()

            if (
                not str(re.search(r"[\s\d](\(AB\))$", self.data.at[i, "host"]))
                == "None"
            ):
                self.data.at[i, "binary"] = "AB"
                self.data.at[i, "host"] = self.data.at[i, "host"][:-4].strip()

        self.data["host"] = self.data.host.apply(
            lambda x: " ".join(x.strip().strip(".").strip(" (").split())
        )

        if "cb_flag" in self.data.columns:
            # SPECIFIC TO NASA
            self.data.loc[self.data.cb_flag == 1, "binary"] = "AB"

        if "binaryflag" in self.data.columns:
            # SPECIFIC TO OEC
            # if unknown host star, be less specific with S-type, otherwise keep the known letter
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

    def create_catalogstatus_string(self) -> None:
        """
        The create_catalogstatus_string function creates a new column in the dataframe called "Catalogstatus"
        which is a concatenation of the Catalog and status columns. This function is used to create an additional
        column for use in creating plots.
        """
        self.data["Catalogstatus"] = self.data.catalog + ": " + self.data.status
        logging.info("Catalogstatus column created.")

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
