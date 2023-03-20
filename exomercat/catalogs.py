import glob
import re
import pandas as pd
import numpy as np
from exomercat.configurations import *
from datetime import date
from pathlib import Path
from typing import Union
import unidecode
import logging


def uniform_string(name: str) -> str:
    """
    The uniform_string function takes a string as input and returns the same string with some common formatting
    errors corrected. The function is used to correct for inconsistencies in the naming of exoplanets, which can be
    caused by different sources using different naming conventions.

    Parameters
    ----------
        name: str
            Specify the string to uniform

    """
    name = name.replace("'", "").replace('"', "")
    if "K0" in name[2:]:
        name = "KOI-" + name.lstrip("K").lstrip("0")
    if not str(re.match("2M[\d ]", name, re.M)) == "None":
        name = "2MASS J" + name[2:].lstrip()
        name = name.replace("JJ", "J").replace("J ", "J")
    if "Gliese" in name:
        name = name.replace("Gliese ", "GJ ")
    if not str(re.match("VHS \d", name, re.M)) == "None":
        name = name.replace("VHS ", "VHS J")
    if "Gl " in name:
        name = name.replace("Gl  ", "GJ ")
    if "KMT-" in name:
        name = name.rstrip("L")
    if "MOA-" in name:
        name = name.replace("MOA-", "MOA ").rstrip("L")
    if "OGLE" in name:
        name = name.replace("OGLE-", "OGLE ").rstrip("L")
    if "KMT-" in name:
        name = name.split("/")[0]
    if "CoRoT-" in name:
        name = name.replace("-", " ")
    if "2MASS" in name:
        name = name.rstrip(" a")
    return name


class Catalog:
    def __init__(self) -> None:
        """
        The __init__ function is called when the class is instantiated. It sets up the object with a data attribute
        that will be used to store the catalog data, and a name attribute that can be used to refer to this
        particular instance of Catalog.
        """
        self.data = None
        self.name = "catalog"

    def download_and_save_cat(self, url: str, filename: str) -> None:
        """
        The download_and_save_cat function downloads a catalog from the web and saves it to disk.
        It takes two arguments: url, which is the URL of the catalog to be downloaded, and filename,
        which is where we want to save that file locally. It returns nothing.

        Parameters
        ----------
            self
                Allow the function to refer to and modify a class instance's attributes
            url:str
                Specify the url of the catalog to download
            filename:str
                Specify the name of the file where we want to store our data
        """
        if os.path.exists(filename + date.today().strftime("%m-%d-%Y") + ".csv"):
            logging.info("Reading existing file")
            self.data = pd.read_csv(
                filename + date.today().strftime("%m-%d-%Y") + ".csv", low_memory=False
            )
        else:
            try:
                os.system(
                    'wget "'
                    + url
                    + '" -O "'
                    + filename
                    + date.today().strftime("%m-%d-%Y")
                    + '.csv"'
                )
                self.data = pd.read_csv(
                    filename + date.today().strftime("%m-%d-%Y") + ".csv"
                )
            except BaseException:
                local_copy = glob.glob(filename + "*.csv")[0]
                logging.warning(
                    "Error fetching the catalog, taking a local copy:", local_copy
                )
                self.data = pd.read_csv(local_copy)

    def convert_datatypes(self) -> None:
        """
        The convert_datatypes function converts the data types of all columns in the DataFrame to more memory
        efficient types. This is done by iterating through each column and checking if it can be converted
        to a more space-efficient type. If so, then that conversion is made.
        """
        self.data = self.data.convert_dtypes()
        logging.info("Converted datatypes.")

    def read_csv_catalog(self, filename: str) -> None:
        """
        The read_csv_catalog function reads a csv file and stores the data in a pandas DataFrame.

        Parameters
        ----------
            self
                Access variables that belongs to the class
            filename:str
                Specify the name of the file that will be read

        """
        self.data = pd.read_csv(filename)

    def keep_columns(self) -> None:
        """
        The keep_columns function removes all columns from the dataframe except for those specified in the keep list.
        """
        keep = [
            "Name",
            "DiscMeth",
            "ra",
            "dec",
            "P",
            "P_max",
            "P_min",
            "a",
            "a_max",
            "a_min",
            "e",
            "e_max",
            "e_min",
            "i",
            "i_max",
            "i_min",
            "Mass",
            "Mass_max",
            "Mass_min",
            "Msini",
            "Msini_max",
            "Msini_min",
            "R",
            "R_max",
            "R_min",
            "YOD",
            "alias",
            "a_url",
            "Mass_url",
            "P_url",
            "Msini_url",
            "R_url",
            "i_url",
            "e_url",
            "Host",
            "Binary",
            "Letter",
            "Status",
            "catalog",
        ]
        self.data = self.data[keep]
        logging.info("Selected columns to keep.")

    def replace_known_mistakes(self) -> None:
        """
        The replace function replaces the values in the dataframe with those specified in replacements.ini
        """
        const = find_const()
        config_name = read_config_replacements("NAME")
        config_host = read_config_replacements("HOST")
        config_hd = read_config_replacements("HD")
        config_binary = read_config_replacements("BINARY")
        for j in self.data.index:
            for i in const.keys():
                if i in self.data.loc[j, "Name"]:
                    self.data.loc[j, "Name"] = self.data.loc[j, "Name"].replace(
                        i, const[i]
                    )
            for i in const.keys():
                if i in self.data.loc[j, "Host"]:
                    self.data.loc[j, "Host"] = self.data.loc[j, "Host"].replace(
                        i, const[i]
                    )
            for i in config_name.keys():
                if i in self.data.loc[j, "Name"]:
                    self.data.loc[j, "Name"] = self.data.loc[j, "Name"].replace(
                        i, config_name[i]
                    )
            for i in config_host.keys():
                if i in self.data.loc[j, "Host"]:
                    self.data.loc[j, "Host"] = self.data.loc[j, "Host"].replace(
                        i, config_host[i]
                    )
            for i in config_hd.keys():
                if i in self.data.loc[j, "Name"]:
                    self.data.loc[j, "Name"] = self.data.loc[j, "Name"].replace(
                        i, config_hd[i]
                    )
            for i in config_binary.keys():
                # does that only the second time it runs
                if i in self.data.loc[j, "Name"] and "Binary" in self.data.columns:
                    self.data.loc[j, "Binary"] = config_binary[i].replace("NaN", "")

        for repl_searchname in ["ra", "dec"]:
            config_replace = read_config_replacements(repl_searchname)
            for name, change in config_replace.items():
                try:
                    self.data.loc[self.data.Host == name, repl_searchname] = float(
                        change
                    )
                except BaseException:
                    pass

        config_replace = read_config_replacements("DROP")
        for check, lis in config_replace.items():
            for drop in lis.split(","):
                self.data = self.data[
                    ~(self.data[check].str.contains(drop.strip(), na=False))
                ]

        self.data["Name"] = self.data["Name"].apply(unidecode.unidecode)
        self.data["Name"] = self.data.Name.apply(lambda x: " ".join(x.split()))
        self.data = self.data.reset_index(drop=True)
        logging.info("Known mistakes replaced.")

    def remove_theoretical_masses(self):
        """
        The remove_theoretical_masses function removes the theoretical masses from the dataframe.
        It is not implemented here because it is catalog-dependent.
        """
        return NotImplementedError

    def handle_reference_format(self):
        """
        The handle_reference_format function is used to create a url for each reference in the references list.
        It is not implemented here because it is catalog-dependent.
        """
        return NotImplementedError

    def uniform_catalog(self):
        """
        The uniform_catalog function is used to standardize the dataframe columns and values.
        It is not implemented here because it is catalog-dependent.
        """
        return NotImplementedError

    def remove_known_brown_dwarfs(self, print: bool) -> None:
        """
        The remove_known_brown_dwarfs function removes all known brown dwarfs from the dataframe.
        It does this by checking if the mass of a planet is less than 20 Mjup, and if it isn't,
        then it will be removed from the dataframe. If print is set to True, then a csv file will be created
        with all of these planets in them.

        Parameters
        ----------
            print: bool
                Specify whether the function should print out a list of brown dwarfs
        """
        if print:
            self.data[
                self.data.Mass.fillna(self.data.Msini.fillna(0))
                .replace("", 0)
                .astype(float)
                > 20.0
            ].to_csv("UniformSources/" + self.name + "_brown_dwarfs.csv")
        self.data = self.data[
            self.data.Mass.fillna(self.data.Msini.fillna(0))
            .replace("", 0)
            .astype(float)
            <= 20.0
        ]

    def make_errors_absolute(self) -> None:
        """
        The make_errors_absolute function takes in a DataFrame and returns a DataFrame where all the columns related
        to errors are made absolute.
        """
        numerics = ["int16", "int32", "int64", "float16", "float32", "float64"]
        for c in [
            c
            for c in [
                "P_max",
                "a_max",
                "e_max",
                "i_max",
                "R_max",
                "Msini_max",
                "Mass_max",
                "P_min",
                "a_min",
                "e_min",
                "i_min",
                "R_min",
                "Msini_min",
                "Mass_min",
            ]
            if self.data[c].dtype in numerics
        ]:
            self.data[c] = self.data[c].abs()
        logging.info("Made all errors absolute values.")

    def uniform_name_host_letter(self) -> None:
        """
        The uniform_name_host_letter function uniforms the names, hosts, and aliases.
        """

        self.data["Name"] = self.data.Name.apply(lambda x: uniform_string(x))
        ind = self.data[self.data.Host == ""].index
        self.data["Host"] = self.data.Host.replace("", np.nan).fillna(self.data.Name)

        for identifier in self.data.loc[ind, "Host"]:
            if not str(re.search(" [b-z]$", identifier, re.M)) == "None":
                self.data.loc[self.data.Host == identifier, "Host"] = identifier[
                    :-1
                ].strip()
        self.data["Host"] = self.data.Host.apply(lambda x: uniform_string(x))

        for i in self.data.index:
            polished_alias = ""
            for al in self.data.at[i, "alias"].split(","):
                if al != "":
                    polished_alias = polished_alias + "," + uniform_string(al)
            self.data.at[i, "alias"] = polished_alias.lstrip(",")

        self.data["Letter"] = self.data.apply(lambda row: row.Name[-1:].strip(), axis=1)

        for identifier in self.data.Name:
            if not str(re.search("(\.0)\\d$", identifier, re.M)) == "None":
                self.data.loc[self.data.Name == identifier, "Letter"] = (
                    identifier[-1:]
                    .replace("1", "b")
                    .replace("2", "c")
                    .replace("3", "d")
                    .replace("4", "e")
                    .replace("5", "f")
                    .replace("6", "g")
                    .replace("7", "h")
                    .replace("8", "i" "" "")
                )
                # self.data.loc[self.data.Name == identifier, "Host"] = identifier[:-3].strip()
            elif str(re.search(" [b-i]$", identifier, re.M)) == "None":
                self.data.loc[self.data.Name == identifier, "Letter"] = "b"

        for i in self.data.index:
            alias_polished = ""
            for al in self.data.at[i, "alias"].split(","):
                al = re.sub(".0\d$", "", al.rstrip())
                al = re.sub(" [b-i]$", "", al.rstrip())
                al = re.sub("^K0", "KOI-", al.lstrip())
                alias_polished = alias_polished + "," + al.rstrip()

            self.data.at[i, "alias"] = alias_polished.lstrip(",")
        logging.info("Name, Host, Letter columns uniformed.")

    def assign_status(self):
        """
        The assign_status function assigns a status to each planet based on the status column.
        It is not implemented here because it is catalog-dependent.
        """
        return NotImplementedError

    def check_koiepic_tables(self, table_path: str) -> None:
        """
        The check_epic_table function checks the dataframe for any objects that have a name
        that matches an entry in the KOI or EPIC catalogs. If there is a match, it will update the
        status of that object to whatever status is listed in the KOI and EPIC catalogs and update its coordinates
        if they are missing from the dataframe.
        """

        tab = pd.read_csv(table_path)
        tab = tab.fillna("")

        for index in self.data.index:
            name = self.data.at[index, "Name"]
            final_alias_total = self.data.at[index, "alias"].split(",")

            sub = tab[tab.aliasplanet.str.contains(name)]
            sub = sub.drop_duplicates().reset_index()

            if len(sub) > 0:
                self.data.at[index, "Status"] = sub.at[0, "disposition"]
                if not list(sub.Name)[0] == "":
                    self.data.at[index, "Name"] = sub.at[0, "Name"]
                if self.data.at[index, "DiscMeth"] == "nan":
                    self.data.at[index, "DiscMeth"] = sub.at[0, "discoverymethod"]

                for internal_alias in sub.alias:
                    for internal_al in internal_alias.split(","):
                        if internal_al not in final_alias_total:
                            final_alias_total.append(internal_al)

            host = self.data.at[index, "Host"]
            # check hosts
            sub = tab[tab.alias.str.contains(host)]
            sub = sub.drop_duplicates().reset_index()
            if len(sub) > 0:
                self.data.at[index, "Status"] = sub.at[0, "disposition"]
                if not list(sub.Name)[0] == "":
                    self.data.at[index, "Name"] = sub.at[0, "Name"]
                if self.data.at[index, "DiscMeth"] == "nan":
                    self.data.at[index, "DiscMeth"] = sub.at[0, "discoverymethod"]

                for internal_alias in sub.alias:
                    for internal_al in internal_alias.split(","):
                        if internal_al not in final_alias_total:
                            final_alias_total.append(internal_al)

            self.data.at[index, "alias"] = ",".join(
                [x for x in set(final_alias_total) if x != "nan"]
            )

        logging.info("Status from " + table_path + " checked.")

    def fill_binary_column(self) -> None:
        """
        The fill_binary_column function fills the binary column of the dataframe with
        the appropriate values. It does this by checking if there is a binary letter at the end of that host
        column. If so, it strips out that letter and puts it into its own column called Binary.
        If not, nothing happens.
        """
        self.data = self.data.reset_index()
        self.data["Binary"] = ""
        for i in self.data.index:
            # protect in case planet name in Host column
            if (
                not str(re.search(r"([ABCNLS\d][a-z])$", self.data.at[i, "Host"]))
                == "None"
            ):
                self.data.at[i, "Host"] = self.data.at[i, "Host"][:-1].strip()

            # strip the binary letter and put it in Binary column
            if (
                not str(re.search(r"[\s\d][ABCNS]\s", self.data.at[i, "Name"]))
                == "None"
            ):
                self.data.at[i, "Binary"] = self.data.at[i, "Name"][-3:-2]

            # specific for circumbinary planets (AB)
            if (
                not str(re.search(r"[\s\d](AB)\s[a-z]$", self.data.at[i, "Name"]))
                == "None"
            ):
                self.data.at[i, "Binary"] = "AB"

            if (
                not str(re.search(r"[\s\d](\(AB\))\s[a-z]$", self.data.at[i, "Name"]))
                == "None"
            ):
                self.data.at[i, "Binary"] = "AB"

            # strip the binary letter and put it in Binary column
            if not str(re.search(r"[\s\d][ABCNS]$", self.data.at[i, "Host"])) == "None":
                self.data.at[i, "Binary"] = self.data.at[i, "Host"][-1:]
                self.data.at[i, "Host"] = self.data.at[i, "Host"][:-1].strip()

            # specific for circumbinary planets (AB)
            if not str(re.search(r"[\s\d](AB)$", self.data.at[i, "Host"])) == "None":
                self.data.at[i, "Binary"] = "AB"
                self.data.at[i, "Host"] = self.data.at[i, "Host"][:-3].strip()

            if (
                not str(re.search(r"[\s\d](\(AB\))$", self.data.at[i, "Host"]))
                == "None"
            ):
                self.data.at[i, "Binary"] = "AB"
                self.data.at[i, "Host"] = self.data.at[i, "Host"][:-4].strip()

        self.data["Host"] = self.data.Host.apply(
            lambda x: " ".join(x.strip().strip(".").strip(" (").split())
        )

        if "cb_flag" in self.data.columns:
            # SPECIFIC TO NASA
            self.data.loc[self.data.cb_flag == 1, "Binary"] = "AB"

        if "binaryflag" in self.data.columns:
            # SPECIFIC TO OEC
            # if unknown host star, be less specific with S-type, otherwise keep the known letter
            self.data.loc[self.data.binaryflag == 2, "Binary"] = self.data.loc[
                self.data.binaryflag == 2, "Binary"
            ].replace("", "S-type")

            self.data.loc[self.data.binaryflag == 1, "Binary"] = self.data.loc[
                self.data.binaryflag == 1, "Binary"
            ].replace("", "AB")
            self.data.loc[self.data.binaryflag == 3, "Binary"] = self.data.loc[
                self.data.binaryflag == 3, "Binary"
            ].replace("", "Rogue")
        logging.info("Fixed planets orbiting binary stars.")

    def create_catalogstatus_string(self) -> None:
        """
        The create_catalogstatus_string function creates a new column in the dataframe called "CatalogStatus"
        which is a concatenation of the Catalog and Status columns. This function is used to create an additional
        column for use in creating plots.
        """
        self.data["CatalogStatus"] = self.data.catalog + ": " + self.data.Status
        logging.info("CatalogStatus column created.")

    def make_uniform_alias_list(self):
        """
        The make_uniform_alias_list function takes in a dataframe and returns a list of aliases for each host.
        The function first groups the data by host, then creates a set of all the aliases associated with that
        host. The set is filtered to remove any None or NaN values, as well as removing the host name from this list.
        Finally, it iterates through each row in the groupby object and sets its alias value equal to this new list.
        """
        for host, group in self.data.groupby(by="Host"):
            final_alias = ""

            for al in group.alias:
                final_alias = final_alias + "," + al
            self.data.loc[self.data.Host == host, "alias"] = ",".join(
                [x for x in set(final_alias.split(",")) if x]
            )
        logging.info("Lists of aliases uniformed.")

    def convert_coordinates(self) -> None:
        """
        The coordinates function takes the RA and Dec columns of a dataframe, and converts them to decimal degrees.
        It is not implemented here because it is catalog-dependent.
        """
        raise NotImplementedError

    def print_catalog(self, filename: Union[str, Path]) -> None:
        """
        The print_cat function prints the dataframe to a csv file.
        It takes one argument, filename, which is the name of the file you want to print it as.

        Parameters
        ----------
            filename:Union[str, Path]
                Specify the location of the file to be written


        """
        self.data = self.data.sort_values(by="Name")
        self.data.to_csv(filename)
        logging.info("Printed catalog.")
