import glob
import logging
import os
import re
from datetime import date, datetime
from pathlib import Path
from typing import Union
import numpy as np
import pandas as pd
import requests
import unidecode

from .utility_functions import UtilityFunctions as Utils


class Catalog:
    """
    The Catalog class contains all methods and attributes that apply to all the other catalogs.
    """

    def __init__(self) -> None:
        """
        This function is called when the class is instantiated. It sets up the object with a data attribute
        that will be used to store the catalog data, and a name attribute that can be used to refer to this
        particular instance of Catalog.

        :param self: An instance of class Catalog
        :type self: Catalog
        :return: None
        :rtype: None
        """
        self.data = None
        self.name = "catalog"
        self.columns = {}

    def download_catalog(
        self, url: str, filename: str, local_date: str, timeout: float = None
    ) -> Path:
        """
        Downloads a catalog from a given URL and saves it to a file. If no local file is found, it will download the
        catalog from the url. If there is an error in downloading or reading the catalog, it will take a previous
        version of that same day's downloaded catalog if one exists.

        :param self: An instance of class Catalog
        :type self: Catalog
        :param url: The URL from which to download the catalog.
        :type url: str
        :param filename: The name of the file to save the catalog to.
        :type filename: str
        :param local_date: The date of the catalog to download.
        :type local_date: str
        :param timeout: The maximum amount of time to wait for the download to complete. Default is None.
        :type timeout: float
        :return: The path to the downloaded file.
        :rtype: Path
        """

        # date is today:
        #  -    check if it exists, if not download
        # - if download fails, get latest available version
        # date is not today:
        # Check if file with that date exists, if so load it up.

        file_path_str = filename + local_date + ".csv"
        # File already exists
        if os.path.exists(file_path_str):
            logging.info("Reading existing file downloaded in date: " + local_date)
        else:
            # File does not exist. If date is today, try downloading it
            if local_date == date.today().strftime("%Y-%m-%d"):
                # Try to download the file
                try:
                    result = requests.get(url, timeout=timeout)
                    with open(file_path_str, "wb") as f:
                        f.write(result.content)
                    # try opening the catalog
                    dat = pd.read_csv(file_path_str)
                    logging.info("Catalog downloaded.")
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
                    pd.errors.ParserError,  # file downloaded but corrupted
                ):
                    # if the file that was downloaded is corrupted, eliminate file
                    if len(glob.glob(file_path_str)) > 0:
                        logging.warning(
                            "File "
                            + file_path_str
                            + " downloaded, but corrupted. Removing file..."
                        )
                        os.system("rm " + file_path_str)
                    # Download failed, try most recent local copy
                    if len(glob.glob(filename + "*.csv")) > 0:
                        li = list(glob.glob(filename + "*.csv"))
                        li = [re.search(r"\d\d\d\d-\d\d-\d\d", l)[0] for l in li]
                        li = [datetime.strptime(l, "%Y-%m-%d") for l in li]
                        # get the most recent compared to the current date. Get only the ones earlier than the date
                        local_date_datetime = datetime.strptime(
                            re.search(r"\d\d\d\d-\d\d-\d\d", file_path_str)[0],
                            "%Y-%m-%d",
                        )
                        li = [l for l in li if l < local_date_datetime]
                        compar_date = max(li).strftime("%Y-%m-%d")
                        file_path_str = filename + compar_date + ".csv"

                        logging.warning(
                            "Error fetching the catalog, taking a local copy: %s",
                            file_path_str,
                        )
                    else:
                        raise ConnectionError(
                            "The catalog could not be downloaded and there is no backup catalog available."
                        )

            else:
                # date is not today and the file does not exist
                raise ValueError(
                    "Could not find catalog with this specific date. Please check your date value."
                )

            # TODO: file does not exist and date is not today. Use an earlier date

        return Path(file_path_str)

    def check_input_columns(self) -> str:
        """
        The check_input_columns ensures that the .csv file contains the columns the script needs later.

        :param self: An instance of class Catalog
        :return: None
        :rtype: None
        """
        # check that the table contains the names of the columns that we need

        missing_columns = ""
        for col in self.columns:
            if col not in self.data.keys():
                missing_columns = ",".join([col, missing_columns])
        return missing_columns.rstrip(",").lstrip(",")

    def check_column_dtypes(self) -> str:
        """
        The check_datatype_columns ensures that the columns have the expected types.

        :param self: An instance of class Catalog
        :return: None
        :rtype: None
        """
        # check that the table contains the names of the columns that we need

        wrong_dtypes = ""
        self.data = self.data.convert_dtypes()
        for column, expected_dtype in self.columns.items():
            actual_dtype = self.data[column].dtype
            if actual_dtype != expected_dtype:
                wrong_dtypes += ",".join([wrong_dtypes, f"{column}[{actual_dtype}]"])

        return wrong_dtypes.rstrip(",").lstrip(",")

    def find_non_ascii(self):
        non_ascii = {}  # Dictionary to store columns and rows with non-ASCII characters
        # Iterate through columns
        for column in self.data.columns:
            if self.data[column].dtype == "object":  # Only check string columns
                # Use regular expression to find non-ASCII characters
                non_ascii_rows = self.data[
                    self.data[column].apply(
                        lambda x: bool(re.search(r"[^\x00-\x7F]", str(x)))
                    )
                ]
                if not non_ascii_rows.empty:
                    non_ascii[column] = non_ascii_rows.index.tolist()
        return non_ascii

    def read_csv_catalog(self, file_path_str: Union[Path, str]) -> None:
        """
        The read_csv_catalog function reads in a csv file and stores it as a pandas dataframe in the self.

        :param self: An instance of class Catalog
        :type self: Catalog
        :param file_path_str: Specify the file path of the csv file
        :type file_path_str: Union[Path, str]
        :return: None
        :rtype: None
        """
        try:
            self.data = pd.read_csv(file_path_str, low_memory=False)
        except:
            raise ValueError("Failed to read the .csv file.")

    def keep_columns(self) -> None:
        """
        The keep_columns function removes all columns from the dataframe except for those specified in the keep list.

        :param self: An instance of class Catalog
        :type self: Catalog
        :return: None
        :rtype: None
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

        It checks if the last character of a planet name is a number or if it ends with an uppercase letter.
        If so, it fills the 'letter' cell with 'BD' to filter it out later.

        The function excludes KOI-like objects by avoiding the patterns ".0d" with d being a digit.

        Special handling:

        - Excludes 'PSR B1257+12' which is a known weird candidate.

        - Handles the special case for the problematic triple BD system DENIS J063001.4-184014 (bc).

        :param self: An instance of class Catalog
        :type self: Catalog
        :return: None
        :rtype: None
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
                if len(re.findall(r"\(.*?\)$", self.data.at[i, "name"])) > 0:
                    self.data.at[i, "letter"] = "BD"
                    self.data.at[i, "binary"] = (
                        re.findall(r"\(.*?\)$", self.data.at[i, "name"])[0]
                        .strip("(")
                        .strip(")")
                    )

        logging.info("Identified possible Brown Dwarfs (no letter for planet name).")

    def replace_known_mistakes(self) -> None:
        """
        Replaces values in the dataframe with those specified in `replacements.ini`.

        This function reads replacements from the `replacements.ini` file and performs the following operations:

        1. Remove rows from the dataframe based on specified values in the DROP section.

        2. Replace values in the name and host columns based on the NAMEtochangeNAME and NAMEtochangeHOST sections.

        3. Replace values in the host column based on the HOSTtochangeHOST section.

        4. Replace values in the ra and dec columns based on the ra and dec sections.

        5. Replace values in the name and host columns based on the const dictionary.

        6. Replace values in the binary column based on the NAMEtochangeBINARY section.

        7. Remove accents and extra spaces from the name column.

        8. Reset the index of the dataframe.

        9. Log the completion of the operation.

        :param self: An instance of class Catalog
        :type self: Catalog
        :return: None
        :rtype: None
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

        # NAMEforNAME
        for name in config_name_for_name.keys():
            if len(self.data[self.data.name == name]) == 0:
                f.write("NAME for NAME: " + name + "\n")
            else:
                self.data.loc[self.data.name == name, "name"] = config_name_for_name[
                    name
                ]

        # NAMEforHOST
        for name in config_name_for_host.keys():
            if len(self.data[self.data.name == name]) == 0:
                f.write("NAME for HOST: " + name + "\n")
            else:
                self.data.loc[self.data.name == name, "host"] = config_name_for_host[
                    name
                ]

        # HOSTforHOST
        for host in config_host_for_host.keys():
            if len(self.data[self.data.host == host]) == 0:
                f.write("HOST for HOST: " + host + "\n")
            else:
                self.data.loc[self.data.host == host, "host"] = config_host_for_host[
                    host
                ]

        # ra and dec
        for coord in ["ra", "dec"]:
            config_replace = Utils.read_config_replacements(coord)
            for name in config_replace.keys():
                if len(self.data.loc[self.data.host == name]) == 0:
                    f.write(coord + ": " + name + "\n")
                else:
                    self.data.loc[self.data.host == name, coord] = float(
                        config_replace[name]
                    )

        # const dictionary
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

        # NAMEtochangeBINARY
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

        # remove accents and extra spaces
        self.data["name"] = self.data["name"].apply(unidecode.unidecode)
        self.data["name"] = self.data.name.apply(lambda x: " ".join(x.split()))
        self.data = self.data.reset_index(drop=True)
        logging.info("Known mistakes replaced.")

    def remove_theoretical_masses(self) -> None:
        """
        Remove theoretical masses from the dataframe.

        This function removes theoretical masses from the dataframe. The removal process is catalog-dependent,
        so it is not implemented in this base class. Subclasses should override this method to provide
        the specific implementation for their respective catalog.

        :param self: An instance of class Catalog
        :return: None
        :rtype: None
        :raises NotImplementedError: This method is not implemented in the base class.
        """
        raise NotImplementedError

    def handle_reference_format(self) -> None:
        """
        The handle_reference_format function is used to create a URL for each reference in the references list.
        This function is left unimplemented here as it is catalog-dependent and should be implemented in subclasses.

        :param self: An instance of class Catalog
        :type self: Catalog
        :return: None
        :rtype: None
        :raises NotImplementedError: This method is not implemented in the base class.
        """
        raise NotImplementedError

    def standardize_catalog(self) -> None:
        """
        Standardize the dataframe columns and values.

        This method is not implemented in the base class. Subclasses should override this method to provide the
        specific implementation for their respective catalog.

        :param self: An instance of class Catalog
        :type self: Catalog
        :return: None
        :rtype: None
        :raises NotImplementedError: This method is not implemented in the base class.
        """
        raise NotImplementedError

    def make_errors_absolute(self) -> None:
        """
        Makes all columns related to errors absolute values.

        This function takes in a DataFrame and returns a DataFrame where all the columns related
        to errors are made absolute. The columns that are modified are:
        p_max, a_max, e_max, i_max, r_max, msini_max, mass_max, p_min, a_min, e_min, i_min, r_min, msini_min, mass_min.

        :param self: An instance of class Catalog
        :type self: Catalog
        :return: None
        :rtype: None

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

    def remove_impossible_values(self) -> None:
        """
        :param self: An instance of class Catalog
        :type self: Catalog
        :return: None
        :rtype: None

        """
        for c in [
            "p",
            "a",
            "e",
            "i",
            "r",
            "msini",
            "mass",
        ]:
            self.data.loc[self.data[c] < 0, c + "_min"] = np.nan
            self.data.loc[self.data[c] < 0, c + "_max"] = np.nan
            self.data.loc[self.data[c] < 0, c] = np.nan

        # impossible value: eccentricity greater than 1
        self.data.loc[self.data["e"] > 1, "e_min"] = np.nan
        self.data.loc[self.data["e"] > 1, "e_max"] = np.nan
        self.data.loc[self.data["e"] > 1, "e"] = np.nan

        logging.info("Removed impossible values of parameters.")

    def standardize_name_host_letter(self) -> None:
        """
        This function standardizes the 'name', 'host', and 'letter' columns in the data.

        It processes the 'name' and 'host' columns by applying a standardized string function,
        filling empty host values with the name, stripping certain characters from host identifiers,
        and cleaning the host column with the standardize_string function.

        It also refines the 'alias' column by removing specific characters and transforming the values.

        Lastly, it assigns the 'letter' column based on certain conditions from the 'name' column.

        The function concludes by logging the completion of the standardization process.

        :param self: An instance of class Catalog
        :type self: Catalog
        :return: None
        :rtype: None
        """

        # standardize name
        self.data["name"] = self.data.name.apply(lambda x: Utils.standardize_string(x))
        ind = self.data[self.data.host == ""].index
        # standardize host
        self.data["host"] = self.data.host.replace("", np.nan).fillna(self.data.name)

        for identifier in self.data.loc[ind, "host"]:
            # .0d cases
            if not str(re.search("(\\.0)\\d$", identifier, re.M)) == "None":
                self.data.loc[self.data.host == identifier, "host"] = identifier[
                    :-3
                ].strip()
            # letter cases
            if not str(re.search(" [b-z]$", identifier, re.M)) == "None":
                self.data.loc[self.data.host == identifier, "host"] = identifier[
                    :-1
                ].strip()
        self.data["host"] = self.data.host.apply(lambda x: Utils.standardize_string(x))

        for i in self.data.index:
            polished_alias = ""
            #
            for al in self.data.at[i, "alias"].split(","):
                # letter cases
                if not str(re.search(" [b-z]$", al, re.M)) == "None":
                    al = al[:-1]
                # .0d cases
                if not str(re.search("(\\.0)\\d$", al, re.M)) == "None":
                    al = al[:-3]
                if al != "":
                    polished_alias = (
                        polished_alias
                        + ","
                        + Utils.standardize_string(al.lstrip().rstrip())
                    )
            self.data.at[i, "alias"] = polished_alias.lstrip(",")

        # standardize letter
        for identifier in self.data.name:
            if not str(re.search("(\\.0)\\d$", identifier, re.M)) == "None":
                self.data.loc[self.data.name == identifier, "letter"] = identifier[-3:]

            else:
                self.data.loc[self.data.name == identifier, "letter"] = identifier[-1:]

        logging.info("name, host, letter columns standardized.")

    def assign_status(self) -> None:
        """
        Assigns a status to each planet based on the status column.

        This function is catalog-dependent and needs to be implemented in subclasses.

        :param self: An instance of class Catalog
        :type self: Catalog
        :return: None
        :rtype: None
        :raises NotImplementedError: If the function is called directly from the base class.

        """
        raise NotImplementedError

    def check_mission_tables(self, table_path_str: str) -> None:
        """
        The check_mission_tables function checks the dataframe for any objects that have a name that matches an entry
        in the KOI or EPIC catalogs.

        If there is a match, it will update the status of that object to whatever status is listed in the KOI,
        TESS and EPIC catalogs and update its coordinates if they are missing from the dataframe.

        :param self: An instance of class Catalog
        :type self: Catalog
        :param table_path_str: the string containing path to the table.
        :type table_path_str: str
        :return: None
        :rtype: None
        """
        # Fill missing values in status column with empty string
        self.data.status = self.data.status.fillna("")

        # Read table into a pandas dataframe
        tab = pd.read_csv(table_path_str)
        tab = tab.fillna("")

        # Loop through each row in the dataframe
        for index in self.data.index:
            name = self.data.at[index, "name"]
            final_alias_total = self.data.at[index, "alias"].split(",")

            # Find the row in the table that matches the name
            sub = tab[tab.aliasplanet.str.contains(name + ",", regex=False)]

            sub = sub.drop_duplicates().reset_index()
            if len(sub) > 0:
                # Check if the status in the dataframe matches the status in the table. If it is different,
                # update the status
                if self.data.at[index, "status"] != sub.at[0, "disposition"]:
                    self.data.at[index, "status"] = sub.at[0, "disposition"]

                # Check if the discovery method in the dataframe matches the discovery method in the table. If it is
                # different, update the discovery method
                if self.data.at[index, "discovery_method"] in [
                    "nan",
                    "Unknown",
                    "Default",
                ]:
                    self.data.at[index, "discovery_method"] = sub.at[
                        0, "discoverymethod"
                    ]

                # Update the alias in the dataframe with the alias in the table
                for internal_alias in sub.alias:
                    for internal_al in internal_alias.split(","):
                        if internal_al not in final_alias_total:
                            final_alias_total.append(internal_al)

            host = self.data.at[index, "host"]
            letter = self.data.at[index, "letter"]

            # Find the row in the table that matches the host
            sub = tab[tab.alias.str.contains(host + ",", regex=False)]
            sub = sub[sub.letter == letter]
            sub = sub.drop_duplicates().reset_index()
            if len(sub) > 0:
                # Check if the status in the dataframe matches the status in the table. If it is different,
                # update the status
                if self.data.at[index, "status"] != sub.at[0, "disposition"]:
                    self.data.at[index, "status"] = sub.at[0, "disposition"]

                # Check if the discovery method in the dataframe. If it is NaN, update the discovery method
                if self.data.at[index, "discovery_method"] == "nan":
                    self.data.at[index, "discovery_method"] = sub.at[
                        0, "discoverymethod"
                    ]

                # Add internal aliases to final alias list
                for internal_alias in sub.alias:
                    for internal_al in internal_alias.split(","):
                        internal_al = (
                            Utils.standardize_string(internal_al)
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

            # Add final alias list to dataframe
            self.data.at[index, "alias"] = ",".join(
                [x for x in set(final_alias_total) if x != "nan"]
            )
        # if there are still empty status strings, use special keyword
        # preliminary i.e. it hasn't been updated in the original catalog yet
        self.data["status"] = self.data.status.replace("", "PRELIMINARY")

        logging.info(table_path_str + " checked.")

    def fill_binary_column(self) -> None:
        """
        The fill_binary_column function fills the binary column of the dataframe with
        the appropriate values. It does this by checking if there is a binary letter at the end of that host
        column. If so, it strips out that letter and puts it into its own column called binary.
        If not, nothing happens.

        :param self: An instance of class Catalog
        :type self: Catalog
        :return: None
        :rtype: None

        """
        # Reset index of the dataframe
        self.data = self.data.reset_index()

        # Initalize binary column
        self.data["binary"] = ""

        for i in self.data.index:
            # Cleanup host if planet name is present
            if len(re.findall(r"([\s\d][b-z])$", self.data.at[i, "host"])) > 0:
                print("planet name in host", self.data.at[i, "host"])
                self.data.at[i, "host"] = self.data.at[i, "host"][:-1].strip()

            if self.data.at[i, "binary"] == "":
                # Check for CIRCUMBINARY NAME or HOST
                if (
                    (
                        len(re.findall(r"[\s\d](AB)\s[b-z]$", self.data.at[i, "name"]))
                        > 0
                    )
                    or (
                        len(
                            re.findall(
                                r"[\s\d](\(AB\))\s[b-z]$", self.data.at[i, "name"]
                            )
                        )
                        > 0
                    )
                    or (len(re.findall(r"[\s\d](AB)$", self.data.at[i, "host"])) > 0)
                    or (
                        len(re.findall(r"[\s\d](\(AB\))$", self.data.at[i, "host"])) > 0
                    )
                    or (len(re.findall(r"[\s\d](AB)$", self.data.at[i, "host"])) > 0)
                ):
                    self.data.at[i, "binary"] = "AB"
                    # Update host column
                    self.data.at[i, "host"] = (
                        self.data.at[i, "host"]
                        .replace("(AB)", "")
                        .replace("AB", "")
                        .strip()
                    )

                # Check for SIMPLE BINARY NAME
                if (
                    len(
                        re.findall(
                            r"[\s\d][ABCNS][\s\d][b-z]$", self.data.at[i, "name"]
                        )
                    )
                    > 0
                ):
                    self.data.at[i, "binary"] = self.data.at[i, "name"][-3:-2]

                # Check for SIMPLE BINARY HOST
                if len(re.findall(r"[\d\s][ABCSN]$", self.data.at[i, "host"])) > 0:
                    self.data.at[i, "binary"] = self.data.at[i, "host"][-1:].strip()
                    self.data.at[i, "host"] = self.data.at[i, "host"][:-1].strip()

        # Clean the host column
        self.data["host"] = self.data.host.apply(
            lambda x: " ".join(x.strip().strip(".").strip(" (").split())
        )

        # Handle NASA specific cases
        if "cb_flag" in self.data.columns:
            self.data.loc[self.data.cb_flag == 1, "binary"] = "AB"

        # Handle OEC specific cases
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

        # Logging
        logging.info("Fixed planets orbiting binary stars.")

    def create_catalogstatus_string(self, string: str) -> None:
        """
        The create_catalogstatus_string function creates a new column in the dataframe which is a concatenation of
        the Catalog and status columns. Depending on when it is called, it can be either formed by the "original"
        status provided by the catalog, or the "checked" status which is the one EMC picks after checking with the
        KOI/K2 catalogs.

        :param self: An instance of class Catalog
        :type self: Catalog
        :param string: The name of the new column
        :type string: str
        :return: None
        :rtype: None
        """

        self.data[string] = self.data.catalog + ": " + self.data.status.fillna("")
        logging.info(string + " column created.")

    def make_standardized_alias_list(self) -> None:
        """
        The make_standardized_alias_list function takes in a dataframe and returns a list of aliases for each host. The
        function first groups the data by host, then creates a set of all the aliases associated with that host. The
        set is filtered to remove any None or NaN values, as well as removing the host name from this list. Finally,
        it iterates through each row in the groupby object and sets its alias value equal to this new list.

        :param self: An instance of class Catalog
        :type self: Catalog
        :return: None
        :rtype: None
        """
        for host, group in self.data.groupby(by="host"):
            final_alias = ""

            for al in group.alias:
                if al not in [np.nan, "NaN", "nan"]:
                    final_alias = final_alias + "," + al
            self.data.loc[self.data.host == host, "alias"] = ",".join(
                [Utils.standardize_string(x) for x in set(final_alias.split(",")) if x]
            )
        logging.info("Lists of aliases standardized.")

    def convert_coordinates(self) -> None:
        """
        Convert the `ra` and `dec` columns of the dataframe to decimal degrees.

        This method is not implemented in the base class. Subclasses should override this method to provide the
        specific implementation for their respective catalog.

        :param self: An instance of class Catalog
        :type self: Catalog
        :raises NotImplementedError: This method is not implemented in the base class.
        :return: None
        :rtype: None
        """
        raise NotImplementedError

    def fill_nan_on_coordinates(self) -> None:
        """
        Fill missing values in the `ra` and `dec` columns of the dataframe with NaN.

        :param self: An instance of class Catalog
        :type self: Catalog
        :return: None
        :rtype: None
        :note:  Currently only used for Open Exoplanet Catalogue, KOI catalogs.
        """
        # Convert RA and Dec columns to numeric data type and replace "nan" and empty strings with NaN
        self.data["ra"] = pd.to_numeric(
            self.data.ra.replace("nan", np.nan).replace("", np.nan)
        )
        self.data["dec"] = pd.to_numeric(
            self.data.dec.replace("nan", np.nan).replace("", np.nan)
        )

        # Logging
        logging.info("Filled empty coordinates with nan.")

    def print_catalog(self, filename: Union[str, Path]) -> None:
        """
        The print_cat function prints the dataframe to a csv file.

        :param self: An instance of class Catalog
        :type self: Catalog
        :param filename: The location of the file to be written
        :type filename: Union[str, Path]
        :return: None
        :rtype: None
        """
        # Save the dataframe to the specified file location
        self.data.to_csv(filename, index=None)

        # Log a message indicating that the catalog has been printed
        logging.info("Printed catalog.")
