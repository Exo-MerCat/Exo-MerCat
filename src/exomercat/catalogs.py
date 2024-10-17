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
    A base class for managing exoplanet catalogs.

    This class provides a foundation for handling various exoplanet catalogs.
    It includes methods for downloading, reading, and processing catalog data,
    as well as standardizing and validating the information.
    """

    def __init__(self) -> None:
        """
        Initialize a Catalog instance.

        Sets up the basic attributes for the catalog, including an empty data attribute
        and a default name.

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
        Download a catalog from a given URL and save it to a file.

        This method attempts to download the catalog for the specified date. If the file
        already exists locally, it uses that file. If downloading fails, it attempts to
        use the most recent local copy.

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

        return Path(file_path_str)

    def check_input_columns(self) -> str:
        """
        Check if the loaded data contains all required columns.

        :param self: An instance of class Catalog
        :return: A comma-separated string of missing column names, if any.
        :rtype: str
        """

        missing_columns = ""
        for col in self.columns:
            if col not in self.data.keys():
                missing_columns = ",".join([col, missing_columns])
        return missing_columns.rstrip(",").lstrip(",")

    def check_column_dtypes(self) -> str:
        """
        Check if the data types of columns match the expected types.

        :param self: An instance of class Catalog
        :return: A comma-separated string of columns with mismatched data types, if any.
        :rtype: str
        """

        wrong_dtypes = ""
        self.data = self.data.convert_dtypes()
        for column, expected_dtype in self.columns.items():
            actual_dtype = self.data[column].dtype
            if actual_dtype != expected_dtype:
                wrong_dtypes += ",".join([wrong_dtypes, f"{column}[{actual_dtype}]"])

        return wrong_dtypes.rstrip(",").lstrip(",")

    def find_non_ascii(self) -> dict:
        """
        Identify non-ASCII characters in string columns of the dataset.

        :param self: An instance of class Catalog

        :return: A dictionary where keys are column names and values are lists of row indices
                 containing non-ASCII characters.
        :rtype: dict
        """
        # Create dictionary to store columns and rows with non-ASCII characters
        non_ascii = {} 

        # Iterate through columns
        for column in self.data.columns:
            # Only check string columns
            if self.data[column].dtype == "object":  
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
        Read a CSV file into the catalog's data attribute.

        :param self: An instance of class Catalog
        :type self: Catalog
        :param file_path_str: Specify the file path of the csv file
        :type file_path_str: Union[Path, str]
        :return: None
        :rtype: None
        :raise ValueError: If reading of the .csv file fails.
        """
        try:
            self.data = pd.read_csv(file_path_str, low_memory=False)
        except:
            raise ValueError("Failed to read the .csv file.")

    def keep_columns(self) -> None:
        """
        Retain only specified columns in the dataframe.

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
        # Check that all columns exist, otherwise raise an error
        except KeyError:
            raise KeyError("Not all columns exist")
        logging.info("Selected columns to keep.")

    def identify_brown_dwarfs(self) -> None:
        """
        Identify possible brown dwarfs in the dataframe based on naming conventions.

        This method marks potential brown dwarfs by setting the 'letter' column to 'BD'
        and handles special cases for certain objects.

        :param self: An instance of class Catalog
        :type self: Catalog
        :return: None
        :rtype: None
        """

        for i in self.data.index:
            # Known weird candidates
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

        # Logging
        logging.info("Identified possible Brown Dwarfs (no letter for planet name).")

    def replace_known_mistakes(self) -> None:
        """
        Replace known errors in the dataframe based on predefined rules.

        This method applies corrections specified in the 'replacements.ini' file,
        including dropping rows, replacing values, and standardizing names and coordinates.

        :param self: An instance of class Catalog
        :type self: Catalog
        :return: None
        :rtype: None
        """

        # Read replacements.ini file
        nomenclature = Utils.get_common_nomenclature()
        config_name_for_name = Utils.read_config_replacements("NAMEtochangeNAME")
        config_name_for_host = Utils.read_config_replacements("NAMEtochangeHOST")
        config_host_for_host = Utils.read_config_replacements("HOSTtochangeHOST")
        config_name_for_binary = Utils.read_config_replacements("NAMEtochangeBINARY")
        config_host_for_coord={}
        config_host_for_coord['ra']=Utils.read_config_replacements("HOSTtochangeRA")
        config_host_for_coord['dec']=Utils.read_config_replacements("HOSTtochangeDEC")

        # DROP section
        config_drop = Utils.read_config_replacements("DROP")
        for check, lis in config_drop.items():
            for drop in lis.split(","):
                self.data = self.data[
                    ~(self.data[check].str.contains(drop.strip(), na=False))
                ]

        # Open file to log unused replacements
        f = open("Logs/replace_known_mistakes.txt", "a")
        f.write("**** UNUSED REPLACEMENTS FOR " + self.name + " ****\n")

        # NAME to change NAME section
        for name in config_name_for_name.keys():
            if len(self.data[self.data.name == name]) == 0:
                f.write("NAME for NAME: " + name + "\n")
            else:
                self.data.loc[self.data.name == name, "name"] = config_name_for_name[
                    name
                ]

        # NAME to change HOST section
        for name in config_name_for_host.keys():
            if len(self.data[self.data.name == name]) == 0:
                f.write("NAME for HOST: " + name + "\n")
            else:
                self.data.loc[self.data.name == name, "host"] = config_name_for_host[
                    name
                ]

        # HOST to change HOST section
        for host in config_host_for_host.keys():
            if len(self.data[self.data.host == host]) == 0:
                f.write("HOST for HOST: " + host + "\n")
            else:
                self.data.loc[self.data.host == host, "host"] = config_host_for_host[
                    host
                ]

        # HOST to change COORDINATE section
        for coord in ["ra", "dec"]:
            config_coord = config_host_for_coord[coord]
            for host in config_coord.keys():
                if len(self.data.loc[self.data.host == host]) == 0:
                    f.write(coord + ": " + host + "\n")
                else:
                    self.data.loc[self.data.host == host, coord] = float(
                        config_coord[host]
                    )

        # Replace common astronomical nomenclature
        for j in self.data.index:
            for i in nomenclature.keys():
                if i in self.data.loc[j, "name"]:
                    self.data.loc[j, "name"] = self.data.loc[j, "name"].replace(
                        i, nomenclature[i]
                    )
            for i in nomenclature.keys():
                if i in self.data.loc[j, "host"]:
                    self.data.loc[j, "host"] = self.data.loc[j, "host"].replace(
                        i, nomenclature[i]
                    )

        # NAME to change BINARY section
        for binary in config_name_for_binary.keys():
            if len(self.data[self.data.name == binary]) == 0:
                f.write("BINARY: " + binary + "\n")
            else:
                self.data.loc[self.data.name == binary, "binary"] = config_name_for_binary[
                    binary
                ].replace("NaN", "")
        f.close()

        # Remove accents and extra spaces
        self.data["name"] = self.data["name"].apply(unidecode.unidecode)
        self.data["name"] = self.data.name.apply(lambda x: " ".join(x.split()))
        self.data = self.data.reset_index(drop=True)
        
        # Logging
        logging.info("Known mistakes replaced.")

    def remove_theoretical_masses(self) -> None:
        """
        Remove theoretical masses from the dataframe.

        This process is catalog-dependent, so it is not implemented in this base class. 
        Subclasses should override this method to provide the specific implementation 
        for their respective catalog.

        :param self: An instance of class Catalog
        :return: None
        :rtype: None
        :raises NotImplementedError: This method is not implemented in the base class.
        """
        raise NotImplementedError

    def handle_reference_format(self) -> None:
        """
        Standardize the reference format for various parameters.

        This process is catalog-dependent, so it is not implemented in this base class. 
        Subclasses should override this method to provide the specific implementation 
        for their respective catalog.

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

        This process is catalog-dependent, so it is not implemented in this base class. 
        Subclasses should override this method to provide the specific implementation 
        for their respective catalog.

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

        # Logging
        logging.info("Made all errors absolute values.")

    def remove_impossible_values(self) -> None:
        """
        Remove impossible or nonsensical values from various parameters in the dataset.

        This method performs the following operations:

        1. Sets negative values of 'p', 'a', 'e', 'i', 'r', 'msini', and 'mass' to NaN.

        2. Sets the corresponding '_min' and '_max' values to NaN when the main value is negative.

        3. Sets eccentricity ('e') values greater than 1 to NaN, along with their '_min' and '_max' values.

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

        # Impossible value: eccentricity greater than 1
        self.data.loc[self.data["e"] > 1, "e_min"] = np.nan
        self.data.loc[self.data["e"] > 1, "e_max"] = np.nan
        self.data.loc[self.data["e"] > 1, "e"] = np.nan

        logging.info("Removed impossible values of parameters.")

    def standardize_name_host_letter(self) -> None:
        """
        Standardize the 'name', 'host', and 'letter' columns in the data.

        This function performs the following operations:

        1. Standardizes the 'name' column using the Utils.standardize_string function.

        2. Fills empty 'host' values with the corresponding 'name' value.

        3. Cleans and standardizes the 'host' column, removing specific suffixes and applying Utils.standardize_string.

        4. Refines the 'alias' column by removing specific characters and standardizing the values.

        5. Assigns the 'letter' column based on specific conditions from the 'name' column.

        :param self: An instance of class Catalog
        :type self: Catalog
        :return: None
        :rtype: None
        """

        # Standardize name
        self.data["name"] = self.data.name.apply(lambda x: Utils.standardize_string(x))
        ind = self.data[self.data.host == ""].index
        
        # Remove nan from host
        self.data["host"] = self.data.host.replace("", np.nan).fillna(self.data.name)

        for identifier in self.data.loc[ind, "host"]:
            # .0d cases
            if not str(re.search("(\\.0)\\d$", identifier, re.M)) == "None":
                self.data.loc[self.data.host == identifier, "host"] = identifier[
                    :-3
                ].strip()
            
            # Letter cases
            if not str(re.search(" [b-z]$", identifier, re.M)) == "None":
                self.data.loc[self.data.host == identifier, "host"] = identifier[
                    :-1
                ].strip()
        
        # Standardize host
        self.data["host"] = self.data.host.apply(lambda x: Utils.standardize_string(x))

        for i in self.data.index:
            polished_alias = ""
            #
            for al in self.data.at[i, "alias"].split(","):
                # Letter cases
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

        # Standardize letter
        for identifier in self.data.name:
            if not str(re.search("(\\.0)\\d$", identifier, re.M)) == "None":
                self.data.loc[self.data.name == identifier, "letter"] = identifier[-3:]

            else:
                self.data.loc[self.data.name == identifier, "letter"] = identifier[-1:]

        # Logging
        logging.info("name, host, letter columns standardized.")

    def assign_status(self) -> None:
        """
        Assign a status to each planet based on the status column.

        This process is catalog-dependent, so it is not implemented in this base class. 
        Subclasses should override this method to provide the specific implementation 
        for their respective catalog.

        :param self: An instance of class Catalog
        :type self: Catalog
        :return: None
        :rtype: None
        :raises NotImplementedError: If the function is called directly from the base class.

        """
        raise NotImplementedError

    def check_mission_tables(self, table_path_str: str) -> None:
        """
        Check and update the dataframe against mission tables.

        This function compares the objects in the dataframe with entries in the specified
        mission table. If a match is found, it updates the object's status, discovery method,
        and aliases. It also handles cases where the object name or host name matches
        entries in the mission table.

        :param self: An instance of class Catalog
        :type self: Catalog
        :param table_path_str: The file path to the mission table (CSV format)
        :type table_path_str: str
        :return: None
        :rtype: None

        :raises FileNotFoundError: If the specified table file does not exist
        :raises pd.errors.EmptyDataError: If the table file is empty

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
        # If there are still empty status strings, use special keyword
        # preliminary i.e. it hasn't been updated in the original catalog yet
        self.data["status"] = self.data.status.replace("", "PRELIMINARY")

        # Logging
        logging.info(table_path_str + " checked.")

    def fill_binary_column(self) -> None:
        """
        Fills the binary column of the dataframe with appropriate values.

        This function performs the following operations:

        1. Initializes the 'binary' column with empty strings.

        2. Cleans up the 'host' column by removing planet names if present.

        3. Identifies and marks circumbinary systems (AB).

        4. Identifies and marks simple binary systems (A, B, C, N, S).

        5. Cleans the 'host' column by removing extra spaces and characters.

        6. Handles NASA-specific cases using the 'cb_flag' column if present.

        7. Handles OEC-specific cases using the 'binaryflag' column if present.

        The function uses regular expressions to identify different binary system patterns
        in the 'name' and 'host' columns. It updates the 'binary' column accordingly and
        adjusts the 'host' column as needed.
        
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
        Creates a new column in the dataframe which is a concatenation of the Catalog and status columns.

        This function generates a new column in the dataframe by combining the 'catalog' and 'status' columns.
        The new column's name is specified by the 'string' parameter. This method can be used to create either
        an "original" status column (as provided by the catalog) or a "checked" status column (as determined by EMC
        after cross-checking with KOI/K2 catalogs).
        
        :param self: An instance of class Catalog
        :type self: Catalog
        :param string: The name of the new column to be created
        :type string: str
        :return: None
        :rtype: None
        """

        self.data[string] = self.data.catalog + ": " + self.data.status.fillna("")
        logging.info(string + " column created.")


    def make_standardized_alias_list(self) -> None:
        """
        Standardize and consolidate alias lists for each host in the catalog.

        The method standardizes and consolidates alias lists for each host in the catalog. 
        It groups the data by host, combines all aliases for each host, removes invalid values, 
        standardizes the aliases, and removes duplicates.
        The resulting 'alias' column will contain a comma-separated string of unique,
        standardized aliases for each host.
        
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
        
        # Logging
        logging.info("Lists of aliases standardized.")


    def convert_coordinates(self) -> None:
        """
        Convert the ra and dec columns of the dataframe to decimal degrees.

        This process is catalog-dependent, so it is not implemented in this base class. 
        Subclasses should override this method to provide the specific implementation 
        for their respective catalog.

        :param self: An instance of class Catalog
        :type self: Catalog
        :return: None
        :rtype: None
        :raises NotImplementedError: This method is not implemented in the base class.
        """
        raise NotImplementedError

    def fill_nan_on_coordinates(self) -> None:
        """
        Fill missing values in the 'ra' and 'dec' columns of the dataframe with NaN.

        :param self: An instance of class Catalog
        :type self: Catalog
        :return: None
        :rtype: None
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
        Print the catalog data to a CSV file.

        This method saves the catalog's dataframe to a CSV file at the specified location.

        :param filename: The path where the CSV file will be saved.
        :type filename: Union[str, Path]
        :return: None
        :rtype: None
        """
        # Save the dataframe to the specified file location
        self.data.to_csv(filename, index=None)

        # Log a message indicating that the catalog has been printed
        logging.info("Printed catalog.")