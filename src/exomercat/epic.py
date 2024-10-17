import logging
import re

import numpy as np
from pandas import Int64Dtype, Float64Dtype, StringDtype

from .catalogs import Catalog
from .utility_functions import UtilityFunctions as Utils


class Epic(Catalog):
    """
    A class representing the EPIC (Exoplanet Population Information Catalog) catalog.

    This class inherits from the Catalog base class and provides specific
    functionality for handling and processing data from the EPIC catalog.
    It includes methods for standardizing the catalog data, converting
    coordinates, handling reference formats, and assigning status to entries.

    Attributes:
        name (str): The name of the catalog, set to "epic".
        data (pandas.DataFrame): The catalog data stored as a DataFrame.
        columns (dict): A dictionary defining the expected columns and their data types.

    Methods:
        standardize_catalog(): Standardize the catalog data format.
        convert_coordinates(): Placeholder method for coordinate conversion (not implemented for EPIC).
        remove_theoretical_masses(): Remove theoretical masses (placeholder method).
        handle_reference_format(): Standardize reference format and create URL columns.
        assign_status(): Assign status to each entry based on disposition.
    """
    def __init__(self) -> None:
        """
        Initialize the Epic class object.

        This method sets up the instance of the Epic class by:
        1. Calling the parent class initializer.
        2. Setting the catalog name to "epic".
        3. Initializing the data attribute as None.
        4. Defining the expected columns and their data types.

        :param self: The instance of the Epic class.
        :type self: Epic
        :return: None
        :rtype: None
        """
        super().__init__()
        self.name = "epic"
        self.data = None
        self.columns = {
            "pl_name": StringDtype(),
            "discoverymethod": StringDtype(),
            "pl_orbper": Float64Dtype(),
            "pl_orbpererr2": Float64Dtype(),
            "pl_orbpererr1": Float64Dtype(),
            "pl_orbsmax": Float64Dtype(),
            "pl_orbsmaxerr2": Float64Dtype(),
            "pl_orbsmaxerr1": Float64Dtype(),
            "pl_orbeccen": Float64Dtype(),
            "pl_orbeccenerr2": Float64Dtype(),
            "pl_orbeccenerr1": Float64Dtype(),
            "pl_orbincl": Float64Dtype(),
            "pl_orbinclerr2": Float64Dtype(),
            "pl_orbinclerr1": Float64Dtype(),
            "pl_radj": Float64Dtype(),
            "pl_radjerr2": Float64Dtype(),
            "pl_radjerr1": Float64Dtype(),
            "disc_year": Int64Dtype(),
            "rv_flag": Int64Dtype(),
            "tran_flag": Int64Dtype(),
            "ttv_flag": Int64Dtype(),
            "pl_massj": Float64Dtype(),
            "pl_massjerr2": Float64Dtype(),
            "pl_massjerr1": Float64Dtype(),
            "pl_msinij": Float64Dtype(),
            "pl_msinijerr2": Float64Dtype(),
            "pl_msinijerr1": Float64Dtype(),
            "hostname": StringDtype(),
            "st_age": Float64Dtype(),
            "st_ageerr1": Float64Dtype(),
            "st_ageerr2": Float64Dtype(),
            "st_mass": Float64Dtype(),
            "st_masserr1": Float64Dtype(),
            "st_masserr2": Float64Dtype(),
            "pl_refname": StringDtype(),
            "hd_name": StringDtype(),
            "hip_name": StringDtype(),
            "tic_id": StringDtype(),
            "gaia_id": StringDtype(),
            "pl_letter": StringDtype(),
        }

    def standardize_catalog(self) -> None:
        """
        Standardize the EPIC catalog data.

        This method performs the following operations:
        1. Sets the catalog name and catalog-specific names.
        2. Renames columns to standard names used across all catalogs.
        3. Filters data based on the default_flag.
        4. Adds and modifies columns such as Kepler_host, letter, and alias.
        5. Converts discovery methods to standard format.

        :param self: The instance of the Epic class.
        :type self: Epic
        :return: None
        :rtype: None
        """

        # Set the catalog name and catalog name
        self.data["catalog"] = self.name
        self.data["catalog_name"] = self.data["pl_name"]
        self.data["catalog_host"] = self.data["hostname"]

        # Rename columns
        self.data = self.data.rename(
            columns={
                "pl_name": "name",
                "discoverymethod": "discovery_method",
                "pl_orbper": "p",
                "pl_orbpererr2": "p_min",
                "pl_orbpererr1": "p_max",
                "pl_orbsmax": "a",
                "pl_orbsmaxerr2": "a_min",
                "pl_orbsmaxerr1": "a_max",
                "pl_orbeccen": "e",
                "pl_orbeccenerr2": "e_min",
                "pl_orbeccenerr1": "e_max",
                "pl_orbincl": "i",
                "pl_orbinclerr2": "i_min",
                "pl_orbinclerr1": "i_max",
                "pl_radj": "r",
                "pl_radjerr2": "r_min",
                "pl_radjerr1": "r_max",
                "disc_year": "discovery_year",
                "rv_flag": "RV",
                "tran_flag": "Transit",
                "ttv_flag": "TTV",
                "pl_massj": "mass",
                "pl_massjerr2": "mass_min",
                "pl_massjerr1": "mass_max",
                "pl_msinij": "msini",
                "pl_msinijerr2": "msini_min",
                "pl_msinijerr1": "msini_max",
                "hostname": "host",
                "st_age": "Age (Gyrs)",
                "st_ageerr1": "Age_max",
                "st_ageerr2": "Age_min",
                "st_mass": "Mstar",
                "st_masserr1": "Mstar_max",
                "st_masserr2": "Mstar_min",
                "pl_refname": "reference",
            }
        )

        # Filter data based on default_flag
        self.data = self.data[self.data.default_flag == 1]

        # Add Kepler_host column
        self.data["Kepler_host"] = self.data.k2_name
        self.data["Kepler_host"].fillna(self.data["host"], inplace=True)
        self.data["Kepler_host"] = self.data.apply(
            lambda row: row["Kepler_host"].rstrip(" bcdefghi"), axis=1
        )

        # Add letter column
        self.data["letter"] = self.data.pl_letter.replace("", np.nan).fillna(
            self.data.name.apply(lambda row: row[-3:])
        )
        self.data.k2_name = self.data.k2_name.fillna(
            self.data.host + " " + self.data.letter
        )
        # Fill missing values in hd_name, hip_name, tic_id, and gaia_id columns
        self.data[["hd_name", "hip_name", "tic_id", "gaia_id"]] = self.data[
            ["hd_name", "hip_name", "tic_id", "gaia_id"]
        ].fillna("")

        # Add alias column
        self.data["alias"] = self.data[
            ["tic_id", "hip_name", "hd_name", "gaia_id", "Kepler_host"]
        ].agg(",".join, axis=1)

        # Remove nan from alias
        for i in self.data.index:
            self.data.at[i, "alias"] = (
                self.data.at[i, "alias"]
                .replace("nan,", "")
                .replace(",,", ",")
                .lstrip(",")
            )

        # Convert discovery methods
        self.data = Utils.convert_discovery_methods(self.data)

        # Logging
        logging.info("Catalog standardized.")

    def convert_coordinates(self) -> None:
        """
        Convert coordinates to decimal degrees.

        This method is a placeholder and does not perform any operations,
        as the EPIC catalog already has coordinates in decimal degrees.

        :param self: An instance of class Epic
        :type self: Epic
        :return: None
        :rtype: None
        :note: This function is not necessary for the EPIC catalog, as the coordinates are already in decimal degrees.

        """
        pass

    def remove_theoretical_masses(self) -> None:
        """
        Remove theoretical masses from the dataframe.

        This method is a placeholder and does not perform any operations,
        as it's not necessary for the EPIC catalog.

        :param self: The instance of the Epic class.
        :type self: Epic
        :return: None
        :rtype: None
        :note: This function is not necessary for the Epic catalog.
        """
        pass

    def handle_reference_format(self) -> None:
        """
        Standardize reference format and create URL columns for parameters.

        This method performs the following operations:
        1. Adds URL columns for each parameter (e, mass, msini, i, a, p, r).
        2. Extracts and standardizes URLs from the reference column.
        3. Replaces null values with empty strings in URL columns.

        :param self: The instance of the Epic class.
        :type self: Epic
        :return: None
        :rtype: None
        """
        # Add url column for each parameter
        for item in ["e", "mass", "msini", "i", "a", "p", "r"]:
            if item + "_url" not in self.data.columns:
                self.data[item + "_url"] = self.data["reference"]

        # Regular expression to extract url
        r = re.compile("href=(.*) target")

        # Loop through each parameter and each row in the dataframe
        for item in ["e", "mass", "msini", "i", "a", "p", "r"]:
            for i in self.data.index:
                # filter finite values by checking if x == x (false for nan, inf). If finite value, replace the
                # string with only the bibcode, else empty string
                if self.data.at[i, item + "_url"] == self.data.at[i, item + "_url"]:
                    url = self.data.at[i, item + "_url"]
                    link = r.findall(url)
                    self.data.at[i, item + "_url"] = (
                        link[0]
                        .replace(
                            "http://adsabs.harvard.edu/cgi-bin/nph-bib_query?bibcode=",
                            "",
                        )
                        .replace("http://adsabs.harvard.edu/abs/", "")
                        .replace("nan", "")
                        .replace("https://ui.adsabs.harvard.edu/abs/", "")
                        .replace("/abstract", "")
                    )

                else:
                    self.data.at[i, item + "_url"] = ""
            # Set all null values to empty string
            self.data.loc[self.data[item].isnull(), item + "_url"] = ""

        # Logging
        logging.info("Reference columns standardized.")

    def assign_status(self) -> None:
        """
        Assign status to each entry based on the 'disposition' column.

        This method maps the values in the 'disposition' column to standard status values:
        - 'CONFIRMED' for confirmed planets
        - 'CANDIDATE' for candidate planets
        - 'FALSE POSITIVE' for false positives and refuted planets

        The method also logs the updated status counts.

        :param self: The instance of the Epic class.
        :type self: Epic
        :return: None
        :rtype: None
        """
        for i in self.data.index:
            if "CONFIRMED" in self.data.at[i, "disposition"]:
                self.data.at[i, "status"] = "CONFIRMED"
            elif "CANDIDATE" in self.data.at[i, "disposition"]:
                self.data.at[i, "status"] = "CANDIDATE"
            elif "FALSE POSITIVE" in self.data.at[i, "disposition"]:
                self.data.at[i, "status"] = "FALSE POSITIVE"
            elif "REFUTED" in self.data.at[i, "disposition"]:
                self.data.at[i, "status"] = "FALSE POSITIVE"

        # Logging
        logging.info("Status column assigned.")
        logging.info("Updated Status:")
        logging.info(self.data.status.value_counts())
