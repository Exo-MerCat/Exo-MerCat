import logging

import numpy as np
import pandas as pd
from pandas import Int64Dtype, Float64Dtype, StringDtype

from .catalogs import Catalog
from .utility_functions import UtilityFunctions as Utils


class Eu(Catalog):
    """
    A class representing the Exoplanet Encyclopedia (EU) catalog.

    This class inherits from the Catalog class and provides specific functionality
    for handling and processing data from the Exoplanet Encyclopedia. It includes
    methods for standardizing the catalog, removing theoretical masses, assigning
    status to planets, handling reference formats, and converting coordinates.
    """
    def __init__(self) -> None:
        """
        Initialize the Eu (Exoplanet Encyclopaedia) class.

        This method sets up the instance of the Eu class by:

        1. Calling the parent class initializer.

        2. Setting the catalog name to "eu".

        3. Defining the expected columns and their data types for this catalog.

        :param self: An instance of class Eu
        :type self: Eu
        :return: None
        :rtype: None
        """
        super().__init__()
        self.name = "eu"
        self.columns = {
            "detection_type": StringDtype(),
            "orbital_period": Float64Dtype(),
            "orbital_period_error_max": Float64Dtype(),
            "orbital_period_error_min": Float64Dtype(),
            "semi_major_axis": Float64Dtype(),
            "semi_major_axis_error_max": Float64Dtype(),
            "semi_major_axis_error_min": Float64Dtype(),
            "eccentricity": Float64Dtype(),
            "eccentricity_error_max": Float64Dtype(),
            "eccentricity_error_min": Float64Dtype(),
            "inclination": Float64Dtype(),
            "inclination_error_max": Float64Dtype(),
            "inclination_error_min": Float64Dtype(),
            "name": StringDtype(),
            "updated": StringDtype(),
            "discovered": Int64Dtype(),
            "mass": Float64Dtype(),
            "mass_error_max": Float64Dtype(),
            "mass_error_min": Float64Dtype(),
            "mass_sini": Float64Dtype(),
            "mass_sini_error_max": Float64Dtype(),
            "mass_sini_error_min": Float64Dtype(),
            "radius": Float64Dtype(),
            "radius_error_max": Float64Dtype(),
            "radius_error_min": Float64Dtype(),
            "mass_measurement_type": StringDtype(),
            "radius_measurement_type": StringDtype(),
            "star_name": StringDtype(),
            "alternate_names": StringDtype(),
            "star_alternate_names": StringDtype(),
            "planet_status": StringDtype(),
            "ra": Float64Dtype(),
            "dec": Float64Dtype(),
        }

    def standardize_catalog(self) -> None:
        """
        Standardize the Exoplanet Encyclopaedia catalog data.

        This method performs the following operations:

        1. Sets the catalog name.

        2. Replaces "None" and "nan" values.

        3. Renames columns to standard names used across all catalogs.

        4. Adds new columns such as catalog_name, catalog_host, and reference.

        5. Processes and standardizes the alias information.

        6. Converts discovery methods to a standard format.

        :param self: An instance of class Eu
        :type self: Eu
        :return: None
        :rtype: None
        """
        # Standardizing data format
        self.data["catalog"] = self.name

        self.data = self.data.replace("None", "").replace("nan", np.nan)

        # Renaming columns
        self.data = self.data.rename(
            columns={
                "detection_type": "discovery_method",
                "orbital_period": "p",
                "orbital_period_error_max": "p_max",
                "orbital_period_error_min": "p_min",
                "semi_major_axis": "a",
                "semi_major_axis_error_max": "a_max",
                "semi_major_axis_error_min": "a_min",
                "eccentricity": "e",
                "eccentricity_error_max": "e_max",
                "eccentricity_error_min": "e_min",
                "inclination": "i",
                "inclination_error_max": "i_max",
                "inclination_error_min": "i_min",
                "name": "name",
                "updated": "Update",
                "discovered": "discovery_year",
                "mass": "mass",
                "mass_error_max": "mass_max",
                "mass_error_min": "mass_min",
                "mass_sini": "msini",
                "mass_sini_error_max": "msini_max",
                "mass_sini_error_min": "msini_min",
                "radius": "r",
                "radius_error_max": "r_max",
                "radius_error_min": "r_min",
                "mass_measurement_type": "MASSPROV",
                "radius_measurement_type": "RADPROV",
                "star_name": "host",
            }
        )

        # Add new columns
        self.data["catalog_name"] = self.data["name"]
        self.data["catalog_host"] = self.data["host"]

        self.data["reference"] = self.name
        self.data["alternate_names"] = self.data["alternate_names"].fillna("")
        self.data["star_alternate_names"] = self.data["star_alternate_names"].fillna("")
        self.data["host"] = self.data["host"].fillna("")

        self.data["alias"] = self.data["alternate_names"].str.cat(
            self.data[["star_alternate_names"]], sep=","
        )

        # String manipulations on alias
        for i in self.data.index:
            alias_polished = ""
            for al in self.data.at[i, "alias"].split(","):
                al = Utils.standardize_string(al)
                alias_polished = alias_polished + "," + al.rstrip().lstrip()

            self.data.at[i, "alias"] = alias_polished.lstrip(",")

        # Convert discovery methods
        self.data = Utils.convert_discovery_methods(self.data)

        # Logging
        logging.info("Catalog standardized.")

    def remove_theoretical_masses(self) -> None:
        """
        Remove theoretical masses and radii from the dataframe.

        This method sets mass, msini, and radius values (including their error ranges) to NaN 
        where the MASSPROV or RADPROV columns indicate theoretical values.

        :param self: An instance of the Eu class.
        :type self: Eu
        :return: None
        :rtype: None
        """
        for value in ["", "_min", "_max"]:
            self.data.loc[
                self.data["MASSPROV"].str.contains("Theoretical", na=False),
                "mass" + value,
            ] = np.nan
            self.data.loc[
                self.data["MASSPROV"].str.contains("Theoretical", na=False),
                "msini" + value,
            ] = np.nan
            self.data.loc[
                self.data["RADPROV"].str.contains("Theoretical", na=False), "r" + value
            ] = np.nan

        # Logging
        logging.info("Theoretical masses/radii removed.")

    def assign_status(self) -> None:
        """
        Assign status to each entry based on the 'planet_status' column.

        This method maps the values in the 'planet_status' column to standard status values:

        - 'CONFIRMED' for confirmed planets

        - 'CANDIDATE' for candidate, unconfirmed, or controversial planets

        - 'FALSE POSITIVE' for retracted planets

        The method also logs the updated status counts.

        :param self: An instance of the Eu class.
        :type self: Eu
        :return: None
        :rtype: None
        """

        # Set all planets with confirmed planets as CONFIRMED
        self.data["status"] = "CONFIRMED"

        #
        self.data.loc[
            self.data["planet_status"].str.contains(
                "Candidate|Unconfirmed|Controversial"
            ),
            "status",
        ] = "CANDIDATE"

        # Set retracted planets as FALSE POSITIVE
        self.data.loc[
            self.data["planet_status"].str.contains("Retracted"), "status"
        ] = "FALSE POSITIVE"

        # Logging
        logging.info("Status column assigned.")
        logging.info("Updated Status:")
        logging.info(self.data.status.value_counts())

    def handle_reference_format(self) -> None:
        """
        Create placeholder URL references for each parameter.

        Since the Exoplanet Encyclopaedia does not provide specific references,
        this method creates a placeholder 'eu' URL for each non-null, finite parameter value.

        Parameters handled: e, mass, msini, i, a, p, r

        :param self: An instance of class Eu
        :type self: Eu
        :return: None
        :rtype: None
        """
        for item in ["e", "mass", "msini", "i", "a", "p", "r"]:
            self.data[item + "_url"] = self.data[item].apply(
                lambda x: "" if pd.isna(x) or np.isinf(x) else "eu"
            )
        logging.info("Reference columns standardized.")

    def convert_coordinates(self) -> None:
        """
        Convert coordinates to decimal degrees.

        This method is a placeholder and does not perform any operations,
        as the Exoplanet Encyclopaedia catalog already has coordinates in decimal degrees.

        :param self: An instance of class Eu
        :type self: Eu
        :return: None
        :rtype: None
        :note:  It is not necessary for Eu, as the coordinates are already in decimal degrees.
        """
        pass
