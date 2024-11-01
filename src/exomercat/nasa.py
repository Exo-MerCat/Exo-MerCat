import logging
import re
from pandas import Int64Dtype, Float64Dtype, StringDtype

import numpy as np

from .catalogs import Catalog
from .utility_functions import UtilityFunctions as Utils


class Nasa(Catalog):
    """
    A class representing the NASA Exoplanet Archive catalog.

    This class inherits from the Catalog base class and provides specific
    functionality for processing and standardizing data from the NASA
    Exoplanet Archive. It includes methods for initializing the catalog,
    standardizing the data format, handling references, assigning planet
    status, and removing theoretical masses and radii.
    """
   

    def __init__(self) -> None:
        """
        Initialize the Nasa class.

        This method sets up the instance of the Nasa class by:

        1. Calling the parent class initializer.

        2. Setting the catalog name to "nasa".

        3. Defining the expected columns and their data types for this catalog.

        :param self: An instance of class Nasa
        :type self: Nasa
        :return: None
        :rtype: None
        """
        super().__init__()
        self.name = "nasa"
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
            "disc_refname": StringDtype(),
            "rv_flag": Int64Dtype(),
            "tran_flag": Int64Dtype(),
            "ttv_flag": Int64Dtype(),
            "pl_bmassj": Float64Dtype(),
            "pl_bmassjerr2": Float64Dtype(),
            "pl_bmassjerr1": Float64Dtype(),
            "pl_bmassprov": StringDtype(),
            "hostname": StringDtype(),
            "pl_radj_reflink": StringDtype(),
            "pl_orbeccen_reflink": StringDtype(),
            "pl_orbsmax_reflink": StringDtype(),
            "pl_orbper_reflink": StringDtype(),
            "pl_orbincl_reflink": StringDtype(),
            "pl_bmassj_reflink": StringDtype(),
            "hd_name": StringDtype(),
            "hip_name": StringDtype(),
            "tic_id": StringDtype(),
            "gaia_id": StringDtype(),
        }

    def standardize_catalog(self) -> None:
        """
        Standardize the NASA Exoplanet Archive catalog data.

        This method performs the following operations:

        1. Sets the catalog name.

        2. Renames columns to standard names used across all catalogs.

        3. Adds new columns such as catalog_name and catalog_host.

        4. Splits best mass into mass and msini.

        5. Processes and standardizes the alias information.

        6. Converts discovery methods to a standard format.
        
        :param self: An instance of class Nasa
        :type self: Nasa
        :return: None
        :rtype: None
        """

        # Standardizing data format
        self.data["catalog"] = self.name

        # Renaming columns
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
                "disc_refname": "reference",
                "rv_flag": "RV",
                "tran_flag": "Transit",
                "ttv_flag": "TTV",
                "pl_bmassj": "bestmass",
                "pl_bmassjerr2": "bestmass_min",
                "pl_bmassjerr1": "bestmass_max",
                "pl_bmassprov": "bestmass_provenance",
                "hostname": "host",
                # "st_age": "Age (Gyrs)",
                # "st_ageerr1": "Age_max",
                # "st_ageerr2": "Age_min",
                # "st_mass": "Mstar",
                # "st_masserr1": "Mstar_max",
                # "st_masserr2": "Mstar_min",
                "pl_radj_reflink": "r_url",
                "pl_orbeccen_reflink": "e_url",
                "pl_orbsmax_reflink": "a_url",
                "pl_orbper_reflink": "p_url",
                "pl_orbincl_reflink": "i_url",
                "pl_bmassj_reflink": "bestmass_url",
            }
        )

        # Add new columns
        self.data["catalog_name"] = self.data["name"]
        self.data["catalog_host"] = self.data["host"]

        # if "default_flag" in self.data.columns:
        #     # this happens when you use PLANETARY SYSTEMS TABLE
        #     self.data = self.data[self.data.default_flag == 1]

        # Split best mass into mass and msini
        self.data["mass"] = np.nan
        self.data["mass_min"] = np.nan
        self.data["mass_max"] = np.nan
        self.data["msini"] = np.nan
        self.data["msini_min"] = np.nan
        self.data["msini_max"] = np.nan

        if "bestmass" in self.data.columns:
            # this happens when you use PLANETARY COMPOSITE PARAMETERS TABLE
            self.sort_bestmass_to_mass_or_msini()

        # String manipulations on alias
        self.data[["hd_name", "hip_name", "tic_id", "gaia_id"]] = self.data[
            ["hd_name", "hip_name", "tic_id", "gaia_id"]
        ].fillna("")

        self.data["alias"] = (
            self.data["hd_name"]
            .str.cat(self.data[["hip_name", "tic_id", "gaia_id"]].fillna(""), sep=",")
            .str.lstrip(",")
        )

        # Convert discovery methods
        self.data = Utils.convert_discovery_methods(self.data)

        # Logging
        logging.info("Catalog standardized.")

    def sort_bestmass_to_mass_or_msini(self) -> None:
        """
        Sort the 'bestmass' values into either 'mass' or 'msini' columns.

        This method categorizes the 'bestmass' values based on the 'bestmass_provenance':

        - If 'Mass', values are placed in the 'mass' columns.

        - If 'Msini', values are placed in the 'msini' columns.

        - If 'M-R relationship' or 'Msin(i)/sin(i)', both 'mass' and 'msini' are set to NaN.

        - For any other provenance, a RuntimeError is raised.

        :param self: An instance of the Nasa class
        :type self: Nasa
        :raise RuntimeError: If 'bestmass' is not a mass or an 'msini'
        :return: None
        :rtype: None
        """

        # Iterate over the rows of the dataframe
        for i in self.data.index:
            # If 'bestmass' is a mass, sort it into 'mass'
            if self.data.at[i, "bestmass_provenance"] == "Mass":
                self.data.at[i, "mass"] = self.data.at[i, "bestmass"]
                self.data.at[i, "mass_max"] = self.data.at[i, "bestmass_max"]
                self.data.at[i, "mass_min"] = self.data.at[i, "bestmass_min"]
                self.data.at[i, "mass_url"] = self.data.at[i, "bestmass_url"]

            # If 'bestmass' is an 'msini', sort it into 'msini'
            elif self.data.at[i, "bestmass_provenance"] == "Msini":
                self.data.at[i, "msini"] = self.data.at[i, "bestmass"]
                self.data.at[i, "msini_max"] = self.data.at[i, "bestmass_max"]
                self.data.at[i, "msini_min"] = self.data.at[i, "bestmass_min"]
                self.data.at[i, "msini_url"] = self.data.at[i, "bestmass_url"]

            # If 'bestmass' is not a mass or an 'msini', set both 'mass' and 'msini' to NaN
            elif (self.data.at[i, "bestmass_provenance"] == "M-R relationship") or (
                self.data.at[i, "bestmass_provenance"] == "Msin(i)/sin(i)"
            ):
                self.data.at[i, "msini"] = np.nan
                self.data.at[i, "msini_max"] = np.nan
                self.data.at[i, "msini_min"] = np.nan
                self.data.at[i, "msini_url"] = np.nan
                self.data.at[i, "mass"] = np.nan
                self.data.at[i, "mass_max"] = np.nan
                self.data.at[i, "mass_min"] = np.nan
                self.data.at[i, "mass_url"] = np.nan

            # If 'bestmass' is not a mass or an 'msini', raise an error
            else:
                print(self.data.at[i, "bestmass_provenance"])
                raise RuntimeError

    def handle_reference_format(self) -> None:
        """
        Standardize the reference format for various parameters.

        This method performs the following for each parameter (e, mass, msini, i, a, p, r):

        1. Ensures a '_url' column exists for each parameter.

        2. Extracts the bibcode from the reference URL.

        3. Standardizes the URL format.

        4. Sets empty strings for null values.

        :param self: The instance of the Nasa class.
        :type self: Nasa
        :return: None
        :rtype: None
        """
        # Add url column for each parameter
        for item in ["e", "mass", "msini", "i", "a", "p", "r"]:
            if item + "_url" not in self.data.columns:
                self.data[item + "_url"] = self.data["reference"]

        # Regular expression to extract url
        r = re.compile("href=(.*) target")

        # Iterate over the rows of the dataframe
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
                # Set all null values to empty string
                else:
                    self.data.at[i, item + "_url"] = ""
            self.data.loc[self.data[item].isnull(), item + "_url"] = ""

        # Logging
        logging.info("Reference columns standardized.")

    def assign_status(self) -> None:
        """
        Assign status to each entry in the catalog.

        For the NASA Exoplanet Archive, all entries are set to "CONFIRMED" by default.

        :param self: An instance of the Nasa class.
        :type self: Nasa
        :return: None
        :rtype: None
        """

        # Set all planets with confirmed planets as CONFIRMED
        self.data["status"] = "CONFIRMED"

        # Logging
        logging.info("Status column assigned.")
        logging.info("Updated Status:")
        logging.info(self.data.status.value_counts())

    def convert_coordinates(self) -> None:
        """
        Convert coordinates to decimal degrees.

        This method is a placeholder and does not perform any operations,
        as the NASA Exoplanet Archive catalog already has coordinates in decimal degrees.

        :param self: An instance of class Nasa
        :type self: Nasa
        :return: None
        :rtype: None
        :note:  It is not necessary for Nasa, as the coordinates are already in decimal degrees.
        """
        pass

    def remove_theoretical_masses(self) -> None:
        """
        Remove theoretical masses and radii from the catalog.

        This method removes values for mass, msini, and radius (including their error ranges and URLs) 
        where the corresponding URL contains "Calculated", indicating a theoretical value.

        :param self: An instance of the Nasa class.
        :type self: Nasa
        :return: None
        :rtype: None
        """

        # Iterate over different value suffixes
        for value in ["", "_min", "_max", "_url"]:
            # Remove theoretical masses
            self.data.loc[
                self.data["mass_url"].str.contains("Calculated", na=False),
                "mass" + value,
            ] = np.nan

            # Remove theoretical msini
            self.data.loc[
                self.data["msini_url"].str.contains("Calculated", na=False),
                "msini" + value,
            ] = np.nan

            # Remove theoretical radii
            self.data.loc[
                self.data["r_url"].str.contains("Calculated", na=False), "r" + value
            ] = np.nan

        # Logging
        logging.info("Theoretical masses/radii removed.")
