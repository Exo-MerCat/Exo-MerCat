import logging
import re

import numpy as np

from exo_mercat.catalogs import Catalog
from exo_mercat.utility_functions import UtilityFunctions as Utils


class Epic(Catalog):
    def __init__(self) -> None:
        """
        Initializes the Epic class object.

        This function is automatically called when an instance of the Epic class is created.
        It sets up the instance of the class by assigning a name and initializing data with an empty DataFrame.

        :param self: The instance of the Epic class.
        :type self: Epic
        :return: None
        :rtype: None
        """
        super().__init__()
        self.name = "epic"
        self.data = None

    def standardize_catalog(self) -> None:
        """
        Standardizes the catalog by renaming columns and adding useful columns derived from existing ones.

        This function takes the data from the K2/EPIC catalog and creates a new table with only the columns we need
        for our analysis. It also adds some useful columns that are derived from existing ones.

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
        Convert the right ascension (RA) and declination (Dec) columns of the dataframe to decimal degrees. This
        function is not implemented as the Epic catalog already has coordinates in decimal degrees.

        :param self: An instance of class Epic
        :type self: Epic
        :return: None
        :rtype: None
        :note: This function is not necessary for the EPIC catalog, as the coordinates are already in decimal degrees.

        """
        pass

    def remove_theoretical_masses(self) -> None:
        """
        Remove theoretical masses from the dataframe. Not used in the Epic catalog.

        :param self: The instance of the Epic class.
        :type self: Epic
        :return: None
        :rtype: None
        :note: This function is not necessary for the Epic catalog.
        """
        pass

    def handle_reference_format(self) -> None:
        """
        This function takes in a dataframe and replaces the reference column with a url column. It also adds columns
        for each of the seven parameters (e, mass, msini, i, a, P, and R) and sets them to be equal to the
        corresponding reference column. It then removes all rows where any of these parameters are null.

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
        Assigns the status of each row in the data DataFrame based on the value in the "disposition" column.

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


        logging.info("Status column assigned.")
        logging.info("Updated Status:")
        logging.info(self.data.status.value_counts())
