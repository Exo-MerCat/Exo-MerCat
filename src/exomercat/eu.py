import logging

import numpy as np
import pandas as pd

from .catalogs import Catalog
from .utility_functions import UtilityFunctions as Utils


class Eu(Catalog):
    def __init__(self) -> None:
        """
        This function is called when the class is instantiated. It sets up the object with a name attribute that can
        be used to refer to this particular instance of Eu.

        :param self: An instance of class Eu
        :type self: Eu
        :return: None
        :rtype: None
        """
        super().__init__()
        self.name = "eu"

    def check_input_columns(self) -> None:
        """
        The check_input_columns ensures that the .csv file contains the columns the script needs later.

        :param self: An instance of class Catalog
        :return: None
        :rtype: None
        """
        error_list=''
        # check that the table contains the names of the columns that we need

        columns= ["detection_type","orbital_period","orbital_period_error_max","orbital_period_error_min",
         "semi_major_axis","semi_major_axis_error_max","semi_major_axis_error_min","eccentricity",
         "eccentricity_error_max","eccentricity_error_min","inclination","inclination_error_max",
         "inclination_error_min","name","updated","discovered","mass","mass_error_max", "mass_error_min",
         "mass_sini","mass_sini_error_max","mass_sini_error_min","radius","radius_error_max",
         "radius_error_min","mass_measurement_type","radius_measurement_type","star_name","name",
         "alternate_names","star_alternate_names","planet_status","ra","dec",
        ]
        missing_columns = ''
        for col in columns:
            if col not in self.data.keys():
                missing_columns = ",".join([col, missing_columns])
        if missing_columns != '':
            print("Check input columns.........FAILED. \n\tMissing columns: " + missing_columns.rstrip(',') + '\n')
        else:
            print('Check input columns.........OK')




    def standardize_catalog(self) -> None:
        """
        This function processes raw data from a catalog. It standardizes the data format, renames columns,
        adds new columns like aliases, discovery methods, and references. Finally, it performs some string
        manipulations on the data and converts discovery methods.

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
        This function removes theoretical masses from the dataframe by setting the mass/msini values to NaN where the
        MASSPROV column contains "Theoretical" and the radii where the RADPROV column contains "Theoretical".

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
        This function sets the status of each planet in the data DataFrame based on the value in the planet_status
        column. It first sets all planets with confirmed planets as CONFIRMED. Then, it looks for candidate,
        unconfirmed, and controversial planets and sets them as CANDIDATE. Finally, it looks for retracted planets
        and sets them as FALSE POSITIVE.

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
        The handle_reference_format function is used to create a URL for each reference in the references list. Since
        the Exoplanet Encyclopaedia table does not provide references, we just use "eu" as a keyword in the url.

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
        Convert the right ascension (RA) and declination (Dec) columns of the dataframe to decimal degrees. This
        function is not implemented as the Eu already has coordinates in decimal degrees.

        :param self: An instance of class Eu
        :type self: Eu
        :return: None
        :rtype: None
        :note:  It is not necessary for Eu, as the coordinates are already in decimal degrees.
        """
        pass
