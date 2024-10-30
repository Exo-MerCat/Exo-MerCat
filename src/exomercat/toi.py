import logging

import numpy as np
import pandas as pd
import pyvo
from astropy import constants as const
from astropy.table import Table
from pandas import Int64Dtype, Float64Dtype, StringDtype

from .catalogs import Catalog
from .utility_functions import UtilityFunctions as Utils



class Toi(Catalog):
    """
    A class representing the TESS Objects of Interest (TOI) catalog.

    This class inherits from the Catalog base class and provides specific
    functionality for handling and processing data from the TOI catalog.
    It includes methods for standardizing the catalog data, converting
    coordinates, handling references, and assigning status to the objects.
    """


    def __init__(self) -> None:
        """
        Initialize the Toi class.

        This method sets up the instance of the Toi class by:

        1. Calling the parent class initializer.

        2. Setting the catalog name to "toi".

        3. Initializing the data attribute as None.

        4. Defining the expected columns and their data types for this catalog.

        :param self: An instance of class Toi
        :type self: Toi
        :return: None
        :rtype: None
       """
        super().__init__()
        self.name = "toi"
        self.data = None
        self.columns = {
            "tid": Int64Dtype(),
            "toi": Float64Dtype(),
            "toidisplay": StringDtype(),
            "toipfx": Int64Dtype(),
            "ctoi_alias": Float64Dtype(),
            "pl_pnum": Int64Dtype(),
            "tfopwg_disp": StringDtype(),
            "ra": Float64Dtype(),
            "dec": Float64Dtype(),
            "pl_orbper": Float64Dtype(),
            "pl_orbpererr1": Float64Dtype(),
            "pl_orbpererr2": Float64Dtype(),
            "pl_orbpersymerr": Int64Dtype(),
            "pl_orbperlim": Int64Dtype(),
            "pl_rade": Float64Dtype(),
            "pl_radeerr1": Float64Dtype(),
            "pl_radeerr2": Float64Dtype(),
            "pl_radesymerr": Int64Dtype(),
            "pl_radelim": Int64Dtype(),
            "toi_created": StringDtype(),
        }

    def standardize_catalog(self) -> None:
        """
        Standardize the TESS Objects of Interest catalog data.

        This method performs the following operations:

        1. Creates TOI, TOI_host, and TIC_host columns.

        2. Runs a TAP query to gather aliases for TIC stars.

        3. Renames columns to standard names used across all catalogs.

        4. Adds missing columns with NaN values.

        5. Converts radius and its uncertainties from Earth to Jupiter units.

        6. Adds discovery year and method.

        7. Sets a reference value for all entries.

        :param self: An instance of class Toi
        :type self: Toi
        :return: None
        :rtype: None
        """

        # Create TOI, TOI_host, TIC_host columns
        self.data["catalog"] = self.name
        self.data["TOI"] = "TOI-" + self.data["toi"].astype(str)

        self.data["TIC_host"] = "TIC " + self.data["tid"].astype(str)
        self.data["TOI_host"] = "TOI-" + self.data["toipfx"].astype(str)

        self.data["TIC_host2"] = "TIC-" + self.data["tid"].astype(str)
        self.data["TOI_host2"] = "TOI " + self.data["toipfx"].astype(str)

        self.data["host"] = self.data["TIC_host"]
        self.data["letter"] = self.data["TOI"].str[-3:]
        self.data["alias"] = self.data[["TOI_host", "TOI_host2", "TIC_host2"]].agg(
            ",".join, axis=1
        )

        # Run the TAP query
        tap_service = pyvo.dal.TAPService(" http://TAPVizieR.cds.unistra.fr/TAPVizieR/tap/")

        query = """SELECT tc.tid,  db.TIC, db.UCAC4, db."2MASS", db.WISEA, db.GAIA, db.KIC, db.HIP, db.TYC FROM "IV/39/tic82" AS db JOIN TAP_UPLOAD.t1 AS tc ON db.TIC = tc.tid"""
        t2 = Table.from_pandas(self.data[["tid"]])
        result = Utils.perform_query(tap_service, query, uploads_dict={"t1": t2})
        result["tid"] = result["tid"].astype(int)
        result = result[["tid", "ids"]]
        result=result.drop_duplicates()
        for tid in result["tid"]:
            self.data.loc[self.data["tid"] == tid, "ids"] = (
                result.loc[result["tid"] == tid, "ids"].values[0]
            )

        # Save to aliases
        self.data["alias"] = self.data["alias"] + "," + self.data["ids"]

        for i in self.data.index:
            self.data.at[i, "alias"] = ",".join(
                sorted([x for x in set(self.data.at[i, "alias"].split(",")) if x])
            )

        # Rename columns
        self.data["name"] = self.data["TOI"]
        self.data["catalog_name"] = self.data["TOI"]
        self.data["catalog_host"] = self.data["host"]
        self.data = self.data.rename(
            columns={
                "pl_orbper": "p",
                "pl_orbpererr2": "p_min",
                "pl_orbpererr1": "p_max",
            }
        )

        # Add missing columns
        self.data["mass"] = np.nan
        self.data["mass_min"] = np.nan
        self.data["mass_max"] = np.nan
        self.data["msini"] = np.nan
        self.data["msini_min"] = np.nan
        self.data["msini_max"] = np.nan
        self.data["a"] = np.nan
        self.data["a_min"] = np.nan
        self.data["a_max"] = np.nan
        self.data["e"] = np.nan
        self.data["e_min"] = np.nan
        self.data["e_max"] = np.nan
        self.data["i"] = np.nan
        self.data["i_min"] = np.nan
        self.data["i_max"] = np.nan

        # Convert to correct units
        self.data["r"] = self.data["pl_rade"] * const.R_earth / const.R_jup
        self.data["r_min"] = self.data["pl_radeerr2"] * const.R_earth / const.R_jup
        self.data["r_max"] = self.data["pl_radeerr1"] * const.R_earth / const.R_jup

        # Add discovery year and method
        self.data["discovery_year"] = self.data["toi_created"].str[:4]

        self.data["discovery_method"] = "Transit"

        # Add reference
        self.data["reference"] = "toi"

        # Remove duplicates (which are relatively common in TOI)
        self.data=self.data.drop_duplicates()
        # Logging
        logging.info("Catalog standardized.")

    def convert_coordinates(self) -> None:
        """
        Convert coordinates to decimal degrees.

        This method is a placeholder and does not perform any operations,
        as the TESS Objects of Interest catalog already has coordinates in decimal degrees.

        :param self: An instance of class Toi
        :type self: Toi
        :return: None
        :rtype: None
        :note: This function is not necessary for the Toi catalog, as the coordinates are already in decimal degrees.

        """
        pass

    def remove_theoretical_masses(self) -> None:
        """
        Remove theoretical masses from the dataframe.

        This method is a placeholder and does not perform any operations,
        as the TESS Objects of Interest catalog does not include mass information.

        :param self: The instance of the Toi class.
        :type self: Toi
        :return: None
        :rtype: None
        :note: This function is not necessary for the Toi catalog, since there are no masses.
        """
        pass

    def handle_reference_format(self) -> None:
        """
        Standardize the reference format for various parameters.

        This method creates a '_url' column for each parameter (e, mass, msini, i, a, p, r),
        setting the value to 'toi' for non-null, finite values and an empty string otherwise.

        :param self: An instance of class Toi
        :type self: Toi
        :return: None
        :rtype: None
        """
        for item in ["e", "mass", "msini", "i", "a", "p", "r"]:
            self.data[item + "_url"] = self.data[item].apply(
                lambda x: "" if pd.isna(x) or np.isinf(x) else "toi"
            )
        logging.info("Reference columns standardized.")

    def assign_status(self):
        """
        Assign status to each entry based on the 'tfopwg_disp' column.

        This method maps the disposition values to standard status categories:

        - "APC" -> "CONTROVERSIAL"

        - "CP"  -> "CONFIRMED"

        - "FA"  -> "FALSE POSITIVE"

        - "FP"  -> "FALSE POSITIVE"

        - "KP"  -> "CONFIRMED"

        - "PC"  -> "CANDIDATE"

        - ""    -> "UNKNOWN"

        The method also logs the updated status counts.

        :param self: An instance of class Toi
        :type self: Toi
        :return: None
        :rtype: None
        """
        replace_dict = {
            "APC": "CONTROVERSIAL",
            "CP": "CONFIRMED",
            "FA": "FALSE POSITIVE",
            "FP": "FALSE POSITIVE",
            "KP": "CONFIRMED",
            "PC": "CANDIDATE",
            "": "UNKNOWN",
        }
        self.data["status"] = self.data["tfopwg_disp"].replace(replace_dict)

        # Logging
        logging.info("Status column assigned.")
        logging.info("Updated Status:")
        logging.info(self.data.status.value_counts())
