import logging

import numpy as np
import pandas as pd
import pyvo
from astropy import constants as const

from .catalogs import Catalog
from .utility_functions import UtilityFunctions as Utils

tap_service = pyvo.dal.TAPService(" http://TAPVizieR.u-strasbg.fr/TAPVizieR/tap/")


class Toi(Catalog):
    """
    The Toi class contains all methods and attributes related to the TESS Objects of Interest catalog.
    """

    def __init__(self) -> None:
        """
        The __init__ function is called when the class is instantiated. It sets up the instance of the class,
        and defines any variables that will be used by all instances of this class.
        """
        super().__init__()
        self.name = "toi"
        self.data = None

    def check_input_columns(self) -> None:
        """
        The check_input_columns ensures that the .csv file contains the columns the script needs later.

        :param self: An instance of class Catalog
        :return: None
        :rtype: None
        """
        # check that the table contains the names of the columns that we need

        columns =['tid', 'toi', 'toidisplay', 'toipfx', 'ctoi_alias', 'pl_pnum',
       'tfopwg_disp', 'ra',
       'dec',  'pl_orbper', 'pl_orbpererr1', 'pl_orbpererr2',
       'pl_orbpersymerr', 'pl_orbperlim',
       'pl_rade', 'pl_radeerr1', 'pl_radeerr2', 'pl_radesymerr', 'pl_radelim',
     'toi_created', ]

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
        Standardize the dataframe columns and values. It assigns new columns to the dataframe. It runs the run_sync
        method to gather the aliases of the TIC stars, to be added to the alias column. It assigns NaN to the missing
        data columns in the dataframe. It calculates the radius and its uncertainties by converting from Earth units
        to Jupiter units.

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
        counter = -1
        for tid in self.data.tid.unique():
            counter += 1
            result = tap_service.run_sync(
                """SELECT tic.TIC,  tic.UCAC4,  tic."2MASS",  tic.WISEA,  tic.GAIA,  tic.KIC,
                      tic.HIP,   tic.TYC
                      FROM "IV/38/tic" as tic
                      WHERE  tic.TIC = """
                + str(tid)
            )
            Utils.print_progress_bar(counter, len(self.data.tid.unique()), prefix='Progress:', suffix='Complete')

            # print(
            #     "Done "
            #     + str(round(counter / len(self.data.tid.unique()), 2) * 100)
            #     + "% of the groups.",
            #     end="\r",
            # )
            result = result.to_table().to_pandas()
            result = result.astype(str)
            result["UCAC4"] = "UCAC4 " + result["UCAC4"].replace("", "<NA>")
            result["2MASS"] = "2MASS J" + result["2MASS"].replace("", "<NA>")
            result["WISEA"] = "WISE " + result["WISEA"].replace("", "<NA>")
            result["GAIA"] = "Gaia DR2 " + result["GAIA"].replace("", "<NA>")
            result["KIC"] = "KIC " + result["KIC"].replace("", "<NA>")
            result["HIP"] = "HIP " + result["HIP"].replace("", "<NA>")
            result["TYC"] = "TYC " + result["TYC"].replace("", "<NA>")
            for col in result.columns:
                result.loc[result[col].str.contains("<NA>"), col] = ""
            result["alias_vizier"] = result[
                ["UCAC4", "2MASS", "WISEA", "GAIA", "KIC", "HIP", "TYC"]
            ].agg(",".join, axis=1)
            self.data.loc[self.data.tid == tid, "alias_vizier"] = result[
                "alias_vizier"
            ][0]

        # Save to aliases
        self.data["alias"] = self.data["alias"] + "," + self.data["alias_vizier"]

        for i in self.data.index:
            self.data.at[i, "alias"] = ",".join(
                sorted([x for x in set(self.data.at[i, "alias"].split(",")) if x])
            )

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
        self.data["Age (Gyrs)"] = np.nan
        self.data["Age_max"] = np.nan
        self.data["Age_min"] = np.nan
        self.data["Mstar"] = np.nan
        self.data["Mstar_max"] = np.nan
        self.data["Mstar_min"] = np.nan

        # Convert to correct units
        self.data["r"] = self.data["pl_rade"] * const.R_earth / const.R_jup
        self.data["r_min"] = self.data["pl_radeerr2"] * const.R_earth / const.R_jup
        self.data["r_max"] = self.data["pl_radeerr1"] * const.R_earth / const.R_jup

        # Add discovery year and method
        self.data["discovery_year"] = self.data["toi_created"].str[:4]

        self.data["discovery_method"] = "Transit"

        self.data["reference"] = "toi"

        # Logging
        logging.info("Catalog standardized.")

    def convert_coordinates(self) -> None:
        """
        Convert the `ra` and `dec columns of the dataframe to decimal degrees. Not implemented for Toi.

        :param self: An instance of class Toi
        :type self: Toi
        :return: None
        :rtype: None
        :note: This function is not necessary for the Toi catalog, as the coordinates are already in decimal degrees.

        """
        pass

    def remove_theoretical_masses(self) -> None:
        """
        Remove theoretical masses from the dataframe. Not used for the Toi catalog, since there are no masses

        :param self: The instance of the Toi class.
        :type self: Toi
        :return: None
        :rtype: None
        :note: This function is not necessary for the Toi catalog, since there are no masses.
        """
        pass

    def handle_reference_format(self) -> None:
        """
        he handle_reference_format function is used to create a URL for each reference in the references list. Since
        the Exoplanet Encyclopaedia table does not provide references, we just use "toi" as a keyword in the url.

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
        Assigns status based on the disposition values.

        The function uses a dictionary to map the disposition values to status categories.

        Status Categories:
        - "APC" -> "CONTROVERSIAL"
        - "CP"  -> "CONFIRMED"
        - "FA"  -> "FALSE POSITIVE"
        - "FP"  -> "FALSE POSITIVE"
        - "KP"  -> "CONFIRMED"
        - "PC"  -> "CANDIDATE"
        - ""    -> "UNKNOWN"

        The function updates the 'status' column in the data attribute using the replace method.

        Logging is used to inform about the assignment of the status column and to display the updated status counts.
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
