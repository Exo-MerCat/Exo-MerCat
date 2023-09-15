import glob
import os
import numpy as np
import re
import pandas as pd
import requests
from exo_mercat.utility_functions import UtilityFunctions as Utils
from exo_mercat.catalogs import Catalog
from datetime import date
from astropy.coordinates import SkyCoord
import astropy.units as u
import logging
from pathlib import Path


class Oec(Catalog):
    def __init__(self) -> None:
        """
        The __init__ function is called when the class is instantiated. It sets up the instance of the class, and
        defines any variables that will be used by all instances of this class.

        Parameters
        ----------
            self
                Represent the instance of the class
        """
        super().__init__()
        self.name = "oec"

    def download_catalog(self, url: str, filename: str, timeout: float = None) -> Path:
        file_path_str = filename + date.today().strftime("%m-%d-%Y") + ".csv"
        file_path_xml_str = filename + date.today().strftime("%m-%d-%Y") + ".xml.gz"

        if os.path.exists(file_path_str):
            logging.info("Reading existing file")

        else:
            try:
                result = requests.get(url, timeout=timeout)
                with open(file_path_xml_str, "wb") as f:
                    f.write(result.content)
                logging.info("Convert from .xml to .csv")
                Utils.convert_xmlfile_to_csvfile(file_path=file_path_xml_str)

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
            ):
                if len(glob.glob(filename + "*.csv")) > 0:
                    file_path_str = glob.glob(filename + "*.csv")[0]

                    logging.warning(
                        "Error fetching the catalog, taking a local copy: %s",
                        file_path_str,
                    )
                else:
                    raise ValueError("Could not find previous catalogs")

            try:
                result = requests.get(url, timeout=timeout)
                with open(file_path_xml_str, "wb") as f:
                    f.write(result.content)
                logging.info("Convert from .xml to .csv")
                Utils.convert_xmlfile_to_csvfile(file_path=file_path_xml_str)

            except BaseException:
                file_path_str = glob.glob(filename + "*.csv")[0]
                logging.warning(
                    "Error fetching the catalog, taking a local copy:", file_path_str
                )

        logging.info("Catalog downloaded.")
        return Path(file_path_str)

    def uniform_catalog(self) -> None:
        """
        The uniform_catalog function is used to standardize the dataframe columns and values.
        """
        self.data["catalog"] = self.name
        self.data["catalog_name"] = self.data["name"]

        self.data = self.data.replace({"None": np.nan})
        self.data = self.data.rename(
            columns={
                "name": "name",
                "discoverymethod": "discovery_method",
                "period": "p",
                "period_min": "p_min",
                "period_max": "p_max",
                "semimajoraxis": "a",
                "semimajoraxis_min": "a_min",
                "semimajoraxis_max": "a_max",
                "eccentricity": "e",
                "eccentricity_min": "e_min",
                "eccentricity_max": "e_max",
                "inclination": "i",
                "inclination_min": "i_min",
                "inclination_max": "i_max",
                "radius": "r",
                "radius_min": "r_min",
                "radius_max": "r_max",
                "discoveryyear": "discovery_year",
                "mass": "mass",
                "mass_min": "mass_min",
                "mass_max": "mass_max",
                "system_rightascension": "ra",
                "system_declination": "dec",
            }
        )
        self.data = self.data.reset_index()
        self.data["alias"] = self.data.alias.fillna("")
        self.data["masstype"] = self.data.masstype.fillna("mass")

        self.data["msini"] = np.nan
        self.data["msini_min"] = np.nan
        self.data["msinis_max"] = np.nan
        for i in self.data.index:
            if self.data.at[i, "masstype"] == "msini":
                self.data.at[i, "msini"] = self.data.at[i, "mass"]
                self.data.at[i, "msini_max"] = self.data.at[i, "mass_max"]
                self.data.at[i, "msini_min"] = self.data.at[i, "mass_min"]

                self.data.at[i, "mass"] = np.nan
                self.data.at[i, "mass_max"] = np.nan
                self.data.at[i, "mass_min"] = np.nan

        self.data["host"] = self.data.name.apply(lambda x: str(x[:-1]).strip())

        for ident in self.data.name:
            if not str(re.search("(\.0)\\d$", ident, re.M)) == "None":
                self.data.loc[self.data.name == ident, "host"] = ident[:-3].strip()
            elif not str(re.search("\\d$", ident, re.M)) == "None":
                self.data.loc[self.data.name == ident, "host"] = ident

        self.data = self.data.replace(
            {
                "astrometry": "Astrometry",
                "microlensing": "Microlensing",
                "imaging": "Imaging",
                "transit": "Transit",
                "timing": "Pulsar Timing",
                "RV": "Radial Velocity",
            }
        )
        logging.info("Catalog uniformed.")

    def remove_theoretical_masses(self) -> None:
        pass  # does not have theoretical masses
        # """
        # """
        # for value in ["", "_min", "_max"]:
        #     #check if nan (nan != nan)
        #     self.data.loc[
        #         self.data["masstype"] != "msini", "mass" + value
        #     ] = self.data.loc[self.data["masstype"] != "msini", "mass" + value]
        #     self.data.loc[
        #         self.data["masstype"] == "msini", "msini" + value
        #     ] = self.data.loc[self.data["masstype"] == "msini", "mass" + value]
        # logging.info("Theoretical masses/radii removed.")

    def assign_status(self) -> None:
        """
        The assign_status function takes the dataframe and assigns a status to each row, based on the value in
        "list" column. The function first checks if "Confirmed" is in the list column of that row, and if so,
        it assigns the status as "CONFIRMED". If not, it then checks for "Controversial", which would assign the
        status as "CANDIDATE". If neither of those are true, then it will check for "Retracted" and assign
        "FALSE POSITIVE". Kepler Objects of Interest will be assigned as candidates.
        """
        for i in self.data.index:
            if "Confirmed" in self.data.at[i, "list"]:
                self.data.at[i, "status"] = "CONFIRMED"
            elif "Controversial" in self.data.at[i, "list"]:
                self.data.at[i, "status"] = "CANDIDATE"
            elif "Retracted" in self.data.at[i, "list"]:
                self.data.at[i, "status"] = "FALSE POSITIVE"
            elif "Kepler Objects of Interest" in self.data.at[i, "list"]:
                self.data.at[i, "status"] = "CANDIDATE"
        logging.info("Status column assigned.")
        logging.info("Updated status:")
        logging.info(self.data.status.value_counts())

    def handle_reference_format(self) -> None:
        """
        The handle_reference_format function is used to create a url for each reference in the references list.
        Since the Open Exoplanet Catalog does not provide references, we just use "OEC" as a keyword.
        """
        for item in ["e", "mass", "msini", "i", "a", "p", "r"]:
            self.data[item + "_url"] = self.data[item].apply(
                lambda x: "" if pd.isna(x) or np.isinf(x) else "oec"
            )
        logging.info("Reference columns uniformed.")

    def convert_coordinates(self) -> None:
        """
        The coordinates function takes the RA and Dec columns of a dataframe,
        and converts them to decimal degrees. It replaces any
        missing values with NaN. Finally, it uses SkyCoord to convert from hour angles and
        degrees into decimal degrees.
        """

        self.data["ra"] = self.data.ra.fillna("").replace("nan", "").replace(np.nan, "")
        self.data["dec"] = (
            self.data.dec.fillna("").replace("nan", "").replace(np.nan, "")
        )
        self.data["ra"] = self.data.apply(
            lambda row: SkyCoord(
                str(row["ra"]) + " " + str(row["dec"]), unit=(u.hourangle, u.deg)
            ).ra.degree
            if not str(row.ra) == ""
            else np.nan,
            axis=1,
        )
        self.data["dec"] = self.data.apply(
            lambda row: SkyCoord(
                str(row["ra"]) + " " + str(row["dec"]), unit=(u.hourangle, u.deg)
            ).dec.degree
            if not str(row.dec) == ""
            else np.nan,
            axis=1,
        )
        logging.info("Converted coordinates from hourangle to deg.")
