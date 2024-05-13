import glob
import logging
import os
import re
from datetime import date
from pathlib import Path

import astropy.units as u
import numpy as np
import pandas as pd
import requests
from astropy.coordinates import SkyCoord

from exo_mercat.catalogs import Catalog
from exo_mercat.utility_functions import UtilityFunctions as Utils


class Oec(Catalog):
    """
    The Oec class contains all methods and attributes related to the Open Exoplanet Catalogue.
    """

    def __init__(self) -> None:
        """
        This function is called when the class is instantiated. It sets up a name attribute that can be used to refer
        to this particular instance of Oec.

        :param self: An instance of class Oec
        :type self: Oec
        :return: None
        :rtype: None
        """
        super().__init__()
        self.name = "oec"

    def download_catalog(self, url: str, filename: str, local_date: str = "", timeout: float = None) -> Path:

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
        :param local_date: The date of the catalog to download. Default is an empty string.
        :type local_date: str
        :param timeout: The maximum amount of time to wait for the download to complete. Default is None.
        :type timeout: float
        :return: The path to the downloaded file.
        :rtype: Path
        """

        # If local_date is not empty, use that date, otherwise use today's date
        if local_date != "":
            file_path_str = filename + local_date + ".csv"
            file_path_xml_str = filename + local_date + ".xml.gz"

            if len(glob.glob(file_path_str)) == 0:
                raise ValueError(
                    "Could not find catalog with this specific date. Please check your date value."
                )
            else:
                logging.info("Reading specific version: " + local_date)
        else:
            file_path_str = filename + date.today().strftime("%m-%d-%Y") + ".csv"
            file_path_xml_str = filename + date.today().strftime("%m-%d-%Y") + ".xml.gz"

        # Check if the file already exists
        if os.path.exists(file_path_str):
            logging.info("Reading existing file")

        else:
            # Download the file
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
                # Check if there is a previous version
                if len(glob.glob(filename + "*.csv")) > 0:
                    file_path_str = glob.glob(filename + "*.csv")[0]

                    logging.warning(
                        "Error fetching the catalog, taking a local copy: %s",
                        file_path_str,
                    )
                else:
                    raise ValueError("Could not find previous catalogs")

        # Read in the csv file
        logging.info("Catalog downloaded.")
        return Path(file_path_str)

    def uniform_catalog(self) -> None:
        """
        Uniforms the catalog by renaming columns and adding useful columns derived from existing ones. It
        standardizes the data format, renames columns, adds new columns like aliases, discovery methods,
        and references. Finally, it performs some string manipulations on the data and converts discovery methods.

        :param self: An instance of class Oec
        :type self: Oec
        :return: None
        """

        self.data["catalog"] = self.name

        # Rename columns
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

        # Add useful columns
        self.data["host"] = self.data.name.apply(lambda x: str(x[:-1]).strip())
        self.data["catalog_name"] = self.data["name"]
        self.data["catalog_host"] = self.data["host"]

        self.data = self.data.reset_index()
        self.data["alias"] = self.data.alias.fillna("")
        self.data["masstype"] = self.data.masstype.fillna("mass")

        self.data["msini"] = np.nan
        self.data["msini_min"] = np.nan
        self.data["msinis_max"] = np.nan

        # Separate mass into msini and mass
        for i in self.data.index:
            if self.data.at[i, "masstype"] == "msini":
                self.data.at[i, "msini"] = self.data.at[i, "mass"]
                self.data.at[i, "msini_max"] = self.data.at[i, "mass_max"]
                self.data.at[i, "msini_min"] = self.data.at[i, "mass_min"]

                self.data.at[i, "mass"] = np.nan
                self.data.at[i, "mass_max"] = np.nan
                self.data.at[i, "mass_min"] = np.nan

        # Clean host names
        for ident in self.data.name:
            if not str(re.search("(\\.0)\\d$", ident, re.M)) == "None":
                self.data.loc[self.data.name == ident, "host"] = ident[:-3].strip()
            elif not str(re.search("\\d$", ident, re.M)) == "None":
                self.data.loc[self.data.name == ident, "host"] = ident

        # Convert discovery methods
        self.data = Utils.convert_discovery_methods(self.data)

        # Logging
        logging.info("Catalog uniformed.")

    def remove_theoretical_masses(self) -> None:
        """
        Remove theoretical masses from the dataframe. Not used in the Oec catalog, since it does not have theoretical
        masses.

        :param self: The instance of the Oec class.
        :type self: Oec
        :return: None
        :rtype: None
        :note: This function is not necessary for the Oec catalog, since it does not have theoretical masses.
        """
        pass  # does not have theoretical masses

    def assign_status(self) -> None:
        """
        Assigns a status to each row in the dataframe based on the value in the "list" column.

        - If "Confirmed" is in the list, assigns "CONFIRMED".
        - If "Controversial" is in the list, assigns "CANDIDATE".
        - If "Retracted" is in the list, assigns "FALSE POSITIVE".
        - Kepler Objects of Interest are assigned "CANDIDATE".

        :param self: An instance of class Oec
        :type self: Oec
        :return: None
        :rtype: None
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

        # Logging
        logging.info("Status column assigned.")
        logging.info("Updated status:")
        logging.info(self.data.status.value_counts())

    def handle_reference_format(self) -> None:
        """
        The handle_reference_format function is used to create a URL for each reference in the references list. Since
        the Open Exoplanet Catalogue table does not provide references, we just use "oec" as a keyword in the url.

        :param self: An instance of class Oec
        :type self: Oec
        :return: None
        :rtype: None
        """
        for item in ["e", "mass", "msini", "i", "a", "p", "r"]:
            self.data[item + "_url"] = self.data[item].apply(
                lambda x: "" if pd.isna(x) or np.isinf(x) else "oec"
            )

        # Logging
        logging.info("Reference columns uniformed.")

    def convert_coordinates(self) -> None:
        """
        Convert the right ascension (RA) and declination (Dec) columns of the dataframe to decimal degrees.

        This function handles missing values by replacing them with empty strings, then converts the RA and Dec
        values to decimal degrees using SkyCoord. If the values are empty strings, NaN is assigned.

        :param self: An instance of class Oec
        :type self: Oec
        :return: None
        :rtype: None
        """

        # Fill missing values with empty strings
        self.data["ra"] = self.data.ra.fillna("").replace("nan", "").replace(np.nan, "")
        self.data["dec"] = (
            self.data.dec.fillna("").replace("nan", "").replace(np.nan, "")
        )

        # Convert RA and Dec to decimal degrees
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

        # Logging
        logging.info("Converted coordinates from hourangle to deg.")
