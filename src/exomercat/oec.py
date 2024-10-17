import glob
import logging
import os
import re
from datetime import date, datetime
from pathlib import Path

import astropy.units as u
import numpy as np
import pandas as pd
from pandas import Int64Dtype, Float64Dtype, StringDtype

import requests
from astropy.coordinates import SkyCoord

from .catalogs import Catalog
from .utility_functions import UtilityFunctions as Utils


class Oec(Catalog):
    """
    The Oec class represents the Open Exoplanet Catalogue.

    This class inherits from the Catalog class and provides specific functionality
    for handling and processing data from the Open Exoplanet Catalogue. It includes
    methods for downloading, standardizing, and manipulating the catalog data.
    """
    

    def __init__(self) -> None:
        """
        Initialize the Oec class.

        This method sets up the instance of the Oec class by:

        1. Calling the parent class initializer.

        2. Setting the catalog name to "oec".

        3. Defining the expected columns and their data types for this catalog.

        :param self: An instance of class Oec
        :type self: Oec
        :return: None
        :rtype: None
        """
        super().__init__()
        self.name = "oec"
        self.columns = {
            "name": StringDtype(),
            "discoverymethod": StringDtype(),
            "period": Float64Dtype(),
            "period_min": Float64Dtype(),
            "period_max": Float64Dtype(),
            "semimajoraxis": Float64Dtype(),
            "semimajoraxis_min": Float64Dtype(),
            "semimajoraxis_max": Float64Dtype(),
            "eccentricity": Float64Dtype(),
            "eccentricity_min": Float64Dtype(),
            "eccentricity_max": Float64Dtype(),
            "inclination": Float64Dtype(),
            "inclination_min": Float64Dtype(),
            "inclination_max": Float64Dtype(),
            "radius": Float64Dtype(),
            "radius_min": Float64Dtype(),
            "radius_max": Float64Dtype(),
            "discoveryyear": Int64Dtype(),
            "mass": Float64Dtype(),
            "mass_min": Float64Dtype(),
            "mass_max": Float64Dtype(),
            "system_rightascension": StringDtype(),
            "system_declination": StringDtype(),
            "binaryflag": Int64Dtype(),
            "masstype": StringDtype(),
            "list": StringDtype(),
        }

    def download_catalog(
        self, url: str, filename: str, local_date: str, timeout: float = None
    ) -> Path:
        """
        Download the Open Exoplanet Catalogue from a given URL and save it to a file.

        This method performs the following operations:

        1. Checks if a local file for the given date already exists.

        2. If not, attempts to download the catalog from the URL.

        3. Converts the downloaded XML file to CSV format.

        4. If download fails, attempts to use the most recent local copy.

        5. Handles various error scenarios and provides appropriate logging.

        :param self: An instance of class Catalog
        :type self: Catalog
        :param url: The URL from which to download the catalog.
        :type url: str
        :param filename: The name of the file to save the catalog to.
        :type filename: str
        :param local_date: The date of the catalog to download.(format: YYYY-MM-DD)
        :type local_date: str
        :param timeout: The maximum amount of time to wait for the download to complete. Default is None.
        :type timeout: float
        :return: The path to the downloaded file.
        :rtype: Path
        """

        # Construct file paths for CSV and XML versions

        file_path_str = filename + local_date + ".csv"

        # Validate URL format
        if ".xml.gz" in url[-7:]:
            file_path_xml_str = filename + local_date + ".xml.gz"
        elif ".xml" in url[-4:]:
            file_path_xml_str = filename + local_date + ".xml"
        else:
            raise ValueError("url not valid. Only .xml or .xml.gz files are accepted.")

        # Check if the CSV file already exists
        if os.path.exists(file_path_str):
            logging.info("Reading existing file downloaded in date: " + local_date)
        
        else:
            
            # If file doesn't exist and the requested date is today, attempt to download
            if local_date == date.today().strftime("%Y-%m-%d"):
                # Download the XML file
                try:
                    result = requests.get(url, timeout=timeout)
                    # Open the input file and parse it as XML

                    with open(file_path_xml_str, "wb") as f:
                        f.write(result.content)
                    logging.info("Convert from .xml to .csv")
                    
                    # Convert XML to CSV
                    Utils.convert_xmlfile_to_csvfile(
                        file_path=file_path_xml_str, output_file=file_path_str
                    )
                    # Verify the CSV file can be read
                    dat = pd.read_csv(file_path_str)
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
                    # Remove corrupted files if they exist
                    if len(glob.glob(file_path_str)) > 0:
                        logging.warning(
                            "File "
                            + file_path_str
                            + " downloaded, but corrupted. Removing file..."
                        )
                        os.system("rm " + file_path_str)
                        os.system("rm " + file_path_xml_str)

                    # Attempt to use the most recent local copy
                    if len(glob.glob(filename + "*.csv")) > 0:
                        li = list(glob.glob(filename + "*.csv"))
                        li = [re.search(r"\d\d\d\d-\d\d-\d\d", l)[0] for l in li]
                        li = [datetime.strptime(l, "%Y-%m-%d") for l in li]

                        # Get the most recent compared to the current date. Get only the ones earlier than the date
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
                # The requested date is not today and the file doesn't exist
                raise ValueError(
                    "Could not find catalog with this specific date. Please check your date value."
                )

            # Note: The TODO for handling dates earlier than today could be implemented here

        logging.info("Catalog downloaded.")

        return Path(file_path_str)

    def standardize_catalog(self) -> None:
        """
        Standardize the Open Exoplanet Catalogue data.

        This method performs the following operations:

        1. Sets the catalog name.

        2. Renames columns to standard names used across all catalogs.

        3. Adds new columns such as host, catalog_name, and catalog_host.

        4. Separates mass into msini and mass based on the masstype column.

        5. Cleans host names.

        6. Converts discovery methods to a standard format.

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
                # "mass": "mass",
                # "mass_min": "mass_min",
                # "mass_max": "mass_max",
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
        logging.info("Catalog standardized.")

    def remove_theoretical_masses(self) -> None:
        """
        Remove theoretical masses from the dataframe.

        This method is a placeholder and does not perform any operations,
        as the Open Exoplanet Catalogue does not include theoretical masses.

        :param self: The instance of the Oec class.
        :type self: Oec
        :return: None
        :rtype: None
        :note: This function is not necessary for the Oec catalog, since it does not have theoretical masses.
        """
        pass  # does not have theoretical masses

    def assign_status(self) -> None:
        """
        Assign status to each entry based on the 'list' column.

        This method assigns status as follows:

        - "CONFIRMED" if 'Confirmed' is in the list.

        - "CANDIDATE" if 'Controversial' is in the list or for Kepler Objects of Interest.

        - "FALSE POSITIVE" if 'Retracted' is in the list.

        The method also logs the updated status counts.

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
        Standardize the reference format for various parameters.

        This method creates a '_url' column for each parameter (e, mass, msini, i, a, p, r),
        setting the value to 'oec' for non-null, finite values and an empty string otherwise.

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
        logging.info("Reference columns standardized.")

    def convert_coordinates(self) -> None:
        """
        Convert right ascension (RA) and declination (Dec) from string format to decimal degrees.

        This method performs the following operations:

        1. Replaces missing values in RA and Dec columns with empty strings.

        2. Converts RA and Dec from string format (HH:MM:SS) to decimal degrees using astropy's SkyCoord.

        3. Assigns NaN to entries where conversion is not possible (empty strings).

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
