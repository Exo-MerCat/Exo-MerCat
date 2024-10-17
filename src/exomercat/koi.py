import logging
import astropy.units as u
import numpy as np
from astropy.coordinates import SkyCoord
from .catalogs import Catalog
from pandas import Int64Dtype, Float64Dtype, StringDtype


class Koi(Catalog):
    """
    A class representing the Kepler Objects of Interest (KOI) catalog.

    This class inherits from the Catalog class and provides specific functionality
    for handling and processing data from the KOI catalog. It includes methods for
    standardizing the catalog data, converting coordinates, and managing KOI-specific
    attributes.

    Attributes:
        name (str): The name of the catalog, set to "koi".
        data (pandas.DataFrame): The catalog data stored as a DataFrame.
        columns (dict): A dictionary defining the expected columns and their data types.

    Methods:
        standardize_catalog(): Standardizes the catalog data format.
        convert_coordinates(): Converts RA and Dec to decimal degrees.
    """

    def __init__(self) -> None:
        """
        Initialize the Koi class.

        This method sets up the instance of the Koi class by:
        1. Calling the parent class initializer.
        2. Setting the catalog name to "koi".
        3. Initializing the data attribute as None.
        4. Defining the expected columns and their data types for this catalog.

        :param self: The instance of the Koi class.
        :type self: Koi
        :return: None
        :rtype: None
        """
        super().__init__()
        self.name = "koi"
        self.data = None
        self.columns = {
            "kepid": Int64Dtype(),
            "kepoi_name": StringDtype(),
            "kepler_name": StringDtype(),
            "koi_disposition": StringDtype(),
            "ra_str": StringDtype(),
            "dec_str": StringDtype(),
        }

    def standardize_catalog(self) -> None:
        """
        Standardize the Kepler Objects of Interest catalog data.

        This method performs the following operations:
        1. Selects relevant columns from the raw data.
        2. Creates new columns: KOI, KOI_host, Kepler_host, KIC_host.
        3. Generates a 'letter' column for planet designation.
        4. Creates 'alias' and 'aliasplanet' columns with various identifiers.
        5. Renames and creates standard columns like 'name', 'disposition', and 'discoverymethod'.
        6. Retains only the standardized columns in the final dataset.

        
        :param self: The instance of the Koi class.
        :type self: Koi
        :return: None
        :rtype: None
        """

        # Selecting relevant columns
        self.data = self.data[
            [
                "kepid",
                "kepoi_name",
                "kepler_name",
                "koi_disposition",
                "ra_str",
                "dec_str",
            ]
        ]
        # Create KOI, KOI_host, Kepler_host, KIC_host columns
        self.data["KOI"] = self.data["kepoi_name"].apply(
            lambda x: "KOI-" + x.lstrip("K").lstrip("0")
        )
        self.data["KOI_host"] = self.data["KOI"].apply(
            lambda x: x[:-3] + x[-3:].rstrip(".01234567")
        )
        self.data["Kepler_host"] = self.data.apply(
            lambda row: row["kepler_name"].rstrip(" bcdefghi")
            if not str(row.kepler_name) == "nan"
            else "nan",
            axis=1,
        )
        self.data["KIC_host"] = self.data["kepid"].apply(lambda x: "KIC " + str(x))

        # Create letter column
        self.data["letter"] = self.data.apply(
            lambda row: row["kepler_name"][-1:]
            if not str(row.kepler_name) == "nan"
            else row["kepoi_name"][-3:],
            axis=1,
        )
        self.data["KIC"] = self.data["KIC_host"] + " " + self.data["letter"]
        self.data = self.data.rename(columns={"ra_str": "ra", "dec_str": "dec"})

        # Put every alias in the alias/aliasplanet columns
        for i in self.data.index:
            self.data.at[i, "alias"] = (
                str(self.data.at[i, "KOI_host"])
                + ","
                + str(self.data.at[i, "Kepler_host"])
                + ","
                + str(self.data.at[i, "KIC_host"])
            )

            self.data.at[i, "alias"] = (
                ",".join(
                    [x for x in set(self.data.at[i, "alias"].split(",")) if x != "nan"]
                )
                + ","
            )

            self.data.at[i, "aliasplanet"] = (
                str(self.data.at[i, "KOI"])
                + ","
                + str(self.data.at[i, "KIC"])
                + ","
                + str(self.data.at[i, "kepler_name"])
            )
            self.data.at[i, "aliasplanet"] = (
                ",".join(
                    [
                        x
                        for x in set(
                            self.data.at[i, "aliasplanet"]
                            .replace(" .0", ".0")
                            .split(",")
                        )
                        if x != "nan"
                    ]
                )
                + ","
            )

        # Create name, disposition, discoverymethod columns
        self.data["name"] = self.data["KOI"]
        self.data["disposition"] = self.data["koi_disposition"]
        self.data["discoverymethod"] = "Transit"
        self.data = self.data[
            [
                "name",
                "alias",
                "aliasplanet",
                "disposition",
                "discoverymethod",
                "letter",
                "ra",
                "dec",
            ]
        ]

        # Logging
        logging.info("Catalog standardized.")

    def convert_coordinates(self) -> None:
        """
        Convert right ascension (RA) and declination (Dec) from string format to decimal degrees.

        This method performs the following operations:
        1. Replaces missing values in RA and Dec columns with empty strings.
        2. Converts RA and Dec from string format (HH:MM:SS) to decimal degrees using astropy's SkyCoord.
        3. Assigns NaN to entries where conversion is not possible (empty strings).

        :param self: An instance of class Koi
        :type self: Koi
        :return: None
        :rtype: None
        """

        # Replace nans
        self.data["ra"] = self.data.ra.fillna("").replace("nan", "").replace(np.nan, "")
        self.data["dec"] = (
            self.data.dec.fillna("").replace("nan", "").replace(np.nan, "")
        )

        # Convert to degrees
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
