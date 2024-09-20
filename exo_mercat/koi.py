import logging
import astropy.units as u
import numpy as np
from astropy.coordinates import SkyCoord
from exo_mercat.catalogs import Catalog


class Koi(Catalog):
    """
    The Koi class contains all methods and attributes related to the Kepler Objects of Interest catalog.
    """

    def __init__(self) -> None:
        """
        Initializes the Koi class object.

        This function is automatically called when an instance of the Emc class is created.
        It sets up the instance of the class by assigning a name.

        :param self: The instance of the Koi class.
        :type self: Koi
        :return: None
        :rtype: None
        """
        super().__init__()
        self.name = "koi"
        self.data = None

    def standardize_catalog(self) -> None:
        """
        This function standardizes the catalog data by selecting relevant columns, creating aliases, renaming columns,
        and logging the standardization process.

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
         Convert the right ascension (RA) and declination (Dec) columns of the dataframe to decimal degrees.

        This function handles missing values by replacing them with empty strings, then converts the RA and Dec
        values to decimal degrees using SkyCoord. If the values are empty strings, NaN is assigned.

        :param self: An instance of class Koi
        :type self: Koi
        :return: None
        :rtype: None
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

        # Logging
        logging.info("Converted coordinates from hourangle to deg.")
