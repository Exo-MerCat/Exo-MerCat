from exo_mercat.catalogs import Catalog
from astropy.coordinates import SkyCoord
import astropy.units as u
import numpy as np
import logging


class Koi(Catalog):
    def __init__(self) -> None:
        """
        The __init__ function is called when the class is instantiated. It sets up the instance of the class,
        and defines any variables that will be used by all instances of this class.
        """
        super().__init__()
        self.name = "koi"
        self.data = None

    def uniform_catalog(self) -> None:
        """
        The uniform_catalog function takes the raw data from the NASA Exoplanet Archive and
        returns a pandas DataFrame with columns for each of the following:
        1. kepid (the Kepler ID)
        2. kepoi_name (the KOI name)
        3. kepler_name (the Kepler host star name, if available)
        4. koi_disposition (CANDIDATE, FALSE POSITIVE, NOT DISPOSITIONED OR CONFIRMED)
        """
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
        self.data["letter"] = self.data.apply(
            lambda row: row["kepler_name"][-1:]
            if not str(row.kepler_name) == "nan"
            else row["kepoi_name"][-3:],
            axis=1,
        )
        self.data["KIC"] = self.data["KIC_host"] + " " + self.data["letter"]
        self.data = self.data.rename(columns={"ra_str": "ra", "dec_str": "dec"})

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

        self.data["name"] = self.data["KOI"]
        self.data["disposition"] = self.data["koi_disposition"]
        self.data["discoverymethod"] = "Transit"
        logging.info("Catalog uniformed.")

    def convert_coordinates(self) -> None:
        """
        The coordinates function takes the RA and Dec columns of a dataframe,
        and converts them to decimal degrees. It replaces any
        missing values with NaN. Finally, it uses SkyCoord to convert from hour angles and
        degrees into decimal degrees.
        Currently only used for Open Exoplanet Catalogue, KOI and EPIC catalogs.
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
