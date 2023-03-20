from exomercat.catalogs import Catalog
from astropy.coordinates import SkyCoord
import astropy.units as u
import numpy as np
import logging


class Epic(Catalog):
    def __init__(self) -> None:
        """
        The __init__ function is called when the class is instantiated.
        It sets up the instance of the class, and defines any variables that will be used by other functions in the class.
        """
        super().__init__()
        self.name = "epic"
        self.data = None

    def uniform_catalog(self) -> None:
        """
        The uniform_catalog function takes the data from the K2targets catalog and creates a new table with only
        the columns we need for our analysis. It also adds some useful columns that are derived from existing ones.
        """
        self.data = self.data[
            [
                "pl_name",
                "pl_letter",
                "k2_name",
                "epic_hostname",
                "hostname",
                "hd_name",
                "hip_name",
                "tic_id",
                "gaia_id",
                "disposition",
                "ra",
                "dec",
                "discoverymethod",
            ]
        ]
        self.data = self.data.drop_duplicates()
        self.data["KEPHOST"] = self.data.k2_name
        self.data["KEPHOST"].fillna(self.data["hostname"], inplace=True)
        self.data["KEPHOST"] = self.data.apply(
            lambda row: row["KEPHOST"].rstrip(" bcdefghi"), axis=1
        )

        self.data["EPICLETTER"] = self.data.apply(
            lambda row: row.pl_name.replace(".01", " b")
            .replace(".02", " c")
            .replace(".03", " d")
            .replace(".04", " e")
            .replace(".05", " f")
            .replace(".06", " g")
            .replace(".07", " h")
            .replace(".08", " i"),
            axis=1,
        )

        self.data["Name"] = self.data["EPICLETTER"]

        self.data["LETTER"] = self.data.EPICLETTER.apply(lambda row: row[-1:])
        self.data["HDLETTER"] = self.data.hd_name + " " + self.data.LETTER
        self.data["HIPLETTER"] = self.data.hip_name + " " + self.data.LETTER
        self.data["TICLETTER"] = self.data.tic_id + " " + self.data.LETTER
        self.data["GAIALETTER"] = self.data.gaia_id + " " + self.data.LETTER
        self.data[["hd_name", "hip_name", "tic_id", "gaia_id"]] = self.data[
            ["hd_name", "hip_name", "tic_id", "gaia_id"]
        ].fillna("")
        for i in self.data.index:
            self.data.at[i, "alias"] = (
                str(self.data.at[i, "tic_id"])
                + ","
                + str(self.data.at[i, "hip_name"])
                + ","
                + str(self.data.at[i, "hd_name"])
                + ","
                + str(self.data.at[i, "gaia_id"])
                + ","
                + str(self.data.at[i, "KEPHOST"])
            )
            self.data.at[i, "alias"] = ",".join(
                [x for x in set(self.data.at[i, "alias"].split(",")) if x != "nan"]
            )
            self.data.at[i, "aliasplanet"] = (
                str(self.data.at[i, "TICLETTER"])
                + ","
                + str(self.data.at[i, "HDLETTER"])
                + ","
                + str(self.data.at[i, "HIPLETTER"])
                + ","
                + str(self.data.at[i, "GAIALETTER"])
                + ","
                + str(self.data.at[i, "k2_name"])
            )
            self.data.at[i, "aliasplanet"] = ",".join(
                [
                    x
                    for x in set(self.data.at[i, "aliasplanet"].split(","))
                    if x != "nan"
                ]
            )
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
