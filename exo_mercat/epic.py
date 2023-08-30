import logging

import numpy as np

from exo_mercat.catalogs import Catalog


class Epic(Catalog):
    def __init__(self) -> None:
        """
        The __init__ function is called when the class is instantiated. It sets up the instance of the class,
        and defines any variables that will be used by other functions in the class.
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
                "default_flag",
            ]
        ]
        self.data = self.data[self.data.default_flag == 1]
        self.data["Kepler_host"] = self.data.k2_name
        self.data["Kepler_host"].fillna(self.data["hostname"], inplace=True)
        self.data["Kepler_host"] = self.data.apply(
            lambda row: row["Kepler_host"].rstrip(" bcdefghi"), axis=1
        )
        # self.data["EPIC_letter"] = self.data.apply(
        #     lambda row: row.pl_name.replace(".01", " b")
        #     .replace(".02", " c")
        #     .replace(".03", " d")
        #     .replace(".04", " e")
        #     .replace(".05", " f")
        #     .replace(".06", " g")
        #     .replace(".07", " h")
        #     .replace(".08", " i"),
        #     axis=1,
        # )
        #
        self.data["name"] = self.data["pl_name"]

        self.data["letter"] = self.data.pl_letter.replace("", np.nan).fillna(
            self.data.pl_name.apply(lambda row: row[-3:])
        )
        self.data.k2_name = self.data.k2_name.fillna(
            self.data.hostname + " " + self.data.letter
        )
        self.data["HD_letter"] = self.data.hd_name + " " + self.data.letter
        self.data["HIP_letter"] = self.data.hip_name + " " + self.data.letter
        self.data["TIC_letter"] = self.data.tic_id + " " + self.data.letter
        self.data["GAIA_letter"] = self.data.gaia_id + " " + self.data.letter
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
                + str(self.data.at[i, "Kepler_host"])
            )
            self.data.at[i, "alias"] = (
                ",".join(
                    [x for x in set(self.data.at[i, "alias"].split(",")) if x != "nan"]
                )
                + ","
            )
            self.data.at[i, "aliasplanet"] = (
                str(self.data.at[i, "TIC_letter"])
                + ","
                + str(self.data.at[i, "HD_letter"])
                + ","
                + str(self.data.at[i, "HIP_letter"])
                + ","
                + str(self.data.at[i, "GAIA_letter"])
                + ","
                + str(self.data.at[i, "k2_name"])
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

        logging.info("Catalog uniformed.")

    def convert_coordinates(self) -> None:
        """
        The coordinates function takes the RA and Dec columns of a dataframe,
        and converts them to decimal degrees. It replaces any
        missing values with NaN. Finally, it uses SkyCoord to convert from hour angles and
        degrees into decimal degrees.
        Currently only used for Open Exoplanet Catalogue, KOI catalogs.
        UPDATE: EPIC now has degrees already.
        """
        pass
