import re
import numpy as np
from exo_mercat.catalogs import Catalog
import logging
from exo_mercat.utility_functions import UtilityFunctions as Utils


class Nasa(Catalog):
    def __init__(self) -> None:
        """
        The __init__ function is called when the class is instantiated.
        It sets up the instance of the class, and defines what attributes it has.
        """
        super().__init__()
        self.name = "nasa"

    def uniform_catalog(self) -> None:
        """
        The uniform_catalog function takes a dataframe and converts it to the format of a catalog.
        The function do es this by renaming columns, changing column names to be more descriptive,
        and adding new columns that are necessary for the catalog.
        """
        self.data["catalog"] = self.name

        self.data = self.data.rename(
            columns={
                "pl_name": "name",
                "discoverymethod": "discovery_method",
                "pl_orbper": "p",
                "pl_orbpererr2": "p_min",
                "pl_orbpererr1": "p_max",
                "pl_orbsmax": "a",
                "pl_orbsmaxerr2": "a_min",
                "pl_orbsmaxerr1": "a_max",
                "pl_orbeccen": "e",
                "pl_orbeccenerr2": "e_min",
                "pl_orbeccenerr1": "e_max",
                "pl_orbincl": "i",
                "pl_orbinclerr2": "i_min",
                "pl_orbinclerr1": "i_max",
                "pl_radj": "r",
                "pl_radjerr2": "r_min",
                "pl_radjerr1": "r_max",
                "disc_year": "discovery_year",
                "disc_refname": "reference",
                "rv_flag": "RV",
                "tran_flag": "Transit",
                "ttv_flag": "TTV",
                # "pl_msinij": "msini",
                # "pl_msinijerr2": "msini_min",
                # "pl_msinijerr1": "msini_max",
                # "pl_massj": "mass",
                # "pl_massjerr2": "mass_min",
                # "pl_massjerr1": "mass_max",
                "pl_bmassj": "bestmass",
                "pl_bmassjerr2": "bestmass_min",
                "pl_bmassjerr1": "bestmass_max",
                "pl_bmassprov": "bestmass_provenance",
                "hostname": "host",
                "st_age": "Age (Gyrs)",
                "st_ageerr1": "Age_max",
                "st_ageerr2": "Age_min",
                "st_mass": "Mstar",
                "st_masserr1": "Mstar_max",
                "st_masserr2": "Mstar_min",
                "pl_radj_reflink": "r_url",
                "pl_orbeccen_reflink": "e_url",
                "pl_orbsmax_reflink": "a_url",
                "pl_orbper_reflink": "p_url",
                "pl_orbincl_reflink": "i_url",
                "pl_bmassj_reflink": "bestmass_url",
            }
        )
        self.data["catalog_name"] = self.data["name"]
        self.data["catalog_host"] = self.data["host"]

        # if "default_flag" in self.data.columns:
        #     # this happens when you use PLANETARY SYSTEMS TABLE
        #     self.data = self.data[self.data.default_flag == 1]
        self.data["mass"] = np.nan
        self.data["mass_min"] = np.nan
        self.data["mass_max"] = np.nan
        self.data["msini"] = np.nan
        self.data["msini_min"] = np.nan
        self.data["msini_max"] = np.nan

        if "bestmass" in self.data.columns:
            # this happens when you use PLANETARY COMPOSITE PARAMETERS TABLE
            self.sort_bestmass_to_mass_or_msini()

        self.data[["hd_name", "hip_name", "tic_id", "gaia_id"]] = self.data[
            ["hd_name", "hip_name", "tic_id", "gaia_id"]
        ].fillna("")

        self.data["alias"] = (
            self.data["hd_name"]
            .str.cat(self.data[["hip_name", "tic_id", "gaia_id"]].fillna(""), sep=",")
            .str.lstrip(",")
        )

        self.data = Utils.convert_discovery_methods(self.data)

        logging.info("Catalog uniformed.")

    def sort_bestmass_to_mass_or_msini(self) -> None:
        """
        The sort_bestmass_to_mass_or_msini function takes in a DataFrame and sorts the values of bestmass
        into either mass or msini.If the value of bestmass is found to be a mass, then it will be sorted
        into mass. If it is found to be an msini value, then it will instead go into msini. If neither
        are true (i.e., if it's some other type of data, e.g. M-R relationship), then both mass and msini
        are set to NaN for that row.
        """
        for i in self.data.index:
            if self.data.at[i, "bestmass_provenance"] == "Mass":
                self.data.at[i, "mass"] = self.data.at[i, "bestmass"]
                self.data.at[i, "mass_max"] = self.data.at[i, "bestmass_max"]
                self.data.at[i, "mass_min"] = self.data.at[i, "bestmass_min"]
                self.data.at[i, "mass_url"] = self.data.at[i, "bestmass_url"]
            elif self.data.at[i, "bestmass_provenance"] == "Msini":
                self.data.at[i, "msini"] = self.data.at[i, "bestmass"]
                self.data.at[i, "msini_max"] = self.data.at[i, "bestmass_max"]
                self.data.at[i, "msini_min"] = self.data.at[i, "bestmass_min"]
                self.data.at[i, "msini_url"] = self.data.at[i, "bestmass_url"]
            elif (self.data.at[i, "bestmass_provenance"] == "M-R relationship") or (
                self.data.at[i, "bestmass_provenance"] == "Msin(i)/sin(i)"
            ):
                self.data.at[i, "msini"] = np.nan
                self.data.at[i, "msini_max"] = np.nan
                self.data.at[i, "msini_min"] = np.nan
                self.data.at[i, "msini_url"] = np.nan
                self.data.at[i, "mass"] = np.nan
                self.data.at[i, "mass_max"] = np.nan
                self.data.at[i, "mass_min"] = np.nan
                self.data.at[i, "mass_url"] = np.nan
            else:
                print(self.data.at[i, "bestmass_provenance"])
                raise RuntimeError

    def handle_reference_format(self) -> None:
        """
        The handle_reference_format function takes in a dataframe and replaces the reference column with
        a url column. The function also adds columns for each of the seven parameters (e, mass, msini, i, a, P and R)
        and sets them to be equal to the corresponding reference column. It then removes all rows where any of these
        parameters are null.
        """
        for item in ["e", "mass", "msini", "i", "a", "p", "r"]:
            if item + "_url" not in self.data.columns:
                self.data[item + "_url"] = self.data["reference"]

        r = re.compile("href=(.*) target")

        for item in ["e", "mass", "msini", "i", "a", "p", "r"]:
            for i in self.data.index:
                # filter finite values by checking if x == x (false for nan, inf). If finite value, replace the
                # string with only the bibcode, else empty string
                if self.data.at[i, item + "_url"] == self.data.at[i, item + "_url"]:
                    url = self.data.at[i, item + "_url"]
                    link = r.findall(url)
                    self.data.at[i, item + "_url"] = (
                        link[0]
                        .replace(
                            "http://adsabs.harvard.edu/cgi-bin/nph-bib_query?bibcode=",
                            "",
                        )
                        .replace("http://adsabs.harvard.edu/abs/", "")
                        .replace("nan", "")
                        .replace("https://ui.adsabs.harvard.edu/abs/", "")
                        .replace("/abstract", "")
                    )

                else:
                    self.data.at[i, item + "_url"] = ""
            self.data.loc[self.data[item].isnull(), item + "_url"] = ""
        logging.info("Reference columns uniformed.")

    def assign_status(self) -> None:
        """
        The assign_status function assigns the status of the candidate (CONFIRMED by DEFAULT for NASA).
        """
        self.data["status"] = "CONFIRMED"
        logging.info("Status column assigned.")
        logging.info("Updated Status:")
        logging.info(self.data.status.value_counts())

    def convert_coordinates(self) -> None:
        """The coordinates function takes the RA and Dec columns of a dataframe,
        and converts them to decimal degrees. It replaces any
        missing values with NaN. Finally, it uses SkyCoord to convert from hour angles and
        degrees into decimal degrees.
        NOT NECESSARY since NASA already has coordinates in degrees"""
        pass

    def remove_theoretical_masses(self) -> None:
        """
        The remove_theoretical_masses function removes mass and radius estimates calculated through M-R relationships.
        It does this by removing all rows where the mass_url, msini_url, and r_url columns contain
        the word "Calculated".
        """
        for value in ["", "_min", "_max", "_url"]:
            self.data.loc[
                self.data["mass_url"].str.contains("Calculated", na=False),
                "mass" + value,
            ] = np.nan

            self.data.loc[
                self.data["msini_url"].str.contains("Calculated", na=False),
                "msini" + value,
            ] = np.nan

            self.data.loc[
                self.data["r_url"].str.contains("Calculated", na=False), "r" + value
            ] = np.nan
        logging.info("Theoretical masses/radii removed.")
