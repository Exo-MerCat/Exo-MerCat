import logging
import re
import numpy as np
import pandas as pd
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
        self.data["catalog"] = self.name
        self.data["catalog_name"] = self.data["pl_name"]

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
                "rv_flag": "RV",
                "tran_flag": "Transit",
                "ttv_flag": "TTV",
                "pl_massj": "mass",
                "pl_massjerr2": "mass_min",
                "pl_massjerr1": "mass_max",
                "pl_msinij": "msini",
                "pl_msinijerr2": "msini_min",
                "pl_msinijerr1": "msini_max",
                "hostname": "host",
                "st_age": "Age (Gyrs)",
                "st_ageerr1": "Age_max",
                "st_ageerr2": "Age_min",
                "st_mass": "Mstar",
                "st_masserr1": "Mstar_max",
                "st_masserr2": "Mstar_min",
                "pl_refname":"reference"
            }
        )


        self.data = self.data[self.data.default_flag == 1]
        self.data["Kepler_host"] = self.data.k2_name
        self.data["Kepler_host"].fillna(self.data["host"], inplace=True)
        self.data["Kepler_host"] = self.data.apply(
            lambda row: row["Kepler_host"].rstrip(" bcdefghi"), axis=1
        )

        self.data["letter"] = self.data.pl_letter.replace("", np.nan).fillna(
            self.data.name.apply(lambda row: row[-3:])
        )
        self.data.k2_name = self.data.k2_name.fillna(
            self.data.host + " " + self.data.letter
        )
        self.data[["hd_name", "hip_name", "tic_id", "gaia_id"]] = self.data[
            ["hd_name", "hip_name", "tic_id", "gaia_id"]
        ].fillna("")
        self.data["alias"] =  self.data[[ 'tic_id','hip_name','hd_name','gaia_id','Kepler_host']].agg(','.join, axis=1)
        for i in self.data.index:
            self.data.at[i, "alias"]=self.data.at[i, "alias"] .replace('nan,','').replace(',,',',').lstrip(',')

        self.data['discovery_method'] = self.data['discovery_method'].fillna("").replace('nan','')
        self.data.loc[self.data.discovery_method == "Primary Transit#TTV", 'discovery_method'] = "TTV"
        self.data.loc[self.data.discovery_method == "Transit Timing Variations", 'discovery_method'] = "TTV"
        self.data.loc[self.data.discovery_method == "Eclipse Timing Variations", 'discovery_method'] = "TTV"
        self.data.loc[self.data.discovery_method == "Primary Transit", 'discovery_method'] = "Transit"
        self.data.loc[self.data.discovery_method == "Pulsar", 'discovery_method'] = "Pulsar Timing"
        self.data.loc[self.data.discovery_method == "Pulsation Timing Variations", 'discovery_method'] = "Pulsar Timing"
        self.data.loc[self.data.discovery_method == "Timing", 'discovery_method'] = "Pulsar Timing"
        self.data.loc[self.data.discovery_method == "disk kinematics", 'discovery_method'] = "Other"
        self.data.loc[self.data.discovery_method == "Kinematic", 'discovery_method'] = "Other"
        self.data.loc[self.data.discovery_method == "Disk Kinematics", 'discovery_method'] = "Other"
        self.data.loc[self.data.discovery_method == "Orbital Brightness Modulation", 'discovery_method'] = "Other"
        self.data.loc[self.data.discovery_method == "astrometry", 'discovery_method'] = "Astrometry"
        self.data.loc[self.data.discovery_method == "microlensing", 'discovery_method'] = "Microlensing"
        self.data.loc[self.data.discovery_method == "imaging", 'discovery_method'] = "Imaging"
        self.data.loc[self.data.discovery_method == "transit", 'discovery_method'] = "Transit"
        self.data.loc[self.data.discovery_method == "timing", 'discovery_method'] = "Pulsar Timing"
        self.data.loc[self.data.discovery_method == "RV", 'discovery_method'] = "Radial Velocity"

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


    def remove_theoretical_masses(self) -> None:
        ''' there are no masses here in the first place'''
        pass

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

    def assign_status(self):

        self.data['status']=self.data['disposition']

        logging.info("Status column assigned.")
        logging.info("Updated Status:")
        logging.info(self.data.status.value_counts())

