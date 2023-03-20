import glob, re
import logging

import pandas as pd
import urllib.request
import numpy as np
from exomercat.configurations import *
from exomercat.catalogs import Catalog
from datetime import date
from astropy.io.votable import parse_single_table
from astropy.io import ascii
import logging

class Eu(Catalog):
    def __init__(self) -> None:
        """
        The __init__ function is called when the class is instantiated.
        It sets up the instance of the class, and defines what attributes it has.
        """
        super().__init__()
        self.name = "eu"

    def download_and_save_cat(self, url: str, filename: str) -> None:
        """
        The download_and_save_cat function downloads a catalog from the web and saves it to disk.
        It takes two arguments: url, which is the URL of the catalog to be downloaded, and filename,
        which is where we want to save that file locally. It returns nothing.

        Parameters
        ----------
            self
                Allow the function to refer to and modify a class instance's attributes
            url:str
                Specify the url of the catalog to download
            filename:str
                Specify the name of the file where we want to store our data

        """
        if os.path.exists(filename + date.today().strftime("%m-%d-%Y") + ".csv"):
            logging.info("Reading existing file")
            self.data = pd.read_csv(
                filename + date.today().strftime("%m-%d-%Y") + ".csv"
            )
        else:
            try:
                urllib.request.urlretrieve(url, "votable.xml")
                table = parse_single_table("votable.xml").to_table()
                ascii.write(
                    table,
                    filename + date.today().strftime("%m-%d-%Y") + ".csv",
                    format="csv",
                    overwrite=True,
                )
                os.remove("votable.xml")
                self.data = pd.read_csv(
                    filename + date.today().strftime("%m-%d-%Y") + ".csv"
                )
            except BaseException:
                local_copy = glob.glob(filename + "*.csv")[0]
                logging.info("Error fetching the catalog, taking a local copy:", local_copy)
                self.data = pd.read_csv(local_copy)
        logging.info('Catalog downloaded.')

    def uniform_catalog(self) -> None:
        """
        The uniform_catalog function takes the raw data from a catalog and converts it into a uniform format.
        The function also adds in columns for aliases, discovery methods, and references.
        """
        self.data["catalog"] = self.name
        self.data = self.data.replace("None", "").replace("nan", np.nan)
        self.data = self.data.rename(
            columns={
                "detection_type": "DiscMeth",
                "orbital_period": "P",
                "orbital_period_error_max": "P_max",
                "orbital_period_error_min": "P_min",
                "semi_major_axis": "a",
                "semi_major_axis_error_max": "a_max",
                "semi_major_axis_error_min": "a_min",
                "eccentricity": "e",
                "eccentricity_error_max": "e_max",
                "eccentricity_error_min": "e_min",
                "inclination": "i",
                "inclination_error_max": "i_max",
                "inclination_error_min": "i_min",
                "name": "Name",
                "updated": "Update",
                "discovered": "YOD",
                "mass": "Mass",
                "mass_error_max": "Mass_max",
                "mass_error_min": "Mass_min",
                "mass_sini": "Msini",
                "mass_sini_error_max": "Msini_max",
                "mass_sini_error_min": "Msini_min",
                "radius": "R",
                "radius_error_max": "R_max",
                "radius_error_min": "R_min",
                "mass_detection_type": "MASSPROV",
                "radius_detection_type": "RADPROV",
                "star_name": "Host",
                "bib_reference": "Reference",
            }
        )

        self.data["alias"] = self.data["alternate_names"].str.cat(
            self.data[["star_alternate_names"]].fillna(""), sep=","
        )

        for i in self.data.index:
            alias_polished = ""
            for al in self.data.at[i, "alias"].split(","):
                al = re.sub(".0\d$", "", al.rstrip())
                al = re.sub(" [b-i]$", "", al.rstrip())
                al = re.sub("^K0", "KOI-", al.lstrip())
                alias_polished = alias_polished + "," + al.rstrip()

            self.data.at[i, "alias"] = alias_polished.lstrip(",")

        self.data = self.data.replace(
            {
                "Primary Transit#TTV": "TTV",
                "Primary Transit": "Transit",
                "Pulsar": "Pulsar Timing",
            }
        )
        logging.info('Catalog uniformed.')

    def remove_theoretical_masses(self) -> None:
        """
        The remove_theoretical_masses function removes theoretical masses from the dataframe.
        It does this by setting to NaN all the mass/msini values where the MASSPROV column contains
        "Theoretical" and the radii where the RADPROV column contains "Theoretical".
        """
        for value in ["", "_min", "_max"]:
            self.data.loc[
                self.data["MASSPROV"].str.contains("Theoretical", na=False),
                "Mass" + value,
            ] = np.nan
            self.data.loc[
                self.data["MASSPROV"].str.contains("Theoretical", na=False),
                "Msini" + value,
            ] = np.nan
            self.data.loc[
                self.data["RADPROV"].str.contains("Theoretical", na=False), "R" + value
            ] = np.nan

        logging.info('Theoretical masses/radii removed.')
    def assign_status(self) -> None:
        """
        The assign_status function assigns a status to each planet based on the
        planet_status column. The function first sets all planets with confirmed
        planets as CONFIRMED, then it looks for candidate and controversial planets,
        and sets them as CANDIDATE. Finally, it looks for retracted planets and sets them
        as FALSE POSITIVE.
        """
        self.data["Status"] = "CONFIRMED"
        self.data.loc[
            self.data["planet_status"].str.contains(
                "Candidate|Unconfirmed|Controversial"
            ),
            "Status",
        ] = "CANDIDATE"
        self.data.loc[
            self.data["planet_status"].str.contains("Retracted"), "Status"
        ] = "FALSE POSITIVE"

        logging.info('Status column assigned.')
        logging.info('Updated Status:')
        logging.info(list(self.data.Status.value_counts()))


    def handle_reference_format(self) -> None:
        """
        The handle_reference_format function is used to create a url for each reference in the references list.
        Since the Exoplanet Encyclopaedia table does not provide references, we just use "EU" as a keyword.
        """
        for item in ["e", "Mass", "Msini", "i", "a", "P", "R"]:
            self.data[item + "_url"] = self.name
        logging.info('Reference columns uniformed.')


    def convert_coordinates(self) -> None:
        """
        The coordinates function takes the RA and Dec columns of a dataframe,
        and converts them to decimal degrees.
        Not necessary for EU.
        """
        pass
