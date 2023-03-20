import re
import numpy as np
from exomercat.catalogs import Catalog
import logging

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

        Parameters
        ----------
            self
                Access variables that belong to a class

        Returns
        -------

            None
        """
        self.data["catalog"] = self.name
        self.data = self.data.rename(
            columns={
                "pl_name": "Name",
                "discoverymethod": "DiscMeth",
                "pl_orbper": "P",
                "pl_orbpererr2": "P_min",
                "pl_orbpererr1": "P_max",
                "pl_orbsmax": "a",
                "pl_orbsmaxerr2": "a_min",
                "pl_orbsmaxerr1": "a_max",
                "pl_orbeccen": "e",
                "pl_orbeccenerr2": "e_min",
                "pl_orbeccenerr1": "e_max",
                "pl_orbincl": "i",
                "pl_orbinclerr2": "i_min",
                "pl_orbinclerr1": "i_max",
                "pl_radj": "R",
                "pl_radjerr2": "R_min",
                "pl_radjerr1": "R_max",
                "disc_year": "YOD",
                "disc_refname": "Reference",
                "rv_flag": "RV",
                "tran_flag": "Transit",
                "ttv_flag": "TTV",
                "pl_msinij": "Msini",
                "pl_msinijerr2": "Msini_min",
                "pl_msinijerr1": "Msini_max",
                "pl_massj": "Mass",
                "pl_massjerr2": "Mass_min",
                "pl_massjerr1": "Mass_max",
                "pl_bmassj": "BestMass",
                "pl_bmassjerr2": "BestMass_min",
                "pl_bmassjerr1": "BestMass_max",
                "pl_bmassprov": "BestMass_provenance",
                "hostname": "Host",
                "st_age": "Age (Gyrs)",
                "st_ageerr1": "Age_max",
                "st_ageerr2": "Age_min",
                "st_mass": "Mstar",
                "st_masserr1": "Mstar_max",
                "st_masserr2": "Mstar_min",
                "pl_radj_reflink": "R_url",
                "pl_orbeccen_reflink": "e_url",
                "pl_orbsmax_reflink": "a_url",
                "pl_orbper_reflink": "P_url",
                "pl_orbincl_reflink": "i_url",
                "pl_bmassj_reflink": "BestMass_url",
            }
        )

        if "default_flag" in self.data.columns:
            # this happens when you use PLANETARY SYSTEMS TABLE
            self.data = self.data[self.data.default_flag == 1]

        if "BestMass" in self.data.columns:
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

        self.data = self.data.replace(
            {
                "Transit Timing Variations": "TTV",
                "Eclipse Timing Variations": "TTV",
                "Pulsation Timing Variations": "Pulsar Timing",
                "Orbital Brightness Modulation": "Other",
            }
        )

        logging.info('Catalog uniformed.')

    def sort_bestmass_to_mass_or_msini(self) -> None:
        """
        The sort_bestmass_to_mass_or_msini function takes in a DataFrame and sorts the values of BestMass into either Mass or Msini.
        If the value of BestMass is found to be a mass, then it will be sorted into Mass. If it is found to be an msini value, then
        it will instead go into Msini. If neither are true (i.e., if it's some other type of data, e.g. M-R relationship), then both
        Mass and Msini are set to NaN for that row.

        Parameters
        ----------
            self
                Access the attributes and methods of the class in python
        """
        for i in self.data.index:
            if self.data.at[i, "BestMass_provenance"] == "Mass":
                self.data.at[i, "Mass"] = self.data.at[i, "BestMass"]
                self.data.at[i, "Mass_max"] = self.data.at[i, "BestMass_max"]
                self.data.at[i, "Mass_min"] = self.data.at[i, "BestMass_min"]
                self.data.at[i, "Mass_url"] = self.data.at[i, "BestMass_url"]
            elif self.data.at[i, "BestMass_provenance"] == "Msini":
                self.data.at[i, "Msini"] = self.data.at[i, "BestMass"]
                self.data.at[i, "Msini_max"] = self.data.at[i, "BestMass_max"]
                self.data.at[i, "Msini_min"] = self.data.at[i, "BestMass_min"]
                self.data.at[i, "Msini_url"] = self.data.at[i, "BestMass_url"]
            elif (self.data.at[i, "BestMass_provenance"] == "M-R relationship") or (
                self.data.at[i, "BestMass_provenance"] == "Msin(i)/sin(i)"
            ):
                self.data.at[i, "Msini"] = np.nan
                self.data.at[i, "Msini_max"] = np.nan
                self.data.at[i, "Msini_min"] = np.nan
                self.data.at[i, "Msini_url"] = np.nan
                self.data.at[i, "Mass"] = np.nan
                self.data.at[i, "Mass_max"] = np.nan
                self.data.at[i, "Mass_min"] = np.nan
                self.data.at[i, "Mass_url"] = np.nan
            else:
                print(self.data.at[i, "BestMass_provenance"])
                raise RuntimeError

    def handle_reference_format(self) -> None:
        """
        The handle_reference_format function takes in a dataframe and replaces the reference column with
        a url column. The function also adds columns for each of the seven parameters (e, Mass, Msini, i, a, P and R)
        and sets them to be equal to the corresponding reference column. It then removes all rows where any of these
        parameters are null.

        Parameters
        ----------
            self
                Access variables that belong to a class

        """
        for item in ["e", "Mass", "Msini", "i", "a", "P", "R"]:
            if item + "_url" not in self.data.columns:
                self.data[item + "_url"] = self.data["Reference"]

        r = re.compile("href=(.*) target")

        for item in ["e", "Mass", "Msini", "i", "a", "P", "R"]:
            for i in self.data.index:
                # filter finite values by checking if x == x (false for nan, inf)
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
        logging.info('Reference columns uniformed.')


    def assign_status(self) -> None:
        """
        The assign_status function assigns the status of the candidate (CONFIRMED by DEFAULT for NASA).

        Parameters
        ----------
            self
                Access the attributes and methods of the class in python

        Returns
        -------
            None

        """
        self.data["Status"] = "CONFIRMED"
        logging.info('Status column assigned.')
        logging.info('Updated Status:')
        logging.info(self.data.Status.value_counts())
    def convert_coordinates(self) -> None:
        '''The coordinates function takes the RA and Dec columns of a dataframe,
        and converts them to decimal degrees. It replaces any
        missing values with NaN. Finally, it uses SkyCoord to convert from hour angles and
        degrees into decimal degrees.
        NOT NECESSARY since NASA already has coordinates in degrees'''
        pass

    def remove_theoretical_masses(self):
        """
        The remove_theoretical_masses function removes mass and radius estimates calculated through M-R relationships.
        It does this by removing all rows where the Mass_url, Msini_url, and R_url columns contain
        the word "Calculated".


        Parameters
        ----------
            self
                Access the attributes and methods of the class in python, it is used to represent a instance of a class

        Returns
        -------

            None

        """
        for value in ["", "_min", "_max", "_url"]:
            self.data.loc[
                self.data["Mass_url"].str.contains("Calculated", na=False),
                "Mass" + value,
            ] = np.nan

            self.data.loc[
                self.data["Msini_url"].str.contains("Calculated", na=False),
                "Msini" + value,
            ] = np.nan

            self.data.loc[
                self.data["R_url"].str.contains("Calculated", na=False), "R" + value
            ] = np.nan
        logging.info('Theoretical masses/radii removed.')
