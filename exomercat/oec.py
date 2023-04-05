import glob
import re
import xml.etree.ElementTree as ET
import gzip
import pandas as pd
import numpy as np
from exomercat.configurations import *
from exomercat.catalogs import Catalog
from datetime import date
from astropy.coordinates import SkyCoord
import astropy.units as u
import logging


def get_parameter(treeobject, parameter: str) -> str:
    """
    The getParameter function takes two arguments:
        1. treeobject - an ElementTree object that is the root of a parsed XML file
        2. parameter - a string representing the name of an element in the XML file

    Parameters
    ----------
        treeobject
            Get the parameter from a tree object
        parameter: str
            Get the parameter from the xml file

    Returns
    -------

        A string containing the parameter value
    """
    if parameter == "alias":
        alias = treeobject.findall("*/name")
        ret = ", ".join([a.text for a in alias])
    else:
        try:
            ret = treeobject.findtext("./" + parameter).strip()
        except BaseException:
            ret = ""
    return ret


def get_attribute(treeobject: ET.Element, parameter: str, attrib: str) -> str:
    """
    The getAttribute function takes three arguments:
        1. treeobject - an ElementTree object, which is the root of a parsed XML file
        2. parameter - a string representing the name of an element in the XML file
        3. attrib - a string representing one of that element's attributes

    Parameters
    ----------
        treeobject
            Specify the tree object to be searched
        parameter
            Specify the parameter that is being searched for
        attrib
            Specify the attribute that is to be returned

    Returns
    -------

        A string containing the value of the attribute
    """
    try:
        retattr = treeobject.find("./" + parameter).attrib[attrib]
    except BaseException:
        retattr = ""
    return retattr


def getParameter_all(treeobject: ET.Element, parameter: str) -> str:
    """
    The getParameter_all function takes two arguments:
        1. treeobject - an ElementTree object, which is the root of a parsed XML file
        2. parameter - a string containing the name of an element in the XML file

    Parameters
    ----------
        treeobject
            Find the parameter in the xml file
        parameter
            Specify which parameter to return

    Returns
    -------

        A list of all values in treeobject for the supplied parameter
    """
    try:
        ret = ", ".join([x.text for x in treeobject.iter(parameter)])
    except Exception as err:
        logging.error(err)
        ret = ""
    return ret


class Oec(Catalog):
    def __init__(self) -> None:
        """
        The __init__ function is called when the class is instantiated. It sets up the instance of the class, and
        defines any variables that will be used by all instances of this class.

        Parameters
        ----------
            self
                Represent the instance of the class
        """
        super().__init__()
        self.name = "oec"

    def download_and_save_cat(self, url: str, filename: str) -> None:
        """
        The download_and_save_cat function downloads a catalog from the web and saves it to disk. It takes two
        arguments: url, which is the URL of the catalog to be downloaded, and filename, which is where we want
        to save that file locally. It returns nothing.

        Parameters
        ----------
            self
                Allow the function to refer to and modify a class instance's attributes
            url:str
                Specify the url of the catalog to download
            filename:str
                Specify the name of the file where we want to store our data

        """
        fields = [
            "name",
            "binaryflag",
            "mass",
            "masstype",
            "mass_min",
            "mass_max",
            "radius",
            "radius_min",
            "radius_max",
            "period",
            "period_min",
            "period_max",
            "semimajoraxis",
            "semimajoraxis_min",
            "semimajoraxis_max",
            "eccentricity",
            "eccentricity_min",
            "eccentricity_max",
            "periastron",
            "longitude",
            "ascendingnode",
            "inclination",
            "inclination_min",
            "inclination_max",
            "temperature",
            "age",
            "discoverymethod",
            "discoveryyear",
            "lastupdate",
            "system_rightascension",
            "system_declination",
            "system_distance",
            "hoststar_mass",
            "hoststar_radius",
            "hoststar_metallicity",
            "hoststar_temperature",
            "hoststar_age",
            "hoststar_magJ",
            "hoststar_magI",
            "hoststar_magU",
            "hoststar_magR",
            "hoststar_magB",
            "hoststar_magV",
            "hoststar_magH",
            "hoststar_magK",
            "hoststar_spectraltype",
            "alias",
            "list",
        ]

        if os.path.exists(filename + date.today().strftime("%m-%d-%Y") + ".xml.gz"):
            logging.info("Reading existing file")
            input_file = gzip.open(
                filename + date.today().strftime("%m-%d-%Y") + ".xml.gz", "r"
            )

        else:
            try:
                os.system(
                    'wget "'
                    + url
                    + '" -O "'
                    + filename
                    + date.today().strftime("%m-%d-%Y")
                    + '.xml.gz"'
                )
                input_file = gzip.open(
                    filename + date.today().strftime("%m-%d-%Y") + ".xml.gz", "r"
                )

            except BaseException:
                local_copy = glob.glob(filename + "*.csv")[0]
                logging.warning(
                    "Error fetching the catalog, taking a local copy:", local_copy
                )
                input_file = gzip.open(local_copy, "r")

        table = ET.parse(input_file)
        tab = pd.DataFrame()

        # read the catalog from XML to Pandas
        for system in table.findall(".//system"):
            planets = system.findall(".//planet")
            stars = system.findall(".//star")

            for planet in planets:
                parameters = pd.DataFrame(
                    [get_parameter(system, "alias")], columns=["alias"]
                )

                for field in fields:
                    parameters[field] = None
                    parameters[field] = get_parameter(planet, field)
                    parameters.alias = get_parameter(system, "alias")
                    if field[0:7] == "system_":
                        parameters[field] = get_parameter(system, field[7:])
                    elif field[0:9] == "hoststar_":
                        parameters[field] = get_parameter(stars, field[9:])
                    elif field == "list":
                        parameters[field] = getParameter_all(planet, field)
                    elif field == "masstype":
                        parameters[field] = get_attribute(planet, field[0:-4], "type")
                    elif field[-4:] == "_min":
                        parameters[field] = get_attribute(
                            planet, field[0:-4], "errorminus"
                        )
                    elif field[-4:] == "_max":
                        parameters[field] = get_attribute(
                            planet, field[0:-4], "errorplus"
                        )

                parameters.binaryflag = 0
                if planet in system.findall(".//binary/planet"):
                    # P type planets
                    parameters.binaryflag = 1
                if planet in system.findall(".//binary/star/planet"):
                    # S type planets
                    parameters.binaryflag = 2
                if len(stars) == 0:
                    # rogue planets
                    parameters.binaryflag = 3

                tab = pd.concat([tab, parameters], sort=False)

        tab.to_csv(filename + date.today().strftime("%m-%d-%Y") + ".csv")
        self.data = tab
        logging.info("Catalog downloaded.")

    def uniform_catalog(self) -> None:
        """
        The uniform_catalog function is used to standardize the dataframe columns and values.
        """
        self.data["catalog"] = self.name
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
                "mass": "M",
                "mass_min": "M_min",
                "mass_max": "M_max",
                "system_rightascension": "ra",
                "system_declination": "dec",
            }
        )
        self.data = self.data.reset_index()
        self.data["alias"] = self.data.alias.apply(
            lambda x: x.strip("[]").replace("'", "").replace("nan", "")
        )
        self.data["host"] = self.data.name.apply(lambda x: str(x[:-1]).strip())

        for ident in self.data.name:
            if not str(re.search("(\.0)\\d$", ident, re.M)) == "None":
                self.data.loc[self.data.name == ident, "host"] = ident[:-3].strip()
            elif not str(re.search("\\d$", ident, re.M)) == "None":
                self.data.loc[self.data.name == ident, "host"] = ident

        self.data = self.data.replace(
            {
                "astrometry": "Astrometry",
                "microlensing": "Microlensing",
                "imaging": "Imaging",
                "transit": "Transit",
                "timing": "Pulsar Timing",
                "RV": "Radial Velocity",
            }
        )
        logging.info("Catalog uniformed.")

    def remove_theoretical_masses(self) -> None:
        # TODO wrong name at least
        """
        The remove_theoretical_masses function is used to remove
        the theoretical masses from the dataframe.
        It does this by replacing all of the values in mass and msini
        with their corresponding M value, depending on whether or
        not it is a msini mass.

        Parameters
        ----------
            self
                Represent the instance of the class

        Returns
        -------

            None

        Doc Author
        ----------
            Trelent
        """
        for value in ["", "_min", "_max"]:
            self.data.loc[
                self.data["masstype"] != "msini", "mass" + value
            ] = self.data.loc[self.data["masstype"] != "msini", "M" + value]
            self.data.loc[
                self.data["masstype"] == "msini", "msini" + value
            ] = self.data.loc[self.data["masstype"] == "msini", "M" + value]
        logging.info("Theoretical masses/radii removed.")

    def assign_status(self) -> None:
        """
        The assign_status function takes the dataframe and assigns a status to each row, based on the value in
        "list" column. The function first checks if "Confirmed" is in the list column of that row, and if so,
        it assigns the status as "CONFIRMED". If not, it then checks for "Controversial", which would assign the
        status as "CANDIDATE". If neither of those are true, then it will check for "Retracted" and assign
        "FALSE POSITIVE". Kepler Objects of Interest will be assigned as candidates.
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
        logging.info("status column assigned.")
        logging.info("Updated status:")
        logging.info(self.data.status.value_counts())

    def handle_reference_format(self) -> None:
        """
        The handle_reference_format function is used to create a url for each reference in the references list.
        Since the Open Exoplanet Catalog does not provide references, we just use "OEC" as a keyword.
        """
        for item in ["e", "mass", "msini", "i", "a", "p", "r"]:
            self.data[item + "_url"] = self.name
        logging.info("Reference columns uniformed.")

    def convert_coordinates(self) -> None:
        """
        The coordinates function takes the RA and Dec columns of a dataframe,
        and converts them to decimal degrees. It replaces any
        missing values with NaN. Finally, it uses SkyCoord to convert from hour angles and
        degrees into decimal degrees.
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
