import logging
import os
import re
from statistics import mode
from datetime import date, datetime
import glob
import numpy as np
import pandas as pd
import pyvo
from astropy import units as u
from astropy.coordinates import SkyCoord
from astropy.table import Table
import socket
from exo_mercat.catalogs import Catalog
from exo_mercat.utility_functions import UtilityFunctions as Utils


class Emc(Catalog):
    """
    The EMC class contains all methods and attributes related to Exo-MerCat catalog.
    """
    def __init__(self) -> None:
        """
        Initializes the Emc class object.

        This function is automatically called when an instance of the Emc class is created.
        It sets up the instance of the class by assigning a name and initializing data with an empty DataFrame.

        :param self: The instance of the Emc class.
        :type self: Emc
        :return: None
        :rtype: None
        """
        super().__init__()
        self.name = "exo_mercat"  # Assigning the name of the class
        self.data = pd.DataFrame()  # Initializing data with an empty DataFrame

    def convert_coordinates(self) -> None:
        """
        Convert the right ascension (RA) and declination (Dec) columns of the dataframe to decimal degrees. This
        function is not implemented as the EMC already has coordinates in decimal degrees.

        :param self: The instance of the Emc class.
        :type self: Emc
        :return: None
        :rtype: None
        :note: This function is not necessary as the EMC already has coordinates in decimal degrees.
        """
        pass

    def alias_as_host(self) -> None:
        """

        The alias_as_host function takes the alias column of a dataframe and checks if any of the aliases are labeled
        as hosts in some other entry. If an alias is labeled as a host, it changes the host to be that of the
        original host. It then adds all aliases of both hosts into one list for each row. The method opens a file
        called "Logs/alias_as_host.txt" in append mode and writes information about the aliases that were changed to
        be hosts. Finally, it updates the dataframe with the new aliases and logs the number of times the aliases
        were changed to be hosts. It is okay if it happens multiple times, as long as it uniforms the host name and
        adds up all the aliases, SIMBAD will find them coherently.

        :param self: The instance of the Emc class.
        :type self: Emc
        :return: None
        :rtype: None
        """
        f = open("Logs/alias_as_host.txt", "a")
        counter = 0
        host: str
        for host, group in self.data.groupby(by="host"):
            # Create an empty set to store unique values
            final_alias = set()

            # Iterate through the list and add non-NaN values to the set
            for alias in list(group.alias):
                alias_list = str(alias).split(",")
                for item in alias_list:
                    if (
                        not pd.isnull(item)
                        and item != "nan"
                        and item not in final_alias
                        and item != host
                    ):
                        final_alias.add(item)

            final_alias_total = set(final_alias)
            for al in final_alias:
                if len(self.data.loc[self.data.host == al]) > 0:
                    counter = counter + 1
                    for internal_alias in self.data.loc[self.data.host == al].alias:
                        alias_list = str(internal_alias).split(",")

                        for internal_al in alias_list:
                            if internal_al not in final_alias_total:
                                final_alias_total.add(internal_al.lstrip().rstrip())
                    self.data.loc[self.data.host == al, "host"] = host
                    f.write("ALIAS: " + al + " AS HOST:" + host + "\n")

            final_alias_total = set(final_alias_total)

            self.data.loc[self.data.host == host, "alias"] = (
                ",".join(sorted(set(final_alias_total))).rstrip(",").lstrip(",")
            )

        f.close()

        logging.info(
            "Aliases labeled as hosts in some other entry checked. It happens "
            + str(counter)
            + " times."
        )

    def check_binary_mismatch(self, keyword: str) -> None:
        """
        The check_binary_mismatch function checks for binary mismatches in the data (planets that orbit a binary but
        arecontroversial in the various catalogs and/or SIMBAD). It also checks if the SIMBAD main_id labels the
        target as binary. Ideally, all of these issues should be fixed by a human for the code to work properly.

        :param self: The instance of the Emc class.
        :type self: Emc
        :return: None
        :rtype: None
        :param keyword: str: Identify the column in the dataframe that is being checked for binary mismatches
        :return: None
        """

        self.data["binary"] = self.data["binary"].fillna("")
        self.data["potential_binary_mismatch"] = 0
        f = open("Logs/check_binary_mismatch.txt", "a")
        f.write("****" + keyword + "****\n")
        # f.write(
        #     "\n****"
        #     + keyword
        #     + "+letter THAT COULD BE UNIFORMED (only if S-type or null)****\n"
        # )
        for (key, letter), group in self.data.groupby(by=[keyword, "letter"]):
            if len(set(group.binary)) > 1:
                # Uniform only S-type
                if len(group[group.binary == "S-type"]) > 0:
                    for i in group[group.binary == "S-type"].index:
                        for j in group[group.binary != "S-type"].index:
                            if (
                                abs(group.at[i, "ra"] - group.at[j, "ra"]) > 0.01
                                and abs(group.at[i, "dec"] - group.at[j, "dec"]) > 0.01
                            ):
                                warning = (
                                    " WARNING, Coordinate Mismatch (potential_binary_mismatch 1) RA: "
                                    + str(list(group.ra))
                                    + " DEC:"
                                    + str(list(group.dec))
                                )
                                self.data.loc[i, "potential_binary_mismatch"] = 1
                                with pd.option_context("display.max_columns", 2000):
                                    f.write(
                                        key
                                        + "\n"
                                        + str(
                                            self.data.loc[group.index][
                                                [
                                                    "name",
                                                    "host",
                                                    "letter",
                                                    "binary",
                                                    "catalog",
                                                ]
                                            ]
                                        )
                                        + warning
                                        + " \n"
                                    )

                    self.data.loc[group[group.binary == "S-type"].index, "binary"] = (
                        group[group.binary != "S-type"].binary.fillna("").mode()[0]
                    )

        for (key, letter), group in self.data.groupby(by=[keyword, "letter"]):
            if len(set(group.binary)) > 1:
                # UNIFORM only null
                if len(group[group.binary == ""]) > 0:
                    warning = ""
                    for i in group[group.binary == ""].index:
                        for j in group[group.binary != ""].index:
                            if (
                                abs(group.at[i, "ra"] - group.at[j, "ra"]) > 0.01
                                and abs(group.at[i, "dec"] - group.at[j, "dec"]) > 0.01
                            ):
                                warning = (
                                    " WARNING, Coordinate Mismatch (potential_binary_mismatch 1) RA: "
                                    + str(list(group.ra))
                                    + " DEC:"
                                    + str(list(group.dec))
                                )
                                self.data.loc[i, "potential_binary_mismatch"] = 1
                                with pd.option_context("display.max_columns", 2000):
                                    f.write(
                                        key
                                        + "\n"
                                        + str(
                                            self.data.loc[group.index][
                                                [
                                                    "name",
                                                    "host",
                                                    "letter",
                                                    "binary",
                                                    "catalog",
                                                ]
                                            ]
                                        )
                                        + warning
                                        + " \n"
                                    )

                    self.data.loc[group[group.binary == ""].index, "binary"] = (
                        group[group.binary != ""].binary.fillna("").mode()[0]
                    )

        # Identify weird systems after applying the correction:
        f.write(
            "\n****"
            + keyword
            + "+letter THAT ARE INCONSISTENTLY LABELED (Potential Mismatch 2). They could be complex systems. If not, "
            "they should be treated manually in replacements.ini ****\n\n"
        )
        for (key, letter), group in self.data.groupby(by=[keyword, "letter"]):
            if len(set(group.binary)) > 1:
                with pd.option_context("display.max_columns", 2000):
                    f.write(
                        key
                        + "\n"
                        + str(
                            self.data.loc[group.index][
                                ["name", "host", "letter", "binary", "catalog"]
                            ]
                        )
                        + "\n"
                    )
                self.data.loc[group.index, "potential_binary_mismatch"] = 2

        f.write(
            "****"
            + keyword
            + " POTENTIAL BINARIES NOT TREATED HERE. They should be treated manually in replacements.ini ****\n"
        )
        for i in self.data.index:
            if (
                not str(re.search(r"([\s\d][ABCNS])$", self.data.at[i, keyword]))
                == "None"
            ):
                if (
                    not self.data.at[i, keyword][-1:].strip()
                    == self.data.at[i, "binary"]
                ):
                    f.write(
                        "MISSED POTENTIAL BINARY Key:"
                        + str(self.data.at[i, keyword])
                        + " name: "
                        + str(self.data.at[i, "name"])
                        + " binary: "
                        + str(self.data.at[i, "binary"])
                        + ".\n"
                    )

        f.close()
        logging.info("Checked potential binaries to be manually corrected.")
        logging.info(
            "Automatic correction results: "
            + str(self.data.potential_binary_mismatch.value_counts())
        )

    def prepare_columns_for_mainid_search(self) -> None:
        """
        Prepares the columns "hostbinary", "aliasbinary", "hostbinary2", "aliasbinary2", "main_id", "list_id", "main_id_ra", "main_id_dec", "angular_separation", "angsep", and "main_id_provenance" for the search of the main identifier. It also initializes the "angular_separation" column.

        :param self: The instance of the Emc class.
        :type self: Emc
        :return: None

        """
        self.data["hostbinary"] = (
            self.data["host"].astype(str)
            + " "
            + self.data["binary"]
            .astype(str)
            .replace("nan", "")
            .replace("Rogue", "")
            .replace("S-type", "")
        )
        self.data["hostbinary"] = self.data.hostbinary.str.rstrip()

        for i in self.data.index:
            if len(self.data.at[i, "alias"]) > 0:
                single_binary = (
                    str(self.data.at[i, "binary"])
                    .replace("nan", "")
                    .replace("Rogue", "")
                    .replace("S-type", "")
                )
                single_aliases = str(self.data.at[i, "alias"]).split(",")
                single_aliases_binary = [
                    s + " " + single_binary for s in single_aliases
                ]
                single_aliases_binary = [
                    s.rstrip().lstrip() for s in single_aliases_binary
                ]
                self.data.at[i, "aliasbinary"] = ",".join(single_aliases_binary)
                self.data["aliasbinary"] = self.data["aliasbinary"].fillna("")

        self.data["hostbinary2"] = self.data["host"].astype(str) + self.data[
            "binary"
        ].astype(str).replace("nan", "").replace("Rogue", "").replace("S-type", "")
        self.data["hostbinary2"] = self.data.hostbinary2.str.rstrip()

        for i in self.data.index:
            if len(self.data.at[i, "alias"]) > 0:
                single_binary = (
                    str(self.data.at[i, "binary"])
                    .replace("nan", "")
                    .replace("Rogue", "")
                    .replace("S-type", "")
                )
                single_aliases = str(self.data.at[i, "alias"]).split(",")
                single_aliases_binary = [s + single_binary for s in single_aliases]
                single_aliases_binary = [
                    s.rstrip().lstrip() for s in single_aliases_binary
                ]
                self.data.at[i, "aliasbinary2"] = ",".join(single_aliases_binary)
                self.data["aliasbinary2"] = self.data["aliasbinary2"].fillna("")

        self.data["main_id"] = ""
        self.data["list_id"] = ""
        self.data["main_id_ra"] = np.nan
        self.data["main_id_dec"] = np.nan

        self.data["angular_separation"] = ""
        self.data["angsep"] = -1.0

        self.data["main_id_provenance"] = ""

    def fill_mainid_provenance_column(self, keyword):
        """
        Fills the 'main_id_provenance' column with the provided keyword if 'main_id_provenance' is empty and 'main_id' is not empty for each relevant index.

        :param self: The instance of the Emc class.
        :type self: Emc
        :param keyword: The keyword to fill the 'main_id_provenance' column with.
        :type keyword: str
        :return: None
        :rtype: None
        """
        for ind in self.data[self.data.main_id_provenance == ""].index:
            if self.data.at[ind, "main_id"] != "":
                self.data.at[ind, "main_id_provenance"] = keyword

    def simbad_list_host_search(self, typed_id: str) -> None:
        """
        The simbad_list_host_search function takes a column name as an argument and searches for the host star in that column in SIMBAD. It then fills in the main_id, IDS, RA, and DEC columns with information from SIMBAD if it finds a match.

        :param self: The instance of the Emc class.
        :type self: Emc
        :param typed_id: The name of the column that contains the host star to search for (host or hostbinary)
        :type typed_id: str
        :return: None
        :rtype: None
        """
        list_of_hosts = self.data[self.data.main_id == ""][[typed_id]].drop_duplicates()
        list_of_hosts[typed_id] = list_of_hosts.loc[
            list_of_hosts[typed_id].str.findall(r"[^\x00-\x7F]+").str.len() == 0,
            typed_id,
        ]

        service = pyvo.dal.TAPService("http://simbad.u-strasbg.fr:80/simbad/sim-tap")

        t2 = Table.from_pandas(list_of_hosts)
        query = (
            """SELECT t.*, basic.main_id, basic.ra as ra_2,basic.dec as dec_2, ids.ids as ids FROM TAP_UPLOAD.tab as 
            t LEFT OUTER JOIN ident ON ident.id = t."""
            + typed_id
            + """ LEFT OUTER JOIN basic ON ident.oidref = basic.oid LEFT OUTER JOIN ids ON basic.oid = ids.oidref"""
        )
        table = Utils.perform_query(service, query, uploads_dict={"tab": t2})

        logging.info(
            "List of unique star names "
            + str(len(list_of_hosts))
            + " of which successful SIMBAD queries "
            + str(len(table))
        )

        for host in table[typed_id]:
            self.data.loc[self.data[typed_id] == host, "main_id_ra"] = float(
                table.loc[table[typed_id] == host, "ra_2"].values[0]
            )
            self.data.loc[self.data[typed_id] == host, "main_id_dec"] = float(
                table.loc[table[typed_id] == host, "dec_2"].values[0]
            )
            self.data.loc[self.data[typed_id] == host, "main_id"] = table.loc[
                table[typed_id] == host, "main_id"
            ].values[0]
            self.data.loc[self.data[typed_id] == host, "list_id"] = (
                table.loc[table[typed_id] == host, "ids"].values[0].replace("|", ",")
            )
            self.data.loc[self.data[typed_id] == host, "angsep"] = 0.0

        self.data.main_id = self.data.main_id.fillna("")
        self.data.list_id = self.data.list_id.fillna("")
        self.data.main_id_ra = self.data.main_id_ra.fillna(np.nan)
        self.data.main_id_dec = self.data.main_id_dec.fillna(np.nan)

    def simbad_list_alias_search(self, column: str) -> None:
        """
        The simbad_list_alias_search function takes a column name as an argument and searches for the main ID of
        each object in that column. The function first splits the string into a list of aliases, then iterates
        through each alias to search SIMBAD for its main ID. If it finds one, it will update the dataframe with
        that information.
        :param column: The name of the column that contains the host star to search for (alias or aliasbinary)
        """

        alias_df = pd.DataFrame(columns=["ind", column])
        self.data[column] = self.data[column].fillna("")
        for i in self.data[self.data.main_id == ""].index:
            if len(self.data.at[i, column].replace("nan", "")) > 0:
                list_of_aliases = pd.DataFrame(
                    self.data.at[i, column].split(","), columns=[column]
                ).drop_duplicates()
                # filter out non-ascii characters (conflicts with pyvo
                # otherwise)
                cleaned_list_of_aliases = pd.DataFrame(columns=[column])
                cleaned_list_of_aliases[column] = list_of_aliases.loc[
                    list_of_aliases[column].str.findall(r"[^\x00-\x7F]+").str.len()
                    == 0,
                    column,
                ]
                cleaned_list_of_aliases["ind"] = i
                alias_df = pd.concat([alias_df, cleaned_list_of_aliases])

        service = pyvo.dal.TAPService("http://simbad.u-strasbg.fr:80/simbad/sim-tap")
        t2 = Table.from_pandas(alias_df)
        query = (
            """SELECT t.*, basic.main_id, basic.ra as ra_2,basic.dec as dec_2, ids.ids FROM TAP_UPLOAD.tab as t LEFT 
            OUTER JOIN ident ON ident.id = t."""
            + column
            + """ LEFT OUTER JOIN basic ON ident.oidref = basic.oid LEFT OUTER JOIN ids ON basic.oid = ids.oidref"""
        )
        table = Utils.perform_query(service, query, uploads_dict={"tab": t2})

        table = table.drop_duplicates(["ind", "main_id", "ra_2", "dec_2", "ids"])
        for i in table["ind"].unique():
            subtable = table[table.ind == i]

            if len(subtable) != 0:
                if len(subtable) > 1:
                    logging.info(
                        "WARNING, MULTIPLE ALIASES NOT IN AGREEMENT "
                        + column
                        + " "
                        + str(sorted(set(subtable[column].unique())))
                        + " main_id "
                        + str(sorted(set(subtable["main_id"].unique())))
                    )
                    subtable = subtable.head(1)

                self.data.at[int(i), "main_id_ra"] = float(subtable["ra_2"].values[0])
                self.data.at[int(i), "main_id_dec"] = float(subtable["dec_2"].values[0])
                self.data.at[int(i), "main_id"] = subtable["main_id"].values[0]
                self.data.at[int(i), "list_id"] = (
                    subtable["ids"].values[0].replace("|", ",")
                )
                self.data.at[int(i), "angsep"] = 0.0

        self.data.main_id = self.data.main_id.fillna("")
        self.data.list_id = self.data.list_id.fillna("")
        self.data.main_id_ra = self.data.main_id_ra.fillna(np.nan)
        self.data.main_id_dec = self.data.main_id_dec.fillna(np.nan)

    def get_host_info_from_simbad(self) -> None:
        """
        The get_host_info_from_simbad function takes the dataframe and extracts all
        unique host star names. It then queries Simbad for each of these names, and
        returns a table with the main ID, alias IDs, RA and DEC.The function merges
        this table with the original dataframe on host name (left join).If there are
        still rows missing main_id values in the merged table, it will query Simbad
        again using all aliases from those rows.
        """

        logging.info("HOST+ +BINARY Simbad Check")
        self.simbad_list_host_search("hostbinary")
        logging.info(
            "Rows still missing main_id after host search "
            + str(len(self.data[self.data.main_id == ""]))
        )

        logging.info("ALIAS+ +BINARY Simbad Check")
        self.simbad_list_alias_search("aliasbinary")
        logging.info(
            "Rows still missing main_id after alias search "
            + str(len(self.data[self.data.main_id == ""]))
        )

        logging.info("HOST+BINARY Simbad Check")
        self.simbad_list_host_search("hostbinary2")
        logging.info(
            "Rows still missing main_id after host search "
            + str(len(self.data[self.data.main_id == ""]))
        )

        logging.info("ALIAS+BINARY Simbad Check")
        self.simbad_list_alias_search("aliasbinary2")
        logging.info(
            "Rows still missing main_id after alias search "
            + str(len(self.data[self.data.main_id == ""]))
        )

        logging.info("PURE HOST Simbad Check")
        self.simbad_list_host_search("host")
        logging.info(
            "Rows still missing main_id after host search "
            + str(len(self.data[self.data.main_id == ""]))
        )

        logging.info("PURE ALIAS Simbad Check")
        self.simbad_list_alias_search("alias")
        logging.info(
            "Rows still missing main_id after alias search "
            + str(len(self.data[self.data.main_id == ""]))
        )

        self.data["list_id"] = self.data["list_id"].apply(
            lambda x: str(x).replace("|", ",")
        )

        self.fill_mainid_provenance_column("SIMBAD")

    def get_coordinates_from_simbad(self) -> None:
        """
        This function takes the dataframe and checks if there are any matches in Simbad for the coordinates of each
        object. It does this by querying Simbad with a circle around each coordinate, starting at 0.01 degrees and
        increasing to 0.5 degrees until it finds a match or gives up.
        """

        # SIMBAD
        service = pyvo.dal.TAPService("http://simbad.u-strasbg.fr:80/simbad/sim-tap")

        tolerance = 1 / 3600  # arcsec in degrees
        t2 = Table.from_pandas(
            self.data[self.data.main_id == ""][["hostbinary", "ra", "dec"]]
        )
        query = (
            """SELECT basic.main_id, basic.dec as dec_2,basic.ra as ra_2, basic.otype as type, t.hostbinary, t.ra, 
            t.dec FROM basic JOIN TAP_UPLOAD.tab AS t on 1=CONTAINS(POINT('ICRS',basic.ra, basic.dec),   
            CIRCLE('ICRS', t.ra, t.dec,"""
            + str(tolerance)
            + """)) """
        )
        table = Utils.perform_query(service, query, uploads_dict={"tab": t2})

        # collect ids too from the previous table
        t2 = Table.from_pandas(table)
        query = """SELECT t.*, ids.ids as ids FROM TAP_UPLOAD.tab as t LEFT OUTER JOIN ident ON ident.id = t.main_id 
        LEFT OUTER JOIN basic ON ident.oidref = basic.oid LEFT OUTER JOIN ids ON basic.oid = ids.oidref"""
        table = Utils.perform_query(service, query, uploads_dict={"tab": t2})
        table = Utils.calculate_angsep(table)

        for host in table["hostbinary"]:
            self.data.loc[self.data["hostbinary"] == host, "main_id_ra"] = float(
                table.loc[table["hostbinary"] == host, "ra_2"].values[0]
            )
            self.data.loc[self.data["hostbinary"] == host, "main_id_dec"] = float(
                table.loc[table["hostbinary"] == host, "dec_2"].values[0]
            )
            self.data.loc[self.data["hostbinary"] == host, "main_id"] = table.loc[
                table["hostbinary"] == host, "main_id"
            ].values[0]
            self.data.loc[self.data["hostbinary"] == host, "list_id"] = (
                table.loc[table["hostbinary"] == host, "ids"]
                .values[0]
                .replace("|", ",")
            )
            self.data.loc[self.data["hostbinary"] == host, "angsep"] = float(
                table.loc[table["hostbinary"] == host, "angsep"].values[0]
            )

        self.data.main_id = self.data.main_id.fillna("")
        self.data.list_id = self.data.list_id.fillna("")
        self.data.main_id_ra = self.data.main_id_ra.fillna(np.nan)
        self.data.main_id_dec = self.data.main_id_dec.fillna(np.nan)
        logging.info(
            "After coordinate check on SIMBAD at tolerance "
            + str(tolerance * 3600)
            + " arcsec, the residuals are: "
            + str(self.data[self.data.main_id == ""].shape[0])
            + ". Maximum angular separation: "
            + str(max(self.data[self.data.angsep == self.data.angsep].angsep))
        )

        self.fill_mainid_provenance_column("SIMBADCOORD")

    def get_host_info_from_tic(self) -> None:
        """
        The simbad_list_host_search function takes a column name as an argument and searches for the host star
        in that column in SIMBAD. It then fills in the main_id, IDS, RA, and DEC columns with information from
        SIMBAD if it finds a match.
        """
        logging.info("TIC host check")
        list_of_hosts = (
            self.data[self.data.main_id == ""][["host"]].drop_duplicates().dropna()
        )
        list_of_hosts = list_of_hosts[list_of_hosts.host.str.contains("TIC")]
        list_of_hosts["host"] = list_of_hosts.loc[
            list_of_hosts["host"].str.findall(r"[^\x00-\x7F]+").str.len() == 0, "host"
        ]
        list_of_hosts["host"] = list_of_hosts["host"].str.replace("TIC ", "")
        timeout = 100000
        socket.setdefaulttimeout(timeout)

        service = pyvo.dal.TAPService("http://TAPVizieR.u-strasbg.fr/TAPVizieR/tap/")

        table = pd.DataFrame()
        for host in list_of_hosts["host"]:
            query = """SELECT tic.RAJ2000 as ra_2, tic.DEJ2000 as dec_2,tic.GAIA, tic.UCAC4, tic."2MASS", tic.WISEA, 
        tic.TIC, tic.KIC, tic.HIP, tic.TYC  FROM "IV/38/tic" as tic WHERE tic.TIC = """ + str(
                host
            )

            single_table = Utils.perform_query(service, query, uploads_dict={})
            single_table["host"] = host
            table = pd.concat([table, single_table])
        table = table.drop_duplicates()
        logging.info(
            "List of unique star names with a TIC host "
            + str(len(list_of_hosts))
            + " of which successful TIC queries "
            + str(len(table))
        )
        for host in table["host"]:
            self.data.loc[self.data["host"] == "TIC " + host, "main_id_ra"] = float(
                table.loc[table["host"] == host, "ra_2"].values[0]
            )
            self.data.loc[self.data["host"] == "TIC " + host, "main_id_dec"] = float(
                table.loc[table["host"] == host, "dec_2"].values[0]
            )
            self.data.loc[self.data["host"] == "TIC " + host, "main_id"] = table.loc[
                table["host"] == host, "main_id"
            ].values[0]
            self.data.loc[self.data["host"] == "TIC " + host, "list_id"] = (
                table.loc[table["host"] == host, "ids"].values[0].replace("|", ",")
            )
            self.data.loc[self.data["host"] == "TIC " + host, "angsep"] = 0.0
        self.data.main_id = self.data.main_id.fillna("")
        self.data.list_id = self.data.list_id.fillna("")
        self.data.main_id_ra = self.data.main_id_ra.fillna("")
        self.data.main_id_dec = self.data.main_id_dec.fillna("")

        logging.info(
            "Rows still missing main_id after TIC host search "
            + str(len(self.data[self.data.main_id == ""]))
        )

        logging.info("TIC alias check")

        alias_df = self.data[self.data.main_id == ""]
        alias_df = alias_df[alias_df.alias.str.contains("TIC")]
        for ind in alias_df.index:
            tic_alias = alias_df.at[ind, "alias"].split(",")
            alias_df.at[ind, "tic_alias"] = [x for x in tic_alias if "TIC" in x][0]
        alias_df["tic_alias"] = alias_df["tic_alias"].str.replace("TIC ", "")

        alias_df = alias_df[["host", "tic_alias"]]

        table = pd.DataFrame()
        for ind in alias_df.index:
            query = (
                """SELECT tic.RAJ2000 as ra_2, tic.DEJ2000 as dec_2,tic.GAIA, tic.UCAC4, tic."2MASS", tic.WISEA, 
        tic.TIC, tic.KIC, tic.HIP, tic.TYC  FROM "IV/38/tic" as tic WHERE tic.TIC = """
                + alias_df.at[ind, "tic_alias"]
            )

            single_table = Utils.perform_query(service, query, uploads_dict={})
            single_table["host"] = alias_df.at[ind, "host"]
            table = pd.concat([table, single_table])
        table = table.drop_duplicates()

        logging.info(
            "List of unique star names with a TIC alias "
            + str(len(list_of_hosts))
            + " of which successful TIC queries "
            + str(len(table))
        )
        for host in table["host"]:
            self.data.loc[self.data["host"] == host, "main_id_ra"] = float(
                table.loc[table["host"] == host, "ra_2"].values[0]
            )
            self.data.loc[self.data["host"] == host, "main_id_dec"] = float(
                table.loc[table["host"] == host, "dec_2"].values[0]
            )
            self.data.loc[self.data["host"] == host, "main_id"] = table.loc[
                table["host"] == host, "main_id"
            ].values[0]
            self.data.loc[self.data["host"] == host, "list_id"] = (
                table.loc[table["host"] == host, "ids"].values[0].replace("|", ",")
            )
            self.data.loc[self.data["host"] == host, "angsep"] = 0.0
        self.data.main_id = self.data.main_id.fillna("")
        self.data.list_id = self.data.list_id.fillna("")
        self.data.main_id_ra = self.data.main_id_ra.fillna("")
        self.data.main_id_dec = self.data.main_id_dec.fillna("")

        logging.info(
            "Rows still missing main_id after TIC alias search "
            + str(len(self.data[self.data.main_id == ""]))
        )
        self.fill_mainid_provenance_column("TIC")

    def get_coordinates_from_tic(self, tolerance=1 / 3600):
        # TIC

        service = pyvo.dal.TAPService(" http://TAPVizieR.u-strasbg.fr/TAPVizieR/tap/")
        t2 = self.data[self.data.main_id == ""][["hostbinary", "ra", "dec"]]
        tolerance = 1 / 3600  # arcsec in degrees

        table = pd.DataFrame()
        for ind in t2.index:
            query = (
                """SELECT tic.RAJ2000 as ra_2, tic.DEJ2000 as dec_2,tic.GAIA, tic.UCAC4, tic."2MASS", tic.WISEA, 
                    tic.TIC, tic.KIC, tic.HIP, tic.TYC  FROM "IV/38/tic" as tic  WHERE 1=CONTAINS(POINT('ICRS',tic.RAJ2000, tic.DEJ2000),   CIRCLE('ICRS',"""
                + str(t2.at[ind, "ra"])
                + ""","""
                + str(t2.at[ind, "dec"])
                + ""","""
                + str(tolerance)
                + """))"""
            )
            single_table = Utils.perform_query(service, query, uploads_dict={})
            if len(single_table) > 0:
                single_table["hostbinary"] = t2.at[ind, "hostbinary"]
                single_table["ra"] = t2.at[ind, "ra"]
                single_table["dec"] = t2.at[ind, "dec"]
                single_table = Utils.calculate_angsep(single_table)

                table = pd.concat([table, single_table])
        table = table.drop_duplicates()

        for host in table["hostbinary"]:
            self.data.loc[self.data["hostbinary"] == host, "main_id_ra"] = float(
                table.loc[table["hostbinary"] == host, "ra_2"].values[0]
            )
            self.data.loc[self.data["hostbinary"] == host, "main_id_dec"] = float(
                table.loc[table["hostbinary"] == host, "dec_2"].values[0]
            )
            self.data.loc[self.data["hostbinary"] == host, "main_id"] = table.loc[
                table["hostbinary"] == host, "main_id"
            ].values[0]
            self.data.loc[self.data["hostbinary"] == host, "list_id"] = (
                table.loc[table["hostbinary"] == host, "ids"]
                .values[0]
                .replace("|", ",")
            )
            self.data.loc[self.data["hostbinary"] == host, "angsep"] = float(
                table.loc[table["hostbinary"] == host, "angsep"].values[0]
            )

        self.data.main_id = self.data.main_id.fillna("")
        self.data.list_id = self.data.list_id.fillna("")
        self.data.main_id_ra = self.data.main_id_ra.fillna("")
        self.data.main_id_dec = self.data.main_id_dec.fillna("")

        logging.info(
            "After coordinate check on TIC "
            "at tolerance "
            + str(tolerance * 3600)
            + " arcsec, the residuals are: "
            + str(self.data[self.data.main_id == ""].shape[0])
            + ". Maximum angular separation: "
            + str(max(self.data[self.data.angsep == self.data.angsep].angsep))
        )
        self.fill_mainid_provenance_column("TICCOORD")

    def check_coordinates(self) -> None:
        """
        The check_coordinates function checks for mismatches in the RA and DEC
        coordinates of a given host (for the targets that cannot rely on SIMBAD
        MAIN_ID because the query was unsuccessful). It does this by grouping all
        entries with the same host name, then checking if any of those entries have
        an RA or DEC that is more than 0.01 degrees away from the mode value for that
        group. If so, it prints out information about those mismatched values
        to a log file called check_coordinates.txt.
        """
        countra = 0
        countdec = 0

        self.data["coordinate_mismatch"] = ""

        f = open("Logs/check_coordinates.txt", "a")
        for host, group in self.data[self.data.main_id == ""].groupby("host"):
            ra = mode(list(round(group.ra, 3)))
            dec = mode(list(round(group.dec, 3)))
            mismatch_string = ""

            if (abs(group["ra"] - ra) > 0.01).any():
                countra = countra + 1
                with pd.option_context("display.max_columns", 2000):
                    f.write(
                        "*** MISMATCH ON RA *** "
                        + str(
                            (
                                group[
                                    [
                                        "name",
                                        "host",
                                        "binary",
                                        "letter",
                                        "catalog",
                                        "ra",
                                    ]
                                ]
                            )
                        )
                        + "\n"
                    )
                mismatch_string = "RA"

            if (abs(group["dec"] - dec) > 0.01).any():
                countdec = countdec + 1
                with pd.option_context("display.max_columns", 2000):
                    f.write(
                        "*** MISMATCH ON DEC *** "
                        + str(
                            (
                                group[
                                    [
                                        "name",
                                        "host",
                                        "binary",
                                        "letter",
                                        "catalog",
                                        "dec",
                                    ]
                                ]
                            )
                        )
                        + "\n"
                    )
                mismatch_string = mismatch_string + "DEC"

            self.data.loc[group.index, "coordinate_mismatch"] = mismatch_string
        f.close()
        logging.info("Found " + str(countra) + " mismatched RA.")
        logging.info("Found " + str(countdec) + " mismatched DEC.")

        logging.info(self.data.coordinate_mismatch.value_counts())

    def replace_old_new_identifier(
        self, identifier, new_identifier, binary=None
    ) -> str:
        output_string = ""
        service = pyvo.dal.TAPService("http://simbad.u-strasbg.fr:80/simbad/sim-tap")

        query = (
            """SELECT  basic.main_id, basic.ra as ra_2,basic.dec as dec_2, ids.ids
    FROM ident JOIN basic ON ident.oidref = basic.oid LEFT OUTER JOIN ids ON basic.oid = ids.oidref
    WHERE id = '"""
            + new_identifier
            + """'"""
        )

        table = Utils.perform_query(service, query, uploads_dict={})
        if len(table) >= 1:
            self.data.loc[self.data.main_id == identifier, "main_id_ra"] = float(
                table.at[0, "ra_2"]
            )
            self.data.loc[self.data.main_id == identifier, "main_id_dec"] = float(
                table.at[0, "dec_2"]
            )
            if binary is not None:
                cumulative_alias = (
                    ",".join(
                        self.data.loc[
                            self.data.main_id == identifier, "list_id"
                        ].unique()
                    )
                    + ","
                    + table.loc[0, "ids"].replace("|", ",")
                )
                cumulative_alias = ",".join(
                    [
                        x
                        for x in set(
                            cumulative_alias.rstrip(",").lstrip(",").split(",")
                        )
                    ]
                )
                self.data.loc[
                    self.data.main_id == identifier, "list_id"
                ] = cumulative_alias
            else:
                self.data.loc[self.data.main_id == identifier, "list_id"] = table.loc[
                    0, "ids"
                ].replace("|", ",")
            output_string = (
                "MAINID corrected " + identifier + " to " + new_identifier + "."
            )
            if binary is not None:
                output_string = output_string + " Binary value: " + binary + "."
                binary_catalog = self.data.loc[
                    self.data.main_id == identifier, "binary"
                ].copy()
                if (binary_catalog.replace("S-type", "") == "").all():
                    self.data.loc[self.data.main_id == identifier, "binary"] = binary
                    output_string = output_string + " Binary could be uniformed."
                elif (binary_catalog != binary).any():
                    output_string = (
                        output_string
                        + " Binary value is not in agreement, please check."
                    )
                elif (binary_catalog == binary).all():
                    output_string = output_string + " Already correct."
            else:
                output_string = output_string + "\n"
            # last thing to be changed since it changes the query
            self.data.loc[self.data.main_id == identifier, "main_id"] = table.at[
                0, "main_id"
            ]
            output_string = output_string
        elif len(table) == 0:
            output_string = (
                "Weird MAINID found: "
                + identifier
                + " but cannot be found when "
                + new_identifier
                + ". "
            )

        return output_string

    def polish_main_id(self) -> None:
        counter = 0
        f = open("Logs/polish_main_id.txt", "a")
        f.write("***** CHECK FOR PLANET LETTER IN MAIN_ID *****\n")
        for identifier in self.data.main_id:
            # PLANET IN SIMBAD, WILL TRY TO LOOK FOR STAR IN SIMBAD AND REPLACE
            if len(re.findall("[\s\d][b-i]$", identifier)) > 0:
                new_identifier = identifier[:-1].strip().replace("NAME ", "")
                counter += 1
                output_string = self.replace_old_new_identifier(
                    identifier, new_identifier, None
                )
                f.write(output_string)
        f.write("\n***** CHECK FOR BINARY LETTER IN MAIN_ID *****\n")
        for identifier in self.data.main_id:
            # CHECK CIRCUMBINARY, WILL TRY TO LOOK FOR HOST
            if len(re.findall(r"[\s\d](\(AB\))$", identifier)) > 0:
                counter += 1
                # it finds AB in main_id. Check if AB is already in binary
                if (
                    self.data.loc[self.data.main_id == identifier, "binary"] != "AB"
                ).any():
                    f.write(
                        "MAIN_ID: "
                        + identifier
                        + ". Selected binary value: AB"
                        + ". Value in catalog: \n"
                        + str(
                            self.data.loc[
                                self.data.main_id == identifier,
                                ["name", "host", "binary", "catalog"],
                            ]
                        )
                        + "\n"
                    )

                new_identifier = identifier[:-4].rstrip().replace("NAME ", "")
                output_string = self.replace_old_new_identifier(
                    identifier, new_identifier, "AB"
                )
                f.write(output_string + "\n***\n")

            if len(re.findall(r"[\[a-z](AB)$|\s(AB)$|\d(AB)$]", identifier)) > 0:
                counter += 1
                if (
                    self.data.loc[self.data.main_id == identifier, "binary"] != "AB"
                ).any():
                    f.write(
                        "MAIN_ID: "
                        + identifier
                        + ". Selected binary value: AB"
                        + ". Value in catalog: \n"
                        + str(
                            self.data.loc[
                                self.data.main_id == identifier,
                                ["name", "host", "binary", "catalog"],
                            ]
                        )
                        + "\n"
                    )

                new_identifier = identifier[:-2].rstrip()
                output_string = self.replace_old_new_identifier(
                    identifier, new_identifier, "AB"
                )
                f.write(output_string + "\n***\n")

            # REGULAR BINARY
            if len(re.findall("[\s\d][ABCSN]$", identifier)) > 0:
                counter += 1
                if (
                    self.data.loc[self.data.main_id == identifier, "binary"]
                    != identifier[-1:]
                ).any():
                    f.write(
                        "MAIN_ID: "
                        + identifier
                        + ". Selected binary value: "
                        + identifier[-1:]
                        + ". Value in catalog: \n"
                        + str(
                            self.data.loc[
                                self.data.main_id == identifier,
                                ["name", "host", "binary", "catalog"],
                            ]
                        )
                        + "\n"
                    )

                new_identifier = identifier[:-1].strip()
                output_string = self.replace_old_new_identifier(
                    identifier, new_identifier, identifier[-1:]
                )

                f.write(output_string + "\n***\n")
        f.close()
        logging.info(
            "Removed planet/binary letter from main_id. It happens "
            + str(counter)
            + " times."
        )

    def fill_missing_main_id(self):
        # ROWS STILL MISSING ID
        self.data["main_id_provenance"] = (
            self.data["main_id_provenance"]
            .replace("", np.nan)
            .fillna(self.data["catalog"])
        )
        self.data["main_id"] = (
            self.data["main_id"].replace("", np.nan).fillna(self.data["host"])
        )
        self.data["main_id_ra"] = (
            self.data["main_id_ra"].replace("", np.nan).fillna(self.data["ra"])
        ).astype(float)
        self.data["main_id_dec"] = (
            self.data["main_id_dec"].replace("", np.nan).fillna(self.data["dec"])
        ).astype(float)
        self.data["angular_separation"] = (
            self.data["catalog"] + ": " + self.data.angsep.astype(str)
        )

    def check_same_host_different_id(self) -> None:
        """
        The check_same_host_different_id function checks to see if there
        are any instances where the same host has multiple SIMBAD main IDs.
        This should _never_ happen unless the SIMBAD search is failing.

        """
        f = open("Logs/check_same_host_different_id.txt", "a")
        for host, group in self.data.groupby("hostbinary"):
            if len(group.main_id.drop_duplicates()) > 1:
                with pd.option_context("display.max_columns", 2000):
                    f.write("SAME HOST+BINARY DIFFERENT MAIN_ID\n")
                    f.write(str(group[["main_id", "binary", "catalog"]]) + "\n")

        for host, group in self.data.groupby("host"):
            if len(group.main_id.drop_duplicates()) > 1:
                with pd.option_context("display.max_columns", 2000):
                    f.write("SAME HOST DIFFERENT MAIN_ID\n")
                    f.write(str(group[["main_id", "binary", "catalog"]]) + "\n")
        logging.info("Checked if host is found under different main_ids.")
        f.close()

    def check_same_coords_different_id(self) -> None:
        """
        The check_same_host_different_id function checks to see if there
        are any instances where the same host has multiple SIMBAD main IDs.
        This should _never_ happen unless the SIMBAD search is failing.

        """
        f = open("Logs/check_same_coords_different_id.txt", "a")
        self.data.main_id_ra = self.data.main_id_ra.astype(float)
        self.data.main_id_dec = self.data.main_id_dec.astype(float)
        self.data["skycoord"] = SkyCoord(
            ra=self.data.main_id_ra * u.degree, dec=self.data.main_id_dec * u.degree
        )
        for i in self.data.index:
            # create a wider rectangle to look into
            sub = self.data.copy()
            sub = self.data[
                self.data.main_id_ra < (self.data.at[i, "main_id_ra"] + 100 / 3600)
            ]
            sub = sub[sub.main_id_ra > (sub.at[i, "main_id_ra"] - 100 / 3600)]
            sub = sub[sub.main_id_dec > (sub.at[i, "main_id_dec"] - 100 / 3600)]
            sub = sub[sub.main_id_ra < (sub.at[i, "main_id_dec"] + 100 / 3600)]
            for j in sub.index:
                sub.at[j, "angsep"] = (
                    self.data.at[i, "skycoord"].separation(sub.at[j, "skycoord"]).value
                )
            sub = sub[sub.angsep < 1 / 3600]
            if len(sub.main_id.unique()) > 1:
                with pd.option_context("display.max_columns", 2000):
                    f.write(
                        "FOUND SAME COORDINATES DIFFERENT MAINID\n"
                        + str(
                            sub[
                                [
                                    "host",
                                    "main_id",
                                    "binary",
                                    "letter",
                                    "catalog",
                                    "angsep",
                                    "main_id_provenance",
                                ]
                            ]
                        )
                        + "\n"
                    )

        logging.info("Checked if same coordinates found in main_ids")
        f.close()

    def group_by_list_id_check_main_id(self) -> None:
        """ """
        f = open("Logs/group_by_list_id_check_main_id.txt", "a")
        count = 0
        for ids, group in self.data.groupby(by="list_id"):
            if ids != "" and len(set(group.main_id)) > 1:
                self.data.loc[self.data.list_id == ids, "main_id"] = list(
                    group.main_id
                )[0]
                with pd.option_context("display.max_columns", 2000):
                    f.write(
                        "*** SAME LIST_ID, DIFFERENT MAIN_ID *** \n"
                        + str(group[["catalog", "status", "letter", "main_id"]])
                        + "\n"
                    )
                count = count + 1
        logging.info(
            "Planets that had a different main_id name but same SIMBAD alias: "
            + str(count)
        )
        f.close()

    def group_by_main_id_set_final_alias(self) -> None:
        """
        The group_by_main_id_set_final_alias function takes the alias and list_id columns from
        the dataframe,and combines them into a single column called final_alias.
        It then removes duplicates from this new column.
        """
        self.data["final_alias"] = ""
        for host, group in self.data.groupby(by="main_id"):
            final_alias = ""
            for al in group.alias:
                final_alias = final_alias + "," + str(al)
            for al in group.list_id:
                final_alias = final_alias + "," + str(al)

            final_alias = ",".join([x for x in set(final_alias.split(",")) if x])
            final_alias = final_alias.replace("nan", "").replace(",,", ",")
            self.data.loc[self.data.main_id == host, "final_alias"] = final_alias

    def cleanup_catalog(self) -> None:
        """
        The cleanup_catalog function is used to replace any rows in
        the catalog that have a value of 0 or inf for any of the
        columns i, mass, msini, a, p and e with NaN.
        """
        for col in ["i", "mass", "msini", "a", "p", "e"]:
            self.data.loc[self.data[col + "_min"] == 0, col + "_min"] = np.nan
            self.data.loc[self.data[col + "_max"] == 0, col + "_max"] = np.nan
            self.data.loc[self.data[col + "_min"] == np.inf, col + "_min"] = np.nan
            self.data.loc[self.data[col + "_max"] == np.inf, col + "_max"] = np.nan
        logging.info("Catalog cleared from zeroes and infinities.")

    def group_by_period_check_letter(self) -> None:
        f1 = open("Logs/group_by_period_check_letter.txt", "a")
        # self.data["working_period_group"] = Utils.round_parameter_bin(self.data["p"])
        # self.data["working_sma_group"] = Utils.round_parameter_bin(self.data["a"])

        grouped_df = self.data.groupby(["main_id", "binary"], sort=True, as_index=False)
        f1.write("TOTAL NUMBER OF GROUPS: " + str(grouped_df.ngroups) + "\n")
        counter = 0
        for (
            mainid,
            binary,
        ), group in grouped_df:
            if len(group) > 1:  # there are multiple planets in the system
                group = Utils.calculate_working_p_sma(group, tolerance=0.1)
                for pgroup in list(set(group.working_p)):
                    subgroup = group[group.working_p == pgroup]
                    warning = ""
                    # if period is zero, group by semimajor axis, if
                    # unsuccessful group by letter
                    if pgroup != -1:
                        # try to fix the letter if it is different
                        if len(list(set(subgroup.letter))) > 1:
                            warning = (
                                "INCONSISTENT LETTER FOR SAME PERIOD \n"
                                + str(
                                    subgroup[
                                        [
                                            "main_id",
                                            "binary",
                                            "letter",
                                            "catalog",
                                            "catalog_name",
                                            "p",
                                        ]
                                    ]
                                )
                                + "\n\n"
                            )
                            adjusted_letter = [
                                l
                                for l in list(set(subgroup.letter.dropna().unique()))
                                if ".0" not in str(l)
                            ]
                            if len(adjusted_letter) == 1:
                                self.data.loc[
                                    subgroup.index, "letter"
                                ] = adjusted_letter[0]
                                warning = "FIXABLE " + warning
                            if "BD" in list(set(subgroup.letter.dropna().unique())):
                                self.data.loc[subgroup.index, "letter"] = "BD"
                                warning = "FORCED BD " + warning
                    else:
                        for agroup in list(set(subgroup.working_a)):
                            subsubgroup = subgroup[subgroup.working_a == agroup]
                            # try to fix the letter if it is different
                            if len(list(set(subsubgroup.letter))) > 1:
                                warning = (
                                    "INCONSISTENT LETTER FOR SAME SMA \n"
                                    + str(
                                        subsubgroup[
                                            [
                                                "main_id",
                                                "binary",
                                                "letter",
                                                "catalog",
                                                "catalog_name",
                                                "a",
                                            ]
                                        ]
                                    )
                                    + "\n\n"
                                )

                                adjusted_letter = [
                                    l
                                    for l in list(
                                        set(subsubgroup.letter.dropna().unique())
                                    )
                                    if ".0" not in str(l)
                                ]
                                if len(adjusted_letter) == 1:
                                    self.data.loc[
                                        subsubgroup.index, "letter"
                                    ] = adjusted_letter[0]
                                    warning = "FIXABLE " + warning
                            if "BD" in list(set(subgroup.letter.dropna().unique())):
                                self.data.loc[subgroup.index, "letter"] = "BD"
                                warning = "FORCED BD " + warning
                    f1.write(warning)
        f1.close()

    @staticmethod
    def merge_into_single_entry(
        group: pd.DataFrame, mainid: str, binary: str, letter: str
    ) -> pd.DataFrame:
        """
        The merge_into_single_entry function takes the dataframe and merges all entries
        with the same main_id and letter (from the different catalogs) into a single entry.
        It does this by grouping by main_id and letter, then iterating through each group.
        It creates an empty dataframe called 'entry' and adds information to it from each group.
        The final entry contains: the official SIMBAD ID and coordinates; the measurements
        that have the smallest relative error with the corresponding reference; the preferred
        name, the preferred status, the preferred binary letter (chosen as the most common
        in the group); year of discovery, method of discovery, and final list of aliases.
        The function then concatenates all of these entries together into a final catalog.

        :param group : A pandas DataFrame containing the duplicate occurrences.
        :param mainid: The main identifier of the group
        :param binary: The binary identifier of the group
        :param letter: The letter identifier of the group
        :return: A pandas Series corresponding to the merged single entry.
        """
        # these are always going to be unique by definition
        f = open("Logs/merge_into_single_entry.txt", "a")
        entry = pd.DataFrame([mainid], columns=["main_id"])
        entry["binary"] = binary
        entry["letter"] = letter

        entry["host"] = list(set(group["host"].mode()))[0]  # gets most common name

        # save catalog name
        entry["nasa_name"] = ""
        entry["toi_name"] = ""
        entry["epic_name"] = ""
        entry["eu_name"] = ""
        entry["oec_name"] = ""

        for catalog in group["catalog"]:
            if catalog == "nasa":
                entry["nasa_name"] = group.loc[
                    group.catalog == catalog, "catalog_name"
                ].tolist()[0]
            elif catalog == "toi":
                entry["toi_name"] = group.loc[
                    group.catalog == catalog, "catalog_name"
                ].tolist()[0]
            elif catalog == "epic":
                entry["epic_name"] = group.loc[
                    group.catalog == catalog, "catalog_name"
                ].tolist()[0]
            elif catalog == "eu":
                entry["eu_name"] = group.loc[
                    group.catalog == catalog, "catalog_name"
                ].tolist()[0]
            elif catalog == "oec":
                entry["oec_name"] = group.loc[
                    group.catalog == catalog, "catalog_name"
                ].tolist()[0]

        # SELECT BEST MEASUREMENT
        params = [
            ["i_url", "i", "i_min", "i_max", "IREL"],
            ["mass_url", "mass", "mass_min", "mass_max", "MASSREL"],
            ["msini_url", "msini", "msini_min", "msini_max", "MSINIREL"],
            ["r_url", "r", "r_min", "r_max", "RADREL"],
            ["a_url", "a", "a_min", "a_max", "AREL"],
            ["p_url", "p", "p_min", "p_max", "PERREL"],
            ["e_url", "e", "e_min", "e_max", "EREL"],
        ]
        for p in params:
            result = pd.DataFrame(columns=p)
            result.loc[0, p] = np.nan
            result.loc[0, p[0]] = ""

            subgroup = group[p[:-1]]
            subgroup[p[1:-1]] = subgroup[p[1:-1]].fillna(np.nan).replace("", np.nan)
            subgroup = subgroup.dropna(subset=[p[1]])
            subgroup = subgroup.dropna(subset=[p[3], p[2]])

            if len(subgroup) > 0:
                subgroup[p[1]] = subgroup[p[1]].astype("float")
                subgroup["maxrel"] = subgroup[p[3]].astype("float") / subgroup[
                    p[1]
                ].astype("float")
                subgroup["minrel"] = subgroup[p[2]].astype("float") / subgroup[
                    p[1]
                ].astype("float")
                subgroup = subgroup.replace(np.inf, np.nan)
                subgroup["maxrel"] = subgroup["maxrel"].fillna(subgroup[p[2]])
                subgroup["minrel"] = subgroup["minrel"].fillna(subgroup[p[2]])
                subgroup[p[-1]] = subgroup[["maxrel", "minrel"]].max(axis=1)

                result = subgroup.loc[subgroup[p[-1]] == subgroup[p[-1]].min(), p]
                result = result.sort_values(by=p[0]).head(1)

                result = result.reset_index().drop(columns=["index"])

            result = result[p]

            entry = pd.concat([entry, result], axis=1)

        # status

        entry["checked_status_string"] = ",".join(
            sorted(group.checked_catalog_status)
        ).rstrip(",")

        entry["original_status_string"] = ",".join(
            sorted(group.original_catalog_status)
        ).rstrip(",")

        entry["confirmed"] = ",".join(set(group.checked_catalog_status)).count(
            "CONFIRMED"
        )

        if len(set(group.status)) == 1:
            entry["status"] = group.status.unique()[0]
        else:
            entry["status"] = "CONTROVERSIAL"
        # YEAR OF DISCOVERY
        if len(sorted(group.discovery_year.dropna().astype("int").unique())) == 1:
            entry["discovery_year"] = sorted(
                group.discovery_year.dropna().astype("int").unique()
            )[0]
        elif len(sorted(group.discovery_year.dropna().astype("int").unique())) > 1:
            entry["discovery_year"] = sorted(
                group.discovery_year.dropna().astype("int").unique()
            )[0]
        else:
            entry["discovery_year"] = ""
        # discovery method

        if len(list(set(group.discovery_method.unique()))) > 1 and "toi" in list(
            set(group.catalog.unique())
        ):
            # if the method is not transit but transit shows because TOI was
            # forced to have transit as method, remove "transit"
            discovery_method = ",".join(
                list(
                    (
                        sorted(
                            group.loc[
                                group.catalog != "toi", "discovery_method"
                            ].unique()
                        )
                    )
                )
            ).rstrip(",")
        else:
            discovery_method = ",".join(
                list((sorted(group.discovery_method.replace(np.nan, "").unique())))
            ).rstrip(",")
        # fix for discovery methods that already have a , in it
        entry["discovery_method"] = ",".join(
            [disc.strip() for disc in sorted(set(discovery_method.split(",")))]
        )

        entry["catalog"] = ",".join(list((sorted(group.catalog.unique())))).rstrip(",")

        # final Alias
        final_alias = ""
        for al in group.final_alias:
            final_alias = final_alias + "," + str(al)
        entry["final_alias"] = ",".join(
            [x for x in sorted(set(final_alias.split(","))) if x not in ["A", "B", ""]]
        )

        entry["potential_binary_mismatch"] = ",".join(
            map(str, group.potential_binary_mismatch.unique())
        ).rstrip(",")

        entry["coordinate_mismatch"] = ",".join(
            map(str, group.coordinate_mismatch.unique())
        ).rstrip(",")

        if "RA" in set(group.coordinate_mismatch.unique()) and "DEC" in set(
            group.coordinate_mismatch.unique()
        ):
            entry["coordinate_mismatch_flag"] = 2
        elif "RA" in set(group.coordinate_mismatch.unique()) or "DEC" in set(
            group.coordinate_mismatch.unique()
        ):
            entry["coordinate_mismatch_flag"] = 1
        else:
            entry["coordinate_mismatch_flag"] = 0

        entry["angular_separation"] = ",".join(
            map(str, sorted(group.angular_separation.unique()))
        )

        entry["angular_separation_flag"] = len(list(set(group.angsep.unique()))) - 1

        if len(group.main_id_provenance.unique()) > 1:
            with pd.option_context("display.max_columns", 2000):
                f.write(
                    "\nWARNING, main_id_provenance not unique for  "
                    + mainid
                    + " "
                    + binary
                    + " "
                    + letter
                    + "\n"
                    + str(
                        group[
                            [
                                "main_id_provenance",
                                "main_id_ra",
                                "main_id_dec",
                                "angular_separation",
                                "p",
                                "a",
                            ]
                        ]
                    )
                    + "\n"
                )
                for item in [
                    "SIMBAD",
                    "SIMBADCOORD",
                    "TIC",
                    "TICCOORD",
                    "toi",
                    "nasa",
                    "epic",
                    "eu",
                    "oec",
                ]:
                    if item in group.main_id_provenance.unique():
                        entry["main_id_provenance"] = item
                        entry["ra_official"] = list(
                            set(group[group.main_id_provenance == item].main_id_ra)
                        )[0]
                        entry["dec_official"] = list(
                            set(group[group.main_id_provenance == item].main_id_dec)
                        )[0]
                        break
        else:
            entry["main_id_provenance"] = group.main_id_provenance.unique()[0]
            entry["ra_official"] = list(set(group.main_id_ra))[0]
            entry["dec_official"] = list(set(group.main_id_dec))[0]

        # Catalog

        if len(group) > len(group.catalog.unique()):
            with pd.option_context("display.max_columns", 2000):
                f.write(
                    "\n*** DUPLICATE ENTRY "
                    + mainid
                    + " "
                    + binary
                    + " "
                    + letter
                    + " ***\n"
                    + str(
                        group[
                            [
                                "catalog",
                                "catalog_name",
                                "status",
                                "angular_separation",
                                "p",
                                "a",
                            ]
                        ]
                    )
                    + "\n"
                )

            entry["duplicate_catalog_flag"] = 1

            entry["duplicate_names"] = ",".join(
                group.catalog + ": " + group.catalog_name
            ).rstrip(",")

        else:
            entry["duplicate_catalog_flag"] = 0
            entry["duplicate_names"] = ""
        f.close()
        return entry

    def group_by_letter_check_period(self, verbose: bool) -> None:
        """
        The group_by_letter_check_period function groups the full catalog per main_id, binary, and letter. Then it
        checks the periods. If the periods exist and are the same, it merges into a single entry. If the periods exist
        but are different it leaves the different entries unmerged. If the period does not exist, it checks for the
        semi-major axis: if the semi-major axes exist and are the same, it merges into a single entry. If the semi-major
        axis does not match, it leaves the entries unmerged. If also the semi-major axis does not exist, it merges all
        the entries together.

        :param verbose: boolean to allow printing of percentage
        """
        final_catalog = pd.DataFrame()
        f1 = open("Logs/group_by_letter_check_period.txt", "a")
        # self.data['working_period_group'] = round_parameter_bin(self.data['p'])
        # self.data['working_sma_group'] = round_parameter_bin(self.data['a'])
        grouped_df = self.data.groupby(
            ["main_id", "binary", "letter"], sort=True, as_index=False
        )
        counter = 0
        for (mainid, binary, letter), group in grouped_df:
            # cases:
            group = Utils.calculate_working_p_sma(group, tolerance=0.1)
            period_list = list(
                set(group.working_p.replace(-1, np.nan).dropna().unique())
            )
            if len(period_list) == 1:
                # (test) CASE 1: period in agreement (drop nan), regular merging
                entry = Emc.merge_into_single_entry(group, mainid, binary, str(letter))

                final_catalog = pd.concat(
                    [final_catalog, entry], sort=False
                ).reset_index(drop=True)
            elif len(period_list) > 1:
                # (test) CASE 4: period in disagreement (but not nan), include both
                f1.write(
                    "DISAGREEMENT \n"
                    + str(
                        group[
                            [
                                "main_id",
                                "binary",
                                "letter",
                                "catalog",
                                "catalog_name",
                                "p",
                            ]
                        ]
                    )
                    + "\n\n"
                )
                for pgroup in period_list:
                    subgroup = group[group.working_p == pgroup]
                    entry = Emc.merge_into_single_entry(
                        subgroup, mainid, binary, str(letter)
                    )
                    final_catalog = pd.concat(
                        [final_catalog, entry], sort=False
                    ).reset_index(drop=True)
            else:
                # only nan, check sma
                sma_list = list(
                    set(group.working_a.replace(-1, np.nan).dropna().unique())
                )

                if len(sma_list) == 1:
                    # (test) CASE 2: sma in agreement (drop nan), regular merging
                    entry = Emc.merge_into_single_entry(
                        group, mainid, binary, str(letter)
                    )

                    final_catalog = pd.concat(
                        [final_catalog, entry], sort=False
                    ).reset_index(drop=True)
                elif len(sma_list) > 1:
                    # (test) CASE 5: sma in disagreement (but not nan), include both
                    f1.write(
                        "DISAGREEMENT \n"
                        + str(
                            group[
                                [
                                    "main_id",
                                    "binary",
                                    "letter",
                                    "catalog",
                                    "catalog_name",
                                    "a",
                                ]
                            ]
                        )
                        + "\n\n"
                    )
                    for agroup in sma_list:
                        subgroup = group[group.working_a == agroup]
                        entry = Emc.merge_into_single_entry(
                            subgroup, mainid, binary, letter
                        )
                        final_catalog = pd.concat(
                            [final_catalog, entry], sort=False
                        ).reset_index(drop=True)
                else:
                    # (test) CASE 3: no period nor sma, merge together
                    if len(group) > 1:
                        f1.write(
                            "FALLBACK, MERGE \n"
                            + str(
                                group[
                                    [
                                        "main_id",
                                        "binary",
                                        "letter",
                                        "catalog",
                                        "catalog_name",
                                    ]
                                ]
                            )
                            + "\n\n"
                        )
                    entry = Emc.merge_into_single_entry(
                        group, mainid, binary, str(letter)
                    )

                    final_catalog = pd.concat(
                        [final_catalog, entry], sort=False
                    ).reset_index(drop=True)

            if verbose:
                print(
                    "Done "
                    + str(round(counter / len(grouped_df), 2) * 100)
                    + "% of the groups.",
                    end="\r",
                )
            counter = counter + 1
        f1.close()

        self.data = final_catalog
        logging.info("Catalog merged into single entries.")

    def potential_duplicates_after_merging(self) -> None:
        """
        The potential_duplicates_after_merging function finds the rows with the same main_id, binary, letter. If more
        than one occurs, then it assigns the "emc_duplicate_entry_flag" to 1.
        """
        f1 = open("Logs/potential_duplicates_after_merging.txt", "a")
        self.data["emc_duplicate_entry_flag"] = 0
        grouped_df = self.data.groupby(
            ["main_id", "binary", "letter"], sort=True, as_index=False
        )
        counter = 0
        for (mainid, binary, letter), group in grouped_df:
            if len(group) > 1:
                # they have controversial period, keep all because different
                # periods
                f1.write("MAINID " + mainid + " " + binary + " " + letter + "\n")
                self.data.loc[group.index, "emc_duplicate_entry_flag"] = 1
        f1.close()
        logging.info("Checked duplicates after merging.")

    def select_best_mass(self) -> None:
        """
        The select_best_mass function is used to select the best mass for each planet.
        The function first checks if the MASSREL value is greater than MSINIREL, and if so,
        it assigns the msini values to bestmass, bestmass_min and bestmass_max. If not,
        it assigns mass values to these columns instead.
        """

        for i in self.data[
            self.data.MASSREL.fillna(1e9) >= self.data.MSINIREL.fillna(1e9)
        ].index:
            self.data.at[i, "bestmass"] = self.data.at[i, "msini"]
            self.data.at[i, "bestmass_min"] = self.data.at[i, "msini_min"]
            self.data.at[i, "bestmass_max"] = self.data.at[i, "msini_max"]
            self.data.at[i, "bestmass_url"] = self.data.at[i, "msini_url"]
            self.data.at[i, "bestmass_provenance"] = "Msini"

        for i in self.data[
            self.data.MASSREL.fillna(1e9) < self.data.MSINIREL.fillna(1e9)
        ].index:
            self.data.at[i, "bestmass"] = self.data.at[i, "mass"]
            self.data.at[i, "bestmass_min"] = self.data.at[i, "mass_min"]
            self.data.at[i, "bestmass_max"] = self.data.at[i, "mass_max"]
            self.data.at[i, "bestmass_url"] = self.data.at[i, "mass_url"]
            self.data.at[i, "bestmass_provenance"] = "Mass"

        for i in self.data[
            (self.data.mass.fillna(1e9) == 1e9) & (self.data.msini.fillna(1e9) == 1e9)
        ].index:
            self.data.at[i, "bestmass"] = np.nan
            self.data.at[i, "bestmass_min"] = np.nan
            self.data.at[i, "bestmass_max"] = np.nan
            self.data.at[i, "bestmass_url"] = np.nan
            self.data.at[i, "bestmass_provenance"] = ""

        logging.info("Bestmass calculated.")

    def set_exo_mercat_name(self) -> None:
        """
        The set_exo_mercat name creates the columns exo_mercat_name by joining the main_id, the binary (if any), and
        the letter.
        """
        self.data["binary"] = self.data["binary"].replace("nan", "")
        self.data["exo_mercat_name"] = self.data.apply(
            lambda row: (
                row["main_id"]
                if str(re.search("[\\s\\d][ABCNS]$", row["main_id"], re.M)) == "None"
                else row["main_id"][:-1].rstrip()
            )
            + (" " + str(row["binary"]) if not row["binary"] == "" else "")
            + " "
            + row["letter"],
            axis=1,
        )
        self.data = self.data.sort_values(by="exo_mercat_name").reset_index()
        logging.info("Exo-MerCat name assigned.")

    def keep_columns(self) -> None:
        """
        The keep_columns function is used to keep only the columns that are needed for the analysis.
        The function takes in a dataframe and returns a new dataframe with only the columns listed above.
        """
        keep = [
            "exo_mercat_name",
            "nasa_name",
            "toi_name",
            "epic_name",
            "eu_name",
            "oec_name",
            "host",
            "letter",
            "main_id",
            "binary",
            "ra_official",
            "dec_official",
            "mass",
            "mass_max",
            "mass_min",
            "mass_url",
            "msini",
            "msini_max",
            "msini_min",
            "msini_url",
            "bestmass",
            "bestmass_max",
            "bestmass_min",
            "bestmass_url",
            "bestmass_provenance",
            "p",
            "p_max",
            "p_min",
            "p_url",
            "r",
            "r_max",
            "r_min",
            "r_url",
            "a",
            "a_max",
            "a_min",
            "a_url",
            "e",
            "e_max",
            "e_min",
            "e_url",
            "i",
            "i_max",
            "i_min",
            "i_url",
            "discovery_method",
            "status",
            "checked_status_string",
            "original_status_string",
            "confirmed",
            "discovery_year",
            "final_alias",
            "catalog",
            "angular_separation",
            "angular_separation_flag",
            "main_id_provenance",
            "coordinate_mismatch",
            "coordinate_mismatch_flag",
            "duplicate_catalog_flag",
            "duplicate_names",
            "emc_duplicate_entry_flag",
            "row_update",
        ]
        try:
            self.data = self.data[keep]
        except KeyError:
            raise KeyError("Not all columns exist")
        logging.info("Selected columns to keep.")

    def remove_known_brown_dwarfs(self, print_flag: bool) -> None:
        """
        The remove_known_brown_dwarfs function removes all known brown dwarfs from the dataframe.
        It does this by checking if the mass of a planet is less than 20 Mjup, and if it isn't,
        then it will be removed from the dataframe. If print_flag is set to True, then a csv file will be created
        with all of these planets in them.

        :param print_flag: Specify whether the function should print out a list of brown dwarfs
        """
        if print_flag:
            self.data[
                (
                    self.data.mass.fillna(self.data.msini.fillna(0))
                    .replace("", 0)
                    .astype(float)
                    > 20.0
                )
                #   | (self.data.letter == "BD")
            ].to_csv("Exo-MerCat/" + self.name + "_brown_dwarfs.csv")

        self.data = self.data[
            (
                self.data.mass.fillna(self.data.msini.fillna(0))
                .replace("", 0)
                .astype(float)
                <= 20.0
            )
            #    & (self.data.letter != "BD")
        ]
        self.data[(self.data.letter == "BD")].to_csv(
            "Exo-MerCat/" + self.name + "_possible_brown_dwarfs.csv"
        )

    def fill_row_update(self, local_date: str):
        if local_date != "":
            update_date = local_date
        else:
            update_date = date.today().strftime("%m-%d-%Y")

        # find if there are older versions present
        if len(glob.glob("Exo-MerCat/exo-mercat*-*.csv")) > 0:
            li = list(glob.glob("Exo-MerCat/exo-mercat_full*-*.csv"))
            li = [re.search(r"\d\d-\d\d-\d\d\d\d", l)[0] for l in li]
            li = [datetime.strptime(l, "%m-%d-%Y") for l in li]

            # get the most recent compared to the current date. Get only the ones earlier than the date
            li = [l for l in li if l < datetime.strptime(update_date, "%m-%d-%Y")]
            compar_date = max(li).strftime("%m-%d-%Y")
            right_merge = pd.read_csv(
                "Exo-MerCat/exo-mercat_full" + compar_date + ".csv"
            )
            right_merge["old_index"] = right_merge.index.copy().astype(int)
            # in case the previous versions did not have this feature
            if "row_update" not in right_merge.columns:
                right_merge["row_update"] = compar_date

            self.data["new_index"] = self.data.index.copy().astype(int)
            # print into csv to make sure that it reads the same way as right_merge
            self.data.to_csv("temp.csv")
            left_merge = pd.read_csv("temp.csv")
            # cleanup
            os.remove("temp.csv")

            all = left_merge.fillna("").merge(
                right_merge.fillna(""),
                on=[
                    x
                    for x in right_merge.columns
                    if x not in ["index", "new_index", "old_index", "row_update"]
                ],
                how="outer",
                indicator=True,
            )
            all = all.dropna(subset=("new_index"))
            # get the previous date for all the entries that did not change
            self.data.loc[all.new_index, "row_update"] = all.loc[
                all.new_index, "row_update"
            ]
            # fill the missing ones with the update date
            self.data["row_update"] = self.data["row_update"].fillna(update_date)
            # clean up
            self.data = self.data.drop("new_index", axis=1)
        else:
            self.data["row_update"] = update_date

    def save_catalog(self, local_date: str == "", postfix: str == ""):
        if local_date == "":
            self.print_catalog(
                "Exo-MerCat/exo-mercat"
                + postfix
                + date.today().strftime("%m-%d-%Y")
                + ".csv"
            )
        else:
            self.print_catalog("Exo-MerCat/exo-mercat" + postfix + local_date + ".csv")
        self.print_catalog("Exo-MerCat/exo-mercat" + postfix + ".csv")
