import numpy as np
import pandas as pd
from astroquery.simbad import Simbad
from statistics import mode
from exo_mercat.catalogs import Catalog
import re
import pyvo
from astropy import units as u
from astropy.coordinates import SkyCoord
import logging
from exo_mercat.utility_functions import UtilityFunctions as Utils


class Emc(Catalog):
    def __init__(self) -> None:
        """
        The __init__ function is called when the class is instantiated. It sets up the instance of the class, and
        defines any variables that will be used by other functions in the class. In this case, we are setting up
        a dataframe to hold our data.
        """
        super().__init__()
        self.name = "exo_mercat"
        self.data = pd.DataFrame()

    def convert_coordinates(self) -> None:
        """The coordinates function takes the RA and Dec columns of a dataframe,
        and converts them to decimal degrees. It replaces any
        missing values with NaN. Finally, it uses SkyCoord to convert from hour angles and
        degrees into decimal degrees.
        NOT NECESSARY since EMC already has coordinates in degrees"""
        pass

    def alias_as_host(self) -> None:
        """
        The alias_as_host function takes the alias column and checks if any of the aliases are labeled as hosts in
        some other entry. If they are, it will change their host to be that of the original host. It will
        then add all aliases of both hosts into one list for each row. It is okay if it happens multiple times,
        as long as it uniforms the host name and adds up all the aliases, SIMBAD will find them coherently.
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

            self.data.loc[self.data.host == host, "alias"] = ",".join(
                sorted(set(final_alias_total))
            )

        f.close()

        logging.info(
            "Aliases labeled as hosts in some other entry checked. It happens "
            + str(counter)
            + " times."
        )

    def simbad_list_host_search(self, column: str) -> None:
        """
        The simbad_list_host_search function takes a column name as an argument and searches for the host star
        in that column in SIMBAD. It then fills in the main_id, IDS, RA, and DEC columns with information from
        SIMBAD if it finds a match.
        :param column: The name of the column that contains the host star to search for (host or hostbinary)
        """
        list_of_hosts = list(
            self.data[self.data.main_id == ""][column].drop_duplicates()
        )
        list_of_hosts = [x for x in list_of_hosts]

        Simbad.add_votable_fields("typed_id", "ids", "ra", "dec")
        Simbad.TIMEOUT = 60000000
        result_table = Simbad.query_objects(list_of_hosts)
        result_table = result_table.to_pandas()
        result_table_successful = result_table[result_table.MAIN_ID != ""]
        logging.info(
            "List of unique star names "
            + str(len(list_of_hosts))
            + " of which successful SIMBAD queries "
            + str(len(result_table_successful))
        )
        for i in self.data[self.data.main_id == ""].index:
            check = self.data.at[i, column]
            if (
                len(result_table_successful[result_table_successful.TYPED_ID == check])
                == 1
            ):
                self.data.at[i, "main_id"] = result_table_successful[
                    result_table_successful.TYPED_ID == check
                ]["MAIN_ID"].to_numpy()[0]
                self.data.at[i, "IDS"] = result_table_successful[
                    result_table_successful.TYPED_ID == check
                ]["IDS"].to_numpy()[0]
                self.data.at[i, "RA"] = result_table_successful[
                    result_table_successful.TYPED_ID == check
                ]["RA"].to_numpy()[0]
                self.data.at[i, "DEC"] = result_table_successful[
                    result_table_successful.TYPED_ID == check
                ]["DEC"].to_numpy()[0]

        self.data.main_id = self.data.main_id.fillna("")
        self.data.IDS = self.data.IDS.fillna("")
        self.data.RA = self.data.RA.fillna("")
        self.data.DEC = self.data.DEC.fillna("")

    def simbad_list_alias_search(self, column: str) -> None:
        """
        The simbad_list_alias_search function takes a column name as an argument and searches for the main ID of
        each object in that column. The function first splits the string into a list of aliases, then iterates
        through each alias to search SIMBAD for its main ID. If it finds one, it will update the dataframe with
        that information.
        :param column: The name of the column that contains the host star to search for (alias or aliasbinary)
        """

        Simbad.add_votable_fields("typed_id", "ids", "ra", "dec")
        Simbad.TIMEOUT = 60000000
        for i in self.data[self.data.main_id == ""].index:
            list_of_aliases = str(self.data.at[i, column]).split(",")
            for a in list_of_aliases:
                result_table = Simbad.query_object(a)
                result_table = result_table.to_pandas()
                result_table_successful = result_table[result_table.MAIN_ID != ""]
                if (
                    len(result_table_successful[result_table_successful.TYPED_ID == a])
                    > 0
                ):
                    self.data.at[i, "main_id"] = result_table_successful[
                        result_table_successful.TYPED_ID == a
                    ]["MAIN_ID"].to_numpy()[0]
                    self.data.at[i, "IDS"] = result_table_successful[
                        result_table_successful.TYPED_ID == a
                    ]["IDS"].to_numpy()[0]
                    self.data.at[i, "RA"] = result_table_successful[
                        result_table_successful.TYPED_ID == a
                    ]["RA"].to_numpy()[0]
                    self.data.at[i, "DEC"] = result_table_successful[
                        result_table_successful.TYPED_ID == a
                    ]["DEC"].to_numpy()[0]

                    break

    def get_host_info_from_simbad(self) -> None:
        """
        The get_host_info_from_simbad function takes the dataframe and extracts all
        unique host star names. It then queries Simbad for each of these names, and
        returns a table with the main ID, alias IDs, RA and DEC.The function merges
        this table with the original dataframe on host name (left join).If there are
        still rows missing main_id values in the merged table, it will query Simbad
        again using all aliases from those rows.
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
                single_aliases_binary = [s.rstrip() for s in single_aliases_binary]
                self.data.at[i, "aliasbinary"] = ",".join(single_aliases_binary)
                self.data["aliasbinary"] = self.data["aliasbinary"].fillna("")

        self.data["main_id"] = ""
        self.data["IDS"] = ""
        self.data["RA"] = ""
        self.data["DEC"] = ""

        logging.info("HOST+BINARY Simbad Check")
        self.simbad_list_host_search("hostbinary")
        logging.info(
            "Rows still missing main_id after host search "
            + str(len(self.data[self.data.main_id == ""]))
        )

        logging.info("ALIAS+BINARY Simbad Check")
        self.simbad_list_alias_search("aliasbinary")
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

        self.data["list_id"] = self.data["IDS"].apply(
            lambda x: str(x).replace("|", ",")
        )
        self.data["ra_simbad"] = self.data.apply(
            lambda row: SkyCoord(
                str(row["RA"]) + " " + str(row["DEC"]), unit=(u.hourangle, u.deg)
            ).ra.degree
            if not str(row.RA) == ""
            else np.nan,
            axis=1,
        )
        self.data["dec_simbad"] = self.data.apply(
            lambda row: SkyCoord(
                str(row["RA"]) + " " + str(row["DEC"]), unit=(u.hourangle, u.deg)
            ).dec.degree
            if not str(row.RA) == ""
            else np.nan,
            axis=1,
        )

    def set_common_alias(self) -> None:
        """
        The set_common_alias function takes the alias and list_id columns from
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
            self.data.loc[self.data.main_id == host, "final_alias"] = ",".join(
                [x for x in set(final_alias.split(",")) if x]
            )

    def set_common_host(self) -> None:
        """
        The set_common_host function is used to set the host name for all planets
        in a given list_id group to be the same. This function is called by the
        check_alias function, which checks if there are any planets with different
        host names but  the same SIMBAD alias. If so, this function will set all
        of those planet's hosts to be equal.
        """
        f = open("Logs/check_alias.txt", "a")
        count = 0
        for ids, group in self.data.groupby(by="list_id"):
            if ids != "" and len(set(group.host)) > 1:
                self.data.loc[self.data.list_id == ids, "host"] = list(group.host)[0]
                f.write(
                    "HOST "
                    + str(list(group.host))
                    + str(list(group.catalog))
                    + str(list(group.status))
                    + str(list(group.letter))
                    + " ID "
                    + str(list(group.main_id))
                )
                count = count + 1
        logging.info(
            "Planets that had a different host name but same SIMBAD alias: "
            + str(count)
        )
        f.close()

    def check_coordinates(self) -> None:
        """
        The check_coordinates function checks for mismatches in the RA and DEC
        coordinates of a given host (for the targets that cannot rely on SIMBAD
        MAIN_ID because the query was unsuccessful). It does this by grouping all
        entries with the same host name, then checking if any of those entries have
        an RA or DEC that is more than 0.01 degrees away from the mode value for that
        group. If so, it prints out information about those mismatched values
        to a log file called coord_errors.txt.
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
                f.write(
                    "*** MISMATCH ON RA *** "
                    + str(
                        (group[["name", "host", "binary", "letter", "catalog", "ra"]])
                    )
                    + "\n"
                )
                mismatch_string = "RA"

            if (abs(group["dec"] - dec) > 0.01).any():
                countdec = countdec + 1
                f.write(
                    "*** MISMATCH ON DEC *** "
                    + str(
                        (group[["name", "host", "binary", "letter", "catalog", "dec"]])
                    )
                    + "\n"
                )
                mismatch_string = mismatch_string + "DEC"

            self.data.loc[group.index, "coordinate_mismatch"] = mismatch_string
        f.close()
        logging.info("Found " + str(countra) + " mismatched RA.")
        logging.info("Found " + str(countdec) + " mismatched DEC.")

        logging.info(self.data.coordinate_mismatch.value_counts())

    def get_coordinates_from_simbad(self) -> None:
        """
        This function takes the dataframe and checks if there are any matches in Simbad for the coordinates of each
        object. It does this by querying Simbad with a circle around each coordinate, starting at 0.01 degrees and
        increasing to 0.5 degrees until it finds a match or gives up.
        """
        self.data["angular_separation"] = 0
        service = pyvo.dal.TAPService("http://simbad.u-strasbg.fr:80/simbad/sim-tap")
        Simbad.add_votable_fields("typed_id", "ids", "ra", "dec")
        for tolerance in [0.0005, 0.001, 0.005, 0.01, 0.05, 0.1, 0.5]:
            for host, group in self.data[self.data.main_id == ""].groupby("host"):
                ind = group.index
                group = group.reset_index()
                query = (
                    "SELECT db.main_id, db.dec as dec_2,db.ra as ra_2  FROM "
                    + "basic AS db WHERE 1=CONTAINS(POINT('ICRS',db.ra, db.dec),   CIRCLE('ICRS',"
                    + str(group.loc[0, "ra"])
                    + ","
                    + str(group.loc[0, "dec"])
                    + ","
                    + str(tolerance)
                    + "))"
                )

                response = service.run_sync(query, timeout=None)
                table = response.to_table().to_pandas()
                if len(table) > 0:
                    for row in table.iterrows():
                        r = row[1]
                        c1 = SkyCoord(
                            group.loc[0, "ra"],
                            group.loc[0, "dec"],
                            frame="icrs",
                            unit=(u.degree, u.degree),
                        )
                        c2 = SkyCoord(
                            r.ra_2, r.dec_2, frame="icrs", unit=(u.degree, u.degree)
                        )
                        angsep = c2.separation(c1).degree
                        table.at[row[0], "angsep"] = angsep

                    selected = table[table.angsep == min(table.angsep)]
                    selected = selected.reset_index()
                    self.data.loc[ind, "main_id"] = selected.loc[0, "main_id"]
                    self.data.loc[ind, "ra_simbad"] = selected.loc[0, "ra_2"]
                    self.data.loc[ind, "dec_simbad"] = selected.loc[0, "dec_2"]
                    self.data.loc[ind, "angular_separation"] = selected.loc[0, "angsep"]
                    result_table = Simbad.query_object(selected.loc[0, "main_id"])
                    result_table = result_table.to_pandas()
                    self.data.loc[ind, "list_id"] = result_table.loc[0, "IDS"].replace(
                        "|", ","
                    )

            logging.info(
                "After coordinate check at tolerance "
                + str(tolerance)
                + " residuals: "
                + str(self.data[self.data.main_id == ""].shape[0])
                + ". Maximum angular separation: "
                + str(max(self.data.angular_separation))
            )
            if len(self.data[self.data.main_id == ""]) == 0:
                break

    def check_same_host_different_id(self) -> None:
        """
        The check_same_host_different_id function checks to see if there
        are any instances where the same host has multiple SIMBAD main IDs.
        This should _never_ happen unless the SIMBAD search is failing.

        """
        f = open("Logs/same_host_different_id.txt", "a")
        for host, group in self.data.groupby("hostbinary"):
            if len(group.main_id.drop_duplicates()) > 1:
                f.write(
                    host
                    + " main_id: "
                    + str(list(group.main_id))
                    + " binary: "
                    + str(list(group.binary))
                    + " Catalog: "
                    + str(list(group.catalog))
                    + "\n"
                )

        for host, group in self.data.groupby("host"):
            if len(group.main_id.drop_duplicates()) > 1:
                f.write(
                    host
                    + " main_id: "
                    + str(list(group.main_id))
                    + " binary: "
                    + str(list(group.binary))
                    + " Catalog: "
                    + str(list(group.catalog))
                    + "\n"
                )
        logging.info("Checked if host is found under different main_ids.")
        f.close()

    def polish_main_id(self) -> None:
        counter = 0
        f = open("Logs/main_id_correction.txt", "a")
        for identifier in self.data["main_id"]:
            if not str(re.search("[\s\d][b-i]$", identifier, re.M)) == "None":
                counter += 1
                self.data.loc[self.data.main_id == identifier, "main_id"] = identifier[
                    :-1
                ].strip()
                f.write(
                    "MAINID corrected " + identifier + " to " + identifier[:-1] + "\n"
                )
        f.close()
        # counter=0
        # for i in self.data.index:
        #     if not str(re.search("[\s\d][a-i]$", self.data.at[i, "main_id"], re.M)) == "None":
        #         counter=counter+1
        #         print('')
        #         self.data.at[i,'main_id']=self.data.at[i,'main_id'][:-1]
        logging.info(
            "Removed planet letter from main_id. It happens " + str(counter) + " times."
        )

    def check_binary_mismatch(self, keyword: str) -> None:
        """
        The check_binary_mismatch function checks for binary mismatches in the data (planets that orbit a binary but are
        controversial in the various catalogs and/or SIMBAD). It also checks if the SIMBAD main_id labels the target as
        a binary. Ideally, all of these issues should be fixed by a human for the code to work properly.
        """
        self.data["binary"] = self.data["binary"].fillna("")
        self.data["potential_binary_mismatch"] = 0
        f = open("Logs/binary_mismatch.txt", "a")
        f.write("****" + keyword + "****\n")
        f.write(
            "\n****"
            + keyword
            + "+letter THAT COULD BE UNIFORMED (only if S-type or null)****\n"
        )
        for (key, letter), group in self.data.groupby(by=[keyword, "letter"]):
            if len(set(group.binary)) > 1:
                # Uniform only S-type
                if len(group[group.binary == "S-type"]) > 0:
                    warning = ""
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

                    f.write(
                        key
                        + " NAME:"
                        + str(list(self.data.loc[group.index, "name"]))
                        + " HOST:"
                        + str(list(self.data.loc[group.index, "host"]))
                        + " LETTER:"
                        + letter
                        + " BINARY:"
                        + str(list(self.data.loc[group.index, "binary"]))
                        + " CATALOG:"
                        + str(str(list(self.data.loc[group.index, "catalog"])))
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

                    f.write(
                        key
                        + " NAME:"
                        + str(list(group.name))
                        + " HOST:"
                        + str(list(group["host"]))
                        + " LETTER:"
                        + letter
                        + " BINARY:"
                        + str(list(group["binary"]))
                        + " CATALOG:"
                        + str(str(list(group["catalog"])))
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
            + "+letter THAT ARE INCONSISTENTLY LABELED (Potential Mismatch 2). They could be complex systems. If not, they should be treated manually in replacements.ini ****\n\n"
        )
        for (key, letter), group in self.data.groupby(by=[keyword, "letter"]):
            if len(set(group.binary)) > 1:
                f.write(
                    key
                    + " NAME:"
                    + str(list(group.name))
                    + " HOST: "
                    + str(list(group.host))
                    + " LETTER:"
                    + letter
                    + " BINARY:"
                    + str(list(group.binary))
                    + " CATALOG:"
                    + str(list(group.catalog))
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

    def fix_letter_by_period(self) -> None:
        f1 = open("Logs/fixed_letters.txt", "a")
        self.data["working_period_group"] = Utils.round_parameter_bin(self.data["p"])
        self.data["working_sma_group"] = Utils.round_parameter_bin(self.data["a"])

        grouped_df = self.data.groupby(["main_id", "binary"], sort=True, as_index=False)
        counter = 0
        for (
            mainid,
            binary,
        ), group in grouped_df:
            if len(group) > 1:  # there are multiple planets in the system
                for pgroup in list(set(group.working_period_group)):
                    subgroup = group[group.working_period_group == pgroup]

                    # if period is zero, group by semimajor axis, if unsuccessful group by letter
                    if pgroup != -1:
                        # try to fix the letter if it is different
                        if len(list(set(subgroup.letter))) > 1:
                            f1.write(
                                "CONTROVERSIAL LETTER ENTRY "
                                + mainid
                                + " "
                                + binary
                                + " PERIOD "
                                + str(list(subgroup.p))
                                + " LETTER "
                                + str(list(subgroup.letter))
                                + "\n"
                            )

                            adjusted_letter = [
                                l
                                for l in list(set(subgroup.letter.dropna().unique()))
                                if ".0" not in l
                            ]

                            if len(adjusted_letter) == 1:
                                self.data.loc[
                                    subgroup.index, "letter"
                                ] = adjusted_letter[0]
                                f1.write("-> FIXABLE\n")
                            if "BD" in list(set(subgroup.letter.dropna().unique())):
                                self.data.loc[subgroup.index, "letter"] = "BD"
                                f1.write("-> FORCED BD\n")

        f1.close()

    def merge_into_single_entry(
        self, group: pd.DataFrame, mainid: str, binary: str, letter: str
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
        entry = pd.DataFrame([mainid], columns=["main_id"])
        entry["binary"] = binary
        entry["letter"] = letter

        entry["host"] = list(set(group["host"]))[0]
        entry["angular_separation"] = ",".join(
            map(str, group.angular_separation.unique())
        )

        entry["ra_official"] = list(set(group.ra_simbad))[0]
        entry["dec_official"] = list(set(group.dec_simbad))[0]

        # save catalog name
        entry["nasa_name"] = ""
        entry["eu_name"] = ""
        entry["oec_name"] = ""
        for catalog in group["catalog"]:
            if catalog == "nasa":
                entry["nasa_name"] = group.loc[
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

            #                result.main_id=mainid
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

        entry["status_string"] = ",".join(group.Catalogstatus).rstrip(",")

        entry["confirmed"] = ",".join(set(group.Catalogstatus)).count("CONFIRMED")

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
        entry["discovery_method"] = ",".join(
            list(
                set(
                    group.discovery_method[group.discovery_method != "Default"].unique()
                )
            )
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

        entry["angular_separation_flag"] = (
            len(list(set(group.angular_separation.unique()))) - 1
        )
        # Catalog

        if len(group) > len(group.catalog.unique()):
            f = open("Logs/duplicate_entries.txt", "a")

            f.write(
                "DUPLICATE ENTRY "
                + mainid
                + " "
                + str(list(group.letter))
                + " CATALOGS "
                + str(list(group.catalog))
                + " STATUS "
                + str(list(group.status))
                + "\n"
            )
            f.close()
            entry["duplicate_flag"] = 1
        else:
            entry["duplicate_flag"] = 0

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
        f1 = open("Logs/contrasting_periods.txt", "a")
        # self.data['working_period_group'] = round_parameter_bin(self.data['p'])
        # self.data['working_sma_group'] = round_parameter_bin(self.data['a'])

        grouped_df = self.data.groupby(
            ["main_id", "binary", "letter"], sort=True, as_index=False
        )
        counter = 0
        for (mainid, binary, letter), group in grouped_df:
            # cases:
            period_list = list(
                set(group.working_period_group.replace(-1, np.nan).dropna().unique())
            )
            if len(period_list) == 1:
                # period in agreement (drop nan), regular merging
                entry = self.merge_into_single_entry(group, mainid, binary, letter)

                final_catalog = pd.concat(
                    [final_catalog, entry], sort=False
                ).reset_index(drop=True)
            elif len(period_list) > 1:
                # period in disagreement (but not nan), include both
                f1.write(
                    "DISAGREEMENT "
                    + mainid
                    + " "
                    + binary
                    + ""
                    + letter
                    + " PERIOD "
                    + str(list(group.p))
                    + "\n"
                )
                for pgroup in period_list:
                    subgroup = group[group.working_period_group == pgroup]
                    entry = self.merge_into_single_entry(
                        subgroup, mainid, binary, letter
                    )
                    final_catalog = pd.concat(
                        [final_catalog, entry], sort=False
                    ).reset_index(drop=True)
            else:
                # only nan, check sma
                sma_list = list(
                    set(group.working_sma_group.replace(-1, np.nan).dropna().unique())
                )

                if len(sma_list) == 1:
                    # sma in agreement (drop nan), regular merging
                    entry = self.merge_into_single_entry(group, mainid, binary, letter)

                    final_catalog = pd.concat(
                        [final_catalog, entry], sort=False
                    ).reset_index(drop=True)
                elif len(sma_list) > 1:
                    # sma in disagreement (but not nan), include both
                    f1.write(
                        "DISAGREEMENT "
                        + mainid
                        + " "
                        + binary
                        + ""
                        + letter
                        + " SMA "
                        + str(list(group.a))
                        + "\n"
                    )
                    for agroup in sma_list:
                        subgroup = group[group.working_sma_group == agroup]
                        entry = self.merge_into_single_entry(
                            subgroup, mainid, binary, letter
                        )
                        final_catalog = pd.concat(
                            [final_catalog, entry], sort=False
                        ).reset_index(drop=True)
                else:
                    # period in disagreement (but not nan), include both
                    f1.write(
                        "FALLBACK, MERGE " + mainid + " " + binary + "" + letter + "\n"
                    )
                    # period in agreement (drop nan), regular merging
                    entry = self.merge_into_single_entry(group, mainid, binary, letter)

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
        than one occurs, then it assigns the "emc_duplicate_flag" to 1.
        """
        f1 = open("Logs/potential_duplicates_after_merging.txt", "a")
        self.data["emc_duplicate_flag"] = 0
        grouped_df = self.data.groupby(
            ["main_id", "binary", "letter"], sort=True, as_index=False
        )
        counter = 0
        for (mainid, binary, letter), group in grouped_df:
            if len(group) > 1:
                # they have controversial period, keep all because different periods
                f1.write("MAINID " + mainid + " " + binary + " " + letter + "\n")
                self.data.loc[group.index, "emc_duplicate_flag"] = 1
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
            self.data.MASSREL.fillna(1e9) > self.data.MSINIREL.fillna(1e9)
        ].index:
            self.data.at[i, "bestmass"] = self.data.at[i, "msini"]
            self.data.at[i, "bestmass_min"] = self.data.at[i, "msini_min"]
            self.data.at[i, "bestmass_max"] = self.data.at[i, "msini_max"]
            self.data.at[i, "bestmass_url"] = self.data.at[i, "msini_url"]
            self.data.at[i, "bestmass_provenance"] = "Msini"

        for i in self.data[
            self.data.MASSREL.fillna(1e9) <= self.data.MSINIREL.fillna(1e9)
        ].index:
            self.data.at[i, "bestmass"] = self.data.at[i, "mass"]
            self.data.at[i, "bestmass_min"] = self.data.at[i, "mass_min"]
            self.data.at[i, "bestmass_max"] = self.data.at[i, "mass_max"]
            self.data.at[i, "bestmass_url"] = self.data.at[i, "mass_url"]
            self.data.at[i, "bestmass_provenance"] = "Mass"

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
                if str(re.search("[\s\d][ABCNS]$", row["main_id"], re.M)) == "None"
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
            "status_string",
            "confirmed",
            "discovery_year",
            "final_alias",
            "catalog",
            "angular_separation",
            "angular_separation_flag",
            "coordinate_mismatch",
            "coordinate_mismatch_flag",
            "duplicate_flag",
            "emc_duplicate_flag",
        ]
        try:
            self.data = self.data[keep]
        except KeyError:
            raise KeyError("Not all columns exist")
        logging.info("Selected columns to keep.")
