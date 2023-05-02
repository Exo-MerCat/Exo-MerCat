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
            final_alias = ""
            for al in group.alias:
                final_alias = (
                    final_alias + "," + str(al).replace(host, "").replace("nan", "")
                )
            final_alias = list(
                set([x.strip() for x in set(final_alias.split(",")) if x])
            )

            final_alias_total = final_alias.copy()
            for al in final_alias:
                if len(self.data.loc[self.data.host == al]) > 0:
                    counter = counter + 1
                    self.data.loc[self.data.host == al, "host"] = host
                    f.write("ALIAS: " + al + " AS HOST:" + host + "\n")
                    for internal_alias in self.data.loc[self.data.host == al].alias:
                        for internal_al in internal_alias.split(","):
                            if internal_al not in final_alias_total:
                                final_alias_total.append(internal_al)

            self.data.loc[self.data.host == host, "alias"] = ",".join(final_alias_total)
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

        Parameters
        ----------
            column: str
                The name of the column that contains the host star to search for (host or hostbinary)
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
        logging.info(
            "List of unique star names "
            + str(len(list_of_hosts))
            + " of which successful SIMBAD queries "
            + str(len(result_table_successful))
        )

    def simbad_list_alias_search(self, column: str) -> None:
        """
        The simbad_list_alias_search function takes a column name as an argument and searches for the main ID of
        each object in that column. The function first splits the string into a list of aliases, then iterates
        through each alias to search SIMBAD for its main ID. If it finds one, it will update the dataframe with
        that information.

        Parameters
        ----------
            column: str
                The name of the column that contains the host star to search for (alias or aliasbinary)
        """

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

        logging.info("HOST+BINARY Simbad Check")
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
                self.data.at[i, "Aliasbinary"] = ",".join(single_aliases_binary)

        self.data["main_id"] = ""
        self.data["IDS"] = ""
        self.data["RA"] = ""
        self.data["DEC"] = ""
        self.simbad_list_host_search("hostbinary")
        logging.info(
            "Rows still missing MAINID after host search "
            + str(len(self.data[self.data.main_id == ""]))
        )

        self.simbad_list_alias_search("Aliasbinary")
        logging.info(
            "Rows still missing main_id after alias search "
            + str(len(self.data[self.data.main_id == ""]))
        )

        logging.info("PURE HOST SIMBAD CHECK")
        self.simbad_list_host_search("host")
        logging.info(
            "Rows still missing MAINID after host search "
            + str(len(self.data[self.data.main_id == ""]))
        )

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

    # def twomass_main_id(self) -> None:
    #     """
    #     The twomass_main_id function is used to find the 2MASS ID of a star.
    #     For each target still missing the main_id, it takes the list of hosts and aliases.
    #     If any of these has "2MASS" in the name, it searches the 2MASS catalog and
    #     returns the 2MASS ID if it exists.
    #     """
    #
    #     for i in self.data[self.data.main_id == ""].index:
    #         list_of_missing = (
    #             str(self.data.at[i, "host"]) + "," + str(self.data.at[i, "alias"])
    #         )
    #         for identifier in list_of_missing.split(","):
    #             if "2MASS" in identifier:
    #                 identifier = (
    #                     identifier.replace("2MASS ", "")
    #                     .replace("2MASSW ", " ")
    #                     .replace("J", "")
    #                     .replace('"', "")
    #                 )
    #
    #                 service = pyvo.dal.TAPService(
    #                     "http://TAPVizieR.u-strasbg.fr/TAPVizieR/tap/"
    #                 )
    #                 query = (
    #                     'SELECT  t.RAJ2000,  t.DEJ2000,  t."2MASS"'
    #                     + ' FROM "II/246/out" as t WHERE t."2MASS" = "'
    #                     + str(identifier)
    #                     + '"'
    #                 )
    #
    #                 response = service.run_sync(query, timeout=None)
    #                 table = response.to_table().to_pandas()
    #                 if len(table) > 0:
    #                     if len(table) > 0:
    #                         self.data.loc[i, "main_id"] = "2MASS " + str(
    #                             table[table["2MASS"] == int(identifier)].ID.to_numpy()[
    #                                 0
    #                             ]
    #                         )
    #                         self.data.loc[i, "ra_simbad"] = table[
    #                             table.ID == int(identifier)
    #                         ].RAJ2000.to_numpy()[0]
    #                         self.data.loc[i, "dec_simbad"] = table[
    #                             table.ID == int(identifier)
    #                         ].DEJ2000.to_numpy()[0]
    #                         break
    #
    #     print("2MASS check unsuccessful", self.data[self.data.main_id == ""].shape)

    def gaia_main_id(self) -> None:
        """
        The gaia_main_id function takes the dataframe and searches for missing main_id values.
        It then checks if there is a Gaia DR2 ID in the host or alias columns, and if so,
        it uses that to query VizieR for the corresponding source_id. If found, it adds this
        value to main_id column of the dataframe.
        """

        for i in self.data[self.data.main_id == ""].index:
            list_of_missing = (
                str(self.data.at[i, "host"]) + "," + str(self.data.at[i, "alias"])
            )
            for identifier in list_of_missing.split(","):
                if "Gaia DR2" in identifier:
                    identifier = identifier.replace("Gaia DR2 ", "")

                    service = pyvo.dal.TAPService(
                        "http://TAPVizieR.u-strasbg.fr/TAPVizieR/tap/"
                    )
                    query = (
                        "SELECT t.designation,  t.ra_epoch2000,  t.solution_id, "
                        + ' t.dec_epoch2000 FROM "I/345/gaia2" as t WHERE t."source_id" = '
                        + str(identifier)
                    )

                    response = service.run_sync(query, timeout=None)

                    table = response.to_table().to_pandas()
                    if len(table) > 0:
                        self.data.loc[i, "main_id"] = str(
                            table[
                                table["designation"] == int(identifier)
                            ].designation.to_numpy()[0]
                        )
                        self.data.loc[i, "ra_simbad"] = table[
                            table["source_id"] == int(identifier)
                        ].ra_epoch2000.to_numpy()[0]
                        self.data.loc[i, "dec_simbad"] = table[
                            table["source_id"] == int(identifier)
                        ].dec_epoch2000.to_numpy()[0]
                        break

        logging.info(
            "Gaia check unsuccessful " + str(self.data[self.data.main_id == ""].shape)
        )

    def tess_main_id(self) -> None:
        """
        The tess_main_id function takes the dataframe and checks if there is a TIC ID
        in the alias column. If so, it will query the TESS Input Catalog for that ID
        and return all of its aliases. It then adds those aliases to the alias column
        of self.data.
        """

        for i in self.data[self.data.main_id == ""].index:
            list_of_missing = (
                str(self.data.at[i, "host"]) + "," + str(self.data.at[i, "alias"])
            )
            for identifier in list_of_missing.split(","):
                if "TIC" in identifier:
                    identifier = identifier.replace("TIC ", "")

                    service = pyvo.dal.TAPService(
                        "http://TAPVizieR.u-strasbg.fr/TAPVizieR/tap/"
                    )
                    query = (
                        'SELECT t.TIC,  t.DEJ2000,  t.HIP,  t.UCAC4,  t."2MASS", '
                        + " t.WISEA,  t.GAIA, t.KIC,  t.ID,  t.objID,  t.RAJ2000, "
                        + ' t.TYC FROM "IV/38/tic" as t WHERE t."TIC" = '
                        + str(identifier)
                    )

                    response = service.run_sync(query, timeout=None)

                    table = response.to_table().to_pandas()
                    if len(table) > 0:
                        self.data.loc[i, "main_id"] = "TIC " + str(
                            table[table["TIC"] == int(identifier)].TIC.to_numpy()[0]
                        )
                        self.data.loc[i, "ra_simbad"] = table[
                            table["TIC"] == int(identifier)
                        ].RAJ2000.to_numpy()[0]
                        self.data.loc[i, "dec_simbad"] = table[
                            table["TIC"] == int(identifier)
                        ].DEJ2000.to_numpy()[0]
                        if (
                            str(
                                table[table.TIC == int(identifier)]["2MASS"].to_numpy()[
                                    0
                                ]
                            )
                            != "<NA>"
                        ):
                            self.data.loc[i, "alias"] = (
                                self.data.loc[i, "alias"]
                                + ", 2MASS J"
                                + str(
                                    table[table.TIC == int(identifier)][
                                        "2MASS"
                                    ].to_numpy()[0]
                                )
                            )
                        if (
                            str(
                                table[table.TIC == int(identifier)]["HIP"].to_numpy()[0]
                            )
                            != "<NA>"
                        ):
                            self.data.loc[i, "alias"] = (
                                self.data.loc[i, "alias"]
                                + ", HIP "
                                + str(
                                    table[table.TIC == int(identifier)].HIP.to_numpy()[
                                        0
                                    ]
                                )
                            )
                        if (
                            str(
                                table[table.TIC == int(identifier)]["GAIA"].to_numpy()[
                                    0
                                ]
                            )
                            != "<NA>"
                        ):
                            self.data.loc[i, "alias"] = (
                                self.data.loc[i, "alias"]
                                + ", Gaia DR2 "
                                + str(
                                    table[table.TIC == int(identifier)].GAIA.to_numpy()[
                                        0
                                    ]
                                )
                            )
                        if (
                            str(
                                table[table.TIC == int(identifier)]["KIC"].to_numpy()[0]
                            )
                            != "<NA>"
                        ):
                            self.data.loc[i, "alias"] = (
                                self.data.loc[i, "alias"]
                                + ", KIC "
                                + str(
                                    table[table.TIC == int(identifier)].KIC.to_numpy()[
                                        0
                                    ]
                                )
                            )

                        break

        logging.info(
            "TESS check unsuccessful " + str(self.data[self.data.main_id == ""].shape)
        )

    def epic_main_id(self) -> None:
        """
        The epic_main_id function takes the dataframe and searches for EPIC IDs
        in the host column. If it finds an EPIC ID, it will search for that ID
        in VizieR and return a table of information about that star. It then adds
        this information to the dataframe.
        """

        for i in self.data[self.data.main_id == ""].index:
            list_of_missing = (
                str(self.data.at[i, "host"]) + "," + str(self.data.at[i, "alias"])
            )
            for identifier in list_of_missing.split(","):
                if "EPIC" in identifier:
                    identifier = identifier.replace("EPIC ", "")

                    service = pyvo.dal.TAPService(
                        "http://TAPVizieR.u-strasbg.fr/TAPVizieR/tap/"
                    )
                    query = (
                        "SELECT  t.ID,  t.RAJ2000,  t.DEJ2000,  t.HIP,  t.TYC, "
                        + 't.UCAC4,  t."2MASS", t.SDSS  FROM "IV/34/epic" as t WHERE t.ID = '
                        + str(identifier)
                    )

                    try:
                        response = service.run_sync(query, timeout=None)

                        table = response.to_table().to_pandas()
                        if len(table) > 0:
                            self.data.loc[i, "main_id"] = "EPIC " + str(
                                table[table.ID == int(identifier)].ID.to_numpy()[0]
                            )
                            self.data.loc[i, "ra_simbad"] = table[
                                table.ID == int(identifier)
                            ].RAJ2000.to_numpy()[0]
                            self.data.loc[i, "dec_simbad"] = table[
                                table.ID == int(identifier)
                            ].DEJ2000.to_numpy()[0]
                            if (
                                str(
                                    table[table.ID == int(identifier)][
                                        "2MASS"
                                    ].to_numpy()[0]
                                )
                                != "<NA>"
                            ):
                                self.data.loc[i, "alias"] = (
                                    self.data.loc[i, "alias"]
                                    + ", 2MASS "
                                    + str(
                                        table[table.ID == int(identifier)][
                                            "2MASS"
                                        ].to_numpy()[0]
                                    )
                                )
                            if (
                                str(
                                    table[table.ID == int(identifier)][
                                        "HIP"
                                    ].to_numpy()[0]
                                )
                                != "<NA>"
                            ):
                                self.data.loc[i, "alias"] = (
                                    self.data.loc[i, "alias"]
                                    + ", HIP "
                                    + str(
                                        table[
                                            table.ID == int(identifier)
                                        ].HIP.to_numpy()[0]
                                    )
                                )
                            break
                    except:
                        pass

        logging.info(
            "EPIC check unsuccessful " + str(self.data[self.data.main_id == ""].shape)
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
        # TODO check that it makes sense
        f = open("Logs/check_alias.txt", "a")
        count = 0
        for ids, group in self.data.groupby(by="list_id"):
            if ids != "" and len(set(group.host)) > 1:
                self.data.loc[self.data.alias == ids, "host"] = list(group.host)[0]
                f.write(
                    "HOST "+str(list(group.host))+str(list(group.catalog))+str(list(group.status))+str(
                    list(group.letter))+" ID "+str(list(group.main_id))
                )
                count = count + 1
        logging.info(
            "Planets that had a different host name but same SIMBAD alias: "
            + str(count)
        )
        f.close()

    def check_coordinates(self):
        """
        The check_coordinates function checks for mismatches in the RA and DEC
        coordinates of a given host. It does this by grouping all entries with
        the same host name, then checking if any of those entries have an RA
        or DEC that is more than 0.01 degrees away from the mode value for that
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

            if (abs(group["ra"] - ra) > 0.1).any():
                countra = countra + 1
                f.write(
                    "*** MISMATCH ON RA *** "
                    + str(
                        (group[["name", "host", "binary", "letter", "catalog", "ra"]])
                    )
                    + "\n"
                )
                mismatch_string = "RA"

            if (abs(group["dec"] - dec) > 0.1).any():
                countdec = countdec + 1
                f.write(
                    "*** MISMATCH ON DEC *** "
                    + str(
                        (group[["name", "host", "binary", "letter", "catalog", "dec"]])
                    )
                    + "\n"
                )
                mismatch_string = mismatch_string + "DEC"

            self.data.loc[group.index,"coordinate_mismatch"] = mismatch_string
        f.close()
        logging.info("Found " + str(countra) + " mismatched RA.")
        logging.info("Found " + str(countdec) + " mismatched DEC.")

        logging.info(self.data.coordinate_mismatch.value_counts())

    def get_coordinates_from_simbad(self):
        """
        This function takes the dataframe and checks if there are any matches in Simbad for the coordinates of each
        object. It does this by querying Simbad with a circle around each coordinate, starting at 0.01 degrees and
        increasing to 0.5 degrees until it finds a match or gives up.
        """
        self.data["angular_separation"] = 0
        service = pyvo.dal.TAPService("http://simbad.u-strasbg.fr:80/simbad/sim-tap")

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
                    self.data.loc[ind, "IDS"] = result_table.loc[0, "IDS"]

            logging.info(
                "After coordinate check at tolerance "
                + str(tolerance)
                + " residuals: "
                + str(self.data[self.data.main_id == ""].shape)
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
        counter=0
        f = open("Logs/main_id_correction.txt", "a")
        for identifier in self.data["main_id"]:
            if not str(re.search("[\s\d][b-i]$", identifier, re.M)) == "None":
                counter+=1
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
        logging.info('Removed planet letter from main_id. It happens '+str(counter)+' times.')

    def check_binary_mismatch(self, keyword: str) -> None:
        """
        The check_binary_mismatch function checks for binary mismatches in the data (planets that orbit a binary but are
        controversial in the various catalogs and/or SIMBAD). It also checks if the SIMBAD main_id labels the target as
        a binary. Ideally, all of these issues should be fixed by a human for the code to work properly.
        """
        self.data['binary']=self.data['binary'].fillna('')
        self.data['potential_binary_mismatch']=0
        f = open("Logs/binary_mismatch.txt", "a")
        f.write("****" + keyword + "****\n")
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
                        + " binary :"
                        + str(self.data.at[i, "binary"])
                        + "\n"
                    )

        f.write("****" + keyword + "+letter THAT COULD BE UNIFORMED****\n")
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
                                warning = " WARNING, Coordinate Mismatch (potential_binary_mismatch 1)"
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

                    self.data.loc[
                            group[group.binary == "S-type"].index, "binary"
                        ] = group[group.binary != "S-type"].binary.fillna('').mode()[0]

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

                    self.data.loc[group[group.binary == ""].index, "binary"] = group[
                        group.binary != ""
                    ].binary.fillna('').mode()[0]

        # Identify weird systems after applying the correction:
        f.write(
            "\n****"
            + keyword
            + "+letter THAT ARE INCONSISTENTLY LABELED (Potential Mismatch 2). They should be treated manually in replacements.ini ****\n\n"
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

        f.close()
        logging.info("Checked potential binaries to be manually corrected.")
        logging.info(
            "Automatic correction results: "
            + str(self.data.potential_binary_mismatch.value_counts())
        )

    # def check_duplicates_in_same_catalog(self) -> None:
    #     """
    #     The check_duplicates_in_same_catalog function checks for
    #     duplicates within the same catalog. It does this by grouping
    #     the dataframe by name and checking if there are any values
    #     that appear more than once . If so, it prints out a list
    #     of all duplicate entries to a text file called
    #     "duplicatedentries.txt" in the EMC logs folder.
    #     """
    #     count = 0
    #     count_mainid = 0
    #     for (name), group in self.data.groupby("name"):
    #         if max(group.catalog.value_counts()) > 1:
    #             count = count + 1
    #             print(
    #                 "DUPLICATES WITHIN THE SAME TARGET",
    #                 name,
    #                 list(group.main_id),
    #                 list(group.binary),
    #                 list(group.catalog),
    #                 list(group.status),
    #                 file=open("Logs/duplicatedentries.txt", "a"),
    #             )
    #     for (identifier, letter), group in self.data.groupby(["main_id", "letter"]):
    #         if max(group.catalog.value_counts()) > 1:
    #             count_mainid = count_mainid + 1
    #             print(
    #                 "DUPLICATES WITHIN THE SAME TARGET (MAINID CHECK)",
    #                 identifier,
    #                 letter,
    #                 list(group.name),
    #                 list(group.binary),
    #                 list(group.catalog),
    #                 list(group.status),
    #                 file=open("Logs/duplicatedentries.txt", "a"),
    #             )
    #     print("Duplicate values", count, "with mainid", count_mainid)

    def cleanup_catalog(self) -> None:
        """
        The cleanup_catalog function is used to replace any rows in
        the catalog that have a value of 0 or inf for any of the
        columns i, mass, msini, a, P and e with NaN.
        """
        for col in ["i", "mass", "msini", "a", "p", "e"]:
            self.data.loc[self.data[col + "_min"] == 0, col + "_min"] = np.nan
            self.data.loc[self.data[col + "_max"] == 0, col + "_max"] = np.nan
            self.data.loc[self.data[col + "_min"] == np.inf, col + "_min"] = np.nan
            self.data.loc[self.data[col + "_max"] == np.inf, col + "_max"] = np.nan
        logging.info("Catalog cleared from zeroes and infinities.")

    def merge_into_single_entry(self, verbose: bool) -> None:
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
        """
        final_catalog = pd.DataFrame()
        f = open("Logs/duplicate_entries.txt", "a")

        grouped_df = self.data.groupby(
            ["main_id", "binary", "letter"], sort=True, as_index=False
        )
        counter = 0
        for (mainid, binary, letter), group in grouped_df:
            entry = pd.DataFrame([mainid], columns=["main_id"])
            entry["binary"] = binary
            entry["letter"] = letter

            entry["angular_separation"] = str(
                list(set(group.angular_separation.unique()))
            )
            entry["ra_official"] = list(set(group.ra))[0]
            entry["dec_official"] = list(set(group.dec))[0]

            # Decide official name

            entry["name"] = group["name"].tolist()[0]

            # prefer names belonging to big surveys
            for name in {"Kepler", "WASP", "GJ", "K2", "HD", "HIP", "CoRoT"}:
                if any(name in s for s in group["name"].tolist()):
                    entry["name"] = next(
                        (s for s in group["name"].tolist() if name in s), None
                    )
                    break

            entry["host"] = list(set(group["host"]))[0]

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

                #                result.main_id=mainid
                subgroup = group[p[:-1]].fillna(np.nan).replace("", np.nan)
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
                entry["status_string"] = ",".join(group.Catalogstatus)

                entry["confirmed"] = ",".join(set(group.Catalogstatus)).count("CONFIRMED")

                entry['status']=  group.status.value_counts().idxmax()
                # YEAR OF DISCOVERY
                try:
                    entry["discovery_year"] = sorted(
                        group.discovery_year.astype("int").dropna().unique()
                    )[0]
                except:
                    entry["discovery_year"] = np.nan
                # discovery method
                try:
                    entry["discovery_method"] = list(
                        set(group.discovery_method[group.discovery_method != "Default"].unique())
                    )
                except:
                    entry["discovery_method"] = "NA"

                entry["catalog"] = str(sorted(group.catalog.unique()))


                # final Alias
                final_alias = ""
                for al in group.final_alias:
                    final_alias = final_alias + "," + str(al)
                entry["final_alias"] = ",".join(
                    [x for x in set(final_alias.split(",")) if x not in ["A", "B", ""]]
                )

                entry["binary_mismatch_flag"] = list(
                        set(group.potential_binary_mismatch.unique())
                    )

                if len(group.potential_binary_mismatch.unique())>1:
                    entry["binary_mismatch_flag"] = entry["binary_mismatch_flag"]+'*'

                entry["coordinate_mismatch"] = list(
                        set(group.coordinate_mismatch.unique())
                    )

                if 'RA' in  set(group.coordinate_mismatch.unique()) or 'DEC' in  set(group.coordinate_mismatch.unique()):
                    entry['coordinate_mismatch_flag']= 1
                else:
                    entry['coordinate_mismatch_flag'] = 0
                entry['angular_separation_flag']=len(list(set(group.angular_separation.unique())))-1
                # Catalog

                if len(group) > len(group.catalog.unique()):
                    f.write(
                        "DUPLICATE ENTRY "
                        + mainid
                        + " "
                        + letter
                        + " CATALOGS "
                        + str(group.catalog.values)
                        + " STATUS "
                        + str(group.status.values)
                        + "\n"
                    )

                    entry['duplicate_flag'] =  1
                else:
                    entry['duplicate_flag'] = 0
            final_catalog = pd.concat([final_catalog, entry], sort=False).reset_index(
                drop=True
            )
            if verbose:
                print(
                    "Done "
                    + str(round(counter / len(grouped_df), 2) * 100)
                    + "% of the groups.",
                    end="\r",
                )
            counter = counter + 1
        f.close()
        self.data = final_catalog
        logging.info("Catalog merged into single entries.")

    def select_best_mass(self) -> None:
        """
        The select_best_mass function is used to select the best mass for each planet.
        The function first checks if the MASSREL value is greater than MSINIREL, and if so,
        it assigns the msini values to bestmass, bestmass_min and bestmass_max. If not,
        it assigns mass values to these columns instead.
        """

        for i in self.data[self.data.MASSREL.fillna(1e9) > self.data.MSINIREL.fillna(1e9)].index:
            self.data.at[i,'bestmass']=self.data.at[i,'msini']
            self.data.at[i, 'bestmass_min'] = self.data.at[i, 'msini_min']
            self.data.at[i, 'bestmass_max'] = self.data.at[i, 'msini_max']
            self.data.at[i, 'bestmass_url'] = self.data.at[i, 'msini_url']
            self.data.at[i, 'bestmass_provenance'] = 'Msini'

        for i in self.data[self.data.MASSREL.fillna(1e9) <= self.data.MSINIREL.fillna(1e9)].index:
            self.data.at[i,'bestmass']=self.data.at[i,'mass']
            self.data.at[i, 'bestmass_min'] = self.data.at[i, 'mass_min']
            self.data.at[i, 'bestmass_max'] = self.data.at[i, 'mass_max']
            self.data.at[i, 'bestmass_url'] = self.data.at[i, 'mass_url']
            self.data.at[i, 'bestmass_provenance'] = 'Mass'

        logging.info("Bestmass calculated.")

    def set_exo_mercat_name(self) -> None:
        self.data['exo-mercat_name'] = self.data.apply(
            lambda row: row['main_id'] + ' ' + str(row['binary']).replace('nan', '') + ' ' + row['letter'], axis=1)
        self.data=self.data.sort_values(by='exo-mercat_name')
    def keep_columns(self) -> None:
        """
        The keep_columns function is used to keep only the columns that are needed for the analysis.
        The function takes in a dataframe and returns a new dataframe with only the columns listed above.
        """
        keep = [
            'exo-mercat_name',
            "name",
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
            "binary_mismatch_flag",
            "coordinate_mismatch",
            "coordinate_mismatch_flag",
            "duplicate_flag",
        ]
        self.data = self.data[keep]
