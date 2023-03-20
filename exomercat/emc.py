import numpy as np
import pandas as pd
import pdb, traceback, sys
from exomercat.configurations import *
from astroquery.simbad import Simbad
from statistics import mode
from exomercat.catalogs import Catalog
import re
import pyvo
from astropy import units as u
from astropy.coordinates import SkyCoord
import logging

class Emc(Catalog):
    def __init__(self):
        """
        The __init__ function is called when the class is instantiated. It sets up the instance of the class, and
        defines any variables that will be used by other functions in the class. In this case, we are setting up
        a dataframe to hold our data.
        """
        super().__init__()
        self.name = "exomercat"
        self.data = pd.DataFrame()

    def alias_as_host(self) -> None:
        """
        The alias_as_host function takes the alias column and checks if any of the aliases are labeled as hosts in
        some other entry. If they are, it will change their host to be that of the original host. It will
        then add all aliases of both hosts into one list for each row.
        """
        host: str
        for host, group in self.data.groupby(by="Host"):
            final_alias = ""
            for al in group.alias:
                final_alias = (
                    final_alias + "," + str(al).replace(host, "").replace("nan", "")
                )
            final_alias = list(
                set([x.strip() for x in set(final_alias.split(",")) if x])
            )

            counter = 0
            final_alias_total = final_alias.copy()
            for al in final_alias:
                if len(self.data.loc[self.data.Host == al]) > 0:
                    counter = counter + 1
                    self.data.loc[self.data.Host == al, "Host"] = host
                    print(
                        "KNOWN ALIAS AS HOST",
                        host,
                        al,
                        file=open("EMClogs/alias_as_host.txt", "a"),
                    )
                    for internal_alias in self.data.loc[self.data.Host == al].alias:
                        for internal_al in internal_alias.split(","):
                            if internal_al not in final_alias_total:
                                final_alias_total.append(internal_al)

            self.data.loc[self.data.Host == host, "alias"] = ",".join(final_alias_total)
        logging.info('Aliases labeled as hosts in some other entry checked.')

    def simbad_list_host_search(self, column: str) -> None:
        """
        The simbad_list_host_search function takes a column name as an argument and searches for the host star
        in that column in SIMBAD. It then fills in the MAIN_ID, IDS, RA, and DEC columns with information from
        SIMBAD if it finds a match.

        Parameters
        ----------
            column: str
                The name of the column that contains the host star to search for (Host or HostBinary)
        """
        list_of_hosts = list(
            self.data[self.data.MAIN_ID == ""][column].drop_duplicates()
        )
        list_of_hosts = [x for x in list_of_hosts]

        Simbad.add_votable_fields("typed_id", "ids", "ra", "dec")
        Simbad.TIMEOUT = 60000000
        result_table = Simbad.query_objects(list_of_hosts)
        result_table = result_table.to_pandas()
        result_table_successful = result_table[result_table.MAIN_ID != ""]

        for i in self.data[self.data.MAIN_ID == ""].index:
            check = self.data.at[i, column]
            if (
                len(result_table_successful[result_table_successful.TYPED_ID == check])
                == 1
            ):
                self.data.at[i, "MAIN_ID"] = result_table_successful[
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

        self.data.MAIN_ID = self.data.MAIN_ID.fillna("")
        self.data.IDS = self.data.IDS.fillna("")
        self.data.RA = self.data.RA.fillna("")
        self.data.DEC = self.data.DEC.fillna("")
        print(
            "List of unique star names",
            len(list_of_hosts),
            "of which successful SIMBAD queries",
            len(result_table_successful),
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
                The name of the column that contains the host star to search for (alias or aliasBinary)
        """

        for i in self.data[self.data.MAIN_ID == ""].index:
            list_of_aliases = str(self.data.at[i, column]).split(",")
            for a in list_of_aliases:
                result_table = Simbad.query_object(a)
                result_table = result_table.to_pandas()
                result_table_successful = result_table[result_table.MAIN_ID != ""]
                if (
                    len(result_table_successful[result_table_successful.TYPED_ID == a])
                    > 0
                ):
                    self.data.at[i, "MAIN_ID"] = result_table_successful[
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
        this table with the original dataframe on Host name (left join).If there are
        still rows missing MAIN_ID values in the merged table, it will query Simbad
        again using all aliases from those rows.
        """

        print("HOST+BINARY Simbad Check")
        self.data["HostBinary"] = (
            self.data["Host"].astype(str)
            + " "
            + self.data["Binary"]
            .astype(str)
            .replace("nan", "")
            .replace("Rogue", "")
            .replace("S-type", "")
        )
        self.data["HostBinary"] = self.data.HostBinary.str.rstrip()

        for i in self.data.index:
            if len(self.data.at[i, "alias"]) > 0:
                single_binary = (
                    str(self.data.at[i, "Binary"])
                    .replace("nan", "")
                    .replace("Rogue", "")
                    .replace("S-type", "")
                )
                single_aliases = str(self.data.at[i, "alias"]).split(",")
                single_aliases_binary = [
                    s + " " + single_binary for s in single_aliases
                ]
                single_aliases_binary = [s.rstrip() for s in single_aliases_binary]
                self.data.at[i, "AliasBinary"] = ",".join(single_aliases_binary)

        self.data["MAIN_ID"] = ""
        self.data["IDS"] = ""
        self.data["RA"] = ""
        self.data["DEC"] = ""
        self.simbad_list_host_search("HostBinary")
        print(
            "Rows still missing MAINID after host search",
            len(self.data[self.data.MAIN_ID == ""]),
        )

        self.simbad_list_alias_search("AliasBinary")
        print(
            "Rows still missing MAIN_ID after alias search",
            len(self.data[self.data.MAIN_ID == ""]),
        )

        print("PURE HOST SIMBAD CHECK")
        self.simbad_list_host_search("Host")
        print(
            "Rows still missing MAINID after host search",
            len(self.data[self.data.MAIN_ID == ""]),
        )

        self.simbad_list_alias_search("alias")
        print(
            "Rows still missing MAIN_ID after alias search",
            len(self.data[self.data.MAIN_ID == ""]),
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
    #     For each target still missing the MAIN_ID, it takes the list of hosts and aliases.
    #     If any of these has "2MASS" in the name, it searches the 2MASS catalog and
    #     returns the 2MASS ID if it exists.
    #     """
    #
    #     for i in self.data[self.data.MAIN_ID == ""].index:
    #         list_of_missing = (
    #             str(self.data.at[i, "Host"]) + "," + str(self.data.at[i, "alias"])
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
    #                         self.data.loc[i, "MAIN_ID"] = "2MASS " + str(
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
    #     print("2MASS check unsuccessful", self.data[self.data.MAIN_ID == ""].shape)

    def gaia_main_id(self) -> None:
        """
        The gaia_main_id function takes the dataframe and searches for missing MAIN_ID values.
        It then checks if there is a Gaia DR2 ID in the Host or alias columns, and if so,
        it uses that to query VizieR for the corresponding source_id. If found, it adds this
        value to MAIN_ID column of the dataframe.
        """

        for i in self.data[self.data.MAIN_ID == ""].index:
            list_of_missing = (
                str(self.data.at[i, "Host"]) + "," + str(self.data.at[i, "alias"])
            )
            for identifier in list_of_missing.split(","):
                if "Gaia DR2" in identifier:
                    identifier = identifier.replace("Gaia DR2 ", "")

                    service = pyvo.dal.TAPService(
                        "http://TAPVizieR.u-strasbg.fr/TAPVizieR/tap/"
                    )
                    query = (
                        "SELECT t.source_id,  t.ra_epoch2000,  t.solution_id, "
                        + ' t.dec_epoch2000 FROM "I/345/gaia2" as t WHERE t."source_id" = '
                        + str(identifier)
                    )

                    response = service.run_sync(query, timeout=None)

                    table = response.to_table().to_pandas()
                    if len(table) > 0:
                        self.data.loc[i, "MAIN_ID"] = "Gaia DR2 " + str(
                            table[
                                table["source_id"] == int(identifier)
                            ].source_id.to_numpy()[0]
                        )
                        self.data.loc[i, "ra_simbad"] = table[
                            table["source_id"] == int(identifier)
                        ].ra_epoch2000.to_numpy()[0]
                        self.data.loc[i, "dec_simbad"] = table[
                            table["source_id"] == int(identifier)
                        ].dec_epoch2000.to_numpy()[0]
                        break

        print("Gaia check unsuccessful", self.data[self.data.MAIN_ID == ""].shape)

    def tess_main_id(self) -> None:
        """
        The tess_main_id function takes the dataframe and checks if there is a TIC ID
        in the alias column. If so, it will query the TESS Input Catalog for that ID
        and return all of its aliases. It then adds those aliases to the alias column
        of self.data.
        """

        for i in self.data[self.data.MAIN_ID == ""].index:
            list_of_missing = (
                str(self.data.at[i, "Host"]) + "," + str(self.data.at[i, "alias"])
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
                        self.data.loc[i, "MAIN_ID"] = "TIC " + str(
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

        print("TESS check unsuccessful", self.data[self.data.MAIN_ID == ""].shape)

    def epic_main_id(self) -> None:
        """
        The epic_main_id function takes the dataframe and searches for EPIC IDs
        in the Host column. If it finds an EPIC ID, it will search for that ID
        in VizieR and return a table of information about that star. It then adds
        this information to the dataframe.
        """

        for i in self.data[self.data.MAIN_ID == ""].index:
            list_of_missing = (
                str(self.data.at[i, "Host"]) + "," + str(self.data.at[i, "alias"])
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
                            self.data.loc[i, "MAIN_ID"] = "EPIC " + str(
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

        print("EPIC check unsuccessful", self.data[self.data.MAIN_ID == ""].shape)

    def set_common_alias(self) -> None:
        """
        The set_common_alias function takes the alias and list_id columns from
        the dataframe,and combines them into a single column called final_alias.
        It then removes duplicates from this new column.
        """
        self.data["final_alias"] = ""
        for host, group in self.data.groupby(by="MAIN_ID"):
            final_alias = ""
            for al in group.alias:
                final_alias = final_alias + "," + str(al)
            for al in group.list_id:
                final_alias = final_alias + "," + str(al)
            self.data.loc[self.data.MAIN_ID == host, "final_alias"] = ",".join(
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
        count = 0
        for ids, group in self.data.groupby(by="list_id"):
            if ids != "" and len(set(group.Host)) > 1:
                self.data.loc[self.data.alias == ids, "Host"] = list(group.Host)[0]
                print(
                    "HOST ",
                    list(group.Host),
                    list(group.catalog),
                    list(group.Status),
                    list(group.Letter),
                    "ID ",
                    list(group.MAIN_ID),
                    file=open("EMClogs/check_alias.txt", "a"),
                )
                count = count + 1
        print("Planets that had a different host name but same SIMBAD alias:", count)

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
        countrasimbad = 0
        countdecsimbad = 0
        self.data["Coordinate_Mismatch"] = ""
        for host, group in self.data[self.data.MAIN_ID == ""].groupby("Host"):
            ra = mode(list(round(group.ra, 3)))
            dec = mode(list(round(group.dec, 3)))
            mismatch_string = ""

            if (abs(group["ra"] - ra) > 0.1).any():
                countra = countra + 1
                print(
                    "*** MISMATCH ON RA *** ",
                    group[["Name", "Host", "Binary", "Letter", "catalog", "ra"]],
                    file=open("EMClogs/check_coordinates.txt", "a"),
                )
                mismatch_string = "ra"
            if (abs(group["dec"] - dec) > 0.1).any():
                countdec = countdec + 1
                print(
                    "*** MISMATCH ON DEC *** ",
                    group[["Name", "Host", "Binary", "Letter", "catalog", "dec"]],
                    file=open("EMClogs/check_coordinates.txt", "a"),
                )
                mismatch_string = mismatch_string + "DEC"

            self.data.loc[group.index]["Coordinate_Mismatch"] = mismatch_string

        print("Mismatched RA", countra, countrasimbad)
        print("Mismatched DEC", countdec, countdecsimbad)
        print(self.data.Coordinate_Mismatch.value_counts())

    def simbad_coordinate_check(self):
        """
        The simbad_coordinate_check function takes the dataframe and checks if
        there are any matches in Simbad for the coordinates of each object.
        It does this by querying Simbad with a circle around each coordinate,
        starting at 0.01 degrees and increasing to 0.5 degrees until it finds
        a match or gives up.
        """
        self.data["angular_separation"] = 0
        service = pyvo.dal.TAPService("http://simbad.u-strasbg.fr:80/simbad/sim-tap")
        for tolerance in [0.01, 0.05, 0.1, 0.5]:
            for host, group in self.data[self.data.MAIN_ID == ""].groupby("Host"):
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
                    self.data.loc[ind, "MAIN_ID"] = selected.loc[0, "main_id"]
                    self.data.loc[ind, "ra_simbad"] = selected.loc[0, "ra_2"]
                    self.data.loc[ind, "dec_simbad"] = selected.loc[0, "dec_2"]
                    self.data.loc[ind, "angular_separation"] = selected.loc[0, "angsep"]
                    result_table = Simbad.query_object(selected.loc[0, "main_id"])
                    result_table = result_table.to_pandas()
                    self.data.loc[ind, "IDS"] = result_table.loc[0, "IDS"]

            print(
                "After coordinate check at tolerance",
                tolerance,
                self.data[self.data.MAIN_ID == ""].shape,
            )
            print(
                "Maximum angular separation at tolerance",
                tolerance,
                max(self.data.angular_separation),
            )
            if len(self.data[self.data.MAIN_ID == ""]) == 0:
                break
        print(self.data.angular_separation.value_counts())

    def check_same_host_different_id(self) -> None:
        """
        The check_same_host_different_id function checks to see if there
        are any instances where the same host has multiple SIMBAD main IDs.
        This should _never_ happen unless the SIMBAD search is failing.

        """
        for host, group in self.data.groupby("HostBinary"):
            if len(group.MAIN_ID.drop_duplicates()) > 1:
                print(
                    "BINARY MISMATCH FOR THE SAME TARGET",
                    host,
                    list(group.MAIN_ID),
                    list(group.Binary),
                    list(group.catalog),
                    file=open("EMClogs/mainid.txt", "a"),
                )

        for host, group in self.data.groupby("Host"):
            if len(group.MAIN_ID.drop_duplicates()) > 1:
                print(
                    "BINARY MISMATCH FOR THE SAME TARGET",
                    host,
                    list(group.MAIN_ID),
                    list(group.Binary),
                    list(group.catalog),
                    file=open("EMClogs/mainid.txt", "a"),
                )

    def check_binary_mismatch(self) -> None:
        """
        The check_binary_mismatch function checks for binary
        mismatches in the data (planets that orbit a binary but are
        controversial in the various catalogs and/or SIMBAD). It
        also checks if the SIMBAD MAIN_ID labels the target as a binary.
        Ideally, all of these issues should be fixed by a human.
        """
        # TODO assume one value per binary
        for (name), group in self.data.groupby(by="Name"):
            if len(set(group.Binary)) > 1:
                print(
                    "BINARY MISMATCH FOR THE SAME TARGET",
                    name,
                    list(group.MAIN_ID),
                    list(group.Binary),
                    list(group.catalog),
                    file=open("EMClogs/binary_mismatch.txt", "a"),
                )
        for (identifier, letter), group in self.data.groupby(by=["MAIN_ID", "Letter"]):
            if len(set(group.Binary)) > 1:
                print(
                    "BINARY MISMATCH FOR THE SAME TARGET (MAIN_ID)",
                    identifier,
                    letter,
                    list(group.Binary),
                    list(group.catalog),
                    file=open("EMClogs/coord_errors.txt", "a"),
                )
        for i in self.data.index:
            if (
                not str(re.search(r"([\s\d][ABCNS])$", self.data.at[i, "MAIN_ID"]))
                == "None"
            ):
                if (
                    not self.data.at[i, "MAIN_ID"][-1:].strip()
                    == self.data.at[i, "Binary"]
                ):
                    print(
                        "MISSED POTENTIAL BINARY ",
                        self.data.at[i, "MAIN_ID"],
                        self.data.at[i, "Name"],
                        "Binary",
                        self.data.at[i, "Binary"],
                        file=open("EMClogs/binary_mismatch.txt", "a"),
                    )

    def check_duplicates_in_same_catalog(self) -> None:
        """
        The check_duplicates_in_same_catalog function checks for
        duplicates within the same catalog. It does this by grouping
        the dataframe by name and checking if there are any values
        that appear more than once . If so, it prints out a list
        of all duplicate entries to a text file called
        "duplicatedentries.txt" in the EMC logs folder.
        """
        # TODO fix conceptually
        count = 0
        count_mainid = 0
        for (name), group in self.data.groupby("Name"):
            if max(group.catalog.value_counts()) > 1:
                count = count + 1
                print(
                    "DUPLICATES WITHIN THE SAME TARGET",
                    name,
                    list(group.MAIN_ID),
                    list(group.Binary),
                    list(group.catalog),
                    list(group.Status),
                    file=open("EMClogs/duplicatedentries.txt", "a"),
                )
        for (identifier, letter), group in self.data.groupby(["MAIN_ID", "Letter"]):
            if max(group.catalog.value_counts()) > 1:
                count_mainid = count_mainid + 1
                print(
                    "DUPLICATES WITHIN THE SAME TARGET (MAINID CHECK)",
                    identifier,
                    letter,
                    list(group.Name),
                    list(group.Binary),
                    list(group.catalog),
                    list(group.Status),
                    file=open("EMClogs/duplicatedentries.txt", "a"),
                )
        print("Duplicate values", count, "with mainid", count_mainid)

    def cleanup_catalog(self) -> None:
        """
        The cleanup_catalog function is used to replace any rows in
        the catalog that have a value of 0 or inf for any of the
        columns i, Mass, Msini, a, P and e with NaN.
        """
        for col in ["i", "Mass", "Msini", "a", "P", "e"]:
            self.data.loc[self.data[col + "_min"] == 0, col + "_min"] = np.nan
            self.data.loc[self.data[col + "_max"] == 0, col + "_max"] = np.nan
            self.data.loc[self.data[col + "_min"] == np.inf, col + "_min"] = np.nan
            self.data.loc[self.data[col + "_max"] == np.inf, col + "_max"] = np.nan

    def merge_into_single_entry(self) -> None:
        """
        The merge_into_single_entry function takes the dataframe and merges all entries
        with the same MAIN_ID and Letter (from the different catalogs) into a single entry.
        It does this by grouping by MAIN_ID and Letter, then iterating through each group.
        It creates an empty dataframe called 'entry' and adds information to it from each group.
        The final entry contains: the official SIMBAD ID and coordinates; the measurements
        that have the smallest relative error with the corresponding reference; the preferred
        name, the preferred status, the preferred binary letter (chosen as the most common
        in the group); year of discovery, method of discovery, and final list of aliases.
        The function then concatenates all of these entries together into a final catalog.
        """
        final_catalog = pd.DataFrame()

        grouped_df = self.data.groupby(["MAIN_ID", "Letter"], sort=True, as_index=False)
        for (mainid, letter), group in grouped_df:
            entry = pd.DataFrame([mainid], columns=["MAIN_ID"])
            entry["Letter"] = letter
            entry['angular_separation']=str(list(set(group.angular_separation.unique())))
            entry["ra_official"] = list(set(group.ra))[0]
            entry["dec_official"] = list(set(group.dec))[0]

            # Decide official name

            entry["Name"] = group["Name"].tolist()[0]

            # prefer names belonging to big surveys
            for name in {"Kepler", "WASP", "GJ", "K2", "HD", "HIP", "CoRoT"}:
                if any(name in s for s in group["Name"].tolist()):
                    entry["Name"] = next(
                        (s for s in group["Name"].tolist() if name in s), None
                    )
                    break

            entry["Host"] = list(set(group["Host"]))[0]
            # TODO: select most probable
            entry["Binary"] = list(set(group["Binary"]))[0]

            # SELECT BEST MEASUREMENT
            params = [
                ["i_url", "i", "i_min", "i_max", "IREL"],
                ["Mass_url", "Mass", "Mass_min", "Mass_max", "MASSREL"],
                ["Msini_url", "Msini", "Msini_min", "Msini_max", "MSINIREL"],
                ["R_url", "R", "R_min", "R_max", "RADREL"],
                ["a_url", "a", "a_min", "a_max", "AREL"],
                ["P_url", "P", "P_min", "P_max", "PERREL"],
                ["e_url", "e", "e_min", "e_max", "EREL"],
            ]
            for p in params:
                result = pd.DataFrame(columns=p)
                result.loc[0, p] = np.nan

                #                result.MAIN_ID=mainid
                subgroup = group[p[:-1]].fillna(np.nan).replace('',np.nan)
                subgroup = subgroup.dropna(subset=[p[1]])
                subgroup = subgroup.dropna(subset=[p[3], p[2]])

                if len(subgroup) > 0:
                    subgroup[p[1]]=subgroup[p[1]].astype('float')
                    subgroup['maxrel'] = subgroup[p[3]].astype('float') / subgroup[p[1]].astype('float')
                    subgroup['minrel'] = subgroup[p[2]].astype('float') / subgroup[p[1]].astype('float')
                    subgroup=subgroup.replace(np.inf, np.nan)
                    subgroup['maxrel'] = subgroup['maxrel'].fillna(subgroup[p[2]])
                    subgroup['minrel'] = subgroup['minrel'].fillna(subgroup[p[2]])
                    subgroup[p[-1]] = subgroup[["maxrel", "minrel"]].max(axis=1)

                    result = subgroup.loc[subgroup[p[-1]] == subgroup[p[-1]].min(),p]
                    result = result.sort_values(by=p[0]).head(1)
                    result=result.reset_index().drop(columns=["index"])

                result=result[p]

                entry = pd.concat([entry, result], axis=1 )

                # Status
                entry["Status_string"] = ",".join(group.CatalogStatus)
                entry["CONFIRMED"] = ",".join(group.CatalogStatus).count("CONFIRMED")
                entry["Status"] = "CONFIRMED"

                if "CANDIDATE" in group.Status.to_list():
                    entry["Status"] = "CANDIDATE"

                if "FALSE POSITIVE" in group.Status.to_list():
                    entry["Status"] = "FALSE POSITIVE"
                # YEAR OF DISCOVERY
                try:
                    entry["YOD"] = sorted(group.YOD.astype('int').dropna().unique())[0]
                except:
                    entry["YOD"]=np.nan
                # Discovery method
                try:
                    entry["DiscMeth"] = list(set(group.DiscMeth[group.DiscMeth!='Default'].unique()))
                except:
                    entry['DiscMeth'] = 'NA'
                # Catalog
                entry["catalog"] = str(sorted(group.catalog.unique()))

                # final Alias
                final_alias = ""
                for al in group.alias:
                    final_alias = final_alias + "," + str(al)
                for al in group.list_id:
                    final_alias = final_alias + "," + str(al)
                entry["final_alias"] = ",".join(
                    [x for x in set(final_alias.split(",")) if x]
                )
            try:
                final_catalog = pd.concat([final_catalog, entry], sort=False).reset_index(drop=True)
            except:
                import ipdb;ipdb.set_trace()
        self.data =final_catalog

    def select_best_mass(self) -> None:
        """
        The select_best_mass function is used to select the best mass for each planet.
        The function first checks if the MASSREL value is greater than MSINIREL, and if so,
        it assigns the Msini values to BestMass, BestMass_min and BestMass_max. If not,
        it assigns Mass values to these columns instead.
        """
        self.data.loc[
            self.data.MASSREL.fillna(1e9) > self.data.MSINIREL.fillna(1e9), "BestMass"
        ] = self.data.Msini
        self.data.loc[
            self.data.MASSREL.fillna(1e9) > self.data.MSINIREL.fillna(1e9),
            "BestMass_min",
        ] = self.data.Msini_min
        self.data.loc[
            self.data.MASSREL.fillna(1e9) > self.data.MSINIREL.fillna(1e9),
            "BestMass_max",
        ] = self.data.Msini_max
        self.data.loc[
            self.data.MASSREL.fillna(1e9) > self.data.MSINIREL.fillna(1e9),
            "BestMass_url",
        ] = self.data.Msini_url
        self.data.loc[
            self.data.MASSREL.fillna(1e9) > self.data.MSINIREL.fillna(1e9),
            "BestMass_provenance",
        ] = 'Msini'

        self.data.loc[
            self.data.MASSREL.fillna(1e9) <= self.data.MSINIREL.fillna(1e9), "BestMass"
        ] = self.data.Mass
        self.data.loc[
            self.data.MASSREL.fillna(1e9) <= self.data.MSINIREL.fillna(1e9),
            "BestMass_min",
        ] = self.data.Mass_min
        self.data.loc[
            self.data.MASSREL.fillna(1e9) <= self.data.MSINIREL.fillna(1e9),
            "BestMass_max",
        ] = self.data.Mass_max
        self.data.loc[
            self.data.MASSREL.fillna(1e9) > self.data.MSINIREL.fillna(1e9),
            "BestMass_url",
        ] = self.data.Mass_url
        self.data.loc[
            self.data.MASSREL.fillna(1e9) > self.data.MSINIREL.fillna(1e9),
            "BestMass_provenance",
        ] = 'Mass'

    def keep_columns(self) -> None:
        """
        The keep_columns function is used to keep only the columns that are needed for the analysis.
        The function takes in a dataframe and returns a new dataframe with only the columns listed above.
        """
        keep = [
            "Name",
            "Host",
            "Letter",
            "MAIN_ID",
            "Binary",
            "ra_official",
            "dec_official",
            "Mass",
            "Mass_max",
            "Mass_min",
            "Mass_url",
            "Msini",
            "Msini_max",
            "Msini_min",
            "Msini_url",
            "BestMass",
            "BestMass_max",
            "BestMass_min",
            "BestMass_url",
            "BestMass_provenance",
            "P",
            "P_max",
            "P_min",
            "P_url",
            "R",
            "R_max",
            "R_min",
            "R_url",
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
            "DiscMeth",
            "Status",
            "Status_string",
            "CONFIRMED",
            "YOD",
            "final_alias",
            "catalog",
            'angular_separation'
            #"MismatchFlagHost",
        ]
        self.data=self.data[keep]