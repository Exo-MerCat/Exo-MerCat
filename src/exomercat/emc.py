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
from .catalogs import Catalog
from .utility_functions import UtilityFunctions as Utils


class Emc(Catalog):
    """
    The Emc (Exo-MerCat) class is a subclass of Catalog that represents the Exo-MerCat catalog.
    It contains methods for processing, merging, and managing exoplanet data from various sources.
    This class provides functionality for data cleaning, standardization, and analysis of exoplanet information.

    Key features include:
    - Merging data from multiple exoplanet catalogs
    - Standardizing identifiers and coordinates
    - Resolving conflicts in planet properties
    - Generating a unified exoplanet catalog

    The class methods cover various aspects of catalog management, from initial data processing
    to final catalog generation and export.

    Attributes:
        name (str): The name of the catalog, set to "exo_mercat".
        data (pd.DataFrame): The main DataFrame containing the exoplanet catalog data.

    Methods:
        __init__(): Initializes the Emc class object.
        convert_coordinates(): Placeholder for coordinate conversion (not implemented).
        alias_as_host(): Checks and standardizes host names based on aliases.
        check_binary_mismatch(keyword, tolerance): Checks for binary mismatches in the dataframe.
        prepare_columns_for_mainid_search(): Prepares columns for the search of the main identifier.
        fill_mainid_provenance_column(keyword): Fills the 'main_id_provenance' column with the provided keyword.
        simbad_list_host_search(typed_id): Searches for host stars in SIMBAD using the specified column.
        simbad_list_alias_search(column): Searches for the main ID of each object in the specified column using SIMBAD.
        get_host_info_from_simbad(): Retrieves host information from SIMBAD.
        get_coordinates_from_simbad(tolerance): Retrieves coordinates from SIMBAD for objects without main IDs.
        get_host_info_from_tic(): Retrieves host information from the TESS Input Catalog (TIC).
        get_coordinates_from_tic(tolerance): Retrieves coordinates from TIC for objects without main IDs.
        check_coordinates(tolerance): Checks for mismatches in RA and DEC coordinates of a given host.
        replace_old_new_identifier(identifier, new_identifier, binary): Replaces old identifiers with new ones.
        polish_main_id(): Polishes the main_id column by removing planet/binary letters.
        fill_missing_main_id(): Fills missing values in main_id related columns.
        check_same_host_different_id(): Checks for instances where the same host has multiple SIMBAD main IDs.
        check_same_coords_different_id(tolerance): Checks for instances where the same coordinates have different main IDs.
        group_by_list_id_check_main_id(): Groups data by 'list_id' and checks for inconsistencies in 'main_id'.
        post_main_id_query_checks(tolerance): Performs a series of checks after querying SIMBAD for main IDs.
        group_by_main_id_set_main_id_aliases(): Groups by main_id and combines aliases into main_id_aliases.
        cleanup_catalog(): Cleans up the catalog by replacing 0 and infinity values with NaN.
        group_by_period_check_letter(): Checks for inconsistencies in the letter column and attempts to fix them.
        group_by_letter_check_period(verbose): Groups by letter and merges entries based on period or semi-major axis agreement.
        select_best_mass(): Selects the best mass estimate for each planet in the catalog.
        set_exomercat_name(): Creates the 'exo-mercat_name' column.
        keep_columns(): Retains only specified columns in the dataframe.
        remove_known_brown_dwarfs(local_date, print_flag): Removes objects with masses greater than 20 Jupiter masses.
        fill_row_update(local_date): Updates the 'row_update' column based on changes from the previous version.
        save_catalog(local_date, postfix): Saves the catalog to CSV files.
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
        Convert the right ascension (RA) and declination (Dec) columns of the dataframe to decimal degrees. 
        
        This function is not implemented as the EMC already has coordinates in decimal degrees.

        :param self: The instance of the Emc class.
        :type self: Emc
        :return: None
        :rtype: None
        :note: This function is not necessary as the EMC already has coordinates in decimal degrees.
        """
        # This function is intentionally left empty (pass) because:
        # 1. The Exo-MerCat (EMC) catalog already contains coordinates in decimal degrees.
        # 2. No conversion is needed for the existing coordinate format.
        # 3. The function is kept as a placeholder in case future modifications to coordinate handling are required.
        pass
        

    def alias_as_host(self) -> None:
        """
        Check if any aliases are labeled as hosts in some other entry and standardize the host name.

        This function takes the alias column of a dataframe and checks if any of the aliases are labeled
        as hosts in some other entry. If an alias is labeled as a host, it changes the host to be that of the
        original host. It then adds all aliases of both hosts into one list for each row. It logs results into
         "Logs/alias_as_host.txt".

        :param self: The instance of the Emc class.
        :type self: Emc
        :return: None
        :rtype: None
        """
        # Open log file
        f = open("Logs/alias_as_host.txt", "a")
        counter = 0
        host: str
        # Iterate through groups of data, grouped by host
        for host, group in self.data.groupby(by="host"):
            # Create an empty set to store unique values
            main_id_aliases = set()

            # Iterate through the list and add non-NaN values to the set
            for alias in list(group.alias):
                alias_list = str(alias).split(",")
                for item in alias_list:
                    if (
                        not pd.isnull(item)
                        and item != "nan"
                        and item not in main_id_aliases
                        and item != host
                    ):
                        main_id_aliases.add(item)

            main_id_aliases_total = set(main_id_aliases)
            # Check if any alias is labeled as a host in another entry
            for al in main_id_aliases:
                if len(self.data.loc[self.data.host == al]) > 0:
                    counter = counter + 1
                    # Add aliases of the host found as an alias to the total set
                    for internal_alias in self.data.loc[self.data.host == al].alias:
                        alias_list = str(internal_alias).split(",")

                        for internal_al in alias_list:
                            if internal_al not in main_id_aliases_total:
                                main_id_aliases_total.add(internal_al.lstrip().rstrip())
                    # Change the host to be that of the original host
                    self.data.loc[self.data.host == al, "host"] = host
                    f.write("ALIAS: " + al + " AS HOST:" + host + "\n")

            main_id_aliases_total = set(main_id_aliases_total)

            # Update the alias column with all unique aliases
            self.data.loc[self.data.host == host, "alias"] = (
                ",".join(sorted(set(main_id_aliases_total))).rstrip(",").lstrip(",")
            )

        f.close()

        # Log the number of times an alias was labeled as a host
        logging.info(
            "Aliases labeled as hosts in some other entry checked. It happens "
            + str(counter)
            + " times."
        )

    def check_binary_mismatch(self, keyword: str, tolerance: float = 1 / 3600):
        """
        Check for binary mismatches in the dataframe.

        This function checks if there are multiple values of binary for a given system (identified by name and letter).
        It attempts to standardize the binary values and flags complex systems or those with coordinate disagreements.

        If there are multiple values of binary for a given system (identified by name and letter), 
        it tries to replace the null or S-type binaries with the value of another non-null entry in that system.
        If all entries have null or S-type binaries, it replaces them with 'S-type'.
        If there are multiple non-null values of binary, it flags this as a complex system and does not try to correct anything.
        It also flags systems where coordinates do not agree within a tolerance (by default 1 arcsecond).

        :param self: The instance of the Emc class.
        :type self: Emc
        :param keyword: The keyword to search for in the dataframe.
        :type keyword: str
        :param tolerance: The tolerance to use for coordinate comparisons. (default is 1/3600)
        :type tolerance: float
        :return: None
        :rtype: None
        """
        # Fill NaN values in 'binary' column with empty string
        self.data["binary"] = self.data["binary"].fillna("")

        # Add 'binary_mismatch_flag' column if it doesn't exist
        if "binary_mismatch_flag" not in self.data.columns:
            self.data["binary_mismatch_flag"] = 0

        # Open log file in append mode
        f = open("Logs/check_binary_mismatch.txt", "a")
        f.write("*************************************************\n")
        f.write("**** CHECKING BINARIES USING: " + keyword + " ****\n")
        f.write("*************************************************\n")

        counter = 0  # Counter for the number of binary mismatches fixed

        # Try to standardize the value of binary from the other entries of the same system.
        for (key, letter), group in self.data.groupby(by=[keyword, "letter"]):
            group.ra = np.round(group.ra.astype(float), 6)
            group.dec = np.round(group.dec.astype(float), 6)
            group.binary = group.binary.replace("", "null")
            group["skycoord"] = SkyCoord(
                ra=group.ra * u.deg, dec=group.dec * u.deg, unit="deg"
            )
            # if len(set(group.binary))==1 there is no issue, the binary values agree with one another
            if len(set(group.binary)) > 1:  # there is a discrepancy
                f.write("\n")
                # Separate null and non-null binaries
                subgroup1 = group[group.binary.str.contains("S-type|null")]
                subgroup2 = group[~group.binary.str.contains("S-type|null")]

                # Case: No non-null binary values to replace with
                if len(set(subgroup2.binary)) == 0:
                    # If there are only null and S-type, prefer S-type
                    if len(set(subgroup1.binary)) > 1:
                        self.data.loc[subgroup1.index, "binary"] = "S-type"
                        counter += 1
                        warning = ""
                        # Check coordinate agreement
                        check_on_coordinates = [
                            subgroup1.at[i, "skycoord"]
                            .separation(subgroup1.at[j, "skycoord"])
                            .value
                            for j in subgroup1.index
                            for i in subgroup1.index
                        ]
                        
                        # If coordinates don't agree within tolerance
                        if all([x <= tolerance for x in check_on_coordinates]) is False:
                            # For main_id, check if flags are changing compared to previous check
                            if keyword == "main_id":
                                if (len(list(set(group.binary_mismatch_flag.values))) > 1) or (
                                    list(set(group.binary_mismatch_flag.values))[0] != 1
                                ):
                                    f.write(
                                        "WARNING: Flags are changing here compared to previous check. Previous value of binary_mismatch_flag was: "
                                        + ','.join([str(l) for l in (set(group.binary_mismatch_flag.values))])
                                        + ". "
                                    )

                            # Flag for failed check on coordinates
                            self.data.loc[group.index, "binary_mismatch_flag"] = 1
                            
                            # Prepare warning message for coordinate disagreement
                            warning = (
                                "WARNING: coordinate agreement exceeds tolerance. Maximum difference: "
                                + str(max(check_on_coordinates))
                                + " (binary_mismatch_flag = 1) . Please check this system:\n"
                                + group[
                                    [
                                        "name",
                                        keyword,
                                        "binary",
                                        "letter",
                                        "catalog",
                                        "ra",
                                        "dec",
                                    ]
                                ].to_string()
                            )
                        
                        # Write resolution to log file
                        f.write(
                            "Only S-type and null in the system for "
                            + key
                            + " "
                            + letter
                            + ". New binary value: S-type. "
                            + warning
                            + "\n\n"
                        )
                # the non-null values are in agreement
                elif len(set(subgroup2.binary)) == 1:
                    # if len(subgroup1)==0: #there is no S-type or null at all. All is well.
                    # If there's only one non-null binary value in subgroup2
                    if len(subgroup1) > 0:
                        # If there are S-type or null binaries in subgroup1, replace them with the non-null binary value
                        self.data.loc[subgroup1.index, "binary"] = list(set(subgroup2.binary))[0]
                        counter += 1

                        warning = ""
                        # Check the angular separation between coordinates in subgroup1 and subgroup2
                        check_on_coordinates = [
                            subgroup1.at[i, "skycoord"].separation(subgroup2.at[j, "skycoord"]).value
                            for j in subgroup2.index
                            for i in subgroup1.index
                        ]
                        if all([x <= tolerance for x in check_on_coordinates]) is False:
                            # If coordinates don't agree within tolerance, set a flag
                            if keyword == "main_id":
                                # Check if flags are changing compared to previous check
                                if (len(list(set(group.binary_mismatch_flag.values))) > 1) or (
                                    list(set(group.binary_mismatch_flag.values))[0] != 1
                                ):
                                    f.write(
                                        "WARNING: Flags are changing here compared to previous check. Previous value of binary_mismatch_flag was: "
                                        + ','.join([str(l) for l in (set(group.binary_mismatch_flag.values))])
                                        + ". "
                                    )

                            # Set binary_mismatch_flag to 1 for failed check on coordinates
                            self.data.loc[group.index, "binary_mismatch_flag"] = 1
                            # Prepare warning message for coordinate disagreement
                            warning = (
                                "WARNING: coordinate agreement exceeds tolerance. Maximum difference: "
                                + str(max(check_on_coordinates))
                                + " (binary_mismatch_flag = 1). Please check this system:\n"
                                + group[
                                    [
                                        "name",
                                        keyword,
                                        "binary",
                                        "letter",
                                        "catalog",
                                        "ra",
                                        "dec",
                                    ]
                                ].to_string()
                            )
                        # Write the result to the log file
                        f.write(
                            "Fixed S-type or null binary for "
                            + key
                            + " "
                            + letter
                            + ". New binary value: "
                            + str(list(set(subgroup2.binary))[0])
                            + ". "
                            + warning
                            + "\n\n"
                        )

                else:
                    # If there are multiple non-null binary values, this is a complex system
                    if keyword == "main_id":
                        # Check if flags are changing compared to previous check
                        if (len(list(set(group.binary_mismatch_flag.values))) > 1) or (
                            list(set(group.binary_mismatch_flag.values))[0] != 2
                        ):
                            f.write(
                                "WARNING: Flags are changing here compared to previous check. Previous value of binary_mismatch_flag was: "
                                +','.join([str(l) for l in (set(group.binary_mismatch_flag.values))])
                                + ". "
                            )

                    # Set binary_mismatch_flag to 2 for complex systems
                    self.data.loc[group.index, "binary_mismatch_flag"] = 2

                    # Write warning about complex system to log file
                    f.write(
                        "WARNING: Either a complex system or a mismatch in the value of binary (binary_mismatch_flag = 2). Please check this system:  \n"
                        + group[
                            [
                                "name",
                                keyword,
                                "binary",
                                "letter",
                                "catalog",
                            ]
                        ].to_string()
                        + "\n\n"
                    )
                    # if len(subgroup1)==0: #there is no S-type or null at all. All is well.

                    if len(subgroup1) > 0:
                        # If there's at least one S-type or null binary, try to replace it based on coordinates
                        for i in subgroup1.index:
                            for j in subgroup2.index:
                                # Calculate angular separation between each pair of coordinates
                                subgroup2.at[j, "angsep"] = (
                                    subgroup1.at[i, "skycoord"]
                                    .separation(subgroup2.at[j, "skycoord"])
                                    .value
                                )
                            # Filter subgroup2 to only include entries within 1 arcsecond
                            sub = subgroup2[subgroup2.angsep <= tolerance]
                            # Select the entry with the minimum angular separation
                            sub = sub[
                                sub.angsep == min(sub.angsep)
                            ]  # minimum angular separation from the unknown source

                            # Update the binary value for the S-type or null entry
                            self.data.loc[subgroup1.index, "binary"] = list(
                                set(sub.binary)
                            )[0]
                            counter += 1

                            # Log the change
                            f.write( "Fixed binary for complex system "
                                + key
                                + letter
                                + " based on angular separation. New binary value: "
                                + str(list(set(sub.binary))[0])
                                + ".\n\n")
                            

        # Log potential binaries that weren't treated automatically
        f.write(
            "****"
            + keyword
            + " POTENTIAL BINARIES NOT TREATED HERE. They should be treated manually in replacements.ini ****\n"
        )
        for i in self.data.index:
            # Check if the keyword ends with a binary indicator (A, B, C, N, or S)
            if (
                not str(re.search(r"([\s\d][ABCNS])$", self.data.at[i, keyword]))
                == "None"
            ):
                # Check if the binary indicator doesn't match the 'binary' column
                if (
                    not self.data.at[i, keyword][-1:].strip()
                    == self.data.at[i, "binary"]
                ):
                    # Log the mismatch
                    f.write(
                        "MISSED POTENTIAL BINARY Key:"
                        + str(self.data.at[i, keyword])
                        + " name: "
                        + str(self.data.at[i, "name"])
                        + " binary: "
                        + str(self.data.at[i, "binary"])
                        + " catalog:"
                        + str(self.data.at[i, "catalog"])
                        + ".\n"
                    )

        f.close()
        # Log summary of changes
        logging.info(
            "Checked potential binaries to be manually corrected. It happens "
            + str(counter)
            + " times."
            f"Checked potential binaries to be manually corrected. It happens {counter} times."
        )
        logging.info(
            "Automatic correction results: "
            + str(self.data.binary_mismatch_flag.value_counts())
        )

    def prepare_columns_for_mainid_search(self) -> None:
        """
        Prepare columns for the search of the main identifier.

        This function prepares various columns including "hostbinary", "aliasbinary", "main_id", "list_id",
        "main_id_ra", "main_id_dec", "angular_separation", and "main_id_provenance" for the search of the main identifier.

        :param self: The instance of the Emc class.
        :type self: Emc
        :return: None
        :rtype: None
        """
        # Create a 'hostbinary' column by combining 'host' and 'binary' information
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

        # Create an 'aliasbinary' column by combining 'alias' and 'binary' information
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

        # Create a 'hostbinary2' column (similar to 'hostbinary' but without spaces)
        self.data["hostbinary2"] = self.data["host"].astype(str) + self.data[
            "binary"
        ].astype(str).replace("nan", "").replace("Rogue", "").replace("S-type", "")
        self.data["hostbinary2"] = self.data.hostbinary2.str.rstrip()

        # Create an 'aliasbinary2' column (similar to 'aliasbinary' but without spaces)
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

        # Initialize columns for main identifier search
        self.data["main_id"] = ""
        self.data["list_id"] = ""
        self.data["main_id_ra"] = np.nan
        self.data["main_id_dec"] = np.nan

        self.data["angular_separation"] = ""
        self.data["angsep"] = -1.0

        self.data["main_id_provenance"] = ""
       
       
    def fill_mainid_provenance_column(self, keyword: str) -> None:
        """
        Fills the 'main_id_provenance' column with the provided keyword if 'main_id_provenance' is empty and
        'main_id' is not empty for each relevant index.

        This function is used to track the source of the main identifier for each entry in the catalog.
        It only updates entries where the main_id_provenance is currently empty but a main_id exists,
        avoiding overwriting any existing provenance information.

        :param self: The instance of the Emc class.
        :type self: Emc
        :param keyword: The keyword to fill the 'main_id_provenance' column with.
        :type keyword: str
        :return: None
        :rtype: None
        """
        # Iterate through indexes where main_id_provenance is empty
        for ind in self.data[self.data.main_id_provenance == ""].index:
            # Check if main_id is not empty for this index
            if self.data.at[ind, "main_id"] != "":
                # Fill main_id_provenance with the provided keyword
                self.data.at[ind, "main_id_provenance"] = keyword
        

    def simbad_list_host_search(self, typed_id: str) -> None:
        """
        Searches for host stars in SIMBAD using the specified column.

        This function takes a column name as an argument and searches for the host star in that
        column in SIMBAD. It then fills in the main_id, list_id, main_id_ra, and main_id_dec columns
        with information from SIMBAD if it finds a match.

        :param self: The instance of the Emc class.
        :type self: Emc
        :param typed_id: The name of the column that contains the host star to search for (host or hostbinary)
        :type typed_id: str
        :return: None
        :rtype: None
        """
        # Get unique host names, excluding those with non-ASCII characters
        list_of_hosts = self.data[self.data.main_id == ""][[typed_id]].drop_duplicates()
        list_of_hosts[typed_id] = list_of_hosts.loc[
            list_of_hosts[typed_id].str.findall(r"[^\x00-\x7F]+").str.len() == 0,
            typed_id,
        ]

        # Set up SIMBAD TAP service
        service = pyvo.dal.TAPService("http://simbad.cds.unistra.fr/simbad/sim-tap")

        # Convert list of hosts to astropy Table
        t2 = Table.from_pandas(list_of_hosts)

        # Construct and execute SIMBAD query
        query = (
            "SELECT t.*, basic.main_id, basic.ra as ra_2,basic.dec as dec_2, ids.ids as ids FROM TAP_UPLOAD.tab as t LEFT OUTER JOIN ident ON ident.id = t."
            + typed_id
            + " LEFT OUTER JOIN basic ON ident.oidref = basic.oid LEFT OUTER JOIN ids ON basic.oid = ids.oidref"
        )
        table = Utils.perform_query(service, query, uploads_dict={"tab": t2})

        # Log query results
        logging.info(
            "List of unique star names "
            + str(len(list_of_hosts))
            + " of which successful SIMBAD queries "
            + str(len(table))
        )

        # Update main catalog with SIMBAD results
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

        # Fill NaN values in relevant columns
        self.data.main_id = self.data.main_id.fillna("")
        self.data.list_id = self.data.list_id.fillna("")
        self.data.main_id_ra = self.data.main_id_ra.fillna(np.nan)
        self.data.main_id_dec = self.data.main_id_dec.fillna(np.nan)

    def simbad_list_alias_search(self, column: str) -> None:
        """
        Searches for the main ID of each object in the specified column using SIMBAD.

        This function performs the following steps:
        1. Creates a DataFrame of aliases from the specified column
        2. Queries SIMBAD for each alias
        3. Updates the main dataframe with the SIMBAD information

        :param self: The instance of the Emc class.
        :type self: Emc
        :param column: The name of the column that contains the host star aliases to search for (e.g., 'alias' or 'aliasbinary')
        :type column: str
        :return: None
        """

        # Initialize DataFrame to store aliases
        alias_df = pd.DataFrame(columns=["ind", column])
        self.data[column] = self.data[column].fillna("")

        # Iterate through rows where main_id is empty
        for i in self.data[self.data.main_id == ""].index:
            if len(self.data.at[i, column].replace("nan", "")) > 0:
                # Split aliases into a list and remove duplicates
                list_of_aliases = pd.DataFrame(
                    self.data.at[i, column].split(","), columns=[column]
                ).drop_duplicates()
                # Filter out non-ASCII characters to avoid conflicts with pyvo
                cleaned_list_of_aliases = pd.DataFrame(columns=[column])
                cleaned_list_of_aliases[column] = list_of_aliases.loc[
                list_of_aliases[column].str.findall(r"[^\x00-\x7F]+").str.len() == 0,
                column,
                ]
                cleaned_list_of_aliases["ind"] = i
                alias_df = pd.concat([alias_df, cleaned_list_of_aliases])

        # Set up SIMBAD TAP service
        service = pyvo.dal.TAPService("http://simbad.cds.unistra.fr/simbad/sim-tap")
        t2 = Table.from_pandas(alias_df)

        # Construct and execute SIMBAD query
        query = (
            "SELECT t.*, basic.main_id, basic.ra as ra_2,basic.dec as dec_2, ids.ids FROM TAP_UPLOAD.tab as t LEFT OUTER JOIN ident ON ident.id = t."
            + column
            + " LEFT OUTER JOIN basic ON ident.oidref = basic.oid LEFT OUTER JOIN ids ON basic.oid = ids.oidref"
        )
        table = Utils.perform_query(service, query, uploads_dict={"tab": t2})

        # Remove duplicate results
        table = table.drop_duplicates(["ind", "main_id", "ra_2", "dec_2", "ids"])

        # Process SIMBAD results and update main dataframe
        for i in table["ind"].unique():
            subtable = table[table.ind == i]

            if len(subtable) != 0:
            # Log a warning if multiple aliases are not in agreement
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

                # Update main dataframe with SIMBAD information
                self.data.at[int(i), "main_id_ra"] = float(subtable["ra_2"].values[0])
                self.data.at[int(i), "main_id_dec"] = float(subtable["dec_2"].values[0])
                self.data.at[int(i), "main_id"] = subtable["main_id"].values[0]
                self.data.at[int(i), "list_id"] = subtable["ids"].values[0].replace("|", ",")
                self.data.at[int(i), "angsep"] = 0.0

        self.data.main_id = self.data.main_id.fillna("")
        self.data.list_id = self.data.list_id.fillna("")
        self.data.main_id_ra = self.data.main_id_ra.fillna(np.nan)
        self.data.main_id_dec = self.data.main_id_dec.fillna(np.nan)

    def get_host_info_from_simbad(self) -> None:
        """
        The get_host_info_from_simbad function takes the dataframe and extracts all unique host star names.
        It then queries SIMBAD for each of these names, and returns a table with the main ID, alias IDs, RA and DEC.
        The function merges this table with the original dataframe on host name (left join). If there are
        still rows missing main_id values in the merged table, it will query SIMBAD again using all aliases from those rows.

        :param self: The instance of the Emc class.
        :type self: Emc
        :return: None
        :rtype: None
        """

        # Search SIMBAD using host name and binary information
        logging.info("HOST+ +BINARY Simbad Check")
        self.simbad_list_host_search("hostbinary")
        logging.info(
            "Rows still missing main_id after host search "
            + str(len(self.data[self.data.main_id == ""]))
        )

        # Search SIMBAD using alias and binary information
        logging.info("ALIAS+ +BINARY Simbad Check")
        self.simbad_list_alias_search("aliasbinary")
        logging.info(
            "Rows still missing main_id after alias search "
            + str(len(self.data[self.data.main_id == ""]))
        )

        # Search SIMBAD using host name and binary information (without space)
        logging.info("HOST+BINARY Simbad Check")
        self.simbad_list_host_search("hostbinary2")
        logging.info(
            "Rows still missing main_id after host search "
            + str(len(self.data[self.data.main_id == ""]))
        )

        # Search SIMBAD using alias and binary information (without space)
        logging.info("ALIAS+BINARY Simbad Check")
        self.simbad_list_alias_search("aliasbinary2")
        logging.info(
            "Rows still missing main_id after alias search "
            + str(len(self.data[self.data.main_id == ""]))
        )

        # Search SIMBAD using only host name
        logging.info("PURE HOST Simbad Check")
        self.simbad_list_host_search("host")
        logging.info(
            "Rows still missing main_id after host search "
            + str(len(self.data[self.data.main_id == ""]))
        )

        # Search SIMBAD using only alias
        logging.info("PURE ALIAS Simbad Check")
        self.simbad_list_alias_search("alias")

        # Log results
        logging.info(
            "Rows still missing main_id after alias search "
            + str(len(self.data[self.data.main_id == ""]))
        )

        # Replace '|' with ',' in list_id column
        self.data["list_id"] = self.data["list_id"].apply(
            lambda x: str(x).replace("|", ",")
        )

        # Update main_id_provenance column with 'SIMBAD'
        self.fill_mainid_provenance_column("SIMBAD")

        
    def get_coordinates_from_simbad(self, tolerance: float = 1 / 3600) -> None:
        """
        The get_coordinates_from_simbad function prepares a query for SIMBAD, executes the query and then merges
         the results with the original dataframe.

        :param self: The instance of the Emc class.
        :type self: Emc
        :param tolerance: The tolerance for the query in degrees (default is 1 arcsecond)
        :type tolerance: float
        :return: None
        :rtype: None
        """

        # Set up SIMBAD TAP service
        service = pyvo.dal.TAPService("http://simbad.cds.unistra.fr/simbad/sim-tap")

        # Prepare data for SIMBAD query: select rows without main_id and relevant columns
        t2 = Table.from_pandas(
            self.data[self.data.main_id == ""][["hostbinary", "ra", "dec"]]
        )

        # Construct SIMBAD query to get basic information based on coordinates
        query = (
            "SELECT basic.main_id, basic.dec as dec_2,basic.ra as ra_2, basic.otype as type, t.hostbinary, t.ra, t.dec FROM basic JOIN TAP_UPLOAD.tab AS t on 1=CONTAINS(POINT('ICRS',basic.ra, basic.dec), CIRCLE('ICRS', t.ra, t.dec,"
            + str(tolerance)
            + ")) "
        )
        # Execute the query
        table = Utils.perform_query(service, query, uploads_dict={"tab": t2})

        # Prepare data for second SIMBAD query to get additional identifiers
        t2 = Table.from_pandas(table)
        query = "SELECT t.*, ids.ids as ids FROM TAP_UPLOAD.tab as t LEFT OUTER JOIN ident ON ident.id = t.main_id LEFT OUTER JOIN basic ON ident.oidref = basic.oid LEFT OUTER JOIN ids ON basic.oid = ids.oidref"
        # Execute the second query
        table = Utils.perform_query(service, query, uploads_dict={"tab": t2})

        # Calculate angular separation between input and SIMBAD coordinates
        table = Utils.calculate_angsep(table)

        # Update main dataframe with SIMBAD results
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

        # Fill NaN values in relevant columns
        self.data.main_id = self.data.main_id.fillna("")
        self.data.list_id = self.data.list_id.fillna("")
        self.data.main_id_ra = self.data.main_id_ra.fillna(np.nan)
        self.data.main_id_dec = self.data.main_id_dec.fillna(np.nan)

        # Log results
        logging.info(
            "After coordinate check on SIMBAD at tolerance "
            + str(tolerance * 3600)
            + " arcsec, the residuals are: "
            + str(self.data[self.data.main_id == ""].shape[0])
            + ". Maximum angular separation: "
            + str(max(self.data[self.data.angsep == self.data.angsep].angsep))
        )

        # Update main_id_provenance column with 'SIMBADCOORD'
        self.fill_mainid_provenance_column("SIMBADCOORD")

    def get_host_info_from_tic(self) -> None:
        """
        Retrieves host information from the TIC (TESS Input Catalog) for hosts with TIC identifiers.

        This function performs the following steps:
        1. Extracts unique host star names that are TIC identifiers.
        2. Queries the TIC for each of these names.
        3. Merges the obtained information with the original dataframe.
        :param self: The instance of the Emc class.
        :type self: Emc
        :return: None
        :rtype: None
        """

        logging.info("TIC host check")

        # Extract unique host names that are TIC identifiers
        list_of_hosts = (
            self.data[self.data.main_id == ""][["host"]].drop_duplicates().dropna()
        )
        list_of_hosts = list_of_hosts[list_of_hosts.host.str.contains("TIC")]

        # Remove any non-ASCII characters from host names
        list_of_hosts["host"] = list_of_hosts.loc[
            list_of_hosts["host"].str.findall(r"[^\x00-\x7F]+").str.len() == 0, "host"
        ]

        # Clean up TIC identifiers and convert to integer
        list_of_hosts["host"] = (
            list_of_hosts["host"].str.replace("TIC ", "").str.replace("TIC-", "")
        ).astype(int)

        # Set socket timeout
        timeout = 100000
        socket.setdefaulttimeout(timeout)

        # Set up TAP service for querying TIC
        service = pyvo.dal.TAPService("http://TAPVizieR.cds.unistra.fr/TAPVizieR/tap/")

        # Prepare data for query
        t2 = Table.from_pandas(pd.DataFrame(list_of_hosts["host"]))

        # Construct query to retrieve TIC information
        query = 'SELECT tc.*, RAJ2000 as ra_2, DEJ2000 as dec_2, GAIA, UCAC4, "2MASS", WISEA, TIC, KIC, HIP, TYC  FROM "IV/39/tic82" AS db JOIN TAP_UPLOAD.t1 AS tc ON db.TIC = tc.host'

        # Execute query
        table = Utils.perform_query(service, query, uploads_dict={"t1": t2})

        # Remove duplicates from results
        table = table.drop_duplicates()

        # Log query results
        logging.info(
            "List of unique star names with a TIC host "
            + str(len(list_of_hosts))
            + " of which successful TIC queries "
            + str(len(table))
        )

        # Update main dataframe with TIC information
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

        # Fill NaN values in main_id, list_id, main_id_ra, and main_id_dec columns with empty strings
        self.data.main_id = self.data.main_id.fillna("")
        self.data.list_id = self.data.list_id.fillna("")
        self.data.main_id_ra = self.data.main_id_ra.fillna("")
        self.data.main_id_dec = self.data.main_id_dec.fillna("")

        # Log results
        logging.info(
            "Rows still missing main_id after TIC host search "
            + str(len(self.data[self.data.main_id == ""]))
        )

        logging.info("TIC alias check")

        # Extract rows where main_id is empty and alias contains 'TIC'
        alias_df = self.data[self.data.main_id == ""]
        alias_df = alias_df[alias_df.alias.str.contains("TIC")]

        # Extract TIC identifiers from the alias column
        for ind in alias_df.index:
            tic_alias = alias_df.at[ind, "alias"].split(",")
            alias_df.at[ind, "tic_alias"] = [x for x in tic_alias if "TIC" in x][0]

        # Clean up TIC identifiers and convert to integer
        alias_df["tic_alias"] = (
            alias_df["tic_alias"].str.replace("TIC ", "").astype(int)
        )

        # Keep only 'host' and 'tic_alias' columns
        alias_df = alias_df[["host", "tic_alias"]]

        # Convert DataFrame to Astropy Table for use in TAP query
        t2 = Table.from_pandas(alias_df[["host", "tic_alias"]])

        # Construct query to retrieve TIC information
        # This query joins the uploaded table (t1) with the TIC catalog (IV/39/tic82)
        # based on the TIC identifier
        query = 'SELECT tc.*, RAJ2000 as ra_2, DEJ2000 as dec_2, GAIA, UCAC4, "2MASS", WISEA, TIC, KIC, HIP, TYC  FROM "IV/39/tic82" AS db JOIN TAP_UPLOAD.t1 AS tc ON db.TIC = tc.tic_alias'

        # Execute the query
        table = Utils.perform_query(service, query, uploads_dict={"t1": t2})
        
        # Log the number of successful TIC queries
        logging.info(
            "List of unique star names with a TIC alias "
            + str(len(list_of_hosts))
            + " of which successful TIC queries "
            + str(len(table))
        )

        # Update the main dataframe with TIC information for each host
        for host in table["host"]:
            # Update right ascension
            self.data.loc[self.data["host"] == host, "main_id_ra"] = float(
                table.loc[table["host"] == host, "ra_2"].values[0]
            )
            # Update declination
            self.data.loc[self.data["host"] == host, "main_id_dec"] = float(
                table.loc[table["host"] == host, "dec_2"].values[0]
            )
            # Update main identifier
            self.data.loc[self.data["host"] == host, "main_id"] = table.loc[
                table["host"] == host, "main_id"
            ].values[0]
            # Update list of identifiers, replacing '|' with ','
            self.data.loc[self.data["host"] == host, "list_id"] = (
                table.loc[table["host"] == host, "ids"].values[0].replace("|", ",")
            )
            # Set angular separation to 0 (as this is a direct match)
            self.data.loc[self.data["host"] == host, "angsep"] = 0.0

        # Fill NaN values in main_id, list_id, main_id_ra, and main_id_dec columns with empty strings
        self.data.main_id = self.data.main_id.fillna("")
        self.data.list_id = self.data.list_id.fillna("")
        self.data.main_id_ra = self.data.main_id_ra.fillna("")
        self.data.main_id_dec = self.data.main_id_dec.fillna("")

        logging.info(
            "Rows still missing main_id after TIC alias search "
            + str(len(self.data[self.data.main_id == ""]))
        )

        # Update the main_id_provenance column with 'TIC' for entries updated in this function
        self.fill_mainid_provenance_column("TIC")


    def get_coordinates_from_tic(self, tolerance: float = 1.0 / 3600.0):
        """
        Retrieves coordinates from the TESS Input Catalog (TIC) for objects without main IDs.

        This function performs the following steps:
        1. Prepares a query for the TIC using objects without main IDs.
        2. Executes the query to retrieve matching TIC entries.
        3. Merges the results with the original dataframe.
        4. Updates the main_id, coordinates, and other relevant fields for matched objects.
        """


        # Set up the TAP service for querying the VizieR TIC catalog
        service = pyvo.dal.TAPService("http://TAPVizieR.cds.unistra.fr/TAPVizieR/tap/")

        # Prepare data for the query: select rows without main_id and relevant columns
        t2 = Table.from_pandas(
            self.data[self.data.main_id == ""][["hostbinary", "ra", "dec"]]
        )

        # Construct the ADQL query to retrieve TIC information
        # This query joins the uploaded table (t) with the TIC catalog (IV/39/tic82)
        # It uses the CONTAINS and CIRCLE functions to match coordinates within the specified tolerance
        query = (
            """SELECT t.*, RAJ2000 as ra_2, DEJ2000 as dec_2, GAIA, UCAC4, "2MASS", WISEA, TIC, KIC, HIP, TYC FROM "IV/39/tic82" JOIN TAP_UPLOAD.tab AS t on 1=CONTAINS(POINT('ICRS',RAJ2000, DEJ2000), CIRCLE('ICRS', t.ra, t.dec,"""
            + str(tolerance)
            + """)) """
        )

        # Execute the query using the utility function
        table = Utils.perform_query(service, query, uploads_dict={"tab": t2})
        table = table.drop_duplicates()
        table = Utils.calculate_angsep(table)

        for host in table["hostbinary"]:
            self.data.loc[self.data["hostbinary"] == host, "main_id_ra"] = float(
                table.loc[table["hostbinary"] == host, "ra_2"].values[0]
            )
            # Update declination
            self.data.loc[self.data["hostbinary"] == host, "main_id_dec"] = float(
                table.loc[table["hostbinary"] == host, "dec_2"].values[0]
            )
            # Update main identifier
            self.data.loc[self.data["hostbinary"] == host, "main_id"] = table.loc[
                table["hostbinary"] == host, "main_id"
            ].values[0]
            self.data.loc[self.data["hostbinary"] == host, "list_id"] = (
                table.loc[table["hostbinary"] == host, "ids"]
                .replace("|", ",")
            )
            # Update angular separation
            self.data.loc[self.data["hostbinary"] == host, "angsep"] = float(
                table.loc[table["hostbinary"] == host, "angsep"].values[0]
            )

        # Fill NaN values in main_id, list_id, main_id_ra, and main_id_dec columns with empty strings
        self.data.main_id = self.data.main_id.fillna("")
        self.data.list_id = self.data.list_id.fillna("")
        self.data.main_id_ra = self.data.main_id_ra.fillna("")
        self.data.main_id_dec = self.data.main_id_dec.fillna("")

        # Log results
        logging.info(
            "After coordinate check on TIC "
            "at tolerance "
            + str(tolerance * 3600)
            + " arcsec, the residuals are: "
            + str(self.data[self.data.main_id == ""].shape[0])
            + ". Maximum angular separation: "
            + str(max(self.data[self.data.angsep == self.data.angsep].angsep))
        )

        # Update the main_id_provenance column with 'TICCOORD' for entries updated in this function
        self.fill_mainid_provenance_column("TICCOORD")

        
    def check_coordinates(self, tolerance: float = 1 / 3600) -> None:
        """
        Check for mismatches in the RA and DEC coordinates of a given host.

        This function is used for targets that cannot rely on SIMBAD or TIC MAIN_ID because the query was unsuccessful.
        It groups all entries with the same host name, then checks if any of those entries have
        a RA or DEC that is more than a given tolerance away from the mode value for that group.
        If so, it logs information about those mismatched values to a file called "check_coordinates.txt".

        :param self: The instance of the Emc class.
        :type self: Emc
        :param tolerance: The tolerance for coordinate mismatch, in degrees. Default is 1 arcsecond (1/3600 degrees).
        :type tolerance: float
        :return: None
        :rtype: None
        """

        # Initialize counters for mismatched RA and DEC
        countra = 0
        countdec = 0

        # Initialize a new column for coordinate mismatches
        self.data["coordinate_mismatch"] = ""

        # Round the tolerance to 6 decimal places
        tolerance = round(tolerance, 6)

        # Open a log file to record coordinate mismatches
        f = open("Logs/check_coordinates.txt", "a")

        # Group the data by host and binary for entries without a main_id
        for (host, binary), group in self.data[self.data.main_id == ""].groupby(
            ["host", "binary"]
        ):
            # Check if there's more than one entry in the group
            if len(group) > 1:
                # Calculate the mode of RA and DEC for the group
                ra = mode(list(round(group.ra, 6)))
                dec = mode(list(round(group.dec, 6)))
                mismatch_string = ""

                if (abs(round(group["ra"], 6) - ra) > tolerance).any():
                    countra = countra + 1
                    with pd.option_context("display.max_columns", 2000):
                        f.write(
                            "*** MISMATCH ON RA at tolerance "
                            + str(tolerance)
                            + " *** \n"
                            + group[
                                [
                                    "name",
                                    "host",
                                    "binary",
                                    "letter",
                                    "catalog",
                                    "ra",
                                ]
                            ].to_string()
                            + "\n"
                        )
                    mismatch_string = "RA"

                # Check for DEC mismatches
                if (abs(round(group["dec"], 6) - dec) > tolerance).any():
                    countdec += 1
                    # Log the DEC mismatch
                    with pd.option_context("display.max_columns", 2000):
                        f.write(
                            "*** MISMATCH ON DEC at tolerance "
                            + str(tolerance)
                            + " *** \n"
                            + group[
                                [
                                    "name",
                                    "host",
                                    "binary",
                                    "letter",
                                    "catalog",
                                    "dec",
                                ]
                            ].to_string()
                            + "\n"
                        )
                    mismatch_string += "DEC"

                # Update the coordinate_mismatch column for the group
                self.data.loc[group.index, "coordinate_mismatch"] = mismatch_string

        # Close the log file
        f.close()
        logging.info("Found " + str(countra) + " mismatched RA.")
        logging.info("Found " + str(countdec) + " mismatched DEC.")

        # Log the value counts of coordinate mismatches
        logging.info(self.data.coordinate_mismatch.value_counts())


    def replace_old_new_identifier(
        self, identifier: str, new_identifier: str, binary: str = None
    ) -> str:
        """
        The replace_old_new_identifier function replaces the old identifier with the new identifier in the dataframe. It also adds additional aliases to the main_id_aliases. If binary is not None, it also replaces the old binary with the new binary in the dataframe as specified by the calling function.

        :param self: The instance of the Emc class.
        :type self: Emc
        :param identifier: The old identifier
        :type identifier: str
        :param new_identifier: The new identifier
        :type new_identifier: str
        :param binary: The binary string
        :type binary: str
        :return: The explanation string for logging purposes
        :rtype: str

        """
        output_string = ""
        service = pyvo.dal.TAPService("http://simbad.cds.unistra.fr/simbad/sim-tap")

        # Construct the ADQL query to retrieve information for the new identifier
        query = (
            """SELECT  basic.main_id, basic.ra as ra_2,basic.dec as dec_2, ids.ids FROM ident JOIN basic ON ident.oidref = basic.oid LEFT OUTER JOIN ids ON basic.oid = ids.oidref  WHERE id = '"""
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
    
            # Check if there are existing list_ids and append new ones if necessary
            if (
                len(
                    self.data.loc[self.data.main_id == identifier, "list_id"]
                    .replace("", np.nan)
                    .dropna()
                    .unique()
                )
                > 0
            ):
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
                # If no existing list_ids, use SIMBAD data
                self.data.loc[self.data.main_id == identifier, "list_id"] = table.loc[
                    0, "ids"
                ].replace("|", ",")
            output_string = (
                "MAINID can be corrected " + identifier + " to " + new_identifier + ". "
            )

            if binary is not None:
                output_string = output_string + " Binary value: " + binary + "."
                binary_catalog = self.data.loc[
                    self.data.main_id == identifier, "binary"
                ].copy()
                if (binary_catalog.replace("S-type", "") == "").all():
                    self.data.loc[self.data.main_id == identifier, "binary"] = binary
                    output_string = (
                        output_string
                        + " Only S-type or null. Binary could be standardized."
                    )
                elif (binary_catalog != binary).any():
                    # Flag disagreement in binary values, no replacement possible.
                    output_string = (
                        output_string
                        + " Binary value is not in agreement, please check:\n"
                        + self.data.loc[
                            self.data.main_id == identifier,
                            ["name", "host", "binary", "catalog"],
                        ].to_string()
                    )
                elif (binary_catalog == binary).all():
                    # No changes needed if binary values already correct
                    output_string = output_string + " Already correct."
            else:
                output_string = output_string + "\n"
            # last thing to be changed since it changes the query
            self.data.loc[self.data.main_id == identifier, "main_id"] = table.at[
                0, "main_id"
            ]
            output_string = output_string
        elif len(table) == 0:
            # Log results
            output_string = (
                "Weird MAINID found: "
                + identifier
                + " but cannot be found when "
                + new_identifier
                + ". No replacement performed."
            )

        return output_string

    def polish_main_id(self) -> None:
        """
        Polish the main_id column in the data by removing planet/binary letters.

        This function iterates over the unique values in the main_id column of the data and performs the following operations:

        1. Check for planet letters in the main_id column. If a planet letter is found, it tries to look for the corresponding star in SIMBAD and replaces the main_id with the star's main_id.

        2. Check for binary letters in the main_id column. If a binary letter is found, it checks if the binary value is already in the binary column. It then replaces the main_id with the modified identifier.

        3. All of the above operations are logged in a text file named "Logs/polish_main_id.txt".

        :param self: The instance of the Emc class.
        :type self: Emc
        :return: None
        :rtype: None
        """

        counter = 0
        f = open("Logs/polish_main_id.txt", "a")
        f.write("***** CHECK FOR PLANET LETTER IN MAIN_ID *****\n")

        # Iterate through unique main_id values to check for planet letters
        for identifier in self.data.main_id.unique():
            # Check if the identifier ends with a planet letter (b-j)
            if len(re.findall("[\s\d][b-j]$", identifier)) > 0:
                # Remove the planet letter and any leading/trailing whitespace
                new_identifier = identifier[:-1].strip().replace("NAME ", "")
                counter += 1
                # Try to replace the old identifier with the new one
                output_string = self.replace_old_new_identifier(
                    identifier, new_identifier
                )
                f.write(output_string)

        f.write("\n***** CHECK FOR BINARY LETTER IN MAIN_ID *****\n")
        
        # Iterate through unique main_id values to check for binary letters
        for identifier in self.data.main_id.unique():
            # Check for circumbinary identifiers (ending with (AB))
            if len(re.findall(r"[\s\d](\(AB\))$", identifier)) > 0:
                counter += 1
                # Remove the (AB) suffix and any leading/trailing whitespace
                new_identifier = identifier[:-4].rstrip().replace("NAME ", "")
                output_string = self.replace_old_new_identifier(
                    identifier, new_identifier, "AB"
                )
                f.write(output_string + "\n")

            # Check for AB suffix without parentheses
            if len(re.findall(r"[\[a-z](AB)$|\s(AB)$|\d(AB)$]", identifier)) > 0:
                counter += 1
                # Remove the AB suffix
                new_identifier = identifier[:-2].rstrip()
                output_string = self.replace_old_new_identifier(
                    identifier, new_identifier, "AB"
                )
                f.write(output_string + "\n")

            # Check for regular binary identifiers (ending with A, B, C, S, or N)
            if len(re.findall("[\s\d][ABCSN]$", identifier)) > 0:
                counter += 1
                # Remove the binary letter
                new_identifier = identifier[:-1].strip()
                output_string = self.replace_old_new_identifier(
                    identifier, new_identifier, identifier[-1:]
                )
                f.write(output_string + "\n")

        f.close()
        # Log the number of changes made
        logging.info(
            "Removed planet/binary letter from main_id. It happens "
            + str(counter)
            + " times."
        )

        
    def fill_missing_main_id(self) -> None:
        """
        Fill missing values in main_id related columns with data from other columns.

        This function performs the following operations:
        1. Fills empty 'main_id_provenance' with values from 'catalog'.
        2. Fills empty 'main_id' with values from 'host'.
        3. Fills empty 'main_id_ra' with values from 'ra', converting to float.
        4. Fills empty 'main_id_dec' with values from 'dec', converting to float.
        5. Creates 'angular_separation' by concatenating 'catalog' and 'angsep'.

        :param self: The instance of the Emc class.
        :type self: Emc
        :return: None
        :rtype: None
        """

        # Fill empty 'main_id_provenance' with values from 'catalog'
        self.data["main_id_provenance"] = (
            self.data["main_id_provenance"]
            .replace("", np.nan)
            .fillna(self.data["catalog"])
        )

        # Fill empty 'main_id' with values from 'host'
        self.data["main_id"] = (
            self.data["main_id"].replace("", np.nan).fillna(self.data["host"])
        )

        # Fill empty 'main_id_ra' with values from 'ra', converting to float
        self.data["main_id_ra"] = (
            self.data["main_id_ra"].replace("", np.nan).fillna(self.data["ra"])
        ).astype(float)

        # Fill empty 'main_id_dec' with values from 'dec', converting to float
        self.data["main_id_dec"] = (
            self.data["main_id_dec"].replace("", np.nan).fillna(self.data["dec"])
        ).astype(float)
        
        # Create 'angular_separation' by concatenating 'catalog' and 'angsep'
        self.data["angular_separation"] = (
            self.data["catalog"] + ": " + self.data.angsep.astype(str)
        )

    def check_same_host_different_id(self) -> None:
        """
        Checks if there are instances where the same host has multiple SIMBAD main IDs.

        :param self: The instance of the Emc class.
        :type self: Emc
        :return: None
        :rtype: None
        """
        # Open a file to log the results
        f = open("Logs/post_main_id_query_checks.txt", "a")
        f.write("**************************************\n")
        f.write("**** CHECK SAME HOST DIFFERENT ID ****\n")
        f.write("**************************************\n")

        # Check for inconsistencies in hostbinary
        for host, group in self.data.groupby("hostbinary"):
            # If there's more than one unique main_id for a hostbinary, log it
            if len(group.main_id.drop_duplicates()) > 1:
                with pd.option_context("display.max_columns", 2000):
                    f.write("SAME HOST+BINARY DIFFERENT MAIN_ID\n")
                    f.write(
                        group[
                            ["hostbinary", "main_id", "binary", "catalog"]
                        ].to_string()
                        + "\n"
                    )

        # Check for inconsistencies in host
        for host, group in self.data.groupby("host"):
            # If there's more than one unique main_id for a host, log it
            if len(group.main_id.drop_duplicates()) > 1:
                with pd.option_context("display.max_columns", 2000):
                    f.write("SAME HOST DIFFERENT MAIN_ID\n")
                    f.write(
                        group[["host", "main_id", "binary", "catalog"]].to_string()
                        + "\n"
                    )

        # Log that the check has been completed
        logging.info("Checked if host is found under different main_ids.")
    
        # Close the log file
        f.close()

    def check_same_coords_different_id(self, tolerance: float = 1 / 3600) -> None:
        """
        The check_same_host_different_id function checks to see if there are any instances where the same host
        has multiple SIMBAD main IDs. This might happen in case of very close stars or binary stars. The user
        should check in Logs/post_main_id_query_checks.txt that the two main ids and coordinates are indeed
        different stars. Otherwise, they can force a replacement.

        :param self: The instance of the Emc class.
        :type self: Emc
        :return: None
        :rtype: None
        """
        # Open a file for logging the results of the checks
        f = open("Logs/post_main_id_query_checks.txt", "a")
        f.write("****************************************\n")
        f.write("**** CHECK SAME COORDS DIFFERENT ID ****\n")
        f.write("****************************************\n")

        # Convert RA and Dec to float type for numerical operations
        self.data.main_id_ra = self.data.main_id_ra.astype(float)
        self.data.main_id_dec = self.data.main_id_dec.astype(float)

        # Create SkyCoord objects for each entry to facilitate astronomical calculations
        self.data["skycoord"] = SkyCoord(
            ra=self.data.main_id_ra * u.deg, dec=self.data.main_id_dec * u.deg, unit="deg"
        )

        # Iterate through each entry in the dataset
        for i in self.data.index:
            # Extract RA and Dec for the current entry
            ra = self.data.at[i, "main_id_ra"]
            dec = self.data.at[i, "main_id_dec"]
    
            # Create a subset of data within a wider rectangle around the current coordinates
            sub = self.data.copy()
            sub = sub[sub.main_id_ra < (ra + 2 * tolerance)]
            sub = sub[sub.main_id_ra > (ra - 2 * tolerance)]
            sub = sub[sub.main_id_dec > (dec - 2 * tolerance)]
            sub = sub[sub.main_id_ra < (dec + 2 * tolerance)]
    
            # Calculate angular separation between the current entry and each entry in the subset
            for j in sub.index:
                sub.at[j, "angsep"] = (
                    self.data.at[i, "skycoord"].separation(sub.at[j, "skycoord"]).value
                )
            # Filter the subset to only include entries within the specified angular separation tolerance
            sub = sub[sub.angsep <= tolerance]

            # Check if there are multiple unique main_ids within the tolerance
            if len(sub.main_id.unique()) > 1:
                # If multiple main_ids are found, log this information
                with pd.option_context("display.max_columns", 2000):
                    f.write(
                        "FOUND SAME COORDINATES DIFFERENT MAINID\n"
                        + sub[
                            [
                                "host",
                                "main_id",
                                "binary",
                                "letter",
                                "catalog",
                                "angsep",
                                "main_id_provenance",
                            ]
                        ].to_string()
                        + "\n"
                    )
                    
        # Log that the check has been completed
        logging.info("Checked if same coordinates found in main_ids.")

        # Close the log file
        f.close()

    def group_by_list_id_check_main_id(self) -> None:
        """
        Groups the data by 'list_id' and checks for inconsistencies in 'main_id'.

        This function performs the following steps:
        1. Groups the data by the 'list_id' column.
        2. For each group, checks if the 'list_id' is not empty and if there are multiple unique 'main_id' values.
        3. If inconsistencies are found, it sets all 'main_id' values in the group to the first unique 'main_id'.
        4. Logs details of any inconsistencies found.

        :param self: The instance of the Emc class.
        :type self: Emc
        :return: None
        :rtype: None
        """

        # Open a file to log the results of the grouping check
        f = open("Logs/post_main_id_query_checks.txt", "a")
        f.write("****************************************\n")
        f.write("**** GROUP BY LIST_ID CHECK MAIN_ID ****\n")
        f.write("****************************************\n")

        count = 0
        # Group the data by list_id and iterate through each group
        for ids, group in self.data.groupby(by="list_id"):
            # Check if the list_id is not empty and there are multiple main_ids in the group
            if ids != "" and len(set(group.main_id)) > 1:
                # Update all rows with this list_id to have the same main_id (the first one in the group)
                self.data.loc[self.data.list_id == ids, "main_id"] = list(
                    group.main_id
                )[0]
                # Log the inconsistency
                with pd.option_context("display.max_columns", 2000):
                    f.write(
                        "*** SAME LIST_ID, DIFFERENT MAIN_ID *** \n"
                        + group[["catalog", "status", "letter", "main_id"]].to_string()
                        + "\n"
                    )
                # Increment the counter for inconsistencies found
                count = count + 1
        
        # Log the total number of inconsistencies found
        logging.info(
            "Planets that had a different main_id name but same SIMBAD alias: "
            + str(count)
        )
        # Close the log file
        f.close()

    def post_main_id_query_checks(
        self, tolerance: float = 1 / 3600
    ) -> None:  # pragma: no cover
        """
        Performs a series of checks after querying SIMBAD for main IDs.

        This function executes three main checks:
        1. Checks for instances where the same host has different main IDs.
        2. Checks for cases where the same coordinates (within a specified tolerance) have different main IDs.
        3. Checks for situations where the same list ID has different main IDs.

        These checks are crucial for identifying potential inconsistencies or errors in the catalog data,
        particularly related to the identification and matching of celestial objects across different sources.

        The results of these checks are logged in the file 'Logs/post_main_id_query_checks.txt' for further
        analysis and review.
        :param self: The instance of the Emc class.
        :type self: Emc
        :param tolerance: The angular separation tolerance in degrees for considering coordinates as the same.
                      Default is 1 arcsecond (1/3600 degrees).
        :type tolerance: float
        :return: None
        :rtype: None

        Note:
        - This function modifies the internal state of the Emc instance by potentially updating flags or log files.
        - The 'pragma: no cover' comment indicates that this function is excluded from code coverage checks,
        typically because it involves complex external interactions or logging that are hard to test automatically.
        """

        # Check for same host with different main IDs
        self.check_same_host_different_id()

        # Check for same coordinates with different main IDs
        self.check_same_coords_different_id(tolerance)

        # Check for same list ID with different main IDs
        self.group_by_list_id_check_main_id()

        
    def group_by_main_id_set_main_id_aliases(self) -> None:
        """
        Groups the dataframe by main_id and combines alias and list_id columns into a single main_id_aliases column.
        This function consolidates all identifiers for each unique main_id.
        :param self: The instance of the Emc class.
        :type self: Emc
        :return: None
        :rtype: None
        """
        # Initialize the main_id_aliases column with empty strings
        self.data["main_id_aliases"] = ""

        # Iterate over each unique main_id in the dataframe
        for host, group in self.data.groupby(by="main_id"):
        # Initialize an empty string to store all aliases
            main_id_aliases = ""

            # Concatenate all aliases from the 'alias' column
            for al in group.alias:
                main_id_aliases += "," + str(al)

            # Concatenate all identifiers from the 'list_id' column
            for al in group.list_id:
                main_id_aliases += "," + str(al)

            # Remove duplicates and unnecessary characters from main_id_aliases
            main_id_aliases = ",".join(
                [x for x in set(main_id_aliases.split(",")) if x]
            )
            main_id_aliases = main_id_aliases.replace("nan", "").replace(",,", ",")

        # Update the main_id_aliases column for all rows with the current main_id
        self.data.loc[self.data.main_id == host, "main_id_aliases"] = main_id_aliases

    def cleanup_catalog(self) -> None:
        """
        Cleans up the catalog by replacing 0 and infinity values with NaN for specific columns.

        This function iterates through a set of columns ('i', 'mass', 'msini', 'a', 'p', 'e')
        and their corresponding '_min' and '_max' errors. It replaces any values that are
        exactly 0 or infinity with NaN (Not a Number). This helps to ensure that these
        extreme values don't skew analyses or cause issues in later processing steps.
        :param self: The instance of the Emc class.
        :type self: Emc
        :return: None
        :rtype: None
        """
        # List of columns to clean up
        columns_to_clean = ["i", "mass", "msini", "a", "p", "e"]

        # Iterate through each column
        for col in columns_to_clean:
            # Replace 0 with NaN for min and max errors
            self.data.loc[self.data[col + "_min"] == 0, col + "_min"] = np.nan
            self.data.loc[self.data[col + "_max"] == 0, col + "_max"] = np.nan

            # Replace infinity with NaN for min and max errors
            self.data.loc[self.data[col + "_min"] == np.inf, col + "_min"] = np.nan
            self.data.loc[self.data[col + "_max"] == np.inf, col + "_max"] = np.nan

        # Log that the cleanup has been completed
        logging.info("Catalog cleared from zeroes and infinities.")


    def group_by_period_check_letter(self) -> None:
        """
        Check for inconsistencies in the letter column and attempt to fix them.

        This function performs the following steps:
        1. Groups the data by main_id and binary.
        2. For each group with multiple planets:
            a. Calculates an estimate of period (p) and semi-major axis (a).
            b. For each unique period (or semi-major axis if period is not available):
                - Checks for inconsistencies in the letter column.
                - Attempts to fix inconsistencies by standardizing the letter.
        3. Logs any inconsistencies and fixes to a file.

        :param self: An instance of the Emc class
        :type self: Emc
        :return: None
        :rtype: None
        """

        # Open log file
        f1 = open("Logs/group_by_period_check_letter.txt", "a")

        # Group by main_id and binary
        grouped_df = self.data.groupby(["main_id", "binary"], sort=True, as_index=False)
        f1.write("TOTAL NUMBER OF GROUPS: " + str(grouped_df.ngroups) + "\n")
        counter = 0

        # Iterate through each group
        for (
            mainid,
            binary,
        ), group in grouped_df:
            # Check if there are multiple planets in the system
            if len(group) > 1:
                # Calculate an estimate of p and a
                group = Utils.calculate_working_p_sma(group, tolerance=0.1)
                
                # Iterate through each unique value of p
                for pgroup in list(set(group.working_p)):
                    subgroup = group[group.working_p == pgroup]
                    warning = ""
                    
                    # If period is available
                    if pgroup != -1:
                        # Try to fix the letter if it is different (e.g. b and .01)
                        if len(list(set(subgroup.letter))) > 1:
                            warning = (
                                "INCONSISTENT LETTER FOR SAME PERIOD \n"
                                + subgroup[
                                    [
                                        "main_id",
                                        "binary",
                                        "letter",
                                        "catalog",
                                        "catalog_name",
                                        "p",
                                    ]
                                ].to_string()
                                + "\n\n"
                            )
                            # Fet the letter-like classification
                            adjusted_letter = [
                                l
                                for l in list(set(subgroup.letter.dropna().unique()))
                                if ".0" not in str(l)
                            ]
                            # If there is only one letter-like classification, fix the group to that letter
                            if len(adjusted_letter) == 1:
                                self.data.loc[
                                    subgroup.index, "letter"
                                ] = adjusted_letter[0]
                                warning = "FIXABLE " + warning

                            # If BD is in the letter, fix all the group to be BD
                            if "BD" in list(set(subgroup.letter.dropna().unique())):
                                self.data.loc[subgroup.index, "letter"] = "BD"
                                warning = "FORCED BD " + warning
                    else:
                        # No period, do the same with semi-major axis

                        # Iterate through each unique value of a
                        for agroup in list(set(subgroup.working_a)):
                            subsubgroup = subgroup[subgroup.working_a == agroup]
                            
                            # Try to fix the letter if it is different (e.g. b and .01)
                            if len(list(set(subsubgroup.letter))) > 1:
                                warning = (
                                    "INCONSISTENT LETTER FOR SAME SMA \n"
                                    + subsubgroup[
                                        [
                                            "main_id",
                                            "binary",
                                            "letter",
                                            "catalog",
                                            "catalog_name",
                                            "a",
                                        ]
                                    ].to_string()
                                    + "\n\n"
                                )
                                # Get the letter-like classification
                                adjusted_letter = [
                                    l
                                    for l in list(
                                        set(subsubgroup.letter.dropna().unique())
                                    )
                                    if ".0" not in str(l)
                                ]

                                # If there is only one letter-like classification, fix the group to that letter
                                if len(adjusted_letter) == 1:
                                    self.data.loc[
                                        subsubgroup.index, "letter"
                                    ] = adjusted_letter[0]
                                    warning = "FIXABLE " + warning
                                # If BD is in the letter, fix all the group to be BD
                                if "BD" in list(set(subgroup.letter.dropna().unique())):
                                    self.data.loc[subgroup.index, "letter"] = "BD"
                                    warning = "FORCED BD " + warning

                    # Write the warning to the log file
                    f1.write(warning)
        f1.close()

    @staticmethod
    def merge_into_single_entry(
        group: pd.DataFrame, mainid: str, binary: str, letter: str
    ) -> pd.DataFrame:
        """
        Merge multiple entries with the same main_id and letter into a single entry.

        This function combines information from different catalogs for a specific exoplanet,
        selecting the best available data and resolving conflicts. It performs the following tasks:

        1. Creates a new entry with the given main_id, binary, and letter.
        2. Selects the most common host name.
        3. Saves catalog-specific names (NASA, TOI, EPIC, EU, OEC).
        4. Selects the best measurement for various parameters (i, mass, msini, r, a, p, e)
        based on the smallest relative error.
        5. Determines the status of the exoplanet.
        6. Selects the earliest discovery year and combines discovery methods.
        7. Combines aliases.
        8. Sets various flags for mismatches and duplicates.
        9. Selects the best source for coordinates (main_id_ra, main_id_dec).
        10. Logs warnings for multiple main_id_provenance and duplicate entries.

        The final entry contains: the official SIMBAD ID and coordinates; the measurements
        that have the smallest relative error with the corresponding reference; the preferred
        name, the preferred status, the preferred binary letter (chosen as the most common
        in the group); year of discovery, method of discovery, and final list of aliases.
        The function then concatenates all of these entries together into a final catalog.

        :param group : A pandas DataFrame containing the duplicate occurrences.
        :type group: pd.DataFrame
        :param mainid: The main identifier of the group
        :type mainid: str
        :param binary: The binary identifier of the group
        :type binary: str
        :param letter: The letter identifier of the group
        :type letter: str
        :return: A pandas Series corresponding to the merged single entry.
        :rtype: pd.DataFrame
        """
        # Open a file to log any issues during the merging process
        f = open("Logs/merge_into_single_entry.txt", "a")

        # Initialize a new DataFrame 'entry' with the main_id
        entry = pd.DataFrame([mainid], columns=["main_id"])

        # Add binary and letter information to the entry
        entry["binary"] = binary
        entry["letter"] = letter

        # Select the most common host name. If there are multiple most common, choose the first alphabetically
        entry["host"] = sorted(list(set(group["host"].mode())))[0]

        # Initialize catalog-specific name fields
        entry["nasa_name"] = ""
        entry["toi_name"] = ""
        entry["epic_name"] = ""
        entry["eu_name"] = ""
        entry["oec_name"] = ""

        # Iterate through each catalog present in the group
        for catalog in group["catalog"]:
            # For each catalog, retrieve the corresponding catalog_name and store it in the appropriate field
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
        
        # Select best measurement for various planetary parameters
        params = [
            ["i_url", "i", "i_min", "i_max", "IREL"],
            ["mass_url", "mass", "mass_min", "mass_max", "MASSREL"],
            ["msini_url", "msini", "msini_min", "msini_max", "MSINIREL"],
            ["r_url", "r", "r_min", "r_max", "RADREL"],
            ["a_url", "a", "a_min", "a_max", "AREL"],
            ["p_url", "p", "p_min", "p_max", "PERREL"],
            ["e_url", "e", "e_min", "e_max", "EREL"],
        ]

        # Iterate through each parameter set
        for p in params:
            # Initialize result DataFrame with NaN values
            result = pd.DataFrame(columns=p)
            result.loc[0, p] = np.nan
            result.loc[0, p[0]] = ""

            # Extract relevant columns from the group
            subgroup = group[p[:-1]]
    
            # Replace empty strings with NaN and drop rows where the main parameter is NaN
            subgroup.loc[:, p[1:-1]] = (
                subgroup.loc[:, p[1:-1]].fillna(np.nan).replace("", np.nan)
            )
            subgroup = subgroup.dropna(subset=[p[1]])

            if len(subgroup) > 0:
                # Convert main parameter to float
                subgroup[p[1]] = subgroup[p[1]].astype("float")

                # Handle cases with NaN uncertainties
                if len(subgroup.dropna(subset=[p[3], p[2]])) > 0:
                    # Keep only rows with non-NaN error bars
                    subgroup = subgroup.dropna(subset=[p[3], p[2]])
                    # Calculate relative errors
                    subgroup["maxrel"] = subgroup[p[3]].astype("float") / subgroup[p[1]].astype("float")
                    subgroup["minrel"] = subgroup[p[2]].astype("float") / subgroup[p[1]].astype("float")
                else:
                    # If all error bars are NaN, set relative error to a large value
                    subgroup["maxrel"] = 1e32
                    subgroup["minrel"] = 1e32

                # Replace infinity with NaN and fill NaN relative errors
                subgroup = subgroup.replace(np.inf, np.nan)
                subgroup["maxrel"] = subgroup["maxrel"].fillna(subgroup[p[2]])
                subgroup["minrel"] = subgroup["minrel"].fillna(subgroup[p[2]])
        
                # Calculate the overall relative error (maximum of maxrel and minrel)
                subgroup[p[-1]] = subgroup[["maxrel", "minrel"]].max(axis=1)

                # Select the row with the minimum relative error
                result = subgroup.loc[subgroup[p[-1]] == subgroup[p[-1]].min(), p]

                # Prefer entries with a real paper associated (assumed to start with a number)
                # (This is only important if more than one entry shows the minimum error, i.e. len(result)>1)
                result = result.sort_values(by=p[0]).head(1)

                # Reset index and drop unnecessary columns
                result = result.reset_index().drop(columns=["index"])

            # Ensure result only contains the specified parameter columns
            result = result[p]

            # Concatenate the result to the entry DataFrame
            entry = pd.concat([entry, result], axis=1)
        # Select status
        # Combine and sort the checked status strings from all entries in the group
        entry["checked_status_string"] = ",".join(
            sorted(group.checked_catalog_status)
        ).rstrip(",")

        # Combine and sort the original status strings from all entries in the group
        entry["original_status_string"] = ",".join(
            sorted(group.original_catalog_status)
        ).rstrip(",")

        # Count how many times 'CONFIRMED' appears in the checked status strings
        entry["confirmed"] = ",".join(set(group.checked_catalog_status)).count(
            "CONFIRMED"
        )

        # Determine the final status for the entry
        if len(set(group.status)) == 1:
            # If all status values are the same, use that status
            entry["status"] = group.status.unique()[0]
        else:
            # If there are different status values, mark as 'CONTROVERSIAL'
            entry["status"] = "CONTROVERSIAL"

        # Select the earliest discovery year
        discovery_years = sorted(group.discovery_year.dropna().astype("int").unique())
        if len(discovery_years) == 1:
            # If there's only one unique year, use it
            entry["discovery_year"] = discovery_years[0]
        elif len(discovery_years) > 1:
            # If there are multiple years, use the earliest one
            entry["discovery_year"] = discovery_years[0]
        else:
            # If no valid year is found, set to empty string
            entry["discovery_year"] = ""

        # Select discovery method(s)
        if len(set(group.discovery_method.unique())) > 1 and "toi" in set(group.catalog.unique()):
            # If multiple methods and 'toi' is present, exclude 'transit' method from 'toi' catalog
            discovery_method = ",".join(
                sorted(
                    group.loc[group.catalog != "toi", "discovery_method"].unique()
                )
            ).rstrip(",")
        else:
            # Otherwise, include all unique discovery methods
            discovery_method = ",".join(
                sorted(group.discovery_method.replace(np.nan, "").unique())
            ).rstrip(",")

        # Clean up and format the discovery method string
        entry["discovery_method"] = ",".join(
            sorted(set(method.strip() for method in discovery_method.split(",")))
        )

        entry["catalog"] = ",".join((sorted(group.catalog.unique()))).rstrip(",")
        
        # Combine all unique catalogs
        entry["catalog"] = ",".join(sorted(group.catalog.unique())).rstrip(",")

        # Select final Alias
        main_id_aliases = ""
        for al in group.main_id_aliases:
            main_id_aliases = main_id_aliases + "," + str(al)
        # Create a sorted, unique list of aliases, excluding 'A', 'B', and empty strings
        entry["main_id_aliases"] = ",".join(
            [
                x
                for x in sorted(set(main_id_aliases.split(",")))
                if x not in ["A", "B", ""]
            ]
        )

        # Set various flags
        # Combine unique binary_mismatch_flag values
        entry["binary_mismatch_flag"] = ",".join(
            map(str, group.binary_mismatch_flag.unique())
        ).rstrip(",")

        # Combine unique coordinate_mismatch values
        entry["coordinate_mismatch"] = ",".join(
            map(str, group.coordinate_mismatch.unique())
        ).rstrip(",")
        # Set coordinate_mismatch_flag based on RA and DEC mismatches
        if "RA" in set(group.coordinate_mismatch.unique()) and "DEC" in set(
            group.coordinate_mismatch.unique()
        ):
            entry["coordinate_mismatch_flag"] = 2  # Both RA and DEC mismatch
        elif "RA" in set(group.coordinate_mismatch.unique()) or "DEC" in set(
            group.coordinate_mismatch.unique()
        ):
            entry["coordinate_mismatch_flag"] = 1  # Either RA or DEC mismatch
        else:
            entry["coordinate_mismatch_flag"] = 0  # No mismatch

        # Combine unique angular_separation values
        entry["angular_separation"] = ",".join(
            map(str, sorted(group.angular_separation.unique()))
        )

        # Set angular_separation_flag based on number of unique angsep values
        entry["angular_separation_flag"] = len(list(set(group.angsep.unique()))) - 1

        # Check if multiple main_id_provenance (user should check that it has been done right)
        if len(group.main_id_provenance.unique()) > 1:
            with pd.option_context("display.max_columns", 2000):
                # Log warning about multiple main_id_provenance
                f.write(
                    "\nWARNING, main_id_provenance not unique for  "
                    + mainid
                    + " "
                    + binary
                    + " "
                    + letter
                    + "\n"
                    + group[
                        [
                            "main_id_provenance",
                            "main_id_ra",
                            "main_id_dec",
                            "angular_separation",
                            "p",
                            "a",
                        ]
                    ].to_string()
                    + "\n"
                )

                # If more than one, prefer the ra and dec in the order listed below
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
                        entry["main_id_ra"] = list(
                            set(group[group.main_id_provenance == item].main_id_ra)
                        )[0]
                        entry["main_id_dec"] = list(
                            set(group[group.main_id_provenance == item].main_id_dec)
                        )[0]
                        break
        else:
            # If there's only one unique main_id_provenance, use it and its corresponding RA and Dec
            entry["main_id_provenance"] = group.main_id_provenance.unique()[0]
            entry["main_id_ra"] = list(set(group.main_id_ra))[0]
            entry["main_id_dec"] = list(set(group.main_id_dec))[0]

        # Check for duplicate entries within the same catalog
        if len(group) > len(group.catalog.unique()):
            # Log the duplicate entry details
            with pd.option_context("display.max_columns", 2000):
                f.write(
                    "\n*** DUPLICATE ENTRY "
                    + mainid
                    + " "
                    + binary
                    + " "
                    + letter
                    + " ***\n"
                    + group[
                        [
                            "catalog",
                            "catalog_name",
                            "status",
                            "angular_separation",
                            "p",
                            "a",
                        ]
                    ].to_string()
                    + "\n"
                )

            # Flag the entry as a duplicate
            entry["duplicate_catalog_flag"] = 1
            group['catalog_and_name']=group.catalog + ": " + group.catalog_name
            entry["duplicate_names"] = ",".join(
                sorted(group.catalog_and_name)
            ).rstrip(",")

        else:
            # If no duplicates, set flag to 0 and leave duplicate_names empty

            entry["duplicate_catalog_flag"] = 0
            entry["duplicate_names"] = ""

        # Close the log file
        f.close()

        # Return the processed entry
        return entry

    def group_by_letter_check_period(self, verbose: bool) -> None:
        """
        Group the catalog by main_id, binary, and letter, then merge entries based on period or semi-major axis agreement.

        This function processes the entire catalog to consolidate multiple entries for the same exoplanet. It performs the following steps:

        1. Groups the catalog by main_id, binary, and letter.
        2. For each group:
            a. Calculates working period and semi-major axis values.
            b. Checks for agreement in period values:
                - If periods agree, merges entries into a single entry.
                - If periods disagree, keeps separate entries and logs the disagreement.
            c. If no period data, checks for agreement in semi-major axis values:
                - If semi-major axes agree, merges entries into a single entry.
                - If semi-major axes disagree, keeps separate entries and logs the disagreement.
            d. If neither period nor semi-major axis data available, merges all entries.
        3. Assigns merging_mismatch_flags:
            - 0: Successful merge (period or semi-major axis agreement)
            - 1: Disagreement in period or semi-major axis
            - 2: Fallback merge (no period or semi-major axis data)
        4. Creates a new catalog with the merged entries.
        5. Logs the merging process and any disagreements.

        :param self: An instance of the Emc class
        :type self: Emc
        :param verbose: If True, displays a progress bar during processing
        :type verbose: bool
        :return: None
        :rtype: None
        """
        # Create an empty DataFrame to store the final catalog
        final_catalog = pd.DataFrame()
        
        # Open a log file to record any disagreements or issues during the process
        f1 = open("Logs/group_by_letter_check_period.txt", "a")

        # Group the data by main_id, binary, and letter
        grouped_df = self.data.groupby(
            ["main_id", "binary", "letter"], sort=True, as_index=False
        )
        
        counter = 0
        # Iterate through each group in the grouped DataFrame
        for (mainid, binary, letter), group in grouped_df:
            # Calculate working period and semi-major axis for the group
            group = Utils.calculate_working_p_sma(group, tolerance=0.1)
            
            # Get a list of unique periods, excluding NaN and -1 values
            period_list = list(
                set(group.working_p.replace(-1, np.nan).dropna().unique())
            )
            
            if len(period_list) == 1:
                # All entries have the same period (excluding NaN values):
                # Merge all entries into a single entry
                entry = Emc.merge_into_single_entry(group, mainid, binary, str(letter))
                entry["merging_mismatch_flag"] = 0  # Indicates successful merge
                final_catalog = pd.concat(
                    [final_catalog, entry], sort=False
                ).reset_index(drop=True)

            elif len(period_list) > 1:
                # Multiple different periods found (excluding NaN values):
                # Log the disagreement for further investigation
                f1.write(
                    "DISAGREEMENT (merging_mismatch_flag=1)\n"
                    + group[
                        [
                            "main_id",
                            "binary",
                            "letter",
                            "catalog",
                            "catalog_name",
                            "p",
                        ]
                    ].to_string()
                    + "\n\n"
                )
                # Create separate entries for each unique period
                for pgroup in period_list:
                    subgroup = group[group.working_p == pgroup]
                    entry = Emc.merge_into_single_entry(
                        subgroup, mainid, binary, str(letter)
                    )
                    entry["merging_mismatch_flag"] = 1  # Indicates disagreement in period

                    final_catalog = pd.concat(
                        [final_catalog, entry], sort=False
                    ).reset_index(drop=True)
            else:
                # No valid period data available, check semi-major axis (sma)
                sma_list = list(
                    set(group.working_a.replace(-1, np.nan).dropna().unique())
                )

                if len(sma_list) == 1:
                    # Semi-major axis values are in agreement (excluding NaN values):
                    # Perform regular merging of entries
                    entry = Emc.merge_into_single_entry(
                        group, mainid, binary, str(letter)
                    )
                    entry["merging_mismatch_flag"] = 0  # Indicates successful merge

                    # Add the merged entry to the final catalog
                    final_catalog = pd.concat(
                        [final_catalog, entry], sort=False
                    ).reset_index(drop=True)

                    
                elif len(sma_list) > 1:
                    # Semi-major axis values disagree (excluding NaN values):
                    # Log the disagreement for further investigation
                    f1.write(
                        "DISAGREEMENT (merging_mismatch_flag=1)\n"
                        + group[
                            [
                                "main_id",
                                "binary",
                                "letter",
                                "catalog",
                                "catalog_name",
                                "a",
                            ]
                        ].to_string()
                        + "\n\n"
                    )
                    # Create separate entries for each unique semi-major axis value
                    for agroup in sma_list:
                        subgroup = group[group.working_a == agroup]
                        entry = Emc.merge_into_single_entry(
                            subgroup, mainid, binary, letter
                        )

                        entry["merging_mismatch_flag"] = 1 # Indicates disagreement in sma

                        # Add the merged entry to the final catalog
                        final_catalog = pd.concat(
                            [final_catalog, entry], sort=False
                        ).reset_index(drop=True)
                else:
                    # No period nor sma: will merge together
                    if len(group) > 1:
                        f1.write(
                            "FALLBACK, MERGE (merging_mismatch_flag=2) \n"
                            + group[
                                [
                                    "main_id",
                                    "binary",
                                    "letter",
                                    "catalog",
                                    "catalog_name",
                                ]
                            ].to_string()
                            + "\n\n"
                        )
                    entry = Emc.merge_into_single_entry(
                        group, mainid, binary, str(letter)
                    )

                    entry["merging_mismatch_flag"] = 2 # Indicates fallback merge

                    # Add the merged entry to the final catalog
                    final_catalog = pd.concat(
                        [final_catalog, entry], sort=False
                    ).reset_index(drop=True)

            # Print progress
            if verbose:
                Utils.print_progress_bar(
                    counter, len(grouped_df), prefix="Progress:", suffix="Complete"
                )

            counter = counter + 1
        f1.close()

        # Assign final catalog
        self.data = final_catalog

        # Logging
        logging.info("\nCatalog merged into single entries.")


    def select_best_mass(self) -> None:
        """
        Selects the best mass estimate for each planet in the catalog.

        This function determines whether to use the mass or minimum mass (msini) as the best mass estimate
        for each planet based on their relative errors. It performs the following steps:

        1. If MASSREL (relative error of mass) is greater than or equal to MSINIREL (relative error of msini),
           it uses msini as the best mass estimate.
        2. If MASSREL is less than MSINIREL, it uses mass as the best mass estimate.
        3. If both mass and msini are missing (NaN), it sets all best mass related fields to NaN or empty string.


        :param self: An instance of the Emc class
        :type self: Emc
        :return: None
        :rtype: None
        """

        # Select best mass estimate based on relative errors
        # If MASSREL (relative error of mass) is greater than or equal to MSINIREL (relative error of msini),
        # use msini as the best mass estimate
        for i in self.data[
            self.data.MASSREL.fillna(1e9) >= self.data.MSINIREL.fillna(1e9)
        ].index:
            self.data.at[i, "bestmass"] = self.data.at[i, "msini"]
            self.data.at[i, "bestmass_min"] = self.data.at[i, "msini_min"]
            self.data.at[i, "bestmass_max"] = self.data.at[i, "msini_max"]
            self.data.at[i, "bestmass_url"] = self.data.at[i, "msini_url"]
            self.data.at[i, "bestmass_provenance"] = "Msini"

        # If MASSREL is less than MSINIREL, use mass as the best mass estimate
        for i in self.data[
            self.data.MASSREL.fillna(1e9) < self.data.MSINIREL.fillna(1e9)
        ].index:
            self.data.at[i, "bestmass"] = self.data.at[i, "mass"]
            self.data.at[i, "bestmass_min"] = self.data.at[i, "mass_min"]
            self.data.at[i, "bestmass_max"] = self.data.at[i, "mass_max"]
            self.data.at[i, "bestmass_url"] = self.data.at[i, "mass_url"]
            self.data.at[i, "bestmass_provenance"] = "Mass"

        # If both mass and msini are missing (NaN), set all best mass related fields to NaN or empty string
        for i in self.data[
            (self.data.mass.fillna(1e9) == 1e9) & (self.data.msini.fillna(1e9) == 1e9)
        ].index:
            self.data.at[i, "bestmass"] = np.nan
            self.data.at[i, "bestmass_min"] = np.nan
            self.data.at[i, "bestmass_max"] = np.nan
            self.data.at[i, "bestmass_url"] = np.nan
            self.data.at[i, "bestmass_provenance"] = ""

        # Log that the best mass calculation is complete
        logging.info("Bestmass calculated.")
        
    def set_exomercat_name(self) -> None:
        """
        Creates the 'exo-mercat_name' column by joining the main_id, binary (if any), and letter.

        :param self: An instance of the class Emc
        :type self: Emc
        :return: None
        :rtype: None
        """
        # Replace 'nan' with empty string in the 'binary' column
        self.data["binary"] = self.data["binary"].replace("nan", "")

        # Create the 'exo-mercat_name' column
        self.data["exo-mercat_name"] = self.data.apply(
            lambda row: (
            # If main_id doesn't end with A, B, C, N, or S preceded by a space or digit,
            # use the full main_id. Otherwise, remove the last character and strip trailing spaces.
                row["main_id"]
                if str(re.search("[\\s\\d][ABCNS]$", row["main_id"], re.M)) == "None"
                else row["main_id"][:-1].rstrip()
            )
            # Add binary information if it exists
            + (" " + str(row["binary"]) if not row["binary"] == "" else "")
            # Add the letter
            + " "
            + row["letter"],
            axis=1,
        )

        # Sort the dataframe by the new 'exo-mercat_name' column and reset the index
        self.data = self.data.sort_values(by="exo-mercat_name").reset_index()

        # Log that the Exo-MerCat name has been assigned
        logging.info("Exo-MerCat name assigned.")

    def keep_columns(self) -> None:
        """
        Retain only specified columns in the dataframe and remove all others.

        This function performs the following operations:
        1. Defines a list of columns to keep, including various exoplanet properties and metadata.
        2. Attempts to filter the dataframe to retain only these specified columns.
        3. If any specified column is missing from the dataframe, it raises a KeyError.

        :param self: An instance of Emc class
        :type self: Emc
        :return: None
        :rtype: None
        :raises KeyError: If any of the specified columns to keep are not present in the dataframe
        """
        # List columns to keep
        keep = [
            "exo-mercat_name",
            "nasa_name",
            "toi_name",
            "epic_name",
            "eu_name",
            "oec_name",
            "host",
            "letter",
            "main_id",
            "binary",
            "main_id_ra",
            "main_id_dec",
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
            "main_id_aliases",
            "catalog",
            "angular_separation",
            "angular_separation_flag",
            "main_id_provenance",
            "binary_mismatch_flag",
            "coordinate_mismatch",
            "coordinate_mismatch_flag",
            "duplicate_catalog_flag",
            "duplicate_names",
            "merging_mismatch_flag",
            "row_update",
        ]

        # Check columns exist, otherwise raise Error
        try:
            self.data = self.data[keep]
        except KeyError:
            raise KeyError("Not all columns exist")

        # Logging
        logging.info("Selected columns to keep.")

    def remove_known_brown_dwarfs(
        self,
        local_date: str,
        print_flag: bool,
    ) -> None:
        """
        Remove objects with masses greater than 20 Jupiter masses (considered brown dwarfs) from the dataset.

        This function performs the following operations:
        1. Identifies objects with mass or minimum mass (msini) greater than 20 Jupiter masses.
        2. Optionally saves these identified objects to CSV files.
        3. Removes the identified objects from the main dataset.

        The mass threshold is applied as follows:
        - Uses 'mass' if available, otherwise uses 'msini'.
        - If both are unavailable, treats the object as having zero mass (thus not removed).
        - Empty strings are treated as zero mass.

        :param self: An instance of the Emc class
        :type self: Emc
        :param local_date: A string representation of the current date, used for naming output files
        :type local_date: str
        :param print_flag: If True, saves the removed objects to CSV files
        :type print_flag: bool
        :return: None
        :rtype: None
        """
        # Identify objects with mass > 20 Jupiter masses
        brown_dwarfs = (
            self.data.mass.fillna(self.data.msini.fillna(0))
            .replace("", 0)
            .astype(float)
            > 20.0
        )
        if print_flag:
            # Save identified brown dwarfs to a CSV file
            self.data[
                brown_dwarfs
            ].to_csv("Exo-MerCat/" + self.name + "_brown_dwarfs.csv", index=None)
            
            
            # Save identified brown dwarfs to a dated CSV file
            self.data[
                brown_dwarfs
            ].to_csv(
                "Exo-MerCat/" + self.name + "_brown_dwarfs" + local_date + ".csv",
                index=None,
            )

        # Remove brown dwarfs from the main dataset
        self.data = self.data[~brown_dwarfs]



    def fill_row_update(self, local_date: str) -> None:
        """
        Update the 'row_update' column in the DataFrame based on changes from the previous version.

        This function performs the following operations:
        1. Uses the provided local_date as the update date.
        2. Checks for previous versions of the catalog.
        3. If previous versions exist, compares the current DataFrame with the most recent previous version.
        4. Updates the 'row_update' column for rows that have changed or are new.
        5. Retains the previous 'row_update' value for unchanged rows.

        The function handles the following scenarios:
        - If no previous versions exist, all rows are considered new and updated with the current date.
        - If previous versions exist, only changed or new rows are updated with the current date.

        :param self: An instance of the Emc class
        :type self: Emc
        :param local_date: The date to use for updating the 'row_update' column
        :type local_date: str
        :return: None
        :rtype: None

        """
        
        # Use the provided local_date as the update date
        update_date = local_date

        # Check if there are older versions of the catalog present
        if len(glob.glob("Exo-MerCat/exo-mercat*-*.csv")) > 0:
            # Get a list of all previous catalog versions
            li = list(glob.glob("Exo-MerCat/exo-mercat_full*-*.csv"))

            # Extract the dates from the filenames
            li = [re.search(r"\d\d\d\d-\d\d-\d\d", l)[0] for l in li]
            
            # Convert the extracted dates to datetime objects
            li = [datetime.strptime(l, "%Y-%m-%d") for l in li]

            # Get the most recent catalog version that's earlier than the current date
            li = [l for l in li if l < datetime.strptime(update_date, "%Y-%m-%d")]
            compar_date = max(li).strftime("%Y-%m-%d")
            
            # Load the most recent previous catalog
            right_merge = pd.read_csv(
                "Exo-MerCat/exo-mercat_full" + compar_date + ".csv"
            )
            right_merge["old_index"] = right_merge.index.copy().astype(int)

            # Hotfix in case the previous versions did not have this feature
            if "row_update" not in right_merge.columns:
                right_merge["row_update"] = compar_date

            self.data["new_index"] = self.data.index.copy().astype(int)

            # Print into csv to make sure that it reads the same way as right_merge
            self.data.to_csv("temp.csv")
            left_merge = pd.read_csv("temp.csv")
            
            # Clean up
            os.remove("temp.csv")

            # Merge current and previous catalog
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
            # Get the previous date for all the entries that did not change
            self.data.loc[all.new_index, "row_update"] = all.loc[
                all.new_index, "row_update"
            ]
            # Fill the missing ones with the update date
            self.data["row_update"] = self.data["row_update"].fillna(update_date)
            # Clean up
            self.data = self.data.drop("new_index", axis=1)
        else:
            # No previous versions found, update all rows with current date

            self.data["row_update"] = update_date

    def save_catalog(self, local_date: str, postfix: str = '') -> None:
        """
        Saves the catalog to csv viles.
        It is saved to the 'Exo-MerCat' folder both as a exo-mercat.csv file and as a exo-mercatYYYY-MM-DD.csv file.

        :param self: An instance of the class Emc
        :type self: Emc
        :param local_date: The date to save the catalog
        :type local_date: str
        :param postfix: The postfix to add to the filename
        :type postfix: str
        :return: None
        :rtype: None

        """
        self.print_catalog("Exo-MerCat/exo-mercat" + postfix + local_date + ".csv")
        self.print_catalog("Exo-MerCat/exo-mercat" + postfix + ".csv")
