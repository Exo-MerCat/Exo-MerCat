#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import glob

from .nasa import Nasa
from .eu import Eu
from .oec import Oec
from .epic import Epic
from .emc import Emc
from .koi import Koi
from .toi import Toi
from .utility_functions import UtilityFunctions as Utils
import socket
import warnings
import pandas as pd
import logging
import os
from argparse import ArgumentParser, ArgumentDefaultsHelpFormatter
from datetime import date, datetime
import numpy as np


def main():  # pragma: no cover
    """
    Main entry point for the Exo-MerCat command-line interface.

    This function parses command-line arguments and executes the appropriate
    function based on the user's input. It supports the following operations:

    - maintenance: Perform sanity checks on the catalog data

    - input: Download and standardize catalog files

    - run: Process and merge catalog data to create the Exo-MerCat catalog

    - check: Perform validation checks on the final Exo-MerCat catalog

    - all: Execute all of the above operations in sequence

    Command-line arguments:

    - function: The operation to perform (maintenance, input, run, check, or all)

    - --verbose (-v): Increase output verbosity (use -v, -vv, or -vvv for more detail)

    - --date (-d): Specify a date for catalog data (format: YYYY-MM-DD)

    This function is not intended to be imported and used directly in other modules.
    """

    # Parse command line arguments
    parser = ArgumentParser(formatter_class=ArgumentDefaultsHelpFormatter)
    parser.add_argument(
        "function",
        help="specify function to be run (options: maintenance, input, run, check, all)",
    )  # positional argument
    parser.add_argument(
        "--verbose",
        "-v",
        action="count",
        default=0,
        help="Increase output verbosity. Use -v, -vv, or -vvv for more verbosity.",
    )
    parser.add_argument("-d", "--date", help="load a specific date (YYYY-MM-DD)")
    args = vars(parser.parse_args())

    # Set up level of verbosity
    warnings.filterwarnings("ignore")
    if args["verbose"] >= 1:
        logging.basicConfig(format="%(asctime)s: %(message)s", level=logging.INFO)
        if args["verbose"] >= 2:
            # Debugging verbose
            warnings.filterwarnings("default")

    # Date: either a specific date, or today's date
    if args["date"]:
        local_date = args["date"]
    else:
        local_date = date.today().strftime("%Y-%m-%d")

    # Initialize service files and folders
    Utils.folder_initialization()

    # Execute the specified function based on the command-line argument
    if args["function"] == "maintenance":

        # Perform sanity checks on the catalog data
        ping(local_date)
    if args["function"] == "input":
        # Remove log file created in input
        os.system("rm Logs/replace_known_mistakes.txt")
        # Download and standardize catalog files
        input(local_date)
    if args["function"] == "run":
        # Remove log files created in run
        for file in glob.glob('Logs/*'):
            if 'replace_known_mistakes.txt' not in file:
                os.system("rm "+file)
        # Process and merge catalog data to create the Exo-MerCat catalog
        run(local_date, args["verbose"])
    if args["function"] == "check":
        # Perform validation checks on the final Exo-MerCat catalog
        check(local_date)
    if args["function"] == "all":
        # Remove all log files
        os.system("rm Logs/*")
        # Execute all operations in sequence
        # 1. Perform sanity checks
        ping(local_date)
        # 2. Download and standardize catalog files
        input(local_date)
        # 3. Process and merge catalog data
        run(local_date, args["verbose"])
        # 4. Perform validation checks on the final catalog
        check(local_date)

    timeout = 100000
    socket.setdefaulttimeout(timeout)


def ping(local_date):  # pragma: no cover
    """
    Perform sanity checks on the input catalog data.

    This function performs the following checks for each catalog:

    1. Attempts to download the catalog

    2. Tries to read the downloaded catalog

    3. Checks if all required columns are present

    4. Verifies the data types of the columns

    5. Checks for non-ASCII characters in the data

    It also checks the connection to SIMBAD and VizieR services.

    :param local_date: The date for which to perform the checks (format: YYYY-MM-DD)
    :type local_date: str
    :raises ValueError: If one or more sanity checks fail
    """
    
    logging.info("STARTING SANITY CHECKS.")
    error = 0
    config_dict = Utils.read_config()

    # List of catalog types to check
    cat_types = [Koi(), Eu(), Nasa(), Oec(), Toi(), Epic()]
    for cat in cat_types:
        logging.info("****** " + cat.name + " ******")
        config_per_cat = config_dict[cat.name]

        # Step 1: Try downloading the catalog
        try:
            file_path = cat.download_catalog(
                config_per_cat["url"], config_per_cat["file"], local_date
            )
            logging.info("Download catalog\t\tOK.")
        except (ConnectionError, ValueError):
            logging.info("Download catalog\t\tFAILED.")
            error += 1
        # Step 2: Try reading the downloaded catalog
        try:
            cat.read_csv_catalog(file_path)
            logging.info("Read .csv file\t\t\tOK.")
        except ValueError:
            logging.info("Read .csv file\t\t\tFAILED.")
            error += 1

        # Step 3: Check if the catalog contains all required columns
        missing_columns = cat.check_input_columns()
        if missing_columns == "":
            logging.info("Check input columns\t\tOK.")
        else:
            logging.info(
                "Check input columns\t\tFAILED. Missing columns: "
                + missing_columns.rstrip(",")
            )
            error += 1

        # Step 4: Check if the columns have the correct data types
        wrong_types = cat.check_column_dtypes()
        if wrong_types == "":
            logging.info("Check dtypes\t\t\tOK.")
        else:
            logging.info(
                "Check dtypes\t\t\tFAILED. Wrong dtypes: " + missing_columns.rstrip(",")
            )
            error += 1

        # Step 5: Check for non-ASCII characters in the data
        non_ascii = cat.find_non_ascii()
        if non_ascii:
            logging.info(
                "Check weird characters\t\tFAILED. Non-ASCII characters found in columns and indexes: {non_ascii} "
            )
            error += 1
        else:
            logging.info("Check weird characters\t\tOK.")

    # Step 6: Check connection to SIMBAD and VizieR services
    logging.info("****** STELLAR CATALOGS ******")
    ping_str = Utils.ping_simbad_vizier()

    for string in ping_str.split("\n"):
        logging.info(string.rstrip("\n"))
    if "FAILED" in ping_str:
        error += 1

    # If any check failed, raise an error
    if error > 0:
        raise ValueError("One or more sanity checks was not successful.")
    

def input(local_date):  # pragma: no cover
    """
    Download and standardize catalog files.

    This function performs the following operations for each catalog:

    1. Downloads the catalog data

    2. Reads the downloaded data

    3. Standardizes the catalog format

    4. Converts coordinates to a standard format

    5. Performs various data cleaning and standardization operations

    6. Saves the standardized catalog

    :param local_date: The date for which to download and process catalogs (format: YYYY-MM-DD)
    :type local_date: str
    """

    # Load configuration
    config_dict = Utils.read_config()

    # Process KOI catalog separately
    for cat in [Koi()]:
        logging.info("****** " + cat.name + " ******")
        config_per_cat = config_dict[cat.name]
        
        # Download and read the catalog
        file_path_str = cat.download_catalog(
            config_per_cat["url"], config_per_cat["file"], local_date
        )
        cat.read_csv_catalog(file_path_str)

        # Standardize the catalog
        cat.standardize_catalog()
        cat.convert_coordinates()
        cat.fill_nan_on_coordinates()
        
        # Save the standardized catalog
        cat.print_catalog("StandardizedSources/" + cat.name + local_date + ".csv")

    # Process other catalogs
    cat_types = [Eu(), Nasa(), Oec(), Toi(), Epic()]
    for cat in cat_types:
        logging.info("****** " + cat.name + " ******")
        config_per_cat = config_dict[cat.name]
        
        # Download and read the catalog
        file_path = cat.download_catalog(
            config_per_cat["url"], config_per_cat["file"], local_date
        )
        cat.read_csv_catalog(file_path)
        # Standardize and clean the catalog
        cat.standardize_catalog()
        cat.convert_coordinates()
        cat.fill_nan_on_coordinates()
        cat.fill_binary_column()
        cat.replace_known_mistakes()
        cat.standardize_name_host_letter()
        cat.identify_brown_dwarfs()
        cat.remove_theoretical_masses()
        cat.make_errors_absolute()
        cat.remove_impossible_values()
        cat.handle_reference_format()
        
        # Assign status and create catalog status strings
        cat.assign_status()
        cat.create_catalogstatus_string("original_catalog_status")
        cat.check_mission_tables("StandardizedSources/koi" + local_date + ".csv")
        cat.create_catalogstatus_string("checked_catalog_status")
        
        # Finalize and save the standardized catalog
        cat.make_standardized_alias_list()
        cat.keep_columns()
        cat.print_catalog("StandardizedSources/" + cat.name + local_date + ".csv")

def run(local_date: str, verbose: int):  # pragma: no cover
    """
    Process and merge catalog data to create the Exo-MerCat catalog.

    This function performs the following operations:

    1. Loads standardized catalog files

    2. Merges data from different catalogs

    3. Matches with stellar catalogs

    4. Performs various data processing and standardization steps

    5. Merges entries for the same exoplanet from different catalogs

    6. Saves the final Exo-MerCat catalog

    :param local_date: The date for which to process the catalogs (format: YYYY-MM-DD)
    :type local_date: str
    :param verbose: The verbosity level for output
    :type verbose: int
    """

    emc = Emc()

    logging.info("Loading standardized files...")
    # Load NASA catalog data
    emc.data = Utils.load_standardized_catalog("StandardizedSources/nasa", local_date)
    # Concatenate data from other catalogs (EU, OEC, TOI, EPIC)
    for catalog in ["eu", "oec", "toi", "epic"]:
        emc.data = pd.concat(
            [
                emc.data,
                Utils.load_standardized_catalog(
                    "StandardizedSources/" + catalog, local_date
                ),
            ]
        )
    # Fix for TOI catalog: convert 'letter' column to string and keep only last 3 characters
    emc.data.letter = emc.data.letter.astype(str)
    emc.data.letter = emc.data.letter.str[-3:]

    emc.data = emc.data.reset_index()

    # Matching with stellar catalogs
    emc.alias_as_host()
    emc.check_binary_mismatch(keyword="host", tolerance=1.0 / 3600.0)
    emc.prepare_columns_for_mainid_search()
    emc.get_host_info_from_simbad()
    emc.get_host_info_from_tic()
    emc.check_coordinates(tolerance=1.0 / 3600.0)
    emc.get_coordinates_from_simbad(tolerance=1.0 / 3600.0)
    emc.get_coordinates_from_tic(tolerance=1.0 / 3600.0)
    emc.fill_missing_main_id()
    emc.polish_main_id()
    emc.post_main_id_query_checks(tolerance=1.0 / 3600.0)
    emc.group_by_main_id_set_main_id_aliases()
    emc.check_binary_mismatch(keyword="main_id", tolerance=1.0 / 3600.0)
    emc.cleanup_catalog()

    # Merging entries
    emc.group_by_period_check_letter()
    emc.group_by_letter_check_period(verbose=verbose)
    emc.select_best_mass()
    emc.set_exomercat_name()
    emc.identify_misnamed_duplicates()
    emc.fill_row_update(local_date)
    emc.keep_columns()
    emc.save_catalog(local_date, "_full")
    emc.remove_known_brown_dwarfs(
        local_date,
        print_flag=True,
    )
    emc.save_catalog(local_date, "")


def check(local_date:str):  # pragma: no cover
    """
    Perform validation checks on the final Exo-MerCat catalog.

    This function loads the final Exo-MerCat catalog and performs various consistency checks, including:
    - Verifying the presence and correctness of catalog-specific name columns
    - Checking the consistency of parameter values and their associated metadata (e.g., errors, URLs)
    - Verifying the completeness of critical fields like discovery method and year

    :param local_date: The date of the catalog to check (format: YYYY-MM-DD)
    :type local_date: str
    """
    error_string = ""
    emc = pd.read_csv("Exo-MerCat/exo-mercat_full" + local_date + ".csv")

    # CHECK nasa_name: check that nasa_name is null only when catalog does not contain 
    # nasa, and that it is not null only when catalog contains nasa.

    if len(emc[emc.nasa_name.isna()]) != len(
        emc[(emc.nasa_name.isna()) & ~(emc.catalog.str.contains("nasa"))]
    ):
        error_string = error_string + "CHECK nasa_name.a\n"
    if len(emc[~(emc.nasa_name.isna()) & ~(emc.catalog.str.contains("nasa"))]) > 0:
        error_string = error_string + "CHECK nasa_name.b\n"

    if len(emc[~emc.nasa_name.isna()]) != len(
        emc[~(emc.nasa_name.isna()) & (emc.catalog.str.contains("nasa"))]
    ):
        error_string = error_string + "CHECK nasa_name.c\n"
    if len(emc[(emc.nasa_name.isna()) & (emc.catalog.str.contains("nasa"))]) > 0:
        error_string = error_string + "CHECK nasa_name.d\n"

    #CHECK eu_name: check that eu_name is null only when catalog does not contain 
    # eu, and that it is not null only when catalog contains eu.

    if len(emc[emc.eu_name.isna()]) != len(
        emc[(emc.eu_name.isna()) & ~(emc.catalog.str.contains("eu"))]
    ):
        error_string = error_string + "CHECK eu_name.a\n"
    if len(emc[~(emc.eu_name.isna()) & ~(emc.catalog.str.contains("eu"))]) > 0:
        error_string = error_string + "CHECK eu_name.b\n"

    if len(emc[~emc.eu_name.isna()]) != len(
        emc[~(emc.eu_name.isna()) & (emc.catalog.str.contains("eu"))]
    ):
        error_string = error_string + "CHECK eu_name.c\n"
    if len(emc[(emc.eu_name.isna()) & (emc.catalog.str.contains("eu"))]) > 0:
        error_string = error_string + "CHECK eu_name.d\n"

    # CHECK oec_name: check that oec_name is null only when catalog does not contain 
    # oec, and that it is not null only  when catalog contains oec.

    if len(emc[emc.oec_name.isna()]) != len(
        emc[(emc.oec_name.isna()) & ~(emc.catalog.str.contains("oec"))]
    ):
        error_string = error_string + "CHECK oec_name.a\n"
    if len(emc[~(emc.oec_name.isna()) & ~(emc.catalog.str.contains("oec"))]) > 0:
        error_string = error_string + "CHECK oec_name.b\n"

    if len(emc[~emc.oec_name.isna()]) != len(
        emc[~(emc.oec_name.isna()) & (emc.catalog.str.contains("oec"))]
    ):
        error_string = error_string + "CHECK oec_name.c\n"
    if len(emc[(emc.oec_name.isna()) & (emc.catalog.str.contains("oec"))]) > 0:
        error_string = error_string + "CHECK oec_name.d\n"

    # CHECK toi_name: check that toi_name is null only when catalog does not contain 
    # toi, and that it is not null only when catalog contains toi.

    if len(emc[emc.toi_name.isna()]) != len(
        emc[(emc.toi_name.isna()) & ~(emc.catalog.str.contains("toi"))]
    ):
        error_string = error_string + "CHECK toi_name.a\n"
    if len(emc[~(emc.toi_name.isna()) & ~(emc.catalog.str.contains("toi"))]) > 0:
        error_string = error_string + "CHECK toi_name.b\n"

    if len(emc[~emc.toi_name.isna()]) != len(
        emc[~(emc.toi_name.isna()) & (emc.catalog.str.contains("toi"))]
    ):
        error_string = error_string + "CHECK toi_name.c\n"
    if len(emc[(emc.toi_name.isna()) & (emc.catalog.str.contains("toi"))]) > 0:
        error_string = error_string + "CHECK toi_name.d\n"

    
    # CHECK mass: mass_max, mass_min and mass_url must be null only when mass is null. 
    

    if len(emc[emc.mass.isna()]) != len(
        emc[
            (emc.mass.isna())
            & (emc.mass_max.isna())
            & (emc.mass_min.isna())
            & (emc.mass_url.isna())
        ]
    ):
        error_string = error_string + "CHECK mass.a\n"
    

    # CHECK msini: msini_max, msini_min and msini_url must be null only when msini is null. 

    if len(emc[emc.msini.isna()]) != len(
        emc[
            (emc.msini.isna())
            & (emc.msini_max.isna())
            & (emc.msini_min.isna())
            & (emc.msini_url.isna())
        ]
    ):
        error_string = error_string + "CHECK msini.a\n"
    
    # check that url is not null whenever value is not null (null errors are possible)
    if len(emc[~emc.msini.isna()]) != len(
        emc[
            ~(emc.msini.isna())
            & ~(emc.msini_url.isna())
        ]
    ):
        error_string = error_string + "CHECK msini.b\n"


    # CHECK p: p_max, p_min and p_url must be null only when p is null. 

    if len(emc[emc.p.isna()]) != len(
        emc[
            (emc.p.isna())
            & (emc.p_max.isna())
            & (emc.p_min.isna())
            & (emc.p_url.isna())
        ]
    ):
        error_string = error_string + "CHECK p.a\n"
    
    # check that url is not null whenever value is not null (null errors are possible)

    if len(emc[~emc.p.isna()]) != len(
        emc[
            ~(emc.p.isna())
            & ~(emc.p_url.isna())
        ]
    ):
        error_string = error_string + "CHECK p.b\n"

  
    # CHECK r: r_max, r_min and r_url must be null only when r is null. 

    if len(emc[emc.r.isna()]) != len(
        emc[
            (emc.r.isna())
            & (emc.r_max.isna())
            & (emc.r_min.isna())
            & (emc.r_url.isna())
        ]
    ):
        error_string = error_string + "CHECK r.a\n"
    
    # check that url is not null whenever value is not null (null errors are possible)

    if len(emc[~emc.r.isna()]) != len(
        emc[
            ~(emc.r.isna())
            & ~(emc.r_url.isna())
        ]
    ):
        error_string = error_string + "CHECK r.b\n"

    # CHECK e: e_max, e_min and e_url must be null only when e is null. 


    if len(emc[emc.e.isna()]) != len(
        emc[
            (emc.e.isna())
            & (emc.e_max.isna())
            & (emc.e_min.isna())
            & (emc.e_url.isna())
        ]
    ):
        error_string = error_string + "CHECK e.a\n"
    
    # check that url is not null whenever value is not null (null errors are possible)

    if len(emc[~emc.e.isna()]) != len(
        emc[
            ~(emc.e.isna())
            & ~(emc.e_url.isna())
        ]
    ):
        error_string = error_string + "CHECK e.b\n"

    # CHECK i: i_max, i_min and i_url must be null only when i is null. 

    if len(emc[emc.i.isna()]) != len(
        emc[
            (emc.i.isna())
            & (emc.i_max.isna())
            & (emc.i_min.isna())
            & (emc.i_url.isna())
        ]
    ):
        error_string = error_string + "CHECK i.a\n"
    
    # check that url is not null whenever value is not null (null errors are possible)

    if len(emc[~emc.i.isna()]) != len(
        emc[
            ~(emc.i.isna())
            & ~(emc.i_url.isna())
        ]
    ):
        error_string = error_string + "CHECK i.b\n"

    #CHECK bestmass: bestmass_max, bestmass_min, and bestmass_url must be null only when 
    # bestmass is null. """

    if len(emc[emc.bestmass.isna()]) != len(
        emc[
            (emc.bestmass.isna())
            & (emc.bestmass_max.isna())
            & (emc.bestmass_min.isna())
            & (emc.bestmass_url.isna())
        ]
    ):
        error_string = error_string + "CHECK bestmass.a\n"

    # check that url is not null whenever value is not null (null errors are possible)

    if len(emc[~emc.bestmass.isna()]) != len(
        emc[
            ~(emc.bestmass.isna())
            & ~(emc.bestmass_url.isna())
        ]
    ):
        error_string = error_string + "CHECK bestmass.b\n"

    # Bestmass must be null only when both mass and msini are null
    if len(emc[emc.bestmass.isna() & (~(emc.mass.isna()) | ~(emc.msini.isna()))]) > 0:
        error_string = error_string + "CHECK bestmass.c\n"


    # CHECK discovery_method: discovery_method should never be null.

    if len(emc[emc.discovery_method.isna()]) > 0:
        error_string = error_string + "CHECK discovery_method.a\n"

    # # CHECK discovery_year: discovery_year should never be null (except when it is null from the source files).

    # if len(emc[emc.discovery_year.isna()]) > 0:
    #     error_string = error_string + "CHECK discovery_year.a (known issue)\n"

    # # CHECK final_alias: final_alias should never be null (except when it is null from the source files).

    # if len(emc[emc.main_id_aliases.isna()]) > 0:
    #     error_string = error_string + "CHECK main_id_aliases.a (known issue)\n"

    if len(error_string) == 0:
        print("All checks passed.")
    else:
        print("The following checks failed:\n" + error_string)


if __name__ == "__main__":  # pragma: no cover
    main()
