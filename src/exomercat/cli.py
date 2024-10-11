#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Oct  7 10:35:12 2019

@author: eleonoraalei
"""

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
from argparse import ArgumentParser, ArgumentDefaultsHelpFormatter
from datetime import date, datetime
import numpy as np


def main():  # pragma: no cover
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

    warnings.filterwarnings("ignore")
    if args["verbose"] >= 1:
        logging.basicConfig(format="%(asctime)s: %(message)s", level=logging.INFO)
        if args["verbose"] >= 2:
            # debugging verbose
            warnings.filterwarnings("default")

    # date: either a specific date, or today's date
    if args["date"]:
        local_date = args["date"]
    else:
        local_date = date.today().strftime("%Y-%m-%d")

    if args["function"] == "maintenance":
        sanity_checks(local_date)
    if args["function"] == "input":
        input(local_date)
    if args["function"] == "run":
        run(local_date, args["verbose"])
    if args["function"] == "check":
        check(local_date)
    if args["function"] == "all":
        sanity_checks(local_date)
        input(local_date)
        run(local_date, args["verbose"])
        check(local_date)

    timeout = 100000
    socket.setdefaulttimeout(timeout)


def sanity_checks(local_date):  # pragma: no cover
    logging.info("STARTING SANITY CHECKS.")
    error = 0
    config_dict = Utils.read_config()

    Utils.service_files_initialization()

    cat_types = [Koi(), Eu(), Nasa(), Oec(), Toi(), Epic()]
    for cat in cat_types:
        logging.info("****** " + cat.name + " ******")
        config_per_cat = config_dict[cat.name]

        # try downloading the catalog
        try:
            file_path = cat.download_catalog(
                config_per_cat["url"], config_per_cat["file"], local_date
            )
            logging.info("Download catalog\t\tOK.")
        except (ConnectionError, ValueError):
            logging.info("Download catalog\t\tFAILED.")
            error += 1
        # try reading the catalog
        try:
            cat.read_csv_catalog(file_path)
            logging.info("Read .csv file\t\t\tOK.")
        except ValueError:
            logging.info("Read .csv file\t\t\tFAILED.")
            error += 1

        # check the goodness of the catalog (if contains all the columns, if there are weird characters...)
        missing_columns = cat.check_input_columns()
        if missing_columns == "":
            logging.info("Check input columns\t\tOK.")
        else:
            logging.info(
                "Check input columns\t\tFAILED. Missing columns: "
                + missing_columns.rstrip(",")
            )
            error += 1

        # check that the columns have the right format
        wrong_types = cat.check_column_dtypes()
        if wrong_types == "":
            logging.info("Check dtypes\t\t\tOK.")
        else:
            logging.info(
                "Check dtypes\t\t\tFAILED. Wrong dtypes: " + missing_columns.rstrip(",")
            )
            error += 1

        # check non-ascii characters
        non_ascii = cat.find_non_ascii()
        if non_ascii:
            logging.info(
                "Check weird characters\t\tFAILED. Non-ASCII characters found in columns and indexes: {non_ascii} "
            )
            error += 1
        else:
            logging.info("Check weird characters\t\tOK.")

    # check simbad and vizier
    logging.info("****** STELLAR CATALOGS ******")
    ping_str = Utils.ping_simbad_vizier()

    for string in ping_str.split("\n"):
        logging.info(string.rstrip("\n"))
    if "FAILED" in ping_str:
        error += 1

    if error > 0:
        raise ValueError("One or more sanity check was not succesful.")


def input(local_date):  # pragma: no cover
    """
    This function downloads and standardizes the files.

    """
    config_dict = Utils.read_config()
    Utils.service_files_initialization()
    for cat in [Koi()]:
        # Ingesting catalogs
        logging.info("****** " + cat.name + " ******")
        config_per_cat = config_dict[cat.name]
        file_path_str = cat.download_catalog(
            config_per_cat["url"], config_per_cat["file"], local_date
        )
        cat.read_csv_catalog(file_path_str)

        # Standardizing catalogs
        cat.standardize_catalog()
        cat.convert_coordinates()
        cat.fill_nan_on_coordinates()
        cat.print_catalog("StandardizedSources/" + cat.name + local_date + ".csv")

    cat_types = [Eu(), Nasa(), Oec(), Toi(), Epic()]
    for cat in cat_types:
        # Ingesting catalogs
        logging.info("****** " + cat.name + " ******")
        config_per_cat = config_dict[cat.name]
        file_path = cat.download_catalog(
            config_per_cat["url"], config_per_cat["file"], local_date
        )
        cat.read_csv_catalog(file_path)
        # Standardizing catalogs
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
        cat.assign_status()
        cat.create_catalogstatus_string("original_catalog_status")
        cat.check_mission_tables("StandardizedSources/koi" + local_date + ".csv")
        cat.create_catalogstatus_string("checked_catalog_status")
        cat.make_standardized_alias_list()
        cat.keep_columns()
        cat.print_catalog("StandardizedSources/" + cat.name + local_date + ".csv")


def run(local_date, verbose):  # pragma: no cover
    """This function runs the emc catalog"""
    emc = Emc()

    logging.info("Loading standardized files...")
    emc.data = Utils.load_standardized_catalog("StandardizedSources/nasa", local_date)
    for catalog in ["eu", "oec", "toi", "epic"]:
        emc.data = pd.concat(
            [
                emc.data,
                Utils.load_standardized_catalog(
                    "StandardizedSources/" + catalog, local_date
                ),
            ]
        )

    # fix for toi catalog that only has ".0x" and it is read as a float
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
    emc.set_exo_mercat_name()
    emc.fill_row_update(local_date)
    emc.keep_columns()
    emc.save_catalog(local_date, "_full")
    emc.remove_known_brown_dwarfs(
        local_date,
        print_flag=True,
    )
    emc.save_catalog(local_date, "")


def check(local_date):  # pragma: no cover
    error_string = ""
    emc = pd.read_csv("Exo-MerCat/exo-mercat_full" + local_date + ".csv")

    """CHECK nasa_name: check that nasa_name is null only when catalog does not contain nasa, and that it is not null 
    only when catalog contains nasa."""

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

    """CHECK eu_name: check that eu_name is null only when catalog does not contain eu, and that it is not null only when 
    catalog contains eu."""

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

    """CHECK oec_name: check that oec_name is null only when catalog does not contain oec, and that it is not null only 
    when catalog contains oec."""

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

    """CHECK toi_name: check that toi_name is null only when catalog does not contain toi, and that it is not null only 
    when catalog contains toi."""

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

    """
    CHECK mass: mass_max, mass_min and mass_url must be null only when mass is null. 
    """

    if len(emc[emc.mass.isna()]) != len(
        emc[
            (emc.mass.isna())
            & (emc.mass_max.isna())
            & (emc.mass_min.isna())
            & (emc.mass_url.isna())
        ]
    ):
        error_string = error_string + "CHECK mass.a\n"
    if len(emc[~emc.mass.isna()]) != len(
        emc[
            ~(emc.mass.isna())
            & ~(emc.mass_max.isna())
            & ~(emc.mass_min.isna())
            & ~(emc.mass_url.isna())
        ]
    ):
        error_string = error_string + "CHECK mass.b\n"

    """
    CHECK msini: msini_max, msini_min and msini_url must be null only when msini is null. 
    """

    if len(emc[emc.msini.isna()]) != len(
        emc[
            (emc.msini.isna())
            & (emc.msini_max.isna())
            & (emc.msini_min.isna())
            & (emc.msini_url.isna())
        ]
    ):
        error_string = error_string + "CHECK msini.a\n"
    if len(emc[~emc.msini.isna()]) != len(
        emc[
            ~(emc.msini.isna())
            & ~(emc.msini_max.isna())
            & ~(emc.msini_min.isna())
            & ~(emc.msini_url.isna())
        ]
    ):
        error_string = error_string + "CHECK msini.b\n"

    """
    CHECK p: p_max, p_min and p_url must be null only when p is null. 
    """

    if len(emc[emc.p.isna()]) != len(
        emc[
            (emc.p.isna())
            & (emc.p_max.isna())
            & (emc.p_min.isna())
            & (emc.p_url.isna())
        ]
    ):
        error_string = error_string + "CHECK p.a\n"
    if len(emc[~emc.p.isna()]) != len(
        emc[
            ~(emc.p.isna())
            & ~(emc.p_max.isna())
            & ~(emc.p_min.isna())
            & ~(emc.p_url.isna())
        ]
    ):
        error_string = error_string + "CHECK p.b\n"

    """
    CHECK r: r_max, r_min and r_url must be null only when r is null. 
    """

    if len(emc[emc.r.isna()]) != len(
        emc[
            (emc.r.isna())
            & (emc.r_max.isna())
            & (emc.r_min.isna())
            & (emc.r_url.isna())
        ]
    ):
        error_string = error_string + "CHECK r.a\n"
    if len(emc[~emc.r.isna()]) != len(
        emc[
            ~(emc.r.isna())
            & ~(emc.r_max.isna())
            & ~(emc.r_min.isna())
            & ~(emc.r_url.isna())
        ]
    ):
        error_string = error_string + "CHECK r.b\n"

    """
    CHECK e: e_max, e_min and e_url must be null only when e is null. 
    """

    if len(emc[emc.e.isna()]) != len(
        emc[
            (emc.e.isna())
            & (emc.e_max.isna())
            & (emc.e_min.isna())
            & (emc.e_url.isna())
        ]
    ):
        error_string = error_string + "CHECK e.a\n"
    if len(emc[~emc.e.isna()]) != len(
        emc[
            ~(emc.e.isna())
            & ~(emc.e_max.isna())
            & ~(emc.e_min.isna())
            & ~(emc.e_url.isna())
        ]
    ):
        error_string = error_string + "CHECK e.b\n"

    """
    CHECK i: i_max, i_min and i_url must be null only when i is null. 
    """

    if len(emc[emc.i.isna()]) != len(
        emc[
            (emc.i.isna())
            & (emc.i_max.isna())
            & (emc.i_min.isna())
            & (emc.i_url.isna())
        ]
    ):
        error_string = error_string + "CHECK i.a\n"
    if len(emc[~emc.i.isna()]) != len(
        emc[
            ~(emc.i.isna())
            & ~(emc.i_max.isna())
            & ~(emc.i_min.isna())
            & ~(emc.i_url.isna())
        ]
    ):
        error_string = error_string + "CHECK i.b\n"

    """CHECK bestmass: bestmass_max, bestmass_min, and bestmass_url must be null only when bestmass is null. Bestmass 
    must be null only when both mass and msini are null"""

    if len(emc[emc.bestmass.isna()]) != len(
        emc[
            (emc.bestmass.isna())
            & (emc.bestmass_max.isna())
            & (emc.bestmass_min.isna())
            & (emc.bestmass_url.isna())
        ]
    ):
        error_string = error_string + "CHECK bestmass.a\n"
    if len(emc[~emc.bestmass.isna()]) != len(
        emc[
            ~(emc.bestmass.isna())
            & ~(emc.bestmass_max.isna())
            & ~(emc.bestmass_min.isna())
            & ~(emc.bestmass_url.isna())
        ]
    ):
        error_string = error_string + "CHECK bestmass.b\n"

    if len(emc[emc.bestmass.isna()]) != len(
        emc[(emc.mass.isna()) & (emc.msini.isna())]
    ):
        error_string = error_string + "CHECK bestmass.c\n"
    if len(emc[emc.bestmass.isna() & (~(emc.mass.isna()) | ~(emc.msini.isna()))]) > 0:
        error_string = error_string + "CHECK bestmass.d\n"
    if len(emc[~emc.bestmass.isna() & (emc.mass.isna()) & (emc.msini.isna())]) > 0:
        error_string = error_string + "CHECK bestmass.d\n"

    """
    CHECK discovery_method: discovery_method should never be null (except when it is null from the source files).
    """
    if len(emc[emc.discovery_method.isna()]) > 0:
        error_string = error_string + "CHECK discovery_method.a\n"
        if emc[emc.discovery_method.isna()].oec_name.tolist() == ["HD 100546 c"]:
            error_string = (
                error_string
                + "FIXED discovery_method.a (known issue with source file OEC: HD 100546 c)\n"
            )

    """
    CHECK discovery_year: discovery_year should never be null (except when it is null from the source files).
    """
    if len(emc[emc.discovery_year.isna()]) > 0:
        error_string = error_string + "CHECK discovery_year.a (known issue)\n"

    """
    CHECK final_alias: final_alias should never be null (except when it is null from the source files).
    """
    if len(emc[emc.main_id_alias.isna()]) > 0:
        error_string = error_string + "CHECK main_id_alias.a (known issue)\n"

    if len(error_string) == 0:
        print("All checks passed.")
    else:
        print("The following checks failed:\n" + error_string)


if __name__ == "__main__":  # pragma: no cover
    main()
