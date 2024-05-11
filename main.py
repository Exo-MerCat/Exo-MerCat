#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Oct  7 10:35:12 2019

@author: eleonoraalei
"""

from exo_mercat.nasa import Nasa
from exo_mercat.eu import Eu
from exo_mercat.oec import Oec
from exo_mercat.epic import Epic
from exo_mercat.emc import Emc
from exo_mercat.koi import Koi
from exo_mercat.toi import Toi
from exo_mercat.utility_functions import UtilityFunctions as Utils
import socket
import warnings
import pandas as pd
from datetime import date
import logging
from argparse import ArgumentParser, ArgumentDefaultsHelpFormatter

# Parse command line arguments
parser = ArgumentParser(formatter_class=ArgumentDefaultsHelpFormatter)
parser.add_argument("-v", "--verbose", action="store_true", help="increase verbosity")
parser.add_argument("-w", "--warnings", action="store_true", help="show UserWarnings")
parser.add_argument(
    "-l", "--local", action="store_false", help="load previously uniformed catalogs"
)
parser.add_argument(
    "-d", "--date", help="load a specific date (MM-DD-YYYY)"
)
args = vars(parser.parse_args())

if args["verbose"]:
    logging.basicConfig(format="%(asctime)s: %(message)s", level=logging.INFO)
if args["warnings"]:
    warnings.filterwarnings("once")
else:
    warnings.filterwarnings("ignore")
local_date = ''
if args['date']:
    local_date=args['date']


timeout = 100000
socket.setdefaulttimeout(timeout)


def main():
    """
    The main function of this module.

    """
    config_dict = Utils.read_config()
    Utils.service_files_initialization()

    emc = Emc()

    #
    if args["local"]:
        for cat in [Koi()]:

            ### Initalizing catalogs
            logging.info("****** " + cat.name + " ******")
            config_per_cat = config_dict[cat.name]
            file_path_str = cat.download_catalog(
                config_per_cat["url"], config_per_cat["file"], local_date
            )
            cat.read_csv_catalog(file_path_str)

            ### Uniforming catalogs
            cat.uniform_catalog()
            cat.convert_coordinates()
            cat.fill_nan_on_coordinates()
            cat.print_catalog("UniformSources/" + cat.name + ".csv")

        cat_types = [Eu(), Nasa(), Oec(),Toi(),Epic()]
        for cat in cat_types:

            ### Initializing catalogs
            logging.info("****** " + cat.name + " ******")
            config_per_cat = config_dict[cat.name]
            file_path = cat.download_catalog(
                config_per_cat["url"], config_per_cat["file"], local_date
            )
            cat.read_csv_catalog(file_path)
            ### Uniforming catalogs
            cat.uniform_catalog()
            cat.convert_coordinates()
            cat.fill_nan_on_coordinates()
            cat.fill_binary_column()
            cat.replace_known_mistakes()
            cat.uniform_name_host_letter()
            cat.identify_brown_dwarfs()
            cat.remove_theoretical_masses()
            cat.make_errors_absolute()
            cat.handle_reference_format()
            cat.assign_status()
            cat.create_catalogstatus_string("original_catalog_status")
            cat.check_mission_tables("UniformSources/koi.csv")
            # cat.check_mission_tables("UniformSources/epic.csv")
            cat.create_catalogstatus_string("checked_catalog_status")
            cat.make_uniform_alias_list()
            cat.keep_columns()
            # cat.convert_datatypes()
            cat.print_catalog("UniformSources/" + cat.name + ".csv")
            emc.data = pd.concat([emc.data, cat.data])
    else:
        logging.info("Loading local files...")
        emc.data = pd.read_csv("UniformSources/eu.csv")
        emc.data = pd.concat([emc.data, pd.read_csv("UniformSources/nasa.csv")])
        emc.data = pd.concat([emc.data, pd.read_csv("UniformSources/oec.csv")])
        emc.data = pd.concat([emc.data, pd.read_csv("UniformSources/toi.csv")])
        emc.data = pd.concat([emc.data, pd.read_csv("UniformSources/epic.csv")])
        #fix for toi catalog that only has ".0x" and it is read as a float
        emc.data.letter = emc.data.letter.astype(str)
        emc.data.letter = emc.data.letter.str[-3:]
    # emc.convert_datatypes()
    emc.data = emc.data.reset_index()
    emc.print_catalog("UniformSources/fullcatalog.csv")
    ### Matching with stellar catalogs
    emc.alias_as_host()
    emc.check_binary_mismatch(keyword="host")
    emc.prepare_columns_for_mainid_search()
    emc.get_host_info_from_simbad()
    emc.get_coordinates_from_simbad()
    emc.get_host_info_from_tic()
    emc.get_coordinates_from_tic()
    emc.fill_missing_main_id()
    emc.check_coordinates()
    emc.polish_main_id()
    emc.data.to_csv('UniformSources/all.csv')

    emc.check_same_host_different_id()
    emc.check_same_coords_different_id()
    emc.group_by_list_id_check_main_id()
    emc.group_by_main_id_set_final_alias()
    emc.check_binary_mismatch(keyword="main_id")
    emc.cleanup_catalog()

    ### Merging entries
    emc.group_by_period_check_letter()
    emc.group_by_letter_check_period(verbose=args["verbose"])
    emc.potential_duplicates_after_merging()
    emc.select_best_mass()
    emc.set_exo_mercat_name()
    # emc.convert_datatypes()
    emc.fill_row_update(local_date)
    emc.keep_columns()
    emc.save_catalog(local_date,'_full')
    emc.remove_known_brown_dwarfs(print_flag=True)
    emc.save_catalog(local_date,'')
if __name__ == "__main__":
    main()
