#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Oct  7 10:35:12 2019

@author: eleonoraalei
"""

from exomercat.nasa import Nasa
from exomercat.eu import Eu
from exomercat.oec import Oec
from exomercat.epic import Epic
from exomercat.emc import Emc
from exomercat.koi import Koi
from exomercat.configurations import *
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
args = vars(parser.parse_args())

if args["verbose"]:
    logging.basicConfig(format="%(asctime)s: %(message)s", level=logging.INFO)
if args["warnings"]:
    warnings.filterwarnings("once")
else:
    warnings.filterwarnings("ignore")


timeout = 100000
socket.setdefaulttimeout(timeout)


def main():
    """
    The main function of this module.

    """
    config_dict = read_config()
    service_files_initialization()

    for cat in [Koi(), Epic()]:
        logging.info("****** " + cat.name + " ******")
        config_per_cat = config_dict[cat.name]
        cat.download_and_save_cat(config_per_cat["url"], config_per_cat["file"])
        cat.uniform_catalog()
        cat.convert_coordinates()
        cat.print_catalog("UniformSources/" + cat.name + ".csv")

    emc = Emc()

    cat_types = [Eu(), Nasa(), Oec()]
    for cat in cat_types:
        logging.info("****** " + cat.name + " ******")
        config_per_cat = config_dict[cat.name]
        cat.download_and_save_cat(config_per_cat["url"], config_per_cat["file"])
        cat.uniform_catalog()
        cat.convert_coordinates()
        cat.replace_known_mistakes()
        cat.uniform_name_host_letter()
        cat.identify_brown_dwarfs()
        cat.remove_theoretical_masses()
        cat.remove_known_brown_dwarfs(print=True)
        cat.make_errors_absolute()
        cat.handle_reference_format()
        cat.assign_status()
        cat.check_koiepic_tables("UniformSources/koi.csv")
        cat.check_koiepic_tables("UniformSources/epic.csv")
        cat.fill_binary_column()
        cat.check_known_binary_mismatches()
        cat.make_uniform_alias_list()
        #cat.replace_known_mistakes()
        cat.keep_columns()
        cat.convert_datatypes()
        cat.print_catalog("UniformSources/" + cat.name + ".csv")
        cat.create_catalogstatus_string()
        emc.data = pd.concat([emc.data, cat.data])

    emc.data = emc.data.reset_index()
    emc.print_catalog("UniformSources/fullcatalog.csv")
    emc.alias_as_host()
    emc.check_binary_mismatch(keyword='Host')
    emc.get_host_info_from_simbad()
    print("Check on other catalogs:")
    emc.tess_main_id()
    emc.gaia_main_id()
    # emc.twomass_main_id()
    emc.epic_main_id()
    emc.check_coordinates()
    emc.get_coordinates_from_simbad()
    emc.check_same_host_different_id()
    emc.set_common_host()
    emc.set_common_alias()
    emc.check_binary_mismatch(keyword='MAIN_ID')
    # emc.check_duplicates_in_same_catalog()
    emc.cleanup_catalog()
    emc.convert_datatypes()
    emc.merge_into_single_entry()
    emc.select_best_mass()
    emc.keep_columns()
    emc.print_catalog(
        "Exo-MerCat/exo-mercat" + date.today().strftime("%m-%d-%Y") + ".csv"
    )
    emc.print_catalog("Exo-MerCat/exo-mercat.csv")



if __name__ == "__main__":
    main()
