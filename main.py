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
from exo_mercat.configurations import *
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
parser.add_argument("-l",'--local',action='store_false',help='load previously uniformed catalogs')
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

    emc = Emc()
    if args['local']:
        for cat in [Koi(), Epic()]:
            logging.info("****** " + cat.name + " ******")
            config_per_cat = config_dict[cat.name]
            file_path_str=cat.download_catalog(config_per_cat['url'], config_per_cat["file"])
            cat.read_csv_catalog(file_path_str)
            cat.uniform_catalog()
            cat.convert_coordinates()
            cat.print_catalog("UniformSources/" + cat.name + ".csv")


        cat_types = [Eu(), Nasa(), Oec()]
        for cat in cat_types:
            logging.info("****** " + cat.name + " ******")
            config_per_cat = config_dict[cat.name]
            file_path=cat.download_catalog(config_per_cat["url"], config_per_cat["file"])
            cat.read_csv_catalog(file_path)
            cat.uniform_catalog()
            cat.convert_coordinates()
            cat.replace_known_mistakes()
            cat.uniform_name_host_letter()
            cat.identify_brown_dwarfs()
            cat.remove_theoretical_masses()
            cat.remove_known_brown_dwarfs(print_flag=True)
            cat.make_errors_absolute()
            cat.handle_reference_format()
            cat.assign_status()
            cat.check_koiepic_tables("UniformSources/koi.csv")
            cat.check_koiepic_tables("UniformSources/epic.csv")
            cat.fill_binary_column()
            cat.check_known_binary_mismatches()
            cat.make_uniform_alias_list()
            cat.keep_columns()
            cat.convert_datatypes()
            cat.create_catalogstatus_string()
            cat.print_catalog("UniformSources/" + cat.name + ".csv")
            emc.data = pd.concat([emc.data, cat.data])
    else:
        logging.info('Loading local files...')
        emc.data=pd.read_csv('UniformSources/eu.csv')
        emc.data=pd.concat([emc.data,pd.read_csv('UniformSources/nasa.csv')])
        emc.data=pd.concat([emc.data,pd.read_csv('UniformSources/oec.csv')])

    emc.data = emc.data.reset_index()
    emc.print_catalog("UniformSources/fullcatalog.csv")
    emc.alias_as_host()
    emc.check_binary_mismatch(keyword='host')
    emc.get_host_info_from_simbad()
    logging.info("Check on other catalogs:")
    #emc.tess_main_id()
    #emc.gaia_main_id()
    # emc.twomass_main_id()
    #emc.epic_main_id()
    emc.check_coordinates()
    emc.get_coordinates_from_simbad()
    emc.polish_main_id()
    emc.check_same_host_different_id()
    emc.set_common_host()
    emc.set_common_alias()
    emc.check_binary_mismatch(keyword='main_id')
    # emc.check_duplicates_in_same_catalog()
    emc.cleanup_catalog()
    emc.convert_datatypes()
    emc.merge_into_single_entry(verbose=args['verbose'])
    emc.select_best_mass()
    emc.set_exo_mercat_name()
    emc.keep_columns()
    emc.print_catalog(
        "Exo-MerCat/exo-mercat" + date.today().strftime("%m-%d-%Y") + ".csv"
    )
    emc.print_catalog("Exo-MerCat/exo-mercat.csv")


if __name__ == "__main__":
    main()
