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
import pandas as pd
from datetime import date
import logging

import warnings
warnings.filterwarnings("ignore", 'This pattern is interpreted as a regular expression')

logging.basicConfig(format='%(levelname)s:%(message)s', level=logging.INFO)


timeout = 100000
socket.setdefaulttimeout(timeout)


def main():
    """
    The main function of this module.

    """
    config_dict = read_config()
    service_files_initialization()

    for cat in [Koi(), Epic()]:
        logging.info('****** '+cat.name+' ******')
        config_per_cat = config_dict[cat.name]
        cat.download_and_save_cat(config_per_cat['url'], config_per_cat['file'])
        cat.uniform_catalog()
        cat.convert_coordinates()
        cat.print_catalog('UniformSources/' + cat.name + '.csv')

    emc = Emc()

    cat_types = [Eu(), Nasa(), Oec()]
    for cat in cat_types:
        logging.info('****** '+cat.name+' ******')
        config_per_cat = config_dict[cat.name]
        cat.download_and_save_cat(config_per_cat['url'], config_per_cat['file'])
        cat.uniform_catalog()
        cat.uniform_name_host_letter()
        cat.remove_theoretical_masses()
        cat.remove_known_brown_dwarfs(print=True)
        cat.make_errors_absolute()
        cat.handle_reference_format()
        cat.replace_known_mistakes()
        cat.assign_status()
        cat.check_koiepic_tables('UniformSources/koi.csv')
        cat.check_koiepic_tables('UniformSources/epic.csv')
        cat.fill_binary_column()
        cat.convert_coordinates()
        cat.make_uniform_alias_list()
        cat.replace_known_mistakes()
        cat.keep_columns()
        cat.convert_datatypes()
        cat.data = cat.data.sort_values(by="Name")
        cat.print_catalog('UniformSources/' + cat.name + '.csv')
        cat.create_catalogstatus_string()
        emc.data = pd.concat([emc.data, cat.data])

    emc.data = emc.data.reset_index()
    emc.alias_as_host()
    # TODO: check binary for same target
    # TODO: check default coordinate mismatch
    emc.get_host_info_from_simbad()
    print('Check on other catalogs:')
    emc.tess_main_id()
    emc.gaia_main_id()
    # emc.twomass_main_id()
    emc.epic_main_id()
    emc.check_coordinates()
    emc.simbad_coordinate_check()
    emc.check_same_host_different_id()
    emc.set_common_host()
    emc.set_common_alias()
    emc.check_binary_mismatch()
    emc.check_duplicates_in_same_catalog()
    emc.convert_datatypes()
    emc.cleanup_catalog()
    emc.convert_datatypes()
    emc.merge_into_single_entry()
    emc.select_best_mass()
    emc.keep_columns()
    emc.print_catalog('Exo-MerCat/exo-mercat'+date.today().strftime("%m-%d-%Y")+'.csv')
    emc.print_catalog('Exo-MerCat/exo-mercat.csv')
    # emc.check_coordinates()
    #

if __name__ == "__main__":
    main()
