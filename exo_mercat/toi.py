import pandas as pd

from exo_mercat.catalogs import Catalog
from astropy.coordinates import SkyCoord
import astropy.units as u
from astropy import constants as const
import numpy as np
import logging
import pyvo

tap_service = pyvo.dal.TAPService(" http://TAPVizieR.u-strasbg.fr/TAPVizieR/tap/")

class Toi(Catalog):
    def __init__(self) -> None:
        """
        The __init__ function is called when the class is instantiated. It sets up the instance of the class,
        and defines any variables that will be used by all instances of this class.
        """
        super().__init__()
        self.name = "toi"
        self.data = None

    def uniform_catalog(self) -> None:
        """

        """

        self.data["catalog"] = self.name
        self.data["TOI"] = 'TOI-' + self.data["toi"].astype(str)

        self.data["TIC_host"] = 'TIC ' + self.data["tid"].astype(str)
        self.data["TOI_host"] = 'TOI-' + self.data["toipfx"].astype(str)

        self.data["TIC_host2"] = 'TIC-' + self.data["tid"].astype(str)
        self.data["TOI_host2"] = 'TOI ' + self.data["toipfx"].astype(str)

        self.data['host']=self.data['TIC_host']
        self.data["letter"] = self.data['TOI'].str[-3:]
        self.data['alias'] = self.data[[ 'TOI_host','TOI_host2', 'TIC_host2']].agg(','.join, axis=1)


        # MORE ALIAS BY QUERYING VIZIER
        for tid in self.data['tid']:
            query = '''SELECT "IV/38/tic".TIC,  "IV/38/tic".UCAC4,  "IV/38/tic"."2MASS",  "IV/38/tic".WISEA,  "IV/38/tic".GAIA,  "IV/38/tic".KIC,
            "IV/38/tic".HIP,  "IV/38/tic".objID,  "IV/38/tic".TYC
            FROM "IV/38/tic"
            WHERE  "IV/38/tic".TIC ='''+str(tid)

            result = tap_service.search(query)
            result=result.to_table().to_pandas()
            result=result.astype(str)
            result['UCAC4']='UCAC4 '+result['UCAC4'].replace('','<NA>')
            result['2MASS']='2MASS J'+result['2MASS'].replace('','<NA>')
            result['WISEA']='WISE '+result['WISEA'].replace('','<NA>')
            result['GAIA']='Gaia DR2 '+result['GAIA'].replace('','<NA>')
            result['KIC']='KIC '+result['KIC'].replace('','<NA>')
            result['HIP']= 'HIP '+result['HIP'].replace('','<NA>')
            result['TYC']='TYC '+result['TYC'].replace('','<NA>')
            for col in result.columns:
                result.loc[result[col].str.contains('<NA>'), col] = ''
            result['alias'] = result[['UCAC4', '2MASS', 'WISEA', 'GAIA', 'KIC', 'HIP', 'TYC']].agg(','.join, axis=1)
            self.data.loc[self.data.tid == tid,'alias_vizier']=result['alias'][0]

        self.data['alias'] = self.data['alias'] + ','+ self.data['alias_vizier']

        for i in self.data.index:
            self.data.at[i, 'alias'] = ",".join(
                sorted([x for x in set(self.data.at[i, 'alias'].split(",")) if x])
            )
        self.data["name"] = self.data["TOI"]
        self.data["catalog_name"] = self.data["TOI"]
        self.data = self.data.rename(
            columns={
                "pl_orbper": "p",
                "pl_orbpererr2": "p_min",
                "pl_orbpererr1": "p_max",

            }
        )

        self.data["mass"] = np.nan
        self.data["mass_min"] = np.nan
        self.data["mass_max"] = np.nan
        self.data["msini"] = np.nan
        self.data["msini_min"] = np.nan
        self.data["msini_max"] = np.nan
        self.data["a"] = np.nan
        self.data["a_min"] = np.nan
        self.data["a_max"] = np.nan
        self.data["e"] = np.nan
        self.data["e_min"] = np.nan
        self.data["e_max"] = np.nan
        self.data["i"] = np.nan
        self.data["i_min"] = np.nan
        self.data["i_max"] = np.nan
        self.data["Age (Gyrs)"] = np.nan
        self.data["Age_max"] = np.nan
        self.data["Age_min"] = np.nan
        self.data["Mstar"] = np.nan
        self.data["Mstar_max"] = np.nan
        self.data["Mstar_min"] = np.nan


        self.data['r']=self.data['pl_rade']*const.R_earth/const.R_jup
        self.data['r_min']=self.data['pl_radeerr2']*const.R_earth/const.R_jup
        self.data['r_max']=self.data['pl_radeerr1']*const.R_earth/const.R_jup


        self.data['discovery_year']=self.data['toi_created'].str[:4]


        self.data['discovery_method'] = 'Transit'

        self.data['reference']='toi'


        logging.info("Catalog uniformed.")

    def convert_coordinates(self)->None:
        pass

    def remove_theoretical_masses(self) -> None:
        ''' there are no masses here in the first place'''
        pass

    def handle_reference_format(self) -> None:
        """
        The handle_reference_format function is used to create a url for each reference in the references list.
        Since the TOI table does not provide references, we just use "TOI" as a keyword.
        """
        for item in ["e", "mass", "msini", "i", "a", "p", "r"]:
            self.data[item + "_url"] = self.data[item].apply(
                lambda x: "" if pd.isna(x) or np.isinf(x) else "toi"
            )
        logging.info("Reference columns uniformed.")

    def assign_status(self):


        ### DISPOSITION
        replace_dict={"APC":"CONTROVERSIAL","CP":"CONFIRMED","FA":"FALSE POSITIVE","FP":"FALSE POSITIVE","KP":"CONFIRMED","PC":"CANDIDATE","":"UNKNOWN"}
        self.data["status"] = self.data["tfopwg_disp"].replace(replace_dict)
        logging.info("Status column assigned.")
        logging.info("Updated Status:")
        logging.info(self.data.status.value_counts())

