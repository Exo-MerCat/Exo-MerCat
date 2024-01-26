import pandas as pd

error_string=""
emc=pd.read_csv('Exo-MerCat/exo-mercat.csv')

'''
CHECK nasa_name: check that nasa_name is null only when catalog does not contain nasa, and that it is not null only when catalog contains nasa.
'''

if len(emc[emc.nasa_name.isna()])!= len(emc[(emc.nasa_name.isna()) & ~(emc.catalog.str.contains('nasa'))]):
    error_string=error_string+"CHECK nasa_name.a\n"
if  len(emc[~(emc.nasa_name.isna()) & ~(emc.catalog.str.contains('nasa'))])>0:
    error_string=error_string+"CHECK nasa_name.b\n"

if len(emc[~emc.nasa_name.isna()])!= len(emc[~(emc.nasa_name.isna()) & (emc.catalog.str.contains('nasa'))]):
    error_string=error_string+"CHECK nasa_name.c\n"
if  len(emc[(emc.nasa_name.isna()) & (emc.catalog.str.contains('nasa'))])>0:
    error_string=error_string+"CHECK nasa_name.d\n"


'''
CHECK eu_name: check that eu_name is null only when catalog does not contain eu, and that it is not null only when catalog contains eu.
'''

if len(emc[emc.eu_name.isna()])!= len(emc[(emc.eu_name.isna()) & ~(emc.catalog.str.contains('eu'))]):
    error_string=error_string+"CHECK eu_name.a\n"
if  len(emc[~(emc.eu_name.isna()) & ~(emc.catalog.str.contains('eu'))])>0:
    error_string=error_string+"CHECK eu_name.b\n"

if len(emc[~emc.eu_name.isna()])!= len(emc[~(emc.eu_name.isna()) & (emc.catalog.str.contains('eu'))]):
    error_string=error_string+"CHECK eu_name.c\n"
if  len(emc[(emc.eu_name.isna()) & (emc.catalog.str.contains('eu'))])>0:
    error_string=error_string+"CHECK eu_name.d\n"

'''
CHECK oec_name: check that oec_name is null only when catalog does not contain oec, and that it is not null only when catalog contains oec.
'''

if len(emc[emc.oec_name.isna()])!= len(emc[(emc.oec_name.isna()) & ~(emc.catalog.str.contains('oec'))]):
    error_string=error_string+"CHECK oec_name.a\n"
if  len(emc[~(emc.oec_name.isna()) & ~(emc.catalog.str.contains('oec'))])>0:
    error_string=error_string+"CHECK oec_name.b\n"

if len(emc[~emc.oec_name.isna()])!= len(emc[~(emc.oec_name.isna()) & (emc.catalog.str.contains('oec'))]):
    error_string=error_string+"CHECK oec_name.c\n"
if  len(emc[(emc.oec_name.isna()) & (emc.catalog.str.contains('oec'))])>0:
    error_string=error_string+"CHECK oec_name.d\n"

'''
CHECK toi_name: check that toi_name is null only when catalog does not contain toi, and that it is not null only when catalog contains toi.
'''

if len(emc[emc.toi_name.isna()])!= len(emc[(emc.toi_name.isna()) & ~(emc.catalog.str.contains('toi'))]):
    error_string=error_string+"CHECK toi_name.a\n"
if  len(emc[~(emc.toi_name.isna()) & ~(emc.catalog.str.contains('toi'))])>0:
    error_string=error_string+"CHECK toi_name.b\n"

if len(emc[~emc.toi_name.isna()])!= len(emc[~(emc.toi_name.isna()) & (emc.catalog.str.contains('toi'))]):
    error_string=error_string+"CHECK toi_name.c\n"
if  len(emc[(emc.toi_name.isna()) & (emc.catalog.str.contains('toi'))])>0:
    error_string=error_string+"CHECK toi_name.d\n"


'''
CHECK mass: mass_max, mass_min and mass_url must be null only when mass is null. 
'''

if len(emc[emc.mass.isna()])!=len(emc[(emc.mass.isna()) & (emc.mass_max.isna()) & (emc.mass_min.isna()) & (emc.mass_url.isna())]) :
    error_string=error_string+"CHECK mass.a\n"
if len(emc[~emc.mass.isna()])!=len(emc[~(emc.mass.isna()) & ~(emc.mass_max.isna()) & ~(emc.mass_min.isna()) & ~(emc.mass_url.isna())]) :
    error_string=error_string+"CHECK mass.b\n"

'''
CHECK msini: msini_max, msini_min and msini_url must be null only when msini is null. 
'''

if len(emc[emc.msini.isna()])!=len(emc[(emc.msini.isna()) & (emc.msini_max.isna()) & (emc.msini_min.isna()) & (emc.msini_url.isna())]) :
    error_string=error_string+"CHECK msini.a\n"
if len(emc[~emc.msini.isna()])!=len(emc[~(emc.msini.isna()) & ~(emc.msini_max.isna()) & ~(emc.msini_min.isna()) & ~(emc.msini_url.isna())]) :
    error_string=error_string+"CHECK msini.b\n"


'''
CHECK p: p_max, p_min and p_url must be null only when p is null. 
'''

if len(emc[emc.p.isna()])!=len(emc[(emc.p.isna()) & (emc.p_max.isna()) & (emc.p_min.isna()) & (emc.p_url.isna())]) :
    error_string=error_string+"CHECK p.a\n"
if len(emc[~emc.p.isna()])!=len(emc[~(emc.p.isna()) & ~(emc.p_max.isna()) & ~(emc.p_min.isna()) & ~(emc.p_url.isna())]) :
    error_string=error_string+"CHECK p.b\n"

'''
CHECK r: r_max, r_min and r_url must be null only when r is null. 
'''

if len(emc[emc.r.isna()])!=len(emc[(emc.r.isna()) & (emc.r_max.isna()) & (emc.r_min.isna()) & (emc.r_url.isna())]) :
    error_string=error_string+"CHECK r.a\n"
if len(emc[~emc.r.isna()])!=len(emc[~(emc.r.isna()) & ~(emc.r_max.isna()) & ~(emc.r_min.isna()) & ~(emc.r_url.isna())]) :
    error_string=error_string+"CHECK r.b\n"


'''
CHECK e: e_max, e_min and e_url must be null only when e is null. 
'''

if len(emc[emc.e.isna()])!=len(emc[(emc.e.isna()) & (emc.e_max.isna()) & (emc.e_min.isna()) & (emc.e_url.isna())]) :
    error_string=error_string+"CHECK e.a\n"
if len(emc[~emc.e.isna()])!=len(emc[~(emc.e.isna()) & ~(emc.e_max.isna()) & ~(emc.e_min.isna()) & ~(emc.e_url.isna())]) :
    error_string=error_string+"CHECK e.b\n"

'''
CHECK i: i_max, i_min and i_url must be null only when i is null. 
'''

if len(emc[emc.i.isna()])!=len(emc[(emc.i.isna()) & (emc.i_max.isna()) & (emc.i_min.isna()) & (emc.i_url.isna())]) :
    error_string=error_string+"CHECK i.a\n"
if len(emc[~emc.i.isna()])!=len(emc[~(emc.i.isna()) & ~(emc.i_max.isna()) & ~(emc.i_min.isna()) & ~(emc.i_url.isna())]) :
    error_string=error_string+"CHECK i.b\n"


'''
CHECK bestmass: bestmass_max, bestmass_min, and bestmass_url must be null only when bestmass is null. Bestmass must be null only when both mass and msini are null 
'''

if len(emc[emc.bestmass.isna()]) != len(emc[(emc.bestmass.isna()) & (emc.bestmass_max.isna()) & (emc.bestmass_min.isna()) & (emc.bestmass_url.isna())]):
    error_string = error_string + "CHECK bestmass.a\n"
if len(emc[~emc.bestmass.isna()]) != len(
        emc[~(emc.bestmass.isna()) & ~(emc.bestmass_max.isna()) & ~(emc.bestmass_min.isna()) & ~(emc.bestmass_url.isna())]):
    error_string = error_string + "CHECK bestmass.b\n"

if len(emc[emc.bestmass.isna()]) != len(emc[(emc.mass.isna()) & (emc.msini.isna())]):
    error_string = error_string + "CHECK bestmass.c\n"
if len(emc[emc.bestmass.isna() & (~(emc.mass.isna())  | ~(emc.msini.isna()))])>0:
        error_string = error_string + "CHECK bestmass.d\n"
if len(emc[~emc.bestmass.isna() & (emc.mass.isna())  & (emc.msini.isna())])>0:
        error_string = error_string + "CHECK bestmass.d\n"


'''
CHECK discovery_method: discovery_method should never be null (except when it is null from the source files).
'''
if len(emc[emc.discovery_method.isna() ])>0:
        error_string = error_string + "CHECK discovery_method.a\n"
        if emc[emc.discovery_method.isna()].oec_name.tolist()==['HD 100546 c']:
            error_string = error_string + "FIXED discovery_method.a (known issue with source file OEC: HD 100546 c)\n"

'''
CHECK discovery_year: discovery_year should never be null (except when it is null from the source files).
'''
if len(emc[emc.discovery_year.isna() ])>0:
        error_string = error_string + "CHECK discovery_year.a (known issue)\n"

'''
CHECK final_alias: final_alias should never be null.
'''
if len(emc[emc.final_alias.isna() ])>0:
        error_string = error_string + "CHECK final_alias.a\n"





if len(error_string)==0:
    print('All checks passed.')
else:
    print('The following checks failed:\n'
          +error_string)