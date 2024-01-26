# Exo-MerCat

## Installation
This version is compatible with Python 3.8.

```pip install -r requirements.txt```

## Usage 
```python main.py [-h] [-v] [-w]```

Optional arguments:
```
  -h, --help      show this help message and exit
  -v, --verbose   increase verbosity (default: False)
  -w, --warnings  show UserWarnings (default: False)
```

## Changelog
Compared to Alei+2020:

- All catalogs are now subclasses of a Parent Class named `Catalog` which is located in `exomercat/catalogs.py`. The `Catalog` class contains all the common methods for all catalogs. Catalog-specific methods can be found as subclass methods in the python files specific for each catalog:
  - `exomercat/eu.py` for the Exoplanet Encyclopaedia,
  - `exomercat/nasa.py` for the NASA Exoplanet Archive, 
  - `exomercat/oec.py` for the Open Exoplanet Catalogue, 
  - `exomercat/koi.py` for the Kepler Objects of Interest Catalog
  - `exomercat/epic.py` for the K2/EPIC Objects of Interest Catalog
  - `exomercat/emc.py` for the cumulative Exo-MerCat catalog object (EMC)
- Added file `replacements.ini` that includes all replacements that must be made to help uniforming the targets. The user should keep filling this file, especially if any of the log files in `EMClogs/` are not empty. Ideally, correct replacement would allow the log files to be empty afterwards. The file is handles in `exomercat/utilityfunctions.py`.
- Ingestion of NASA Exoplanet Archive and Kepler objects of interest (KOI) catalog was changed from the source. Exo-Mercat was adapted accordingly. 
- The K2 objects of interest (EPIC) catalog was ingested in addition to the KOI catalog.
- Exo-MerCat now excludes all targets whose (minimum) mass is higher than 20 Jovian masses using `remove_known_brown_dwarfs(print=True)`. If `print=True` the brown dwarfs are printed separately and saved in `UniformSources/`.
- The alias and coordinate checks were redundant at that stage of the script (see sec. 4.5) and were removed, since they are treated consistently in following functions.
- A first binary check is performed at this stage to identify potential missing binaries. Whenever the values of the `Binary` column were inconsistent for the same Host+Letter, the script automatically replaces the empty strings and the S-type strings with more specific values of Binary. It also checks if the coordinates are consistent up to 0.01 degrees. If this is not the case, it fills the `PotentialMismatch` flag to 1. If there are weird binary systems (e.g. coexistence of A, B, and/or AB binary values), the script does not perform any change but warns the user in the log file and by filling the `PotentialMismatch` flag to 2.
- The MAIN_ID query is performed by querying SIMBAD only. The first query is done using "Host+Binary". For all unsuccessful queries, a second query is performed using "Alias+Binary" for each available alias. Unsuccessful queries are repeated using only "Host" as keyword. Finally, all remaining unsuccessful queries are queried using "Host". For the ones still unsuccessful, SIMBAD is queried using "Alias" as keyword, for all available aliases. 
- Exo-MerCat now assumes the SIMBAD coordinates as official coordinates. For this reason, a coordinate check among all the entries is no longer performed. A coordinate check is performed only for the targets still missing a MAIN_ID to ensure consistency of the coordinates prior to the ConeSearch to be performed. Troublesome coordinates are printed in the log file in `EMClogs/`.
- If targets are still missing MAIN_ID, a ConeSearch is performed on SIMBAD only (and no longer the other catalogs mentioned in Alei+2020) at increasing tolerance radius. The script selects the host star with the smallest angular separation and saves the value in the column `angular_separation`.
- A second check on the binary value is performed on MAIN_ID + Letter. It is used the same way as previously described.
- The check on duplicate entries is performed once when creating the single entry. Duplicate entries are stored in `EMClogs/duplicate_entries.txt`. 
- The `AXDXEXCX` string was made more understandable. it now is contained in the `Status_string` column, which shows how the target was classified in each catalog. The most likely status is then shown in the `Status` column.
- The `YOD` and `DiscMeth` columns were renamed into `Discovery_year` and `Discovery_method` for clarity.