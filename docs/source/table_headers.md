# Table headers

We refer to Alei et al2024 for the translation between Exo-MerCat 1.0 and 2.x.


| Column headers             | Meaning                                                                                                             | Type    |
|----------------------------|---------------------------------------------------------------------------------------------------------------------|---------|
| `exo_mercat_name`          | Planet name chosen by Exo-MerCat                                                                                    | `str`   |     
| `nasa_name`                | Planet name in the NASA Exoplanet Archive                                                                           | `str`   |
| `toi_name`                 | Planet name in the TESS Objects of Interest Table                                                                   | `str`   |
| `epic_name`                | Planet name in the Kepler/K2 Objects of Interest Table                                                              | `str`   |
| `eu_name`                  | Planet name in the Exoplanet Encyclopaedia                                                                          | `str`   |
| `host`                     | Host star name                                                                                                      | `str`   |     
| `letter`                   | Letter labeling the planet                                                                                          | `str`   |
| `main_id`                  | Main identifier of the host star from SIMBAD or TIC catalogs                                                        | `str`   |
| `binary`                   | String labeling the binary host star, if any                                                                        | `str`   |
| `main_id_ra`               | J2000 right ascension in degrees from SIMBAD                                                                        | `str`   |
| `main_id_dec`              | J2000 declination in degrees from SIMBAD                                                                            | `str`   |
| `mass`                     | Planet mass in Jovian masses                                                                                        | `float` | 
| `mass_max`                 | Positive error on the mass in Jovian masses                                                                         | `float` | 
| `mass_min`                 | Negative error on the mass in Jovian masses (absolute value)                                                        | `float` | 
| `mass_url`                 | Bibcode of the reference paper for the mass value                                                                   | `str`   |
| `msini`                    | Planet minimum mass in Jovian masses                                                                                | `float` | 
| `msini_max`                | Positive error on the minimum mass in Jovian masses                                                                 | `float` | 
| `msini_min`                | Negative error on the moinimum mass in Jovian masses (absolute value)                                               | `float` | 
| `msini_url`                | Bibcode of the reference paper for the minimum mass value                                                           | `str`   |
| `bestmass`                 | Most precise value between mass and minimum mass in Jovian masses                                                   | `float` | 
| `bestmass_max`             | Positive error on the best mass in Jovian masses                                                                    | `float` | 
| `bestmass_min`             | Negative error on the best mass in Jovian masses (absolute value)                                                   | `float` | 
| `bestmass_url`             | Bibcode of the reference paper for the best mass value                                                              | `str`   |
| `bestmass_provenance`      | String labeling the origin of the best mass (mass or minimum mass)                                                  | `str`   |
| `p`                        | Planet orbital period in days                                                                                       | `float` | 
| `p_max`                    | Positive error on the period in days                                                                                | `float` | 
| `p_min`                    | Negative error on the period in days (absolute value)                                                               | `float` | 
| `p_url`                    | Bibcode of the reference paper for the period value                                                                 | `str`   |
| `r`                        | Planet radius in Jovian radii                                                                                       | `float` | 
| `r_max`                    | Positive error on the radius in Jovian radii                                                                        | `float` | 
| `r_min`                    | Negative error on the radius in Jovian radii (absolute value)                                                       | `float` | 
| `r_url`                    | Bibcode of the reference paper for the radius value                                                                 | `str`   |
| `a`                        | Planet semi-major axis in au                                                                                        | `float` | 
| `a_max`                    | Positive error on the semi-major axis in au                                                                         | `float` | 
| `a_min`                    | Negative error on the semi-major axis in au (absolute value)                                                        | `float` | 
| `a_url`                    | Bibcode of the reference paper for the semi-major axis value                                                        | `str`   |
| `e`                        | Eccentricity of the planet (scalar between 0 and 1)                                                                 | `float` | 
| `e_max`                    | Positive error on the eccentricity (scalar)                                                                         | `float` | 
| `e_min`                    | Negative error on the eccentricity (scalar)                                                                         | `float` | 
| `e_url`                    | Bibcode of the reference paper for the eccentricity value                                                           | `str`   |
| `i`                        | Planet inclination in degrees                                                                                       | `float` | 
| `i_max`                    | Positive error on the inclination in degrees                                                                        | `float` | 
| `i_min`                    | Negative error on the minimum mass in degrees (absolute value)                                                      | `float` | 
| `i_url`                    | Bibcode of the reference paper for the inclination value                                                            | `str`   |
| `discovery_method`         | Planet discovery method                                                                                             | `str`   |
| `status`                   | Planet status of the planet preferred by Exo-MerCat                                                                 | `str`   |
| `original_status_string`   | String listing the planet status in all available input sources                                                     | `str`   |
| `checked_status_string`    | String listing the planet status in all available input sources after the KOI check                                 | `str`   |
| `confirmed`                | Number of CONFIRMED status in `checked_status_string`                                                               | `str`   |
| `discovery_year`           | Planet discovery year                                                                                               | `int`   |
| `main_id_aliases`          | String listing all known aliases for the host star                                                                  | `str`   |
| `catalog`                  | String listing the catalogs in which the target appears                                                             | `str`   |
| `angular_separation`       | String listing the unique angular separations of the merged entries                                                 | `str`   |
| `angular_separation_flag`  | Flag for non-null values of `angular_separation`                                                                    | `int`   |
| `main_id_provenance`       | Provenance of the main identifier                                                                                   | `str`   |
| `binary_mismatch_flag`     | Flag for possible binary mismatches                                                                                 | `int`   |
| `coordinate_mismatch_flag` | Flag for possible coordinate mismatches                                                                             | `int`   |
| `coordinate_mismatch`      | String showing in what coordinate the binary mismatch occurred  (if `coordinate_mismatch_flag` is not null)         | `str`   |
| `duplicate_catalog_flag`   | Flag for duplicates during merging                                                                                  | `int`   |
| `duplicate_names`          | String listing the planet name in all available input sources (only if `duplicate_catalog_flag` is not null)        | `str`   |
|`merging_mismatch_flag` | Flag for potential duplicates in final catalog (because of mismatching \inline{p} or \inline{a}, or fallback merge) | `int`   | 
| `row_update` | Date of the last update of the row                                                                                  | `str`   | 
