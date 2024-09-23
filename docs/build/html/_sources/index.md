% Exo-MerCat documentation master file, created by
% sphinx-quickstart on Wed Aug 30 15:42:35 2023.
% You can adapt this file completely to your liking, but it should at least
% contain the root `toctree` directive.



```{include} ../../README.md
:relative-images:
```


```{toctree}
:caption: 'Contents:'
:maxdepth: 1

run_tap
run_locally
classes

```

Exo-MerCat is a Python software designed to generate a catalog of known and candidate exoplanets, where data is merged from various exoplanet catalogs. 

Exo-MerCat retrieves a common name for the planet target, linking its host star name with the preferred identifier in the most well-known stellar databases. This makes the merging of different input sources possible and it has allowed the release of a standardized catalog containing data from the major input catalogs used in the community: the *Exoplanets Encyclopaedia*[^1], the *NASA Exoplanet Archive*[^2], the *Open Exoplanet Catalogue*[^3], the *TESS Objects of Interest Catalog*[^4] and the *EPIC/K2 Objects of Interest Catalog*[^5].

``{warning}
    Because of the nature of the merge, this catalog is not self-consistent (i.e., all measurements for the same target don't belong to the same reference paper). 
    Rather, it is a collection of the most precise estimates for each planetary parameter (mass, minimum mass, radius, period, semi-major axis, eccentricity, inclination), based on the lowest relative error, and the corresponding reference for each preferred estimate. 
``
> [!NOTE]
> Useful information that users should know, even when skimming content.


[^1]:http://exoplanet.eu/ 
[^2]:https://exoplanetarchive.ipac.caltech.edu/
[^3]:https://openexoplanetcatalogue.com/ 
[^4]:tess
[^5]:epic


Standardizing the data is crucial, as it involves comparing entries from different catalogs, a task that requires careful selection of software tools. One major challenge is identifying aliases and verifying host star coordinates due to discrepancies in notation across catalogs.

The program includes a Graphical User Interface for catalog filtering and automatic plotting. Dependencies include the pandas package for flexible data manipulation and astropy for handling astronomical coordinates and parameters. Additional packages like astroquery and pyvo are used for accessing and retrieving data from various sources conforming to International Virtual Observatory Alliance standards.

The software utilizes the Table Access Protocol and Astronomical Data Query Language for querying and filtering catalogs, enabling spatial cross-matching and custom manipulations.

Datasets from different sources are retrieved using varied methods, such as wget for some catalogs like the Exoplanet Orbit Database and NASA Exoplanet Archive. Specific columns are selected during retrieval to minimize data download and ensure essential information is included.

There is a standardization process necessary to merge diverse exoplanet catalogs effectively. Key operations include selecting relevant columns, renaming columns for easy merging, handling aliases, removing unnecessary spaces, and standardizing target names. Notable steps include labeling planets within the same system and handling exceptions like Kepler Objects of Interest and binary systems.

Additionally, it addresses the retrieval of calculated values, standardizing retrieval method names, and hierarchical indexing. The process includes checks for Kepler Objects of Interest status, updating with the official Kepler identifier if available.

There are discrepancies in the status of confirmed, candidate, and false positive exoplanets across catalogs, potentially due to delays in updates or misinterpretations. The routine also addresses missing coordinates for some Kepler candidates and discusses plans to adapt the process for future TESS candidates.


Initially, the function identifies potential aliases for host stars across the catalogs and standardizes them for consistency. It queries the SIMBAD database to confirm host star identities and resolves duplicate entries by selecting a common identifier. This process successfully resolves many alias inconsistencies but also encounters cases where no matches are found.

Subsequently, the software checks the consistency of coordinates to avoid mismatches during merging. It identifies and corrects discrepancies in coordinates, flagging instances where differences exceed a certain threshold.

Additionally, the function retrieves main identifiers for host stars by querying various astronomical databases such as SIMBAD, VizieR, and GAIA. This process significantly reduces the number of targets without identifiers, with remaining cases addressed through iterative queries with increasing tolerances. Duplicate entries within catalogs are automatically identified and logged for further review.

Overall, these checks ensure the accuracy and completeness of host star identifiers and coordinates, facilitating the merging process.

In catalog retrieval, the merged catalog is indexed hierarchically, and multiple entries for each planet are collapsed into a single entry based on measurement precision. Relative errors for each parameter are calculated, and the dataset with the smallest relative error is selected for each parameter. Host star and planet names are standardized and prioritized based on commonly known identifiers. Duplicate entries are collapsed into one row, with the original catalogs retained in a separate column. The status of each target is stored as a string indicating its presence, candidacy, or confirmation in various catalogs.

The latest version of the catalog is made available through a dedicated TAP service, with subsequent versions generated as workflow runs.

In the performance evaluation section, the software's performance is assessed by comparing the results of two merging runs: one without any preprocessing (referred to as "Simple") and one with the full suite of functions applied (referred to as "EMC"). By comparing the two runs, improvements in the final catalog achieved by the software are analyzed.

Overall, the software effectively standardizes and merges multiple catalogs, improving the consistency and accuracy of the resulting merged catalog.




For further info, check out {doc}`run_tap`. Read installation instructions in {doc}`run_locally`.
