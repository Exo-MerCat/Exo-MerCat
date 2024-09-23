

# Exo-MerCat: The Exoplanet Merged Catalog

Exo-MerCat is a Python software designed to generate a catalog of known and candidate exoplanets, where data is merged from various exoplanet catalogs. 

Exo-MerCat retrieves a common name for the planet target, linking its host star name with the preferred identifier in the most well-known stellar databases. This makes the merging of different input sources possible and it has allowed the release of a standardized catalog containing data from the major input catalogs used in the community: the *Exoplanets Encyclopaedia*[^1], the *NASA Exoplanet Archive*[^2], the *Open Exoplanet Catalogue*[^3], the *TESS Objects of Interest Catalog*[^4] and the *EPIC/K2 Objects of Interest Catalog*[^5].

```{warning}
Because of the nature of the merge, this catalog is not self-consistent (i.e., all measurements for the same target don't belong to the same reference paper). 
Rather, it is a collection of the most precise estimates for each planetary parameter (mass, minimum mass, radius, period, semi-major axis, eccentricity, inclination), based on the lowest relative error, and the corresponding reference for each preferred estimate. 
```

Exo-MerCat assigns a main identifier from SIMBAD or TESS Input Catalog v8.2, performing name- and coordinate-based queries. The information on each target collected from any of the original input sources is then collapsed to form a single Exo-MerCat entry. 

The catalog is available via TAP (see {doc}`run_tap` for more info). The user can also download the source file from the GitHub and run Exo-MerCat locally (see {doc}`run_locally`).

```{note}
If you use Exo-MerCat, please cite the following papers: [Alei et al. 2020](https://ui.adsabs.harvard.edu/abs/2020A%26C....3100370A/abstract), Alei et al. 2024
```

[^1]:http://exoplanet.eu/ 
[^2]:https://exoplanetarchive.ipac.caltech.edu/
[^3]:https://openexoplanetcatalogue.com/ 
[^4]:tess
[^5]:epic

