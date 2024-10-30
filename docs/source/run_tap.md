# Get Exo-MerCat from TAP

The easiest way to access Exo-MerCat is through Table Access protocol (for more information, see [the IVOA TAP standard page](https://www.ivoa.net/documents/TAP/)). This can be done through [TOPCAT](#topcat) or [pyvo](#pyvo).

## TOPCAT

Exo-MerCat can be queried using [TOPCAT](https://www.star.bris.ac.uk/~mbt/topcat/). To download the Exo-MerCat table by performing a TAP query on TOPCAT the user should: 

- Select VO --> Table Access Protocol (TAP) Query 
- Paste this link TAP URL into the Selected TAP Service: [http://archives.ia2.inaf.it/vo/tap/projects/](http://archives.ia2.inaf.it/vo/tap/projects/) and click Use Service (tip: you can also search for `exomercat` and it will find the TAP address autonomously).
- Paste your ADQL query in the ADQL section. (for example `SELECT TOP 1000 * FROM exomercat.exomercat`) and click Run Query.

The Exo-MerCat table will be then downloaded and available on TOPCAT. More ADQL examples can be found in [Examples](#examples).

## pyvo

Exo-MerCat can be accessed in Python through [pyvo](https://pyvo.readthedocs.io/en/latest/). Once installed the package, the user can run:

```{code}
import pyvo as vo
service = vo.dal.TAPService("http://archives.ia2.inaf.it/vo/tap/projects/")
resultset = service.search("SELECT TOP 1000 * FROM exomercat.exomercat")
```

For more information, please see the [pyvo documentation](https://pyvo.readthedocs.io/en/latest/).

## Examples

In this section, we provide some useful ADQL queries that might be useful to the user.

### Download the full catalog
Using the following TAP service: [http://archives.ia2.inaf.it/vo/tap/projects/](http://archives.ia2.inaf.it/vo/tap/projects/)

```{code} 
SELECT * 
FROM exomercat.exomercat
```

### Get original EU entry
Using the following TAP service: [http://voparis-tap-planeto.obspm.fr/tap](http://voparis-tap-planeto.obspm.fr/tap)
```{code}
SELECT * 
FROM exoplanet.epn_core AS db
JOIN TAP_UPLOAD.t1 AS tc 
  ON tc.eu_name = db.target_name
  ```

### Get more info on the star from SIMBAD
Using the following TAP service: [http://simbad.cds.unistra.fr/simbad/sim-tap](http://simbad.cds.unistra.fr/simbad/sim-tap)
```{code}
SELECT t1.*, 
       basic.sp_type, 
       basic.pmra, 
       basic.pmdec, 
       basic.plx_value, 
       basic.rvz_redshift, 
       basic.otype_txt
FROM TAP_UPLOAD.t1 AS t1
LEFT OUTER JOIN ident ON ident.id = t1.main_id
LEFT OUTER JOIN basic ON ident.oidref = basic.oid
LEFT OUTER JOIN ids ON basic.oid = ids.oidref
```