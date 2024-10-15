# Run Exo-MerCat locally

The script is written in Python v3.x and it has been tested for Python v3.8 **test**.

## Create virtual environment

You can use [conda](https://conda.io/projects/conda/en/latest/user-guide/getting-started.html), [pyenv](https://github.com/pyenv/pyenv), or [venv](https://docs.python.org/3/library/venv.html) to create a virtual environment.


::::{tab-set}

:::{tab-item} conda
    conda env create -f environment.yml #  Create a virtual environment
    conda activate exomercat2 # Activate the virtual environment (on Linux/macOS)
:::

:::{tab-item} pyenv
    pyenv install 3.8.0  # Or any specific version
    pyenv virtualenv 3.8.0 exomercat2  # Create a virtual environment named exomercat2
    pyenv activate exomercat2  # Activate the virtual environment
    pip3 install -r requirements.txt


:::

:::{tab-item} venv
    python3 -m venv exomercat2  # Create a virtual environment named exomercat2
    source exomercat2/bin/activate  # Activate the virtual environment (on Linux/macOS)
    pip3 install -r requirements.txt


::::

## Install Exo-MerCat

Once activated the virtual environment, you can install Exo-MerCat in the folder that contains the `pyproject.toml` file.

```{code}
pip install -I .
```

## Run Exo-MerCat


 Once installed, the script can be launched with the following command:

```{code}
exomercat [-h] [-v] [-d DATE] function
```

The user can select optional arguments: 

- `-h` (or `--help`) to print a help message; 
- `-v` (or `--verbose`) to increase output verbosity. Use `-vv` or `-vvv` to increase verbosity;
- `-d YYYY-MM-DD` (or `--date YYYY-MM-DD`) to load the input sources at a specific date in YYYY-MM-DD format.

Possible functions to be run are: 
- `maintenance`, which executes sanity checks on the input sources to check if they are currently available for download;
- `input`, which executes the download of the input sources and their standardization; 
- `run`, which joins the input sources to generate the Exo-MerCat catalog; 
- `check`, which performs sanity checks on the output; 
- `all`, which runs all of the functions above.

To run tests on the code, the user can run:

```{code}
pytest
```

This will produce a html page in the folder `htmlcov` showing if the cose is currently covered by the implemented tests.



You can run Exo-MerCat in any folder, as long as it contains the `replacements.ini` and the `input_sources.ini` files. You can change the `.ini` files according to your preferences.

## Modify the `.ini` files

### The `input_sources.ini` file

The `input_sources.ini` file contains the links to the input sources and the path to where they should be saved. In an ideal scenario, this file will never need to be changed. However, in case the input sources change the location of their tables, or if the user wants to add a new resource and/or change the location of the files that are downloaded, this is the file to modify. 

Here is an example of the `input_sources.ini` file.

```{code}
[nasa]
url = https://exoplanetarchive.ipac.caltech.edu/TAP/sync?query=select+*+from+pscomppars&format=csv
file = InputSources/nasa_init

[oec]
url = https://github.com/OpenExoplanetCatalogue/oec_gzip/raw/master/systems.xml.gz
file = InputSources/oec_init

[eu]
url =  https://exoplanet.eu/catalog/csv/?query_f=planet_status%3D%22confirmed%22%20or%20planet_status%3D%22candidate%22%20or%20planet_status%3D%22unconfirmed%22%20or%20planet_status%3D%22controversial%22%20or%20planet_status%3D%22retracted%22
file = InputSources/eu_init

[koi]
url = https://exoplanetarchive.ipac.caltech.edu/cgi-bin/nstedAPI/nph-nstedAPI?table=cumulative
file = InputSources/koi_init

[epic]
url = https://exoplanetarchive.ipac.caltech.edu/TAP/sync?query=select+*+from+k2pandc&format=csv
file = InputSources/epic_init

[toi]
url = https://exoplanetarchive.ipac.caltech.edu/TAP/sync?query=select+*+from+TOI&format=csv
file = InputSources/toi_init
```



The file is split in sections, one per catalog. The section name is between brackets and it must be the same as the assigned name of each catalog. Then, the URL must be specified as well as the preferred file name/path.

### The `replacements.ini` file

Here is an excerpt of a valid `replacements.ini` file:

```{code}
[NAMEtochangeNAME] 
KOI-7209 = KOI-7209 b
gam1 Leo b = gam01 Leo b

[NAMEtochangeHOST]
2MASS 0359+2009 b = 2MASS J03590986+2009361
2M 0103-55 (AB) b = 2MASS J01033563-5515561

[NAMEtochangeBINARY]
XO-2N c = A
XO-2N b = A

[HOSTtochangeHOST]
2M 0103-55 (AB) = SCR J0103-5515 (AB)
1SWASP J1407 = 1SWASP J140747.93-394542.6

[HOSTtochangeRA]
K2-2016-BLG-0005L = 269.879166677

[HOSTtochangeDEC]
M62H b =-30.1069833

[DROP]
name = Trojan,Candidate
name = Oumuamua
alias= Sun
host = Sun
```

There are six different sections, depending on the preferred key that needs to be used to change a specific value. Specifically:

- `NAMEtochangeHOST` contains all entries for which we need to use the planet name to change the host star name;
- `NAMEtochangeHOST` contains all entries for which we need to use the planet name to change the host star name;
- `NAMEtochangeBINARY` contains all entries for which we need to use the planet name to change the binary letter value;
- `HOSTtochangeHOST` contains all entries for which we need to use the host star name to change the host star name itself;
- `HOSTtochangeRA` contains all entries for which we need to use the host star name to change the host star right ascension;
- `HOSTtochangeDEC` contains all entries for which we need to use the host star name to change the host star declination;
- `DROP` contains all entries that need to be dropped, searching through the following keys: `name` (the planet name), `host` (the star name), `alias` (a stellar alias). All items to be discarded can be listed one by one (e.g. `name=Candidate` and `name=Trojan`) or as a comma-separated list (i.e.`name=Candidate,Trojan`).

You can find a summary of the used replacements for each catalogs in the logfile `replace_known_mistakes.txt`.

