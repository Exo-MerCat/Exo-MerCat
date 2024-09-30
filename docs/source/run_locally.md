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
exomercat [ [-l] [-d DATE] function
```

The user can select optional arguments: 

- `-h` (or `--help`) to print a help message; 
- `-v` (or `--verbose`) to increase the verbosity of the output';
- `-d YYYY-MM-DD` (or `--date YYYY-MM-DD`) to load the input sources at a specific date in YYYY-MM-DD format.

Possible functions to be run are: 
- `input`, which executes the download of the input sources and their standardization; 
- `run`, which joins the input sources to generate the Exo-MerCat catalog; 
- `check`, which performs sanity checks on the output; 
- `tests`, which performs unit tests on the code.


You can run Exo-MerCat in any folder, as long as it contains the `replacements.ini` and the `input_sources.ini` files. You can change the `.ini` files according to your preferences.

## Modify the `.ini` files

### The `replacements.ini` file

