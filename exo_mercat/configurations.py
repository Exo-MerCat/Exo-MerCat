import os
import configparser


def service_files_initialization() -> None:
    """
    The service_files_initialization function creates the Exo-MerCat, InputSources, UniformedSources and Logs
    folders if they do not exist, and deletes all files in the Logs folder.
    """
    # CREATE OUTPUT FOLDERS
    if not os.path.exists("Exo-MerCat/"):
        os.makedirs("Exo-MerCat")

    if not os.path.exists("InputSources/"):
        os.makedirs("InputSources")

    if not os.path.exists("UniformSources/"):
        os.makedirs("UniformSources")

    # CREATE LOG FOLDER
    if not os.path.exists("Logs/"):
        os.makedirs("Logs")
    os.system("rm Logs/*")


def find_const() -> dict:
    # TODO fix so that it does the operation inside the function
    """
    The find_const function takes a string and returns the same string with all of its
    constellation abbreviations replaced by their full names. The function uses a dictionary
    to map each abbreviation to its corresponding constellation name. The dictionary is
    created using the find_const() function, which creates a dataframe containing two columns:
    the first column contains all possible constellation abbreviations, while the second column
    contains their corresponding full names.
    """
    constants = {
        "alfa ": "alf ",
        "beta ": "bet ",
        "gamma ": "gam ",
        "delta ": "del ",
        "zeta ": "zet ",
        "teta ": "tet ",
        "iota ": "iot ",
        "kappa ": "kap ",
        "lambda ": "lam ",
        "omicron ": "omi ",
        "sigma ": "sig ",
        "upsilon ": "ups ",
        "omega ": "ome ",
        "alpha ": "alf ",
        "epsilon ": "eps ",
        "theta ": "tet ",
        "mu ": "miu ",
        "nu ": "niu ",
        "xi ": "ksi ",
        "chi ": "khi ",
        "Alfa ": "alf ",
        "Beta ": "bet ",
        "Gamma ": "gam ",
        "Delta ": "del ",
        "Eps ": "eps ",
        "Zeta ": "zet ",
        "Eta ": "eta ",
        "Teta ": "tet ",
        "Iota ": "iot ",
        "Kappa ": "kap ",
        "Lambda ": "lam ",
        "Miu ": "miu ",
        "Niu ": "niu ",
        "Ksi ": "ksi ",
        "Omicron ": "omi ",
        "Pi ": "pi ",
        "Rho ": "rho ",
        "Sigma ": "sig ",
        "Upsilon ": "ups ",
        "Phi ": "phi ",
        "Khi ": "khi ",
        "Psi ": "psi ",
        "Omega ": "ome ",
        "Alpha ": "alf ",
        "Bet ": "bet ",
        "Gam ": "gam ",
        "Del ": "del ",
        "Epsilon ": "eps ",
        "Zet ": "zet ",
        "Theta ": "tet ",
        "Iot ": "iot ",
        "Kap ": "kap ",
        "Lam ": "lam ",
        "Mu ": "miu ",
        "Nu ": "niu ",
        "Xi ": "ksi ",
        "Omi ": "omi ",
        "Sig ": "sig ",
        "Ups ": "ups ",
        "Chi ": "khi ",
        "Ome ": "ome ",
        "Andromedae": "And",
        "Antliae": "Ant",
        "Apodis": "Aps",
        "Aquarii": "Aqr",
        "Aquilae": "Aql",
        "Arae": "Ara",
        "Arietis": "Ari",
        "Aurigae": "Aur",
        "Bootis": "Boo",
        "Caeli": "Cae",
        "Camelopardalis": "Cam",
        "Cancri": "Cnc",
        "Canum Venaticorum": "CVn",
        "Canis Majoris": "CMa",
        "Canis Minoris": "CMi",
        "Capricorni": "Cap",
        "Carinae": "Car",
        "Cassiopeiae": "Cas",
        "Centauri": "Cen",
        "Cephei": "Cep",
        "Cepi": "Cep",
        "Ceti": "Cet",
        "Chamaeleontis": "Cha",
        "Circini": "Cir",
        "Columbae": "Col",
        "Comae Berenices": "Com",
        "Coronae Australis": "CrA",
        "Coronae Borealis": "CrB",
        "Corvi": "Crv",
        "Crateris": "Crt",
        "Crucis": "Cru",
        "Cygni": "Cyg",
        "Delphini": "Del",
        "Doradus": "Dor",
        "Draconis": "Dra",
        "Equulei": "Equ",
        "Eridani": "Eri",
        "Fornacis": "For",
        "Geminorum": "Gem",
        "Gruis": "Gru",
        "Herculis": "Her",
        "Horologii": "Hor",
        "Hydrae": "Hya",
        "Hydri": "Hyi",
        "Indi": "Ind",
        "Lacertae": "Lac",
        "Leonis": "Leo",
        "Leonis Minoris": "LMi",
        "Leporis": "Lep",
        "Librae": "Lib",
        "Lupi": "Lup",
        "Lyncis": "Lyn",
        "Lyrae": "Lyr",
        "Mensae": "Men",
        "Microscopii": "Mic",
        "Monocerotis": "Mon",
        "Muscae": "Mus",
        "Normae": "Nor",
        "Octantis": "Oct",
        "Ophiuchi": "Oph",
        "Orionis": "Ori",
        "Pavonis": "Pav",
        "Pegasi": "Peg",
        "Persei": "Per",
        "Phoenicis": "Phe",
        "Pictoris": "Pic",
        "Piscium": "Psc",
        "Piscis Austrini": "PsA",
        "Puppis": "Pup",
        "Pyxidis": "Pyx",
        "Reticuli": "Ret",
        "Sagittae": "Sge",
        "Sagittarii": "Sgr",
        "Scorpii": "Sco",
        "Sculptoris": "Scl",
        "Scuti": "Sct",
        "Serpentis": "Ser",
        "Sextantis": "Sex",
        "Tauri": "Tau",
        "Telescopii": "Tel",
        "Trianguli": "Tri",
        "Trianguli Australis": "TrA",
        "Tucanae": "Tuc",
        "Ursae Majoris": "UMa",
        "Uma": "UMa",
        "Ursae Minoris": "UMi",
        "Umi": "UMi",
        "Velorum": "Vel",
        "Virginis": "Vir",
        "Volantis": "Vol",
        "Vulpeculae": "Vul",
        "2M ": "2MASS ",
        "KOI ": "KOI-",
        "Kepler ": "Kepler-",
        "BD ": "BD",
        "OGLE-": "OGLE ",
        "MOA-": "MOA ",
        "gam 1 ": "gam ",
        "EPIC-": "EPIC ",
        "Pr 0": "Pr ",
        "TOI ": "TOI-",
        "kepler": "Kepler",
        "Gliese": "GJ",
        "p ": "pi ",
    }

    return constants


def read_config():
    """
    The read_config function reads the config.ini file and returns a dictionary of
    the configuration parameters.
    """
    config = configparser.RawConfigParser(inline_comment_prefixes="#")
    config.read("config.ini")

    return config


def read_config_replacements(section: str):
    """
    The read_config_replacements function reads the replacements.ini file and returns a dictionary of
    replacement values for use in the replace_text function.

    Parameters
    ----------
        section: str
            Specify which section of the replacements
    """
    config = configparser.RawConfigParser(inline_comment_prefixes="#")
    config.optionxform = str
    config.read("replacements.ini")
    config_replace = dict(config.items(section))
    return config_replace
