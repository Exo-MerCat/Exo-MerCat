import os
import configparser


def service_files_initialization() -> None:

    """
    The service_files_initialization function creates the EMC and EMClogs folders if they do not exist,
    and deletes all files in the EMClogs folder. It then creates check_alias.txt, coord_errors.txt, binary_mismatch.txt,
    mainid.txt and duplicatedentries.txt in the EMClogs folder.

    Parameters
    ----------

    Returns
    -------

        The mainid

    Doc Author
    ----------
        Trelent
    """
    # CREATE OUTPUT FOLDER
    if not os.path.exists("Exo-MerCat/"):
        os.makedirs("Exo-MerCat")

    # CREATE LOG FOLDER
    if not os.path.exists("EMClogs/"):
        os.makedirs("EMClogs")
    os.system("rm EMClogs/*")

def find_const() -> None:
    # TODO fix so that it does the operation inside the function
    """
    The find_const function takes a string and returns the same string with all of its
    constellation abbreviations replaced by their full names. The function uses a dictionary
    to map each abbreviation to its corresponding constellation name. The dictionary is
    created using the find_const() function, which creates a dataframe containing two columns:
    the first column contains all possible constellation abbreviations, while the second column
    contains their corresponding full names.
    """
    dict = {
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

    c = {
        0: [
            "alfa ",
            "beta ",
            "gamma ",
            "delta ",
            "zeta ",
            "teta ",
            "iota ",
            "kappa ",
            "lambda ",
            "omicron ",
            "sigma ",
            "upsilon ",
            "omega ",
            "alpha ",
            "epsilon ",
            "theta ",
            "mu ",
            "nu ",
            "xi ",
            "omicron ",
            "chi ",
            "Alfa ",
            "Beta ",
            "Gamma ",
            "Delta ",
            "Eps ",
            "Zeta ",
            "Eta ",
            "Teta ",
            "Iota ",
            "Kappa ",
            "Lambda ",
            "Miu ",
            "Niu ",
            "Ksi ",
            "Omicron ",
            "Pi ",
            "Rho ",
            "Sigma ",
            "Upsilon ",
            "Phi ",
            "Khi ",
            "Psi ",
            "Omega ",
            "Alpha ",
            "Bet ",
            "Gam ",
            "Del ",
            "Epsilon ",
            "Zet ",
            "Theta ",
            "Iot ",
            "Kap ",
            "Lam ",
            "Mu ",
            "Nu ",
            "Xi ",
            "Omi ",
            "Sig ",
            "Ups ",
            "Chi ",
            "Ome ",
            "Andromedae",
            "Antliae",
            "Apodis",
            "Aquarii",
            "Aquilae",
            "Arae",
            "Arietis",
            "Aurigae",
            "Bootis",
            "Caeli",
            "Camelopardalis",
            "Cancri",
            "Canum Venaticorum",
            "Canis Majoris",
            "Canis Minoris",
            "Capricorni",
            "Carinae",
            "Cassiopeiae",
            "Centauri",
            "Cephei",
            "Cepi",
            "Ceti",
            "Chamaeleontis",
            "Circini",
            "Columbae",
            "Comae Berenices",
            "Coronae Australis",
            "Coronae Borealis",
            "Corvi",
            "Crateris",
            "Crucis",
            "Cygni",
            "Delphini",
            "Doradus",
            "Draconis",
            "Equulei",
            "Eridani",
            "Fornacis",
            "Geminorum",
            "Gruis",
            "Herculis",
            "Horologii",
            "Hydrae",
            "Hydri",
            "Indi",
            "Lacertae",
            "Leonis",
            "Leonis Minoris",
            "Leporis",
            "Librae",
            "Lupi",
            "Lyncis",
            "Lyrae",
            "Mensae",
            "Microscopii",
            "Monocerotis",
            "Muscae",
            "Normae",
            "Octantis",
            "Ophiuchi",
            "Orionis",
            "Pavonis",
            "Pegasi",
            "Persei",
            "Phoenicis",
            "Pictoris",
            "Piscium",
            "Piscis Austrini",
            "Puppis",
            "Pyxidis",
            "Reticuli",
            "Sagittae",
            "Sagittarii",
            "Scorpii",
            "Sculptoris",
            "Scuti",
            "Serpentis",
            "Sextantis",
            "Tauri",
            "Telescopii",
            "Trianguli",
            "Trianguli Australis",
            "Tucanae",
            "Ursae Majoris",
            "Uma",
            "Ursae Minoris",
            "Umi",
            "Velorum",
            "Virginis",
            "Volantis",
            "Vulpeculae",
            "2M ",
            "KOI ",
            "Kepler ",
            "BD ",
            "OGLE-",
            "MOA-",
            "gam 1 ",
            "EPIC-",
            "Pr 0",
            "TOI ",
            "kepler",
            "Gliese",
            "π ",
        ],
        1: [
            "alf ",
            "bet ",
            "gam ",
            "del ",
            "zet ",
            "tet ",
            "iot ",
            "kap ",
            "lam ",
            "omi ",
            "sig ",
            "ups ",
            "ome ",
            "alf ",
            "eps ",
            "tet ",
            "miu ",
            "niu ",
            "ksi ",
            "omi ",
            "khi ",
            "alf ",
            "bet ",
            "gam ",
            "del ",
            "eps ",
            "zet ",
            "eta ",
            "tet ",
            "iot ",
            "kap ",
            "lam ",
            "miu ",
            "niu ",
            "ksi ",
            "omi ",
            "pi ",
            "rho ",
            "sig ",
            "ups ",
            "phi ",
            "khi ",
            "psi ",
            "ome ",
            "alf ",
            "bet ",
            "gam ",
            "del ",
            "eps ",
            "zet ",
            "tet ",
            "iot ",
            "kap ",
            "lam ",
            "miu ",
            "niu ",
            "ksi ",
            "omi ",
            "sig ",
            "ups ",
            "khi ",
            "ome ",
            "And",
            "Ant",
            "Aps",
            "Aqr",
            "Aql",
            "Ara",
            "Ari",
            "Aur",
            "Boo",
            "Cae",
            "Cam",
            "Cnc",
            "CVn",
            "CMa",
            "CMi",
            "Cap",
            "Car",
            "Cas",
            "Cen",
            "Cep",
            "Cep",
            "Cet",
            "Cha",
            "Cir",
            "Col",
            "Com",
            "CrA",
            "CrB",
            "Crv",
            "Crt",
            "Cru",
            "Cyg",
            "Del",
            "Dor",
            "Dra",
            "Equ",
            "Eri",
            "For",
            "Gem",
            "Gru",
            "Her",
            "Hor",
            "Hya",
            "Hyi",
            "Ind",
            "Lac",
            "Leo",
            "LMi",
            "Lep",
            "Lib",
            "Lup",
            "Lyn",
            "Lyr",
            "Men",
            "Mic",
            "Mon",
            "Mus",
            "Nor",
            "Oct",
            "Oph",
            "Ori",
            "Pav",
            "Peg",
            "Per",
            "Phe",
            "Pic",
            "Psc",
            "PsA",
            "Pup",
            "Pyx",
            "Ret",
            "Sge",
            "Sgr",
            "Sco",
            "Scl",
            "Sct",
            "Ser",
            "Sex",
            "Tau",
            "Tel",
            "Tri",
            "TrA",
            "Tuc",
            "UMa",
            "UMa",
            "UMi",
            "UMi",
            "Vel",
            "Vir",
            "Vol",
            "Vul",
            "2MASS ",
            "KOI-",
            "Kepler-",
            "BD",
            "OGLE ",
            "MOA ",
            "gam ",
            "EPIC ",
            "Pr ",
            "TOI-",
            "Kepler",
            "GJ",
            "pi ",
        ],
    }
    return dict


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
