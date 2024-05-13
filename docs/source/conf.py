# Configuration file for the Sphinx documentation builder.
#
# For the full list of built-in configuration values, see the documentation:
# https://www.sphinx-doc.org/en/master/usage/configuration.html

# -- Project information -----------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#project-information
import os
import sys

sys.path.insert(0, os.path.abspath(os.path.join("..", "..")))

project = "Exo-MerCat"
copyright = "2024, Alei et al."
author = "Alei et al."
release = "2.0"

# -- General configuration ---------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#general-configuration

extensions = [
    # 'sphinx.ext.napoleon',
    "sphinx.ext.autodoc",
    "sphinx_autodoc_typehints",
    "sphinx.ext.autosummary",
    "myst_parser",
    #           'sphinx.ext.intersphinx',
    "sphinx.ext.duration",
    "sphinx.ext.autosectionlabel"
    # 'sphinx.ext.doctest',
    # # 'sphinx.ext.autodoc',
]
autodoc_member_order = "bysource"

autodoc_default_options = {
    "members": True,
    "undoc-members": True,
    "show-inheritance": True,
}

templates_path = ["_templates"]
exclude_patterns = []


# -- Options for HTML output -------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#options-for-html-output

html_theme = "sphinx_book_theme"
html_logo = "_static/Acquerello_scritta.png"

html_theme_options = {
    "logo": {
        # Because the logo is also a homepage link, including "home" in the
        # alt text is good practice
        "alt_text": "Exo-MerCat 2.0",
        "image_light": "_static/Acquerello.png",
        "image_dark": "_static/Acquerello.png",
    }
}
html_static_path = ["_static"]
html_css_files = ["custom.css"]
html_theme_options = {"show_nav_level": 1}
