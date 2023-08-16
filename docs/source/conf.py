# Configuration file for the Sphinx documentation builder.
#
# For the full list of built-in configuration values, see the documentation:
# https://www.sphinx-doc.org/en/master/usage/configuration.html

# -- Project information -----------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#project-information

import sys
from pathlib import Path

BASE_PATH = Path(__file__).resolve().parents[2] / "src"

sys.path.insert(0, BASE_PATH)

with open(BASE_PATH / "pyethnicity/__init__.py", "r") as f:
    version_no = f.readlines()[0].split("=")[1].strip()

project = "pyethnicity"
copyright = "2023, Cangyuan Li"
author = "Cangyuan Li"
release = version_no

# -- General configuration ---------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#general-configuration

extensions = [
    "sphinx.ext.napoleon",
    "sphinx.ext.mathjax",
    "sphinx_copybutton",
    "sphinx.ext.autodoc",
    "sphinx.ext.autodoc.typehints",
]

templates_path = ["_templates"]
exclude_patterns = []

add_module_names = False
autodoc_typehints = "description"
pygments_style = "sphinx"
copybutton_exclude = ".linenos, .gp"
# autoapi_dirs = [BASE_PATH]

# -- Options for HTML output -------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#options-for-html-output

html_theme = "alabaster"
html_static_path = ["_static"]
htmlhelp_basename = "pyethnicity_doc"
