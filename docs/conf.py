"""Sphinx configuration for jump-diffusion-estimation."""

import os
import sys

sys.path.insert(0, os.path.abspath("../src"))

project = "jump-diffusion-estimation"
copyright = "2026, Juan David Ospina Arango"
author = "Juan David Ospina Arango"

extensions = [
    "sphinx.ext.autodoc",
    "sphinx.ext.napoleon",
    "sphinx.ext.viewcode",
    "sphinx.ext.intersphinx",
]

napoleon_google_docstring = True
napoleon_numpy_docstring = True

templates_path = ["_templates"]
exclude_patterns = ["_build", "Thumbs.db", ".DS_Store"]

autodoc_default_options = {
    "members": True,
    "undoc-members": True,
    "show-inheritance": True,
}
autodoc_member_order = "bysource"

intersphinx_mapping = {
    "python": ("https://docs.python.org/3", None),
    "numpy": ("https://numpy.org/doc/stable/", None),
    "scipy": ("https://docs.scipy.org/doc/scipy/", None),
    "pandas": ("https://pandas.pydata.org/docs/", None),
}

html_theme = "sphinx_rtd_theme"
html_static_path = ["_static"]
