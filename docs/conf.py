"""Sphinx configuration for stratix documentation."""

import os
import sys
from importlib.metadata import version as get_version
from pathlib import Path
from typing import Any

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

project = "stratix"
release = get_version("stratix")
copyright = "Copyright &copy; 2026, Oslumen Community"
author = "Oslumen Community"


extensions = [
    "sphinx.ext.autodoc",
    "sphinx.ext.extlinks",
    "sphinx.ext.viewcode",
    "sphinx.ext.autosummary",
    "sphinx_copybutton",
    "sphinx_design",
    "myst_parser",
    "sphinx.ext.napoleon",
    "sphinx_sitemap",
    "sphinx_autodoc_typehints",
    "sphinx_gallery.gen_gallery",
]

html_theme = "shibuya"
html_title = "stratix"
html_static_path = ["_static"]
html_css_files = ["custom.css"]
html_baseurl = "https://github.com/oslumen/stratix"
html_copy_source = False
html_show_sourcelink = False

html_extra_path: list[str] = []
html_theme_options = {}

html_theme_options = {
    "accent_color": "red",
    "github_url": "https://github.com/oslumen/stratix",
    "globaltoc_expand_depth": 1,
    "open_in_perplexity": True,
}

html_context = {
    "source_type": "github",
    "source_user": "oslumen",
    "source_repo": "stratix",
}


templates_path = ["_templates"]

# sphinx-autodoc-typehints
autodoc_typehints = "description"
typehints_document_rtype = False
typehints_use_rtype = False
always_use_bars_union = True
typehints_fully_qualified = False
typehints_defaults = "comma"
always_document_param_types = True
autodoc_typehints_description_target = "documented"

typehints_use_signature = True
typehints_use_signature_return = False

# Napoleon
napoleon_use_rtype = False

autodoc_default_options = {
    "members": True,
    "undoc-members": True,
    "show-inheritance": True,
    "exclude-members": "__init__, __new__, __init_subclass__",
}

language = "en"
exclude_patterns = [
    "_build",
    "Thumbs.db",
    ".DS_Store",
    "examples/*.ipynb",
    "examples/*/*.ipynb",
    "benchmarks/*.ipynb",
    "benchmarks/*/*.ipynb",
    "adr",
    "agents",
]


intersphinx_mapping = {
    "python": ("https://docs.python.org/3", None),
    "numpy": ("https://numpy.org/devdocs/", None),
}

_basedir = Path(__file__).parent
sphinx_gallery_conf = {
    "examples_dirs": "../examples",
    "gallery_dirs": "examples",
    "filename_pattern": "/plot_",
    "remove_config_comments": True,
    "download_all_examples": False,
    "write_computation_times": True,
    "min_reported_time": 0,
    "show_memory": True,
    "reference_url": {"stratix": None},
}
