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

html_theme = "oslumen"
html_title = "stratix"

html_logo = "_static/_assets/stratix-name.svg"
html_favicon = "_static/_assets/stratix.svg"
html_static_path = ["_static"]
html_css_files = ["custom.css"]
html_baseurl = "https://github.com/oslumen/stratix"

html_copy_source = True
html_show_sourcelink = True

html_extra_path: list[str] = []

html_theme_options = {
    # "accent_color": "red",
    "light_logo": "_static/_assets/stratix-name.svg",
    "dark_logo": "_static/_assets/stratix-name-dark.svg",
    "slack_url": "https://www.slack.com/oslumen",
    "youtube_url": "https://www.youtube.com/oslumen",
    "linkedin_url": "https://www.linkedin.com/oslumen",
    "bluesky_url": "https://www.bluesky.com/oslumen",
    "discussion_url": "https://github.com/oslumen/stratix/discussions",
    "github_url": "https://github.com/oslumen/stratix",
    # "banner": "This a community-driven project. Your <a href='https://github.com/oslumen/sphinx_theme_oslumen'>contributions</a> are welcome!",
    # "carbon_ads_code": "REPLACE_WITH_CODE",
    "globaltoc_expand_depth": 1,
    "open_in_perplexity": True,
    "use_root_redirect": False,
    "repository_url": "https://github.com/oslumen/stratix",
    "repository_branch": "main",
    "launch_buttons": {
        "binderhub_url": "https://mybinder.org",
        "colab_url": "https://colab.research.google.com/",
        "deepnote_url": "https://deepnote.com/",
        "notebook_interface": "jupyterlab",
        "thebe": True,
        "jupyterlite_url": "https://jupyterlite.github.io/demo/lab/",
        # "jupyterhub_url": "https://datahub.berkeley.edu",  # For testing
    },
    "use_edit_page_button": True,
    "use_source_button": True,
    "use_issues_button": True,
    "use_download_button": True,
    "use_fullscreen_button": True,
    "globaltoc_expand_depth": 1,
    "path_to_docs": "docs",
}

html_context = {
    "source_type": "github",
    "source_user": "oslumen",
    "source_repo": "stratix",
}


templates_path = ["_templates"]


extlinks = {
    "pull": (
        "https://github.com/oslumen/stratix/pull/%s",
        "pull request #%s",
    ),
    "issue": ("https://github.com/oslumen/stratix/issues/%s", "issue #%s"),
}


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
