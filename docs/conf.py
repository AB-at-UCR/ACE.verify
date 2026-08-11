# Configuration file for the Sphinx documentation builder.
#
# Read The Docs configuration for ACE.verify documentation.
# Built with Sphinx + sphinxcontrib-mermaid for diagram rendering.

import os
import sys

# -- Path setup --------------------------------------------------------------

# If extensions (or modules to document with autodoc) are in another directory,
# add these directories to sys.path here.
sys.path.insert(0, os.path.abspath('..'))

# -- Project information -----------------------------------------------------

project = 'ACE.verify'
author = 'Aditya Bhardwaj, Emily Clark, Connor Claborn'
copyright = '2026, ACE.verify Contributors'

# The version info for the project being documented
version = '1.0'
release = '1.0.0'

# -- General configuration ---------------------------------------------------

# Add any Sphinx extension module names here, as strings.
extensions = [
    'sphinxcontrib.mermaid',
    'sphinx_rtd_theme',
]

# Mermaid configuration
mermaid_version = '10'
mermaid_init_js = "mermaid.initialize({startOnLoad:true, theme:'default', securityLevel:'loose'});"

# Add any paths that contain templates here, relative to this directory.
templates_path = ['_templates']

# List of patterns, relative to source directory, that match files and
# directories to ignore when looking for source files.
exclude_patterns = ['_build', 'Thumbs.db', '.DS_Store']

# The master toctree document.
master_doc = 'index'

# -- Options for HTML output -------------------------------------------------

# The theme to use for HTML and HTML Help pages.
html_theme = 'sphinx_rtd_theme'

# Theme options for sphinx_rtd_theme.
html_theme_options = {
    'logo_only': False,
    'display_version': True,
    'prev_next_buttons_location': 'bottom',
    'style_external_links': False,
    'collapse_navigation': True,
    'sticky_navigation': True,
    'navigation_depth': 4,
    'includehidden': True,
    'titles_only': False,
}

# The name of an image file (relative to this directory) to place at the top
# of the sidebar.
# html_logo = '_static/logo.png'

# -- Options for HTMLHelp output ---------------------------------------------

htmlhelp_basename = 'ACEverifydoc'

# -- Extension configuration -------------------------------------------------

# Mermaid output format (svg or png)
mermaid_output_format = 'svg'
