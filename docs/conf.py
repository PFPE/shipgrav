# Configuration file for the Sphinx documentation builder.
#
# For the full list of built-in configuration values, see the documentation:
# https://www.sphinx-doc.org/en/master/usage/configuration.html

import sys, os

# extensions/moduels are in another dir; use abspath to make it work
root = os.path.abspath('../')
sys.path.insert(0, root)

# import modules for autodoc
import shipgrav
import shipgrav.io
import shipgrav.grav
import shipgrav.nav
import shipgrav.utils

# -- Project information -----------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#project-information

project = 'shipgrav'
copyright = '2024, Hannah Mark'
author = 'Hannah Mark'
release = '0.0'

# -- General configuration ---------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#general-configuration

extensions = ['sphinx.ext.autodoc']

templates_path = ['_templates']
exclude_patterns = ['_build', 'Thumbs.db', '.DS_Store']



# -- Options for HTML output -------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#options-for-html-output

html_theme = 'alabaster'
html_static_path = ['_static']
html_sidebars = {
    '**': [
        'about.html',
        'mynavigation.html',
        'searchbox.html',
    ]
}
html_theme_options = {
    'github_user': 'PFPE',
    'github_repo': 'shipgrav',
    'description': 'Marine gravity processing for UNOLS',
    #'show_powered_by': False,
    #'page_width': '1240px',
    #'sidebar_width': '220px',
    'extra_nav_links': {'Project on Github': 'https://github.com/PFPE/shipgrav'}

}

## -- Options for LaTeX output -------------------------------------------------
#latex_additional_files = ['tex/pfpe-manual.cls']
#latex_docclass = {'manual': 'pfpe-manual'}
