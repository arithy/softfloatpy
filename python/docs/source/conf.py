# Configuration file for the Sphinx documentation builder.
#
# For the full list of built-in configuration values, see the documentation:
# https://www.sphinx-doc.org/en/master/usage/configuration.html

import softfloatpy

# -- Project information -----------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#project-information

project = 'SoftFloatPy'
author = 'Arihiro Yoshida'
copyright = '2024-%Y ' + author
version = softfloatpy.__version__
release = version

# -- General configuration ---------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#general-configuration

extensions = [
    'sphinx.ext.autodoc',
    'sphinx.ext.napoleon',
    'sphinx_rtd_theme',
    'myst_parser'
]

templates_path = ['_templates']
exclude_patterns = []
suppress_warnings = ['myst.header']

autodoc_warningiserror = True
autodoc_default_options = {
    'special-members': '__init__',
    'undoc-members': True
}
autodoc_member_order = 'bysource'

# -- Options for HTML output -------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#options-for-html-output

html_theme = 'sphinx_rtd_theme'
html_static_path = ['_static']
