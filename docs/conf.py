import os
import sys
sys.path.insert(0, os.path.abspath('..'))

project = 'wofryimpl'
copyright = '2024, Manuel Sanchez del Rio, Luca Rebuffi'
author = 'Manuel Sanchez del Rio, Luca Rebuffi'

extensions = [
    'sphinx.ext.autodoc',
    'sphinx.ext.autosummary',
    'sphinx.ext.napoleon',
    'sphinx.ext.viewcode',
]

autosummary_generate = True
autodoc_default_options = {'members': True, 'undoc-members': True}

templates_path = ['_templates']
exclude_patterns = ['_build', 'Thumbs.db', '.DS_Store']

html_theme = 'sphinx_rtd_theme'
html_static_path = []

html_context = {
    'display_github': True,
    'github_user': 'oasys-kit',
    'github_repo': 'wofryimpl',
    'github_version': 'main',
    'conf_py_path': '/docs/',
}
