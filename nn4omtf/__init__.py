# -*- coding: utf-8 -*-
from nn4omtf import dataset
from nn4omtf import network
from nn4omtf import utils

def info():
    print("""
--- INFO ---
    Neural Networks tools for OMTF@CMS
    Jacek Lysiak 2018
    Due to high overhead with installing ROOT and its python packages
    utility methods which uses ROOT are not imported by default.
    Call 'import_root_utils()' method to import them.
--- END OF INFO ---
    """)

def import_root_utils():
    """Import part of package which uses ROOT."""
    from . import root_utils

# Show info...
# info()
