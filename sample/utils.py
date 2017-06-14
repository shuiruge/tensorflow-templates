"""
Description
---------

Collection of utils.
"""

# For compatible with Python2
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function


import os

def ensure_directory(directory):
    """
    Check whether the `directory` exists or not. If not, then create
    this directory.

    Args:
        directory: str
    Returns:
        None
    """
    try:
        directory = os.path.expanduser(directory)

        os.makedirs(directory)
    
    except FileExistsError:
        
        pass

    return None