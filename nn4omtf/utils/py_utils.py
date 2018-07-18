# -*- coding: utf-8 -*-
"""
    Copyright (C) 2018 Jacek Łysiak
    MIT License

    Python language utilities.
"""

import importlib.util
import importlib.machinery
import os
import inspect


def obj_elems(x):
    return [e for e, _ in vars(x).items() if not e.startswith('_')]


def import_module_from_path(path):
    """Import python code from path."""
    name = os.path.basename(path).split(".")[0]
    loader = importlib.machinery.SourceFileLoader(name, path)
    spec = importlib.util.spec_from_loader(loader.name, loader)
    mod = importlib.util.module_from_spec(spec)
    loader.exec_module(mod)
    return mod


def get_from_module_by_name(mod, name):
    """Extract object from compiled module.
    Returns:
        Python object if exists in module dict or None otherwise.
    """
    try:
        return mod.__dict__[name]
    except Exception:
        return None


def get_source_of_obj(obj):
    """Get source code of compiled Python object."""
    lines = inspect.getsourcelines(obj)
    return "".join(lines[0])


def dict_to_object(dictionary):
    """Prepare object-like dictionary.
    Args:
        dictionary: dictionary to convert
    Returns:
        object with fields same as dictionary entries
    """
    class Dict2Obj:
        def __init__(self, d):
            self.__dict__ = d
    return Dict2Obj(dictionary)

