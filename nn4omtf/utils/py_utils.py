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


__all__ = ['import_module_from_path', 'get_from_module_by_name', 'get_source_of_obj']


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

