# -*- coding: utf-8 -*-

import os

SCRIPTS_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.abspath(os.path.join(SCRIPTS_DIR, '..'))


def resolve_path(path, base=None):
    """Rozwiaz sciezke wzgledna: cwd, potem `base`, potem katalog pakietu."""
    if not path or not isinstance(path, str):
        return path

    if os.path.isabs(path):
        return path

    roots = [os.getcwd(), base or SCRIPTS_DIR, PROJECT_ROOT]
    for root in roots:
        candidate = os.path.abspath(os.path.join(root, path))
        if os.path.exists(candidate):
            return candidate
    return path
