#: Channel order written by ``precompute_shape.py`` and assumed by the model.
SHAPE_FEATURES = ("MGW", "Shear", "Stretch", "Stagger", "Buckle", "ProT",
                  "Opening", "Shift", "Slide", "Rise", "Tilt", "Roll", "HelT")

SHAPE_RC_ANTISYMMETRIC = ("Shear", "Buckle", "Shift", "Tilt")

SHAPE_STEP_FEATURES = frozenset(
    {"Shift", "Slide", "Rise", "Tilt", "Roll", "HelT"})
