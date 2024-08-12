import warnings

# Monkey-patch the line away from warnings, as it is rather irritating.
formatwarning_orig = warnings.formatwarning
warnings.formatwarning = lambda msg, cat, fname, lineno, line=None: \
    formatwarning_orig(msg, cat, fname, lineno, line='')

__version__ = '0.0.1'
