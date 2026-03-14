"""cathode_ml: Machine learning for lithium cathode materials."""

# matgl locks its backend on first import (defaults to PYG).
# M3GNet requires the DGL backend, so we must set this before
# matgl is ever imported — this is the earliest reliable point.
import os as _os

_os.environ.setdefault("MATGL_BACKEND", "dgl")

__version__ = "0.1.0"
