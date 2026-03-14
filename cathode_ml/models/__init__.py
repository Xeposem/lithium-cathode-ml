# Baseline and deep learning models for cathode property prediction.

# matgl locks its backend on first import. M3GNet requires the DGL backend,
# so we must set this before matgl is ever imported in this process.
import os as _os

_os.environ.setdefault("MATGL_BACKEND", "dgl")
