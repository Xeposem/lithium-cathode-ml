"""Enable running the cathode ML pipeline as a module.

Usage:
    python -m cathode_ml
    python -m cathode_ml --help
    python -m cathode_ml --skip-fetch --models rf cgcnn
"""

from cathode_ml.pipeline import main

if __name__ == "__main__":
    main()
