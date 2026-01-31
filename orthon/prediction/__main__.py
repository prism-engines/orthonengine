"""
Enable running the prediction module directly.

Usage:
    python -m orthon.prediction rul /path/to/prism/output
    python -m orthon.prediction health /path/to/prism/output
    python -m orthon.prediction anomaly /path/to/prism/output
"""

import sys
from .cli import main

if __name__ == "__main__":
    sys.exit(main())
