import sys
from pathlib import Path

# Make "src" importable as a package regardless of where pytest is invoked from.
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
