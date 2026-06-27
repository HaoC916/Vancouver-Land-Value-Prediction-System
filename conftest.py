import sys
from pathlib import Path

# Ensure the repo root is importable (so `import src...` works) and is the cwd
# the predictor's relative artifact paths resolve against.
ROOT = Path(__file__).resolve().parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
