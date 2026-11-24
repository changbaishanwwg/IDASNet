"""Train ResNet1D on the reference IDASNet outer folds (Table S2C)."""
from pathlib import Path
import sys
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from idasnet.evaluate import main
if __name__ == "__main__":
    main("ResNet1D")
