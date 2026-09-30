from pathlib import Path
import sys

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))

from s6_simulation import main

if __name__ == "__main__":
    main()
