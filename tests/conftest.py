from pathlib import Path

CACHEPATH = Path(__file__).parent.parent / 'cobaya' / 'cache'
if not CACHEPATH.exists():
    CACHEPATH.mkdir()