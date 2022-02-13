import pytest
from pathlib import Path

CACHEPATH = Path(__file__).parent.parent / 'cobaya' / 'cache'
if not CACHEPATH.exists():
    CACHEPATH.mkdir()


def pytest_addoption(parser):
    parser.addoption(
        "--diffbird", action="store_true", default=False,
        help="mark test to enable comparison to pybird_dev"
    )


def pytest_collection_modifyitems(config, items):
    if config.getoption("--diffbird"):
        return
    skip = pytest.mark.skip(reason="need --diffbird option to run")
    for item in items:
        if "diffbird" in item.keywords:
            item.add_marker(skip)
