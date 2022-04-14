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
    parser.addoption(
        "--fcompare", action="store_true", default=False,
        help="show failed comparison",
    )
    parser.addoption(
        "--atol", type=float, default=0,
        help="absolute tolerance, by default 0",
    )
    parser.addoption(
        "--rtol", type=float, default=1e-8,
        help="relative tolerance, by default 1e-8",
    )


def pytest_collection_modifyitems(config, items):
    if not config.getoption("--diffbird"):
        skip = pytest.mark.skip(reason="need --diffbird option to run")
        for item in items:
            if "diffbird" in item.keywords:
                item.add_marker(skip)
    if not config.getoption("--fcompare"):
        skip = pytest.mark.skip(reason="need --fcompare option to run")
        for item in items:
            if "fcompare" in item.keywords:
                item.add_marker(skip)


@pytest.fixture(scope='session')
def atol(request):
    yield request.config.getoption("--atol")


@pytest.fixture(scope='session')
def rtol(request):
    yield request.config.getoption("--rtol")


@pytest.fixture(scope='session')
def yamlroot():
    yield Path(__file__).parent / 'yamls'