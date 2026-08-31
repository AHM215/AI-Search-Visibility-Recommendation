from __future__ import annotations

from pathlib import Path

import pytest


def pytest_addoption(parser: pytest.Parser) -> None:
    parser.addoption(
        "--gold-database",
        action="store",
        default=None,
        help="Path to a recorded corpus database for the opt-in Gold Set suite.",
    )


def pytest_configure(config: pytest.Config) -> None:
    config.addinivalue_line(
        "markers", "gold_set: opt-in Judge agreement suite; needs --gold-database"
    )


def pytest_collection_modifyitems(
    config: pytest.Config, items: list[pytest.Item]
) -> None:
    if config.getoption("--gold-database"):
        return
    skip = pytest.mark.skip(reason="Gold Set suite is opt-in: pass --gold-database <corpus.db>")
    for item in items:
        if "gold_set" in item.keywords:
            item.add_marker(skip)


@pytest.fixture
def gold_database(request: pytest.FixtureRequest) -> Path:
    value = request.config.getoption("--gold-database")
    if value is None:
        pytest.skip("needs --gold-database")
    return Path(str(value))
