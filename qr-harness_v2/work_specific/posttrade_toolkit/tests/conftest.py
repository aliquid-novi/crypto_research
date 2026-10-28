import os
import sys

import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from posttrade import reference, synthetic  # noqa: E402


@pytest.fixture(scope="session")
def day():
    """8 hours of synthetic data (shared by all tests). Returns (tables, truth)."""
    return synthetic.generate_day(synthetic.SynthConfig(hours=8))


@pytest.fixture(scope="session")
def clean(day):
    tables, truth = day
    f = synthetic.clean_fills(tables["fills"], truth)
    return f


@pytest.fixture(scope="session")
def mid_ref(day):
    return reference.mid_ref(day[0]["book_l1"])


@pytest.fixture(scope="session")
def quote_events(clean):
    from posttrade import markouts

    f = clean[clean["fill_id"].str.startswith("F")]
    return f, markouts.aggregate_events(f)
