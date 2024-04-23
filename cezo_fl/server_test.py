from cezo_fl.server import SeedAndGradientRecords

import pytest


def test_seed_records():
    sr = SeedAndGradientRecords()

    sr.add_records([1, 2, 3], [None] * 3)  # iter 0
    sr.add_records([2, 3, 4], [None] * 3)  # iter 1
    assert sr.fetch_seed_records(earliest_record_needs=0) == [[1, 2, 3], [2, 3, 4]]

    sr.add_records([3, 4, 5], [None] * 3)  # iter 2
    sr.remove_too_old(earliest_record_needs=1)
    assert sr.fetch_seed_records(earliest_record_needs=2) == [[3, 4, 5]]
    assert sr.fetch_seed_records(earliest_record_needs=3) == []
