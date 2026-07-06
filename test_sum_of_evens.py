from sum_of_evens import sum_of_evens


def test_empty_list_returns_zero():
    assert sum_of_evens([]) == 0


def test_all_odd_returns_zero():
    assert sum_of_evens([1, 3, 5, 7]) == 0


def test_all_even_sums_correctly():
    assert sum_of_evens([2, 4, 6]) == 12


def test_mixed_sums_only_evens():
    assert sum_of_evens([1, 2, 3, 4, 5]) == 6


def test_negative_evens_included():
    assert sum_of_evens([-2, -4, 1, 3]) == -6


def test_zero_is_even():
    assert sum_of_evens([0, 1]) == 0


def test_single_even_element():
    assert sum_of_evens([4]) == 4


def test_single_odd_element():
    assert sum_of_evens([7]) == 0
