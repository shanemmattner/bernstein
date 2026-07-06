from examples.sum_even_numbers.sum_even_numbers import sum_even_numbers


def test_empty_list_returns_zero():
    assert sum_even_numbers([]) == 0


def test_all_even():
    assert sum_even_numbers([2, 4, 6]) == 12


def test_all_odd():
    assert sum_even_numbers([1, 3, 5]) == 0


def test_mixed_even_and_odd():
    assert sum_even_numbers([1, 2, 3, 4, 5]) == 6


def test_negative_evens_included():
    assert sum_even_numbers([-2, -4, 1, 3]) == -6


def test_zero_counts_as_even():
    assert sum_even_numbers([0, 1, 3]) == 0


def test_single_even_element():
    assert sum_even_numbers([4]) == 4


def test_single_odd_element():
    assert sum_even_numbers([7]) == 0
