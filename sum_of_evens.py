from __future__ import annotations


def sum_of_evens(numbers: list[int]) -> int:
    """Return the sum of all even integers in the input list.

    Args:
        numbers: A list of integers to filter and sum.

    Returns:
        The sum of even integers; 0 if none exist or the list is empty.
    """
    return sum(n for n in numbers if n % 2 == 0)
