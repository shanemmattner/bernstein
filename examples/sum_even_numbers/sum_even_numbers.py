def sum_even_numbers(numbers: list[int]) -> int:
    """Return the sum of all even integers in `numbers`.

    Empty input returns 0. Zero is considered even.
    """
    return sum(n for n in numbers if n % 2 == 0)
