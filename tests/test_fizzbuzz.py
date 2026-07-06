"""Pytest suite for the fizzbuzz() function."""

import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from fizzbuzz import fizzbuzz


def _lines(capsys) -> list[str]:
    fizzbuzz()
    captured = capsys.readouterr()
    return captured.out.strip().splitlines()


def test_prints_100_lines(capsys):
    assert len(_lines(capsys)) == 100


def test_first_number_is_1(capsys):
    assert _lines(capsys)[0] == "1"


def test_line_3_is_fizz(capsys):
    assert _lines(capsys)[2] == "Fizz"


def test_line_5_is_buzz(capsys):
    assert _lines(capsys)[4] == "Buzz"


def test_line_15_is_fizzbuzz(capsys):
    assert _lines(capsys)[14] == "FizzBuzz"


def test_line_30_is_fizzbuzz(capsys):
    assert _lines(capsys)[29] == "FizzBuzz"


def test_line_100_is_buzz(capsys):
    assert _lines(capsys)[99] == "Buzz"


def test_no_fizz_for_non_multiples(capsys):
    assert _lines(capsys)[6] == "7"
