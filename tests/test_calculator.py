"""Pytest suite for the calculator module's add() and multiply()."""

import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from calculator import add, multiply


def test_add_positive_integers():
    assert add(2, 3) == 5


def test_add_negative_and_positive():
    assert add(-1, 1) == 0


def test_add_zero():
    assert add(0, 0) == 0
    assert add(5, 0) == 5


def test_add_floats():
    assert add(1.5, 2.5) == 4.0


def test_multiply_positive_integers():
    assert multiply(4, 5) == 20


def test_multiply_by_zero():
    assert multiply(0, 100) == 0
    assert multiply(100, 0) == 0


def test_multiply_negative():
    assert multiply(-3, 4) == -12


def test_multiply_floats():
    assert multiply(2.5, 4) == 10.0
