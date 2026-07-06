"""Tests for the greet_world() function in greet_world.py."""

from greet_world import greet_world


def test_greet_world_returns_expected_string() -> None:
    assert greet_world("Alice") == "Hello, Alice! Welcome to the world."


def test_greet_world_with_different_name() -> None:
    assert greet_world("World") == "Hello, World! Welcome to the world."


def test_greet_world_returns_str() -> None:
    assert isinstance(greet_world("X"), str)


def test_greet_world_with_empty_string() -> None:
    assert greet_world("") == "Hello, ! Welcome to the world."
