"""Tests for the hello() function in hello.py."""

from hello import hello


def test_hello_returns_hello_world() -> None:
    assert hello() == "hello world"


def test_hello_returns_str() -> None:
    assert isinstance(hello(), str)
