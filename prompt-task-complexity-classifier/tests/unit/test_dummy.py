#!/usr/bin/env python3
"""Dummy unit test to ensure CI can run unit tests"""


def test_dummy():
    """Dummy test that always passes"""
    assert True


def test_basic_math():
    """Basic math test"""
    assert 2 + 2 == 4


def test_string_operations():
    """Basic string operations test"""
    test_string = "hello world"
    assert test_string.upper() == "HELLO WORLD"
    assert len(test_string) == 11
