import unittest
from utils.validation import Validation


class TestRequireKeys(unittest.TestCase):
    def test_pass(self):
        Validation.require_keys({"a": 1}, ["a"])

    def test_fail(self):
        with self.assertRaises(ValueError):
            Validation.require_keys({}, ["a"])


class TestRequirePositive(unittest.TestCase):
    def test_pass(self):
        Validation.require_positive(1, name="x")

    def test_fail(self):
        with self.assertRaises(ValueError):
            Validation.require_positive(0, name="x")


if __name__ == "__main__":
    unittest.main()