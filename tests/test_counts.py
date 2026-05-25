import unittest

from persona_eval import expected_call_count


class CountTests(unittest.TestCase):
    def test_sample_count(self):
        self.assertEqual(expected_call_count(10, 6, 2, 1), 120)

    def test_smoke_count(self):
        self.assertEqual(expected_call_count(20, 6, 2, 1), 240)

    def test_dev_count(self):
        self.assertEqual(expected_call_count(50, 6, 2, 2), 1200)

    def test_full_count(self):
        self.assertEqual(expected_call_count(200, 6, 2, 2), 4800)


if __name__ == "__main__":
    unittest.main()
