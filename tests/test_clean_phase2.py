import unittest

from src.data.clean_phase2 import _clean_caption


class CaptionCleaningTests(unittest.TestCase):
    def test_length_changing_fix_does_not_shift_later_matches(self):
        cleaned, counts = _clean_caption("bil whtie and larsus")

        self.assertEqual(cleaned, "bill white and tarsus")
        self.assertEqual(counts["typo:bil->bill"], 1)
        self.assertEqual(counts["typo:whtie->white"], 1)
        self.assertEqual(counts["typo:larsus->tarsus"], 1)


if __name__ == "__main__":
    unittest.main()
