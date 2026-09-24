import unittest
from scripts.research_experiment import score
from scripts.paired_research_analysis import compare

class PairedTests(unittest.TestCase):
    def test_special_answers(self):
        self.assertEqual(score('yes indeed','yes')['f1'],0)
        self.assertEqual(score('The dog','dog')['em'],1)
    def test_mismatched_ids_rejected(self):
        with self.assertRaises(ValueError): compare({'a':{}},{'b':{}})
    def test_gold_mismatch_rejected(self):
        with self.assertRaises(ValueError): compare({'a':{'gold_answer':'yes'}},{'a':{'gold_answer':'no'}})

if __name__=='__main__':unittest.main()
