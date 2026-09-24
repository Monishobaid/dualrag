import unittest
from src.rag.sufficiency import evidence_view, parse_object, sufficient_decision, short_answer

class SufficiencyTests(unittest.TestCase):
    def setUp(self):
        self.text,self.sources=evidence_view([{'_pid':2,'title':'Film','text':'A. B.', 'sentences':['A.','B.']}])
    def test_stable_ids_when_order_changes(self):
        self.assertEqual(self.sources['p2s1']['sentence_index'],1)
    def test_false_string_cannot_stop(self):
        self.assertFalse(sufficient_decision({'sufficient':'false','evidence_ids':['p2s0']},self.sources))
    def test_missing_relation_or_bad_id_cannot_stop(self):
        self.assertFalse(sufficient_decision({'sufficient':True,'evidence_ids':['p2s0'],'missing_information':'birthplace'},self.sources))
        self.assertFalse(sufficient_decision({'sufficient':True,'evidence_ids':['p3s0']},self.sources))
    def test_valid_structural_stop(self):
        self.assertTrue(sufficient_decision({'sufficient':True,'evidence_ids':['p2s0'],'missing_information':''},self.sources))
    def test_truncated_json_not_recovered_as_success(self):
        self.assertEqual(parse_object('{"answer":"3"'),(None,False))
    def test_fenced_json_is_non_strict(self):
        self.assertEqual(parse_object('```json\n{"answer":"3"}\n```'),({'answer':'3'},False))
    def test_display_brackets_are_supported(self):
        self.assertTrue(sufficient_decision({'sufficient':True,'evidence_ids':['[p2s0]']},self.sources))
    def test_numeric_and_initials_are_preserved(self):
        self.assertEqual(short_answer({'answer':'1'}),'1')
        self.assertEqual(short_answer({'answer':'J. K. Rowling'}),'J. K. Rowling')
    def test_non_string_not_cast_to_answer(self):
        self.assertEqual(short_answer({'answer':['one','two']}),'')

if __name__=='__main__': unittest.main()
