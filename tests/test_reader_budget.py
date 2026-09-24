import unittest
from scripts.research_experiment import run_reader

class Ranker:
    def rank(self,query,paragraphs,top_k):
        return [{'paragraph':x} for x in list(reversed(paragraphs))[:top_k]]
class LLM:
    last_generation={'input_tokens':10,'hit_token_limit':False}
    def generate_with_token_count(self,*args,**kwargs):
        return '{"answer":"a","evidence_ids":["p9s0"],"explanation":"supported"}',10

class BudgetTests(unittest.TestCase):
    def setUp(self):
        self.paras=[{'_pid':i,'title':str(i),'text':'a','sentences':['a']} for i in range(10)]
    def test_fixed_controls_use_one_call_with_explicit_budget(self):
        for arm,count in [('top5',5),('top9',9),('ranked_all',10)]:
            r=run_reader('question',self.paras,arm,Ranker(),LLM())
            self.assertEqual(len(r['calls']),1)
            self.assertEqual(len(r['visible_titles']),count)
            self.assertEqual(r['visible_titles'][0],'9')
    def test_all_context_keeps_dataset_order(self):
        r=run_reader('question',self.paras,'all_context',Ranker(),LLM())
        self.assertEqual(r['visible_titles'],[str(i) for i in range(10)])

if __name__=='__main__':unittest.main()
