"""A verifier must actually receive the candidate it is asked to verify."""
from src.data.hotpot_loader import HotpotQAExample
from scripts.run_verification_experiment import run_one


def test_verifier_gets_claims_and_reports_malformed_json():
    class Ranker:
        def rank(self, question, paragraphs, top_k):
            return [{'paragraph': p} for p in paragraphs]
    class LLM:
        last_generation = {'hit_token_limit': False}
        prompts = []
        def generate_with_token_count(self, prompt, **kwargs):
            self.prompts.append(prompt)
            if len(self.prompts) == 1:
                return '{"answer":"Ada","answer_type":"entity","operation":"lookup","claims":[{"text":"Ada wrote it.","evidence_ids":["p0s0"]}],"explanation":"Ada wrote it."}', 40
            return 'malformed verifier response', 5
    llm=LLM()
    ex=HotpotQAExample('x','Who wrote it?','Ada',[['Doc',['Ada wrote it.']]],[['Doc',0]])
    row=run_one(ex,'verified',Ranker(),llm,192)
    assert 'CANDIDATE_JSON' in llm.prompts[1]
    assert '"text": "Ada wrote it."' in llm.prompts[1]
    assert row['calls'][1]['parse_failed']
    assert not row['calls'][1]['strict_json']
    assert row['verification']['any_uncertain']
    assert row['final_answer']=='Ada'
