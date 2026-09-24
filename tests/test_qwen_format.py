"""Verify the actual Qwen wrapper without downloading or loading a checkpoint."""
import unittest
try:
    import torch
    from src.llm.qwen import QwenModel
except ImportError:
    torch=None

@unittest.skipIf(torch is None,'torch/transformers available on Renku only')
class QwenFormatTests(unittest.TestCase):
    def setUp(self):
        class Batch(dict):
            def to(self,device):return self
        class Tokenizer:
            def apply_chat_template(self,messages,**kwargs):
                self.messages=messages;return 'chat'
            def __call__(self,prompt,**kwargs):
                self.prompt=prompt;self.kwargs=kwargs
                return Batch(input_ids=torch.ones((1,6),dtype=torch.long))
        self.wrapper=QwenModel.__new__(QwenModel)
        self.wrapper.tokenizer=Tokenizer()
        self.wrapper.model=type('Device',(),{'device':'cpu'})()
        self.wrapper.max_input_tokens=8
        self.wrapper.use_chat_template=True
    def test_real_roles_and_no_silent_cut(self):
        self.wrapper.prepare_inputs('question','system')
        self.assertEqual([x['role'] for x in self.wrapper.tokenizer.messages],['system','user'])
        self.assertNotIn('truncation',self.wrapper.tokenizer.kwargs)
    def test_over_budget_fails_visibly(self):
        self.wrapper.max_input_tokens=5
        with self.assertRaises(ValueError):self.wrapper.prepare_inputs('question')
    def test_legacy_control_preserved(self):
        self.wrapper.use_chat_template=False
        self.wrapper.prepare_inputs('question','system')
        self.assertEqual(self.wrapper.tokenizer.prompt,'system\n\nquestion')
        self.assertEqual(self.wrapper.tokenizer.kwargs['max_length'],4096)

if __name__=='__main__':unittest.main()
