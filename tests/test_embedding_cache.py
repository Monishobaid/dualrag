import tempfile
import unittest
try:
    import numpy as np
    from src.retrieval.index import EmbeddingCache
except ImportError:
    np=None

@unittest.skipIf(np is None,'numpy unavailable locally')
class CacheTests(unittest.TestCase):
    def test_same_text_different_models_do_not_share_vectors(self):
        with tempfile.TemporaryDirectory() as folder:
            first=EmbeddingCache(folder,'model-a');first.put('same',np.array([1.,0.]));first.save()
            second=EmbeddingCache(folder,'model-b')
            self.assertIsNone(second.get('same'))
            reloaded=EmbeddingCache(folder,'model-a')
            np.testing.assert_array_equal(reloaded.get('same'),[1.,0.])

if __name__=='__main__':unittest.main()
