"""Research integrity tests: corpus isolation, addressing, ranking and coverage."""
import bz2
import io
import json
import tarfile
import pytest
from src.retrieval.corpus import Corpus, build_corpus, context_documents, fuse, retrieval_metrics, wiki_documents


def corpus(tmp_path, docs, words=120):
    path = tmp_path / 'index.sqlite'
    build_corpus(path, docs, 'test', {}, words)
    return Corpus(path)


def test_index_does_not_read_answer_or_question(tmp_path):
    path = tmp_path / 'input.json'
    path.write_text(json.dumps([{'question':'secretquestion','answer':'secretanswer',
        'supporting_facts':[['Gold',0]],'context':[['Other',['A harmless sentence.']]]}]))
    c = corpus(tmp_path, context_documents([path]))
    assert c.lexical('secretquestion secretanswer Gold') == []
    assert c.lexical('harmless')


def test_dedup_does_not_merge_same_title_versions(tmp_path):
    c = corpus(tmp_path,[{'title':'T','sentences':['First.']},
                       {'title':'T','sentences':['First.']},
                       {'title':'T','sentences':['Second.']}])
    assert c.metadata['documents'] == 2
    assert c.metadata['chunks'] == 2


def test_sentence_offsets_and_context_are_not_reader_evidence(tmp_path):
    c = corpus(tmp_path,[{'title':'Title','sentences':['Lead information.', 'Later target.', 'Final fact.']}],words=2)
    chunks = list(c.chunks())
    assert [x['start'] for x in chunks] == [0,1,2]
    assert 'Lead information.' in chunks[2]['contextual']
    assert chunks[2]['sentences'] == ['Final fact.']
    assert len(c.lexical('Title', contextual=True)) == 3
    assert c.lexical('Title', contextual=False) == []


def test_metrics_reject_wrong_text_same_title(tmp_path):
    c = corpus(tmp_path,[{'title':'T','sentences':['Actual support.']},
                       {'title':'T','sentences':['Wrong version.']}])
    ex = {'context':[['T',['Actual support.']]],'supporting_facts':[['T',0]]}
    m = retrieval_metrics(ex,[c.get(2)],c)
    assert m['title_recall'] == 1
    assert m['sentence_recall'] == 0
    assert not m['complete_support']
    assert m['corpus_support_availability'] == 1
    assert retrieval_metrics(ex,[c.get(1)],c)['complete_support']


def test_fusion_uses_ranks_not_score_magnitudes():
    a = fuse([[(2,999),(1,12)],[(1,-20),(3,-90)]])
    assert [x[0] for x in a] == [1,2,3]
    assert a == fuse([[(2,1),(1,0)],[(1,1),(3,0)]])


def test_fts_query_operators_are_data(tmp_path):
    c = corpus(tmp_path,[{'title':'T','sentences':['Quoted corpus evidence.']}])
    assert c.lexical('" OR * (NEAR) corpus')
    assert c.lexical('!!!') == []


def test_refuse_overwrite(tmp_path):
    c = corpus(tmp_path,[{'title':'T','sentences':['Text.']}])
    with pytest.raises(FileExistsError):
        build_corpus(c.path, [], 'test', {})


def test_fullwiki_stream_does_not_extract_paths(tmp_path):
    path = tmp_path / 'archive.tar.bz2'
    data = bz2.compress((json.dumps({'title':'Title','text':['Sentence.']})+'\n').encode())
    with tarfile.open(path,'w:bz2') as tar:
        member = tarfile.TarInfo('../../outside.bz2'); member.size = len(data)
        tar.addfile(member,io.BytesIO(data))
    assert list(wiki_documents(path)) == [{'title':'Title','sentences':['Sentence.']}]
    assert not (tmp_path.parent/'outside.bz2').exists()


def test_dense_exact_search_with_nonconsecutive_ids(tmp_path):
    import numpy as np
    from src.retrieval.dense_corpus import DenseCorpus
    index = DenseCorpus.__new__(DenseCorpus)
    index.ids = np.array([3,7,12])
    index.vectors = np.array([[1,0],[0,1],[1,0]], dtype=float)
    assert index.search_vector(np.array([1,0]),2) == [(3,1.0),(12,1.0)]


def test_reader_evidence_keeps_global_addresses_and_excludes_prefix():
    from scripts.read_corpus_evidence import evidence_prompt
    prompt,sources=evidence_prompt('Question?', [{'id':7,'title':'T','start':3,
        'sentences':['Original text.'],'contextual':'Invented indexing-only text.'}])
    assert 'c7s3' in sources
    assert sources['c7s3']['sentence_index'] == 3
    assert 'Invented' not in prompt
    assert 'Original text.' in prompt
