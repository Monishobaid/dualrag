#!/usr/bin/env python3
"""Restore the standard dev file from the HotpotQA Hugging Face distribution.

The CMU source can be unreachable. Require canonical equality to our previously
frozen dataset before permitting any matched run; do not silently change data.
"""
import argparse
import hashlib
import json
from pathlib import Path

CANONICAL_SHA256 = '8c45526920c1b11d0f16d59775c8a53b7530a48a6309f423c8f79cecbe9e8b3c'


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--output',default='data/hotpotqa/hotpot_dev_distractor_v1.json')
    a=p.parse_args()
    from huggingface_hub import HfApi, hf_hub_download
    import pyarrow.parquet as pq
    revision=HfApi().dataset_info('hotpotqa/hotpot_qa').sha
    path=hf_hub_download('hotpotqa/hotpot_qa', 'distractor/validation-00000-of-00001.parquet',
                         repo_type='dataset',revision=revision)
    records=pq.read_table(path).to_pylist()
    data=[]
    for r in records:
        context=r['context']; support=r['supporting_facts']
        data.append({'_id':r['id'],'question':r['question'],'answer':r['answer'],
            'type':r['type'],'level':r['level'],
            'context':list(map(list,zip(context['title'],context['sentences']))),
            'supporting_facts':list(map(list,zip(support['title'],support['sent_id'])))})
    # Match the canonical hashing convention used by the September 24 audit.
    canonical=json.dumps(data,sort_keys=True,separators=(',',':'),ensure_ascii=False).encode()
    actual=hashlib.sha256(canonical).hexdigest()
    if actual!=CANONICAL_SHA256:
        raise ValueError(f'Canonical dataset mismatch: {actual}; do not run a matched comparison')
    out=Path(a.output)
    if out.exists():raise FileExistsError(out)
    out.parent.mkdir(parents=True,exist_ok=True)
    out.write_text(json.dumps(data,ensure_ascii=False))
    print(json.dumps({'records':len(data),'hf_revision':revision,'canonical_sha256':actual,
                     'file_sha256':hashlib.sha256(out.read_bytes()).hexdigest()},indent=2))


if __name__=='__main__':main()
