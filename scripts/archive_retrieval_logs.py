#!/usr/bin/env python3
"""Preserve full retrieval rankings and original evidence as compressed JSONL."""
import argparse
import gzip
import hashlib
import json
from pathlib import Path
import shutil


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--run', required=True)
    p.add_argument('--output', required=True)
    a = p.parse_args()
    root, out = Path(a.run), Path(a.output)
    protocol = json.loads((root/'protocol.json').read_text())
    if protocol['status'] != 'complete':
        raise ValueError('Incomplete retrieval run')
    out.mkdir(parents=True, exist_ok=True)
    manifest = {}
    for arm in protocol['arms_completed']:
        source, dest = root/(arm+'.jsonl'), out/(arm+'.jsonl.gz')
        if dest.exists():
            raise FileExistsError(dest)
        with source.open('rb') as src, dest.open('wb') as dst:
            with gzip.GzipFile(filename='', fileobj=dst, mode='wb', mtime=0) as compressed:
                shutil.copyfileobj(src, compressed)
        manifest[dest.name] = {'compressed_sha256': hashlib.sha256(dest.read_bytes()).hexdigest(),
            'uncompressed_sha256': hashlib.sha256(source.read_bytes()).hexdigest()}
    (out/'raw_trace_sha256.json').write_text(json.dumps(manifest, indent=2))


if __name__ == '__main__':
    main()
