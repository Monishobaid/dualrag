# Embedding input-length audit

Post-run CPU audit using the tokenizers and encoder limits at the recorded checkpoint revisions. This does not change any retrieval result or prompt. Overflow means the encoder cannot represent the complete paragraph; it does not establish whether the annotated supporting sentence lies in the removed tail.

| Model | Title prefix | Cohort | Encoder limit | Paragraphs | Overflow | Gold-title paragraphs overflowing | Maximum length |
|---|---|---|---:|---:|---:|---:|---:|
| MiniLM | False | original50 | 256 | 500 | 17 | 0 | 703 |
| MiniLM | False | confirmation100 | 256 | 1000 | 41 | 0 | 1509 |
| MiniLM | True | original50 | 256 | 500 | 18 | 0 | 707 |
| MiniLM | True | confirmation100 | 256 | 1000 | 44 | 1 | 1515 |
| BGE-base | True | original50 | 512 | 500 | 2 | 0 | 707 |
| BGE-base | True | confirmation100 | 512 | 1000 | 3 | 0 | 1515 |

Both title-prefix variants on confirmation100 are tokenizer audits only; no title-prefix or BGE generation benchmark was run on that cohort. Reader context uses original complete sentences independently of embedding truncation. A model upgrade changes both representation and encoder length, so its gain cannot be attributed solely to one of those factors.

None of the gold-title paragraphs overflowed in the actual plain-MiniLM original50/confirmation100 or BGE original50 configurations. Thus this audit does not support blaming their retrieval misses on truncated gold paragraph tails; long distractor representations can still be affected.

Sources: [MiniLM model card](https://huggingface.co/sentence-transformers/all-MiniLM-L6-v2), [BGE model card](https://huggingface.co/BAAI/bge-base-en-v1.5).
