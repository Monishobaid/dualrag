import os
import re

files_in_order = [
    'src/utils/text.py',
    'src/utils/logging.py',
    'src/eval/metrics.py',
    'src/config.py',
    'src/llm/prompts.py',
    'src/llm/azure_openai.py',
    'src/llm/qwen.py',
    'src/llm/generation.py',
    'src/retrieval/embedder.py',
    'src/retrieval/index.py',
    'src/retrieval/ranker.py',
    'src/llm/early_stop.py',
    'src/rag/stopping.py',
    'src/rag/iterative_rag.py',
    'src/rag/baselines.py',
    'src/data/hotpot_loader.py',
    'src/eval/hotpot_eval.py',
    'scripts/run_experiment.py'
]

output_file = 'colab_dual_stopping_rag.py'

imports = set()
code_blocks = []

def process_file(filepath):
    if not os.path.exists(filepath):
        print(f"Skipping {filepath} - not found")
        return
        
    with open(filepath, 'r') as f:
        lines = f.readlines()
        
    code_lines = []
    
    for line in lines:
        stripped = line.strip()
        
        # Skip local imports
        if stripped.startswith('from src.') or stripped.startswith('import src.'):
            continue
            
        # Collect standard/third-party imports
        if stripped.startswith('import ') or stripped.startswith('from '):
            if not ('__main__' in stripped or stripped.startswith('from . ')):
                imports.add(stripped)
                continue
                
        code_lines.append(line)
        
    code_blocks.append(f"\n# {'='*40}\n# From: {filepath}\n# {'='*40}\n")
    code_blocks.append("".join(code_lines))

for f in files_in_order:
    process_file(f)
    
with open(output_file, 'w') as f:
    f.write("# Colab-ready single file for Dual-Stopping RAG\n\n")
    f.write("# --- IMPORTS ---\n")
    for imp in sorted(list(imports)):
        f.write(imp + '\n')
    f.write("\n\n# --- CODE ---\n")
    for block in code_blocks:
        f.write(block)
        
print(f"Successfully wrote everything to {output_file}")
